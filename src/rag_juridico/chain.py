"""Chain de pergunta e resposta: recuperação, portão de relevância, LLM e guardrails.

Fluxo:
1. Recupera os TOP_K trechos mais similares.
2. Se o melhor trecho tiver similaridade < MIN_SCORE, recusa sem chamar o LLM.
3. O LLM responde em JSON, citando os ids dos trechos, ou declara que não há base.
4. Guardrails determinísticos validam as citações e os artigos mencionados.
5. Tudo é registrado em logs/interactions.jsonl.
"""

import json
import time
from dataclasses import asdict, dataclass, field

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import settings
from .guardrails import check_answer
from .logbook import log_interaction
from .vectorstore import retrieve

REFUSAL_MESSAGE = (
    "Não encontrei base no material para responder com segurança. "
    "Reformule a pergunta ou consulte um monitor."
)

SYSTEM_PROMPT = """Você é um assistente de estudos para a 1ª fase da OAB. Responde dúvidas \
de alunos usando exclusivamente os TRECHOS fornecidos (Estatuto da Advocacia e da OAB, \
Código de Ética e Disciplina e Constituição Federal).

Por que as regras abaixo importam: o aluno vai estudar e fazer a prova com base na sua \
resposta. Um artigo inventado ou um prazo errado é pior do que não responder.

Regras:
- Use somente informações presentes nos TRECHOS. Não use conhecimento externo, nem para \
completar lacunas.
- Cite os trechos que fundamentam a resposta pelos ids entre colchetes (ex.: T1, T3).
- No texto da resposta, indique a norma e o artigo (ex.: "art. 25 do Estatuto da OAB").
- Se os TRECHOS não respondem à pergunta, ou só respondem parcialmente de um jeito que \
poderia induzir o aluno a erro, marque answerable como false. Não tente adivinhar.
- Seja direto: 2 a 6 frases, linguagem clara, em português do Brasil.

Responda apenas com um objeto JSON, sem texto antes ou depois, neste formato:
{{"answerable": true | false, "answer": "resposta ou motivo de não responder", \
"citations": ["T1", "T2"]}}"""

HUMAN_PROMPT = """TRECHOS:
{context}

PERGUNTA DO ALUNO: {question}"""


@dataclass
class Answer:
    question: str
    status: str  # respondida | recusada_sem_contexto | recusada_pelo_modelo | bloqueada_guardrail
    answer: str
    citations: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    guardrail_violations: list[str] = field(default_factory=list)
    raw_model_answer: str | None = None
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0
    model: str = ""

    @property
    def refused(self) -> bool:
        return self.status != "respondida"

    def to_dict(self) -> dict:
        return asdict(self) | {"refused": self.refused}


def _format_context(hits: list[tuple[Document, float]]) -> tuple[str, dict[str, Document]]:
    by_tid, blocks = {}, []
    for i, (doc, _) in enumerate(hits, start=1):
        tid = f"T{i}"
        by_tid[tid] = doc
        blocks.append(f"[{tid}] {doc.page_content}")
    return "\n\n".join(blocks), by_tid


def _cite(doc: Document) -> dict:
    m = doc.metadata
    return {
        "chunk_id": m["chunk_id"],
        "ref": m["ref"],
        "label": f"{m['article_label']} ({m['source_short']})",
        "source_title": m["source_title"],
        "source_url": m["source_url"],
        "text": doc.page_content,
    }


def build_llm(model: str | None = None) -> ChatAnthropic:
    return ChatAnthropic(model=model or settings.llm_model, max_tokens=4000, max_retries=3)


_prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
_parser = JsonOutputParser()


def ask(question: str, *, log: bool = True) -> Answer:
    started = time.perf_counter()
    hits = retrieve(question)
    retrieved = [
        {"chunk_id": doc.metadata["chunk_id"], "ref": doc.metadata["ref"], "score": round(float(score), 4)}
        for doc, score in hits
    ]
    result = Answer(question=question, status="", answer="", retrieved=retrieved, model=settings.llm_model)

    best = max((score for _, score in hits), default=0.0)
    if best < settings.min_score:
        result.status = "recusada_sem_contexto"
        result.answer = REFUSAL_MESSAGE
    else:
        context, by_tid = _format_context(hits)
        message = (_prompt | build_llm()).invoke({"context": context, "question": question})
        result.raw_model_answer = message.text
        result.usage = dict(message.usage_metadata or {})
        try:
            parsed = _parser.parse(message.text)
        except Exception:
            parsed = {"answerable": False, "answer": "", "citations": []}
            result.guardrail_violations.append("resposta do modelo não é JSON válido")

        cited = [c for c in parsed.get("citations", []) if isinstance(c, str)]
        if not parsed.get("answerable"):
            result.status = "recusada_pelo_modelo"
            result.answer = REFUSAL_MESSAGE
        elif violations := check_answer(parsed.get("answer", ""), cited, by_tid):
            result.status = "bloqueada_guardrail"
            result.answer = REFUSAL_MESSAGE
            result.guardrail_violations += violations
        else:
            result.status = "respondida"
            result.answer = parsed["answer"].strip()
            result.citations = [_cite(by_tid[c]) for c in dict.fromkeys(cited)]

    result.latency_ms = int((time.perf_counter() - started) * 1000)
    if log:
        log_interaction(result.to_dict())
    return result


def format_answer(ans: Answer) -> str:
    lines = [ans.answer]
    if ans.citations:
        lines.append("")
        lines.append("Fontes:")
        lines += [f"  - {c['label']}: {c['source_title']}" for c in ans.citations]
    lines.append("")
    lines.append(f"[status: {ans.status} | {ans.latency_ms} ms]")
    if ans.guardrail_violations:
        lines.append(f"[guardrails: {json.dumps(ans.guardrail_violations, ensure_ascii=False)}]")
    return "\n".join(lines)
