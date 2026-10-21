"""Avaliação automatizada: recuperação, recusa, citação, fidelidade e acerto.

Métricas:
- hit@k (recuperação): algum artigo esperado está entre os TOP_K trechos. Roda sem LLM.
- acerto de recusa: recusou quando devia e respondeu quando devia.
- precisão de citação: a resposta citou o artigo esperado.
- fidelidade (LLM juiz): toda afirmação da resposta está apoiada nos trechos citados.
- acerto (LLM juiz): a resposta é consistente com a resposta de referência.

O juiz recebe só os trechos citados (não os 5 recuperados): uma afirmação apoiada
num trecho que não foi citado conta como não fundamentada para o aluno.
"""

import json
from datetime import datetime

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .chain import ask, build_llm
from .config import settings
from .vectorstore import retrieve

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Você avalia respostas de um assistente jurídico para estudantes da OAB.
Seja rigoroso: um prazo, número ou artigo diferente do texto conta como erro.

Avalie dois critérios de forma independente:
1. faithful: toda afirmação da RESPOSTA está apoiada nos TRECHOS CITADOS? Afirmações \
corretas no mundo real, mas ausentes dos trechos, tornam faithful = false.
2. correct: a RESPOSTA é consistente com a RESPOSTA DE REFERÊNCIA? Omitir detalhes \
secundários é aceitável; contradizer ou trocar prazos, números ou artigos não é.

Responda apenas com JSON:
{{"faithful": true | false, "unsupported_claims": ["..."], "correct": true | false, \
"reason": "uma frase"}}"""),
    ("human", """PERGUNTA: {question}

TRECHOS CITADOS:
{cited}

RESPOSTA DO ASSISTENTE:
{answer}

RESPOSTA DE REFERÊNCIA:
{reference}"""),
])


def load_dataset(limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in settings.eval_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def _judge(item: dict, answer_text: str, cited_texts: list[str]) -> dict:
    cited = "\n\n".join(cited_texts) or "(nenhum)"
    msg = (JUDGE_PROMPT | build_llm(settings.judge_model)).invoke({
        "question": item["question"],
        "cited": cited,
        "answer": answer_text,
        "reference": item["reference_answer"],
    })
    try:
        return JsonOutputParser().parse(msg.text)
    except Exception:
        return {"faithful": None, "correct": None, "reason": "juiz não retornou JSON válido"}


def _rate(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(bool(v) for v in vals) / len(vals), 3) if vals else None


def run(limit: int | None = None, retrieval_only: bool = False) -> dict:
    items = load_dataset(limit)
    rows = []
    for item in items:
        expected = set(item["expected_refs"])
        hits = retrieve(item["question"])
        retrieved_refs = [d.metadata["ref"] for d, _ in hits]
        row = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "should_refuse": item["should_refuse"],
            "expected_refs": sorted(expected),
            "retrieved_refs": retrieved_refs,
            "top_score": round(float(hits[0][1]), 4) if hits else 0.0,
            "hit_at_k": bool(expected & set(retrieved_refs)) if expected else None,
        }
        if not retrieval_only:
            ans = ask(item["question"])
            cited_refs = [c["ref"] for c in ans.citations]
            row |= {
                "status": ans.status,
                "refused": ans.refused,
                "refusal_ok": ans.refused == item["should_refuse"],
                "answer": ans.answer,
                "cited_refs": cited_refs,
                "citation_ok": bool(expected & set(cited_refs)) if expected and not ans.refused else None,
                "guardrail_violations": ans.guardrail_violations,
                "latency_ms": ans.latency_ms,
            }
            if not ans.refused:
                verdict = _judge(item, ans.answer, [c["text"] for c in ans.citations])
                row |= {
                    "faithful": verdict.get("faithful"),
                    "correct": verdict.get("correct"),
                    "unsupported_claims": verdict.get("unsupported_claims", []),
                    "judge_reason": verdict.get("reason", ""),
                }
        rows.append(row)
        print(f"  {row['id']}: hit@k={row['hit_at_k']} status={row.get('status', '-')}", flush=True)

    answerable = [r for r in rows if not r["should_refuse"]]
    summary = {
        "n": len(rows),
        "hit_at_k": _rate([r["hit_at_k"] for r in answerable]),
        "top_k": settings.top_k,
        "embedding_model": settings.embedding_model,
    }
    if not retrieval_only:
        summary |= {
            "model": settings.llm_model,
            "refusal_accuracy": _rate([r["refusal_ok"] for r in rows]),
            "false_refusals": sum(r["refused"] for r in answerable),
            "answered_out_of_scope": sum(not r["refused"] for r in rows if r["should_refuse"]),
            "citation_accuracy": _rate([r.get("citation_ok") for r in rows]),
            "faithfulness": _rate([r.get("faithful") for r in rows]),
            "correctness": _rate([r.get("correct") for r in rows]),
            "guardrail_blocks": sum(r["status"] == "bloqueada_guardrail" for r in rows),
            "min_score": settings.min_score,
        }
    _write_report(summary, rows, retrieval_only)
    return summary


def _write_report(summary: dict, rows: list[dict], retrieval_only: bool) -> None:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    kind = "retrieval" if retrieval_only else "full"
    base = settings.reports_dir / f"eval-{kind}-{stamp}"
    base.with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    lines = [f"# Avaliação ({kind}) - {stamp}", "", "## Resumo", "", "| Métrica | Valor |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary.items()]
    lines += ["", "## Casos com problema", ""]
    problems = [
        r for r in rows
        if r["hit_at_k"] is False
        or r.get("refusal_ok") is False
        or r.get("citation_ok") is False
        or r.get("faithful") is False
        or r.get("correct") is False
    ]
    if not problems:
        lines.append("Nenhum.")
    for r in problems:
        lines += [
            f"### {r['id']} ({r['category']}): {r['question']}",
            f"- esperado: {r['expected_refs'] or 'recusa'} | recuperado: {r['retrieved_refs']} (top score {r['top_score']})",
        ]
        if not retrieval_only:
            lines += [
                f"- status: {r['status']} | citou: {r['cited_refs']}",
                f"- resposta: {r['answer']}",
            ]
            if r.get("judge_reason"):
                lines.append(f"- juiz: faithful={r.get('faithful')} correct={r.get('correct')}: {r['judge_reason']}")
            if r.get("unsupported_claims"):
                lines.append(f"- afirmações sem base: {r['unsupported_claims']}")
            if r.get("guardrail_violations"):
                lines.append(f"- guardrails: {r['guardrail_violations']}")
        lines.append("")
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRelatório: {base.with_suffix('.md')}")
