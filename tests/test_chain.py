"""Testa o fluxo da chain com LLM e busca simulados (sem rede, sem chave)."""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from rag_juridico import chain

DOC = Document(
    page_content="[EAOAB, Art. 25] Estatuto\nArt. 25. Prescreve em cinco anos a ação de cobrança de honorários.",
    metadata={
        "chunk_id": "EAOAB:25", "ref": "EAOAB:25", "article": "25", "article_label": "Art. 25",
        "source_short": "EAOAB", "source_title": "Estatuto", "source_url": "http://exemplo",
    },
)


def _run(monkeypatch, model_output: str, score: float = 0.9):
    monkeypatch.setattr(chain, "retrieve", lambda q: [(DOC, score)])
    monkeypatch.setattr(
        chain, "build_llm", lambda model=None: GenericFakeChatModel(messages=iter([AIMessage(content=model_output)]))
    )
    return chain.ask("Prazo de cobrança de honorários?", log=False)


def test_grounded_answer_is_returned_with_citation(monkeypatch):
    ans = _run(monkeypatch, '{"answerable": true, "answer": "Cinco anos (art. 25 do Estatuto).", "citations": ["T1"]}')
    assert ans.status == "respondida"
    assert [c["ref"] for c in ans.citations] == ["EAOAB:25"]


def test_json_inside_code_fence_is_accepted(monkeypatch):
    ans = _run(monkeypatch, '```json\n{"answerable": true, "answer": "Cinco anos (art. 25).", "citations": ["T1"]}\n```')
    assert ans.status == "respondida"


def test_invented_article_is_blocked(monkeypatch):
    ans = _run(monkeypatch, '{"answerable": true, "answer": "Dez anos, art. 205 do Estatuto.", "citations": ["T1"]}')
    assert ans.status == "bloqueada_guardrail"
    assert ans.answer == chain.REFUSAL_MESSAGE


def test_model_refusal(monkeypatch):
    ans = _run(monkeypatch, '{"answerable": false, "answer": "Sem base.", "citations": []}')
    assert ans.status == "recusada_pelo_modelo"


def test_invalid_json_is_treated_as_refusal(monkeypatch):
    ans = _run(monkeypatch, "Claro! São cinco anos.")
    assert ans.refused
    assert "resposta do modelo não é JSON válido" in ans.guardrail_violations


def test_low_similarity_skips_llm(monkeypatch):
    ans = _run(monkeypatch, "não deveria ser chamado", score=0.5)
    assert ans.status == "recusada_sem_contexto"
    assert ans.raw_model_answer is None
