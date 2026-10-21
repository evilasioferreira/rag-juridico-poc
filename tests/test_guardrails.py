from langchain_core.documents import Document

from rag_juridico.guardrails import check_answer, mentioned_articles


def _doc(article: str, text: str) -> Document:
    return Document(page_content=text, metadata={"article": article})


CONTEXT = {
    "T1": _doc("37", "Art. 37. A suspensão é aplicável nos casos de infrações do art. 34."),
    "T2": _doc("35", "Art. 35. As sanções disciplinares consistem em censura, suspensão, exclusão e multa."),
}


def test_mentioned_articles_handles_lists_and_suffixes():
    assert mentioned_articles("arts. 34, 36 e 37 e o artigo 2º-A") == {"34", "36", "37", "2-A"}


def test_valid_answer_passes():
    answer = "Pelo art. 37 do Estatuto, a suspensão se aplica às infrações do art. 34."
    assert check_answer(answer, ["T1"], CONTEXT) == []


def test_invented_article_is_blocked():
    problems = check_answer("Conforme o art. 99 do Estatuto, o prazo é de 60 dias.", ["T1"], CONTEXT)
    assert any("99" in p for p in problems)


def test_citation_outside_context_is_blocked():
    problems = check_answer("Pelo art. 35, há quatro sanções.", ["T7"], CONTEXT)
    assert any("T7" in p for p in problems)


def test_answer_without_citation_is_blocked():
    assert check_answer("Há quatro sanções.", [], CONTEXT) == ["resposta sem nenhuma citação"]
