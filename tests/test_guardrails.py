from langchain_core.documents import Document

from rag_juridico.guardrails import article_mentions, check_answer, mentioned_articles, quantities


def _doc(short: str, article: str, text: str) -> Document:
    return Document(page_content=text, metadata={"source_short": short, "article": article})


CONTEXT = {
    "T1": _doc("EAOAB", "37", "Art. 37. A suspensão é aplicável nos casos de infrações do art. 34 desta Lei. "
                              "§ 1º A suspensão dura de trinta dias a doze meses."),
    "T2": _doc("EAOAB", "35", "Art. 35. As sanções disciplinares consistem em censura, suspensão, exclusão e multa."),
    "T3": _doc("EAOAB", "25", "Art. 25. Prescreve em cinco anos a ação de cobrança de honorários de advogado."),
    "T4": _doc("CF", "134", "Art. 134. A Defensoria Pública, na forma do inciso LXXIV do art. 5º desta Constituição."),
}


def test_mentioned_articles_handles_lists_and_suffixes():
    assert mentioned_articles("arts. 34, 36 e 37 e o artigo 2º-A") == {"34", "36", "37", "2-A"}


def test_article_mentions_detect_norm_for_whole_list():
    assert article_mentions("arts. 34 e 36 do Estatuto e art. 35 do Código de Ética") == [
        ("34", "EAOAB"), ("36", "EAOAB"), ("35", "CED"),
    ]
    assert article_mentions("art. 133 da CF")[0] == ("133", "CF")
    assert article_mentions("art. 1º da Lei 8.906/1994")[0] == ("1", "EAOAB")
    assert article_mentions("art. 335 do CPC")[0] == ("335", "OTHER")


def test_quantities_normalize_words_and_digits():
    assert quantities("quinze dias, prorrogável") == {(15, "dia")}
    assert quantities("15 (quinze) dias") == {(15, "dia")}
    assert quantities("de trinta dias a doze meses") == {(30, "dia"), (12, "mes")}
    assert quantities("dois terços dos membros") == {(2, "terco")}
    assert quantities("vinte e cinco anos") == {(25, "ano")}
    assert quantities("em 1º de janeiro") == set()


def test_valid_answer_passes():
    answer = "Pelo art. 37 do Estatuto, a suspensão vai de 30 dias a 12 meses e alcança as infrações do art. 34."
    assert check_answer(answer, ["T1"], CONTEXT) == []


def test_internal_reference_resolves_to_own_norm():
    assert check_answer("A Defensoria atua na forma do art. 5º da CF (art. 134).", ["T4"], CONTEXT) == []


def test_invented_article_is_blocked():
    problems = check_answer("Conforme o art. 99 do Estatuto, há sanções.", ["T2"], CONTEXT)
    assert any("99" in p for p in problems)


def test_swapped_norm_is_blocked():
    problems = check_answer("Prescreve em cinco anos (art. 25 do Código de Ética).", ["T3"], CONTEXT)
    assert any("25 (CED)" in p for p in problems)


def test_article_from_law_outside_base_is_blocked():
    problems = check_answer("O prazo é o do art. 335 do CPC.", ["T3"], CONTEXT)
    assert any("fora da base" in p for p in problems)


def test_wrong_deadline_is_blocked():
    problems = check_answer("Prescreve em três anos (art. 25 do Estatuto).", ["T3"], CONTEXT)
    assert any("3 ano" in p for p in problems)


def test_deadline_must_be_in_cited_chunk_not_just_retrieved():
    # "trinta dias" está em T1, mas a resposta só citou T3.
    problems = check_answer("Prescreve em cinco anos, e a suspensão dura 30 dias (art. 25 do Estatuto).", ["T3"], CONTEXT)
    assert any("30 dia" in p for p in problems)


def test_citation_outside_context_is_blocked():
    problems = check_answer("Pelo art. 35, há quatro sanções.", ["T7"], CONTEXT)
    assert any("T7" in p for p in problems)


def test_answer_without_citation_is_blocked():
    assert check_answer("Há quatro sanções.", [], CONTEXT) == ["resposta sem nenhuma citação"]
