from rag_juridico.hybrid import BM25, rrf, tokenize


def test_tokenize_removes_accents_stopwords_and_plural():
    assert tokenize("Os honorários de sucumbência pertencem ao advogado") == [
        "honorario", "sucumbencia", "pertencem", "advogado",
    ]


def test_bm25_ranks_exact_term_first():
    bm25 = BM25({
        "a": "O advogado postula em juízo fazendo prova do mandato, com procuração.",
        "b": "A suspensão acarreta interdição do exercício profissional.",
        "c": "Prescreve em cinco anos a ação de cobrança de honorários.",
    })
    assert bm25.search("prazo para cobrar honorários prescreve", k=2)[0] == "c"
    assert bm25.search("procuração", k=3) == ["a"]


def test_rrf_rewards_agreement_between_rankings():
    assert rrf([["x", "y", "z"], ["y", "x", "w"]])[:2] in (["x", "y"], ["y", "x"])
    assert rrf([["x", "y"], ["y", "z"]])[0] == "y"
