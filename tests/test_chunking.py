from rag_juridico.chunking import article_label, split_articles
from rag_juridico.sources import normalize

SAMPLE = normalize("""
TÍTULO I Da Advocacia CAPÍTULO I Da Atividade de Advocacia
Art. 1º São atividades privativas de advocacia: I - a postulação; II - a consultoria.
§ 1º Não se inclui o habeas corpus (Vide ADIN 1.127-8).
Art. 2º O advogado é indispensável à administração da justiça, nos termos do art. 1º.
Art. 2º-A. O advogado pode contribuir com o processo legislativo. (Incluído pela Lei nº 14.365, de 2022)
CAPÍTULO II Da Inscrição
Art. 8º Para inscrição como advogado é necessário: I - capacidade civil;
Art. 8º Versão repetida que deve ser ignorada.
Brasília, 4 de julho de 1994; 173º da Independência.
""")


def test_splits_one_chunk_per_article_with_suffixes():
    keys = [a["key"] for a in split_articles(SAMPLE)]
    assert keys == ["1", "2", "2-A", "8"]


def test_internal_lowercase_reference_does_not_split():
    art2 = next(a for a in split_articles(SAMPLE) if a["key"] == "2")
    assert "nos termos do art. 1º" in art2["text"]


def test_heading_path_and_trailing_heading_removed():
    arts = {a["key"]: a for a in split_articles(SAMPLE)}
    assert arts["1"]["heading"] == "Título I - Da Advocacia > Capítulo I - Da Atividade de Advocacia"
    assert arts["8"]["heading"] == "Título I - Da Advocacia > Capítulo II - Da Inscrição"
    assert "CAPÍTULO II" not in arts["2-A"]["text"]


def test_normalize_strips_annotations_and_signature():
    assert "Vide ADIN" not in SAMPLE
    assert "Incluído pela" not in SAMPLE
    assert "Brasília" not in SAMPLE


def test_article_label():
    assert article_label("1") == "Art. 1º"
    assert article_label("2-A") == "Art. 2º-A"
    assert article_label("25") == "Art. 25"
