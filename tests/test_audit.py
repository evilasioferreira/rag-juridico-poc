"""Auditoria dos guardrails com os trechos reais do índice versionado (sem rede, sem LLM)."""

from rag_juridico.audit import run


def test_reference_answers_pass_and_all_mutations_are_blocked():
    result = run()
    assert result["falsos_bloqueios"] == []
    assert result["referencias_aprovadas"] == result["referencias"] == 25
    assert result["mutacoes_que_passaram"] == []
    for kind, summary in result["mutacoes"].items():
        blocked, total = summary.split()[0].split("/")
        assert blocked == total and int(total) > 0, kind
