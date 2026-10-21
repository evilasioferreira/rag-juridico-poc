"""Verificações determinísticas que rodam depois do LLM.

O prompt pede para citar só os trechos fornecidos, mas instrução não é garantia.
Estas checagens não dependem do modelo: se a resposta cita um trecho que não foi
recuperado, ou menciona um artigo que não aparece no contexto, ela é bloqueada.
"""

import re

from langchain_core.documents import Document

# "art. 25", "arts. 34 e 36", "artigo 2º-A", "Art. 7º, inciso II"
_ART_MENTION = re.compile(r"\bart(?:igo)?s?\.?\s*(\d+)\s*(?:º|o)?(?:\s*-\s*([A-Z])\b)?", re.IGNORECASE)
_ART_LIST_CONT = re.compile(r"^\s*(?:,|e)\s*(\d+)\s*(?:º|o)?(?:\s*-\s*([A-Z])\b)?", re.IGNORECASE)


def mentioned_articles(text: str) -> set[str]:
    """Números de artigo mencionados em um texto, incluindo listas ("arts. 34, 36 e 37")."""
    found: set[str] = set()
    for m in _ART_MENTION.finditer(text):
        found.add(_key(m.group(1), m.group(2)))
        rest = text[m.end():]
        while (cont := _ART_LIST_CONT.match(rest)):
            found.add(_key(cont.group(1), cont.group(2)))
            rest = rest[cont.end():]
    return found


def _key(num: str, suffix: str | None) -> str:
    return f"{int(num)}-{suffix.upper()}" if suffix else str(int(num))


def check_answer(answer: str, cited_ids: list[str], context: dict[str, Document]) -> list[str]:
    """Retorna a lista de violações (vazia = resposta aprovada)."""
    problems = []
    unknown = [c for c in cited_ids if c not in context]
    if unknown:
        problems.append(f"citou trechos que não foram recuperados: {unknown}")
    if not cited_ids:
        problems.append("resposta sem nenhuma citação")

    # Artigos que a resposta pode mencionar: os dos trechos recuperados e os referidos dentro deles.
    allowed: set[str] = set()
    for doc in context.values():
        allowed.add(doc.metadata["article"])
        allowed |= mentioned_articles(doc.page_content)
    invented = sorted(mentioned_articles(answer) - allowed, key=lambda k: int(k.split("-")[0]))
    if invented:
        problems.append(f"mencionou artigos ausentes do contexto: {invented}")
    return problems
