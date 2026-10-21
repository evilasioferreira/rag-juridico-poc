"""Verificações determinísticas que rodam depois do LLM.

O prompt pede para citar só os trechos fornecidos, mas instrução não é garantia.
Estas checagens não dependem do modelo. A resposta é bloqueada se:

1. citar um trecho que não foi recuperado, ou não citar nenhum;
2. mencionar um artigo que não está no contexto, considerando a NORMA, não só o número
   ("art. 25 do Código de Ética" não passa quando o contexto só tem o art. 25 do Estatuto);
3. afirmar um prazo ou quórum (dias, meses, anos, horas, terços, %) que não aparece
   nos trechos citados ("três anos" não passa quando o trecho diz "cinco anos").
"""

import re
import unicodedata

from langchain_core.documents import Document

# "art. 25", "arts. 34 e 36", "artigo 2º-A", "Art. 7º, inciso II"
_ART_MENTION = re.compile(r"\bart(?:igo)?s?\.?\s*(\d+)\s*(?:º|o)?(?:\s*-\s*([A-Z])\b)?", re.IGNORECASE)
_ART_LIST_CONT = re.compile(r"^\s*(?:,|e)\s*(\d+)\s*(?:º|o)?(?:\s*-\s*([A-Z])\b)?", re.IGNORECASE)

# Norma indicada logo depois da menção ("art. 25 do Estatuto", "art. 133 da CF").
# SELF = referência interna do próprio texto ("desta Lei", "deste Código").
_NORMS = [
    ("SELF", r"\b(?:desta|nesta|da presente)\s+(?:lei|constitui[cç][aã]o|resolu[cç][aã]o)\b"
             r"|\bdeste\s+(?:c[oó]digo|estatuto)\b"),
    ("EAOAB", r"\bestatuto\b|\beaoab\b|\blei\s*(?:n[º°o.]*\s*)?8\.?906\b"),
    ("CED", r"\bc[oó]digo\s+de\s+[eé]tica\b|\bced\b"),
    ("CF", r"\bconstitui[cç][aã]o\b|\bcf(?:/88)?\b|\bcarta\s+magna\b"),
    ("OTHER", r"\bregulamento\s+geral\b|\bc[oó]digo\s+(?:civil|penal|de\s+processo)\b"
              r"|\b(?:cpc|cpp|clt)\b|\blei\s*(?:n[º°o.]*\s*)?(?!8\.?906)\d"),
]
_NORM_WINDOW = 80


def _key(num: str, suffix: str | None) -> str:
    return f"{int(num)}-{suffix.upper()}" if suffix else str(int(num))


def _norm_after(text: str) -> str | None:
    found = []
    for name, pattern in _NORMS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            found.append((m.start(), name))
    return min(found)[1] if found else None


def article_mentions(text: str) -> list[tuple[str, str | None]]:
    """[(artigo, norma)] mencionados no texto. Listas ("arts. 34, 36 e 37 do Estatuto")
    herdam a norma indicada depois do último item."""
    matches = list(_ART_MENTION.finditer(text))
    result = []
    for i, m in enumerate(matches):
        keys = [_key(m.group(1), m.group(2))]
        end = m.end()
        while (cont := _ART_LIST_CONT.match(text[end:])):
            keys.append(_key(cont.group(1), cont.group(2)))
            end += cont.end()
        limit = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        norm = _norm_after(text[end:min(limit, end + _NORM_WINDOW)])
        result += [(k, norm) for k in keys]
    return result


def mentioned_articles(text: str) -> set[str]:
    return {key for key, _ in article_mentions(text)}


# ---- prazos e quóruns -------------------------------------------------------

_UNITS = {
    "dia": "dia", "dias": "dia", "mes": "mes", "meses": "mes", "ano": "ano", "anos": "ano",
    "hora": "hora", "horas": "hora", "semana": "semana", "semanas": "semana",
    "terco": "terco", "tercos": "terco", "%": "pct", "por cento": "pct",
}
_WORDS = {
    "zero": 0, "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5,
    "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10, "onze": 11, "doze": 12, "treze": 13,
    "quatorze": 14, "catorze": 14, "quinze": 15, "dezesseis": 16, "dezessete": 17, "dezoito": 18,
    "dezenove": 19, "vinte": 20, "trinta": 30, "quarenta": 40, "cinquenta": 50, "sessenta": 60,
    "setenta": 70, "oitenta": 80, "noventa": 90, "cem": 100, "cento": 100,
}
_WORD = "|".join(sorted(_WORDS, key=len, reverse=True))
_QUANTITY = re.compile(
    rf"\b(\d+|(?:{_WORD})(?:\s+e\s+(?:{_WORD}))*)\s*(?:\([^)]*\)\s*)?"
    r"(dias?|mes(?:es)?|anos?|horas?|semanas?|tercos?|%|por cento)(?![a-z])"
)


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def quantities(text: str) -> set[tuple[int, str]]:
    """Prazos e quóruns do texto, normalizados: "quinze dias" e "15 dias" viram (15, "dia")."""
    found = set()
    for m in _QUANTITY.finditer(_plain(text)):
        raw = m.group(1)
        value = int(raw) if raw.isdigit() else sum(_WORDS[w] for w in raw.split() if w != "e")
        found.add((value, _UNITS[m.group(2)]))
    return found


def _body(doc: Document) -> str:
    """Texto do trecho sem a primeira linha de cabeçalho ("[EAOAB, Art. 25] ...")."""
    return doc.page_content.split("\n", 1)[-1]


# ---- verificação ---------------------------------------------------------------

def check_answer(answer: str, cited_ids: list[str], context: dict[str, Document]) -> list[str]:
    """Retorna a lista de violações (vazia = resposta aprovada)."""
    problems = []
    unknown = [c for c in cited_ids if c not in context]
    if unknown:
        problems.append(f"citou trechos que não foram recuperados: {unknown}")
    if not cited_ids:
        problems.append("resposta sem nenhuma citação")

    # Artigos permitidos: os dos trechos recuperados e os referidos dentro deles, com a norma.
    pairs: set[tuple[str, str]] = set()
    numbers: set[str] = set()
    for doc in context.values():
        own = doc.metadata["source_short"]
        pairs.add((own, doc.metadata["article"]))
        numbers.add(doc.metadata["article"])
        for key, norm in article_mentions(_body(doc)):
            pairs.add((own if norm in (None, "SELF") else norm, key))
            numbers.add(key)

    for key, norm in article_mentions(answer):
        if norm == "OTHER":
            problems.append(f"citou artigo de norma fora da base: art. {key}")
        elif norm in ("EAOAB", "CED", "CF"):
            if (norm, key) not in pairs:
                problems.append(f"mencionou artigo ausente do contexto: art. {key} ({norm})")
        elif key not in numbers:
            problems.append(f"mencionou artigo ausente do contexto: art. {key}")

    cited_docs = [context[c] for c in cited_ids if c in context]
    supported = set().union(*(quantities(_body(d)) for d in cited_docs)) if cited_docs else set()
    invented = sorted(quantities(answer) - supported)
    if invented:
        listed = ", ".join(f"{v} {u}" for v, u in invented)
        problems.append(f"prazo ou quórum ausente dos trechos citados: {listed}")
    return problems
