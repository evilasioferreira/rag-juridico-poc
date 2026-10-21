"""Chunking orientado à estrutura da lei: um chunk por artigo.

Cortar por número de caracteres separa o caput dos parágrafos e mistura artigos
vizinhos, o que leva o modelo a citar o artigo errado. Aqui a unidade é o artigo
(com seus parágrafos e incisos) e cada chunk carrega o título/capítulo de origem.
Só artigos muito longos (ex.: CF art. 5º) são subdivididos, mantendo o rótulo.
"""

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .sources import load_sources

MAX_CHARS = 1800

# "Art. 1º", "Art. 2º-A.", "Art. 25.", "Art. 7o" (só maiúsculo: referências internas usam "art.").
ARTICLE = re.compile(r"\bArt\.\s*(\d+)\s*(?:º|o)?\s*(?:-\s*([A-Z])\b)?\s*\.?\s+")
HEADING = re.compile(
    r"\b(TÍTULO|CAPÍTULO|Capítulo|Seção|SEÇÃO|Subseção|SUBSEÇÃO)\s+([IVXLC]+(?:-[A-Z])?)\b\s+(.+?)"
    r"(?=\s+(?:TÍTULO|CAPÍTULO|Capítulo|Seção|SEÇÃO|Subseção|SUBSEÇÃO)\s+[IVXLC]+\b|\s+Art\.\s*\d)"
)
TRAILING_HEADING = re.compile(
    r"\s+(?:TÍTULO|CAPÍTULO|Capítulo|Seção|SEÇÃO|Subseção|SUBSEÇÃO)\s+[IVXLC]+(?:-[A-Z])?"
    r"(?:\s+D[AaOoEe]S?\b[^.;:]{0,150})?\s*$"
)
_LEVEL = {"título": 0, "capítulo": 1, "seção": 2, "subseção": 3}


def article_key(number: str, suffix: str | None) -> str:
    return f"{int(number)}-{suffix}" if suffix else str(int(number))


def article_label(key: str) -> str:
    num, _, suffix = key.partition("-")
    ordinal = "º" if int(num) < 10 else ""
    return f"Art. {num}{ordinal}" + (f"-{suffix}" if suffix else "")


def _pretty(text: str) -> str:
    """Quebra linhas antes de parágrafos e incisos para leitura (não afeta a busca)."""
    text = re.sub(r"\s+(§\s*\d+|Parágrafo único)", r"\n\1", text)
    text = re.sub(r"\s+([IVXLC]+\s+-\s+)", r"\n\1", text)
    return text.strip()


def _in_ranges(num: int, ranges: list[list[int]] | None) -> bool:
    return ranges is None or any(lo <= num <= hi for lo, hi in ranges)


def split_articles(text: str) -> list[dict]:
    """Retorna [{key, heading, text}] na ordem do documento, sem artigos duplicados."""
    headings = [(m.start(), m.group(1).lower(), f"{m.group(1).title()} {m.group(2)} - {m.group(3).strip()}")
                for m in HEADING.finditer(text)]
    starts = list(ARTICLE.finditer(text))
    articles, seen = [], set()
    path: dict[int, str] = {}
    h_idx = 0
    for i, m in enumerate(starts):
        while h_idx < len(headings) and headings[h_idx][0] < m.start():
            _, kind, label = headings[h_idx]
            level = _LEVEL[kind]
            path = {lv: lb for lv, lb in path.items() if lv < level}
            path[level] = label
            h_idx += 1
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        body = text[m.start():end]
        # Remove cabeçalhos de título/capítulo/seção que ficaram colados no fim do artigo.
        prev = None
        while prev != body:
            prev = body
            body = TRAILING_HEADING.sub("", body).strip()
        key = article_key(m.group(1), m.group(2))
        if key in seen:  # versão repetida (ex.: redação antiga não tachada): fica a primeira
            continue
        seen.add(key)
        articles.append({"key": key, "heading": " > ".join(path[k] for k in sorted(path)), "text": body})
    return articles


def build_documents() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHARS - 200, chunk_overlap=150, separators=["\n§", "\n", "; ", ". ", " "]
    )
    docs: list[Document] = []
    for src in load_sources():
        path = settings.raw_dir / f"{src['id']}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} não existe. Rode `python -m rag_juridico.cli download` antes.")
        for art in split_articles(path.read_text(encoding="utf-8")):
            num = int(art["key"].split("-")[0])
            if not _in_ranges(num, src.get("articles")):
                continue
            body = _pretty(art["text"])
            parts = [body] if len(body) <= MAX_CHARS else splitter.split_text(body)
            label = article_label(art["key"])
            for n, part in enumerate(parts, start=1):
                part_note = f" (parte {n}/{len(parts)})" if len(parts) > 1 else ""
                header = f"[{src['short']}, {label}{part_note}] {src['title']}"
                if art["heading"]:
                    header += f" | {art['heading']}"
                chunk_id = f"{src['short']}:{art['key']}" + (f"#p{n}" if len(parts) > 1 else "")
                docs.append(Document(
                    id=chunk_id,
                    page_content=f"{header}\n{part}",
                    metadata={
                        "chunk_id": chunk_id,
                        "source_id": src["id"],
                        "source_short": src["short"],
                        "source_title": src["title"],
                        "source_url": src["url"],
                        "article": art["key"],
                        "article_label": label,
                        "ref": f"{src['short']}:{art['key']}",
                        "heading": art["heading"],
                    },
                ))
    return docs
