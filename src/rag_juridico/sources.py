"""Download e extração de texto das fontes públicas (Planalto e OAB).

Cada fonte vira um arquivo de texto normalizado em data/raw/<id>.txt.
"""

import io
import json
import re
import urllib.request

from bs4 import BeautifulSoup
from pypdf import PdfReader

from .config import settings

USER_AGENT = "Mozilla/5.0 (rag-juridico-poc)"

# Anotações editoriais do Planalto que não fazem parte do texto normativo.
_ANNOTATION = re.compile(
    r"\((?:Incluíd[oa]|Redação dada|Vide|Renumerad[oa]|Acrescid[oa]|Regulamento|"
    r"Produção de efeitos|Vigência|Promulgação)[^)]*\)",
    re.IGNORECASE,
)


def load_sources() -> list[dict]:
    return json.loads(settings.sources_file.read_text(encoding="utf-8"))


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def _decode_html(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def html_to_text(raw: bytes) -> str:
    soup = BeautifulSoup(_decode_html(raw), "html.parser")
    # Texto revogado aparece tachado no Planalto. Remover evita responder com lei velha.
    for tag in soup.find_all(["strike", "s", "del", "script", "style"]):
        tag.decompose()
    return soup.get_text(" ")


def pdf_to_text(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    pages = []
    for page in reader.pages:
        # O modo "layout" evita palavras quebradas ("p ublicidade") do modo padrão.
        text = page.extract_text(extraction_mode="layout")
        lines = [ln for ln in text.splitlines() if not ln.strip().isdigit()]
        pages.append("\n".join(lines))
    return "\n".join(pages)


def normalize(text: str, start_marker: str | None = None, end_marker: str | None = None) -> str:
    text = _ANNOTATION.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if start_marker and start_marker in text:
        text = text[text.index(start_marker):]
    if end_marker and end_marker in text:
        text = text[: text.index(end_marker)]
    # Remove o fecho de assinatura da lei ("Brasília, 4 de julho de 1994; ...").
    match = list(re.finditer(r"\sBrasília,\s+\d{1,2}\s+de\s+\w+\s+de\s+\d{4}", text))
    if match:
        text = text[: match[-1].start()]
    return text


def download_all() -> list[str]:
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for src in load_sources():
        raw = _fetch(src["url"])
        text = html_to_text(raw) if src["format"] == "html" else pdf_to_text(raw)
        text = normalize(text, src.get("start_marker"), src.get("end_marker"))
        path = settings.raw_dir / f"{src['id']}.txt"
        path.write_text(text, encoding="utf-8")
        written.append(f"{path.name}: {len(text):,} caracteres")
    return written
