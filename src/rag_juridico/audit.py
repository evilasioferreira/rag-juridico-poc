"""Auditoria offline dos guardrails, com os trechos reais do índice e sem LLM.

Para cada pergunta respondível do dataset:
- a resposta de referência, citando os trechos esperados, precisa PASSAR (sem falso bloqueio);
- versões adulteradas dela precisam ser BLOQUEADAS:
  * norma trocada ("art. 25 do Estatuto" -> "art. 25 do Código de Ética");
  * prazo ou quórum alterado ("cinco anos" -> "seis anos");
  * artigo inventado acrescentado ("art. 99 do Estatuto").
"""

import json
import re

from langchain_core.documents import Document

from .evaluate import load_dataset
from .guardrails import _QUANTITY, _plain, check_answer, quantities
from .vectorstore import local_index_path

_SWAPS = [("do Estatuto", "do Código de Ética"), ("do Código de Ética", "do Estatuto"), ("da CF", "do Estatuto")]
_NUMBER_SWAP = {"um": "dois", "dois": "tres", "tres": "quatro", "cinco": "seis", "dez": "onze",
                "quinze": "vinte", "trinta": "sessenta", "doze": "dezoito"}


def _context(raw: dict, refs: list[str]) -> dict[str, Document]:
    ids = [i for i in raw if i.split("#")[0] in refs]
    return {f"T{n}": Document(page_content=raw[i]["text"], metadata=raw[i]["metadata"])
            for n, i in enumerate(ids, start=1)}


def _swap_norm(answer: str) -> str | None:
    for old, new in _SWAPS:
        if old in answer:
            return answer.replace(old, new, 1)
    return None


def _swap_number(answer: str) -> str | None:
    plain = _plain(answer)
    m = _QUANTITY.search(plain)
    if not m:
        return None
    raw = m.group(1)
    new = str(int(raw) + 1) if raw.isdigit() else _NUMBER_SWAP.get(raw.split()[0])
    if not new:
        return None
    # _plain só remove acentos (um caractere vira um caractere), então as posições batem.
    start, end = m.span(1)
    return answer[:start] + new + answer[end:]


def run() -> dict:
    raw = json.loads(local_index_path().read_text(encoding="utf-8"))
    rows = [r for r in load_dataset() if not r["should_refuse"]]
    result = {"referencias_aprovadas": 0, "referencias": len(rows), "falsos_bloqueios": [],
              "mutacoes": {"norma_trocada": [0, 0], "prazo_alterado": [0, 0], "artigo_inventado": [0, 0]},
              "mutacoes_que_passaram": []}

    for row in rows:
        ctx = _context(raw, row["expected_refs"])
        cited = list(ctx)
        reference = row["reference_answer"]
        problems = check_answer(reference, cited, ctx)
        if problems:
            result["falsos_bloqueios"].append({"id": row["id"], "problemas": problems})
        else:
            result["referencias_aprovadas"] += 1

        supported = set().union(*(quantities(d.page_content) for d in ctx.values()))
        mutations = {
            "norma_trocada": _swap_norm(reference),
            "prazo_alterado": _swap_number(reference),
            "artigo_inventado": reference + " Ver também o art. 99 do Estatuto.",
        }
        for kind, mutated in mutations.items():
            if not mutated or mutated == reference:
                continue
            if kind == "prazo_alterado" and quantities(mutated) <= supported:
                continue  # o novo valor também existe no trecho: não é uma adulteração
            result["mutacoes"][kind][1] += 1
            if check_answer(mutated, cited, ctx):
                result["mutacoes"][kind][0] += 1
            else:
                result["mutacoes_que_passaram"].append({"id": row["id"], "tipo": kind, "texto": mutated})

    result["mutacoes"] = {k: f"{b}/{t} bloqueadas" for k, (b, t) in result["mutacoes"].items()}
    return result
