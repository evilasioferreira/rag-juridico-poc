"""Log estruturado de cada interação (JSONL): base para monitorar qualidade em produção.

Cada linha tem pergunta, trechos recuperados com score, resposta, status, violações
de guardrail, tokens e latência. Em produção isso iria para um banco ou observabilidade
(Langfuse, Datadog, Postgres); o formato é o mesmo.
"""

import json
from datetime import datetime, timezone

from .config import settings

LOG_FILE = "interactions.jsonl"


def log_interaction(record: dict) -> None:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat()} | record
    with open(settings.logs_dir / LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_logs() -> dict:
    path = settings.logs_dir / LOG_FILE
    if not path.exists():
        return {"total": 0}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    latencies = sorted(r["latency_ms"] for r in rows)
    return {
        "total": len(rows),
        "por_status": statuses,
        "taxa_recusa": round(sum(r["refused"] for r in rows) / len(rows), 3),
        "latencia_p50_ms": latencies[len(latencies) // 2],
        "latencia_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        "tokens_entrada": sum(r.get("usage", {}).get("input_tokens", 0) for r in rows),
        "tokens_saida": sum(r.get("usage", {}).get("output_tokens", 0) for r in rows),
    }
