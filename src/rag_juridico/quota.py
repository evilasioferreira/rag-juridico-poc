"""Limites de uso para a demo pública: protegem a chave da API de abuso.

- Tamanho máximo da pergunta.
- Perguntas por sessão de navegador.
- Chamadas ao LLM por dia somando todos os visitantes (contadas no log de interações).
"""

import json
import os
from datetime import datetime, timezone

from .config import settings
from .logbook import LOG_FILE

MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "400"))
SESSION_LIMIT = int(os.getenv("SESSION_LIMIT", "15"))
DAILY_LLM_LIMIT = int(os.getenv("DAILY_LLM_LIMIT", "200"))


def llm_calls_today() -> int:
    path = settings.logs_dir / LOG_FILE
    if not path.exists():
        return 0
    today = datetime.now(timezone.utc).date().isoformat()
    count = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["ts"].startswith(today) and row.get("raw_model_answer") is not None:
                count += 1
    return count


def check(question: str, session_count: int) -> str | None:
    """Retorna a mensagem de bloqueio, ou None se a pergunta pode seguir."""
    if len(question) > MAX_QUESTION_CHARS:
        return f"A pergunta passou de {MAX_QUESTION_CHARS} caracteres. Tente resumir."
    if session_count >= SESSION_LIMIT:
        return f"Você atingiu o limite de {SESSION_LIMIT} perguntas desta demonstração. Obrigado por testar!"
    if llm_calls_today() >= DAILY_LLM_LIMIT:
        return "A demonstração atingiu o limite de uso de hoje. Volte amanhã."
    return None
