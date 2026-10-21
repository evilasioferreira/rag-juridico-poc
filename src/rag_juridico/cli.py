"""Linha de comando.

    python -m rag_juridico.cli download          # baixa as fontes públicas
    python -m rag_juridico.cli ingest            # chunking + embeddings + vector store
    python -m rag_juridico.cli ask "pergunta"    # responde com citação ou recusa
    python -m rag_juridico.cli eval [--retrieval-only] [--limit N]
    python -m rag_juridico.cli stats             # resumo dos logs de produção
"""

import argparse
import json
import sys

from .config import settings


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="rag_juridico")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    sub.add_parser("ingest")
    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    p_ask.add_argument("--json", action="store_true", help="imprime o registro completo")
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--retrieval-only", action="store_true", help="só mede a recuperação (sem LLM)")
    p_eval.add_argument("--limit", type=int)
    sub.add_parser("stats")
    sub.add_parser("audit-guardrails", help="testa os guardrails com trechos reais e respostas adulteradas")
    args = parser.parse_args()

    if args.cmd == "download":
        from .sources import download_all

        for line in download_all():
            print(line)
    elif args.cmd == "ingest":
        from .chunking import build_documents
        from .vectorstore import build_store

        docs = build_documents()
        build_store(docs)
        print(f"{len(docs)} trechos indexados em '{settings.vector_store}' com {settings.embedding_model}")
    elif args.cmd == "ask":
        from .chain import ask, format_answer

        ans = ask(args.question)
        print(json.dumps(ans.to_dict(), ensure_ascii=False, indent=2) if args.json else format_answer(ans))
    elif args.cmd == "eval":
        from .evaluate import run

        summary = run(limit=args.limit, retrieval_only=args.retrieval_only)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.cmd == "audit-guardrails":
        from .audit import run as audit

        print(json.dumps(audit(), ensure_ascii=False, indent=2))
    elif args.cmd == "stats":
        from .logbook import summarize_logs

        print(json.dumps(summarize_logs(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
