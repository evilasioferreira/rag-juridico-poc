import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    llm_model: str = os.getenv("LLM_MODEL", "claude-opus-5-5")
    judge_model: str = os.getenv("JUDGE_MODEL", "claude-opus-5-5")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    vector_store: str = os.getenv("VECTOR_STORE", "local")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "rag-juridico-poc")
    top_k: int = int(os.getenv("TOP_K", "5"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.835"))

    root: Path = ROOT
    sources_file: Path = ROOT / "sources.json"
    raw_dir: Path = ROOT / "data" / "raw"
    index_dir: Path = ROOT / "index"
    logs_dir: Path = ROOT / "logs"
    reports_dir: Path = ROOT / "reports"
    eval_file: Path = ROOT / "eval" / "dataset.jsonl"


settings = Settings()
