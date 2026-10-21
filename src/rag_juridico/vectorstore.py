"""Embeddings locais + vector store intercambiável (local ou Pinecone).

A troca de banco vetorial é uma variável de ambiente (VECTOR_STORE). A chain só
conhece a interface VectorStore do LangChain, então pgvector/Weaviate entram aqui
sem mexer no resto.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from .config import settings
from .hybrid import BM25, rrf

CANDIDATES = 20  # quantos resultados de cada busca entram na fusão

def local_index_path() -> Path:
    # Um arquivo por modelo: trocar EMBEDDING_MODEL nunca reaproveita vetores de outro modelo.
    slug = settings.embedding_model.split("/")[-1]
    return settings.index_dir / f"{slug}.json"


class FastEmbedEmbeddings(Embeddings):
    """Embeddings ONNX locais: sem custo por consulta e sem enviar o material a terceiros."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(settings.root / ".models"))
        # Modelos da família E5 foram treinados com prefixos distintos para consulta e trecho.
        self._e5 = "e5" in model_name.lower()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._e5:
            texts = [f"passage: {t}" for t in texts]
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        if self._e5:
            text = f"query: {text}"
        return next(iter(self._model.embed([text]))).tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    return FastEmbedEmbeddings(settings.embedding_model)


def _pinecone_store(create_with_dim: int | None = None) -> VectorStore:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone()
    if create_with_dim and not pc.has_index(settings.pinecone_index):
        pc.create_index(
            name=settings.pinecone_index,
            dimension=create_with_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return PineconeVectorStore(index=pc.Index(settings.pinecone_index), embedding=get_embeddings())


def build_store(docs: list[Document]) -> VectorStore:
    embeddings = get_embeddings()
    if settings.vector_store == "pinecone":
        dim = len(embeddings.embed_query("dimensão"))
        store = _pinecone_store(create_with_dim=dim)
        store.add_documents(docs, ids=[d.id for d in docs])
        return store

    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(docs, ids=[d.id for d in docs])
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    store.dump(str(local_index_path()))
    return store


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    if settings.vector_store == "pinecone":
        return _pinecone_store()
    path = local_index_path()
    if not path.exists():
        raise FileNotFoundError(f"Índice não encontrado em {path}. Rode `python -m rag_juridico.cli ingest`.")
    return InMemoryVectorStore.load(str(path), embedding=get_embeddings())


@lru_cache(maxsize=1)
def _local_corpus() -> tuple[list[str], np.ndarray, BM25]:
    store = get_store()
    ids = list(store.store)
    vectors = np.array([store.store[i]["vector"] for i in ids], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return ids, vectors, BM25({i: store.store[i]["text"] for i in ids})


def _hybrid(question: str, k: int) -> list[tuple[Document, float]]:
    ids, vectors, bm25 = _local_corpus()
    q = np.array(get_embeddings().embed_query(question), dtype=np.float32)
    cosine = vectors @ (q / np.linalg.norm(q))
    by_id = dict(zip(ids, cosine.tolist()))
    dense = [ids[i] for i in np.argsort(-cosine)[:CANDIDATES]]
    fused = rrf([dense, bm25.search(question, CANDIDATES)])[:k]
    return [(d, by_id[d.id]) for d in get_store().get_by_ids(fused)]


def retrieve(question: str, k: int | None = None) -> list[tuple[Document, float]]:
    """Retorna (chunk, similaridade de cosseno), na ordem de relevância.

    No store local, combina busca vetorial e BM25 (RRF). O score devolvido é sempre
    a similaridade de cosseno, usada pelo portão de relevância.
    """
    k = k or settings.top_k
    if settings.hybrid and settings.vector_store == "local":
        return _hybrid(question, k)
    return get_store().similarity_search_with_score(question, k=k)
