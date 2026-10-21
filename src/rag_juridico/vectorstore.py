"""Embeddings locais + vector store intercambiável (local ou Pinecone).

A troca de banco vetorial é uma variável de ambiente (VECTOR_STORE). A chain só
conhece a interface VectorStore do LangChain, então pgvector/Weaviate entram aqui
sem mexer no resto.
"""

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from .config import settings

LOCAL_INDEX = "store.json"


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
    store.dump(str(settings.index_dir / LOCAL_INDEX))
    return store


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    if settings.vector_store == "pinecone":
        return _pinecone_store()
    path = settings.index_dir / LOCAL_INDEX
    if not path.exists():
        raise FileNotFoundError(f"Índice não encontrado em {path}. Rode `python -m rag_juridico.cli ingest`.")
    return InMemoryVectorStore.load(str(path), embedding=get_embeddings())


def retrieve(question: str, k: int | None = None) -> list[tuple[Document, float]]:
    """Retorna (chunk, similaridade de cosseno), do mais para o menos similar."""
    return get_store().similarity_search_with_score(question, k=k or settings.top_k)
