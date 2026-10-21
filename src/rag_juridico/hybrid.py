"""Busca lexical (BM25) para combinar com a vetorial.

Texto jurídico tem termos exatos que pesam muito ("procuração", "prescreve",
"sucumbência"). Um modelo de embeddings pequeno às vezes perde esses termos; o BM25
não perde. As duas listas são combinadas por Reciprocal Rank Fusion (RRF).
"""

import math
import re
import unicodedata
from collections import Counter

STOPWORDS = set(
    "a ao aos as com como da das de do dos e em entre esta este ha isso mais mas na nas no nos "
    "o os ou para pela pelas pelo pelos por qual quais quando que se sem ser seu sua sao so "
    "um uma umas uns lei art arts artigo".split()
)


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z0-9]{2,}", text)
    # Normalização mínima de plural ("honorarios" -> "honorario").
    return [t[:-1] if len(t) > 4 and t.endswith("s") else t for t in tokens if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = {doc_id: Counter(tokenize(text)) for doc_id, text in docs.items()}
        self.len = {doc_id: sum(tf.values()) for doc_id, tf in self.tf.items()}
        self.avg_len = sum(self.len.values()) / max(len(self.len), 1)
        df = Counter(term for tf in self.tf.values() for term in tf)
        n = len(self.tf)
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def search(self, query: str, k: int) -> list[str]:
        terms = set(tokenize(query))
        scores = {}
        for doc_id, tf in self.tf.items():
            s = 0.0
            for t in terms:
                if t in tf:
                    f = tf[t]
                    norm = f + self.k1 * (1 - self.b + self.b * self.len[doc_id] / self.avg_len)
                    s += self.idf[t] * f * (self.k1 + 1) / norm
            if s > 0:
                scores[doc_id] = s
        return sorted(scores, key=scores.get, reverse=True)[:k]


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: soma 1/(k + posição) de cada lista."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for pos, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + pos)
    return sorted(scores, key=scores.get, reverse=True)
