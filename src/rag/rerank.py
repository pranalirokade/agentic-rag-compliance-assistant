"""Second-stage reranking. Every reranker returns (chunk, relevance in [0,1])."""
from __future__ import annotations

import math

from .bm25 import BM25
from .chunking import Chunk
from .text import tokens


class LexicalReranker:
    """Offline fallback: IDF-weighted query-term coverage. Query terms that never occur in the
    corpus stay in the denominator, so out-of-scope questions score low -> abstention."""

    def __init__(self, bm25: BM25):
        self.bm25 = bm25

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5):
        q = set(tokens(query))
        total = sum(self.bm25.idf(t) for t in q) or 1.0
        scored = []
        for c in chunks:
            ct = set(tokens(c.section + " " + c.text))
            scored.append((c, sum(self.bm25.idf(t) for t in q & ct) / total))
        return sorted(scored, key=lambda x: -x[1])[:top_k]


class CrossEncoderReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5):
        if not chunks:
            return []
        logits = self.model.predict([(query, c.text) for c in chunks])
        scored = [(c, 1 / (1 + math.exp(-float(s)))) for c, s in zip(chunks, logits)]
        return sorted(scored, key=lambda x: -x[1])[:top_k]


def make_reranker(kind: str, index):
    return CrossEncoderReranker() if kind == "cross-encoder" else LexicalReranker(index.bm25)
