"""Dependency-free BM25 (Okapi)."""
from __future__ import annotations

import math
from collections import Counter

from .text import tokens


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokens(t)) for t in corpus]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        self.n = len(corpus)
        self.df = Counter(t for d in self.docs for t in d)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        q = tokens(query)
        out = []
        for d, ln in zip(self.docs, self.lens):
            s = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out
