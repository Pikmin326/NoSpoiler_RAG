"""벡터 인덱스와 BM25 키워드 인덱스.

두 인덱스 모두 **pre-filter** 방식이다. 점수를 계산하기 전에 `sequence_end <= max_sequence`
조건으로 후보 집합을 먼저 자르므로, top-k가 미래 chunk로 채워지는 일이 없다.

임베딩은 외부 의존성 없이 동작하는 문자 n-gram TF-IDF 벡터를 기본값으로 쓴다(baseline).
`Embedder` 프로토콜을 구현하면 다국어 신경망 임베딩 모델로 교체할 수 있다.
"""

from __future__ import annotations

import bisect
import math
import zlib
from collections import Counter
from typing import Protocol

from .models import Chunk
from .text import bm25_tokens, char_ngrams

SparseVec = dict[int, float]


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> None: ...
    def embed(self, text: str) -> SparseVec: ...


class CharNgramTfidfEmbedder:
    """해시된 문자 2·3-gram의 TF-IDF 벡터 (L2 정규화)."""

    def __init__(self, dim: int = 1 << 18):
        self.dim = dim
        self.idf: dict[int, float] = {}
        self.default_idf = 1.0

    def _features(self, text: str) -> Counter[int]:
        # Python의 hash()는 실행마다 달라지므로 안정적인 crc32를 쓴다.
        return Counter(zlib.crc32(g.encode("utf-8")) % self.dim for g in char_ngrams(text))

    def fit(self, texts: list[str]) -> None:
        df: Counter[int] = Counter()
        for t in texts:
            df.update(set(self._features(t)))
        n = len(texts)
        self.idf = {f: math.log((1 + n) / (1 + c)) + 1.0 for f, c in df.items()}
        self.default_idf = math.log(1 + n) + 1.0

    def embed(self, text: str) -> SparseVec:
        tf = self._features(text)
        vec = {f: (1 + math.log(c)) * self.idf.get(f, self.default_idf) for f, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {f: v / norm for f, v in vec.items()}


def _cosine(a: SparseVec, b: SparseVec) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(f, 0.0) for f, v in a.items())


class _PositionalIndex:
    """sequence_end 순으로 정렬된 chunk 목록과 이분 탐색 기반 pre-filter."""

    def __init__(self, chunks: list[Chunk]):
        missing = [c.chunk_id for c in chunks if not c.has_position_metadata]
        if missing:
            raise ValueError(f"위치 메타데이터가 없는 chunk는 색인할 수 없습니다: {missing}")
        self.chunks = sorted(chunks, key=lambda c: (c.sequence_end, c.sequence_start))
        self._ends = [c.sequence_end for c in self.chunks]

    def _allowed_count(self, max_sequence: int) -> int:
        return bisect.bisect_right(self._ends, max_sequence)


class VectorIndex(_PositionalIndex):
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        super().__init__(chunks)
        self.embedder = embedder or CharNgramTfidfEmbedder()
        self.embedder.fit([c.text for c in self.chunks])
        self._vectors = [self.embedder.embed(c.text) for c in self.chunks]

    def search(self, query: str, max_sequence: int, top_k: int = 20) -> list[tuple[Chunk, float]]:
        n = self._allowed_count(max_sequence)
        q = self.embedder.embed(query)
        scored = [(self.chunks[i], _cosine(q, self._vectors[i])) for i in range(n)]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda s: s[1], reverse=True)
        return scored[:top_k]


class BM25Index(_PositionalIndex):
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        super().__init__(chunks)
        self.k1, self.b = k1, b
        self._tfs = [Counter(bm25_tokens(c.text)) for c in self.chunks]
        self._lens = [sum(tf.values()) for tf in self._tfs]

    def search(self, query: str, max_sequence: int, top_k: int = 20) -> list[tuple[Chunk, float]]:
        n = self._allowed_count(max_sequence)
        if n == 0:
            return []
        # 문서 빈도·평균 길이도 허용 범위 안의 chunk만으로 계산한다.
        tfs, lens = self._tfs[:n], self._lens[:n]
        avgdl = sum(lens) / n
        terms = set(bm25_tokens(query))
        df = {t: sum(1 for tf in tfs if t in tf) for t in terms}

        scored = []
        for i in range(n):
            tf, dl, score = tfs[i], lens[i], 0.0
            for t in terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
            if score > 0:
                scored.append((self.chunks[i], score))
        scored.sort(key=lambda s: s[1], reverse=True)
        return scored[:top_k]
