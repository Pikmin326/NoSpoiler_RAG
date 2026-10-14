"""진행도 경계를 지키는 검색기 모음.

모든 메서드는 `cutoff`를 필수 인자로 받는다. 검색 경로별로 경계를 적용하고,
최종 결과는 파이프라인에서 firewall을 한 번 더 통과한다.
"""

from __future__ import annotations

from .entities import EntityCatalog
from .index import BM25Index, Embedder, VectorIndex
from .models import Chunk

RRF_K = 60


class SpoilerSafeRetriever:
    def __init__(self, chunks: list[Chunk], catalog: EntityCatalog, embedder: Embedder | None = None):
        self.chunks = sorted(chunks, key=lambda c: c.sequence_end)
        self.catalog = catalog
        self.vector = VectorIndex(chunks, embedder)
        self.bm25 = BM25Index(chunks)

    def _in_range(self, cutoff: int) -> list[Chunk]:
        return [c for c in self.chunks if c.has_position_metadata and c.sequence_end <= cutoff]

    def hybrid(
        self,
        query: str,
        cutoff: int,
        top_k: int = 6,
        boost_entities: list[str] | None = None,
    ) -> list[Chunk]:
        """벡터 + BM25 결과를 Reciprocal Rank Fusion으로 통합한다."""
        scores: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for results in (
            self.vector.search(query, max_sequence=cutoff, top_k=20),
            self.bm25.search(query, max_sequence=cutoff, top_k=20),
        ):
            for rank, (chunk, _) in enumerate(results):
                by_id[chunk.chunk_id] = chunk
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        if boost_entities:
            wanted = set(boost_entities)
            for cid, chunk in by_id.items():
                # cutoff 시점에 알 수 있는 엔티티 연결만 가산점에 사용한다.
                hits = len(wanted & chunk.entity_ids_known_at(cutoff))
                scores[cid] += hits * 0.5 / RRF_K

        ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [by_id[cid] for cid in ranked]

    def by_entities(
        self, entity_ids: list[str], cutoff: int, require_all: bool = False, limit: int = 12
    ) -> list[Chunk]:
        """엔티티가 언급된 chunk를 서술 순서대로 반환한다 (구조화 조회)."""
        wanted = set(entity_ids)
        out = []
        for c in self._in_range(cutoff):
            known = c.entity_ids_known_at(cutoff)
            if (wanted <= known) if require_all else (wanted & known):
                out.append(c)
        if len(out) > limit:
            out = _spread(out, limit)
        return out

    def first_mention(self, entity_id: str, cutoff: int) -> Chunk | None:
        for c in self._in_range(cutoff):
            if entity_id in c.entity_ids_known_at(cutoff):
                return c
        return None

    def timeline(self, cutoff: int, limit: int = 16) -> list[Chunk]:
        """줄거리 요약용: 허용 범위 전체를 서술 순서로, 장마다 최소 하나씩 고르게 뽑는다."""
        allowed = self._in_range(cutoff)
        if len(allowed) <= limit:
            return allowed
        per_chapter: dict[tuple[int, int], Chunk] = {}
        for c in allowed:
            per_chapter.setdefault((c.book_number, c.chapter_number), c)
        picked = {c.chunk_id for c in per_chapter.values()}
        for c in _spread(allowed, limit):
            if len(picked) >= limit:
                break
            picked.add(c.chunk_id)
        return [c for c in allowed if c.chunk_id in picked]


def _spread(items: list[Chunk], limit: int) -> list[Chunk]:
    if limit <= 0:
        return []
    if len(items) <= limit:
        return items
    step = (len(items) - 1) / max(limit - 1, 1)
    return [items[round(i * step)] for i in range(limit)]
