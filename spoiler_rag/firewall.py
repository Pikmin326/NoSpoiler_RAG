"""Retrieval Firewall: 진행도 경계를 강제하는 단일 지점.

모든 검색 경로(벡터·키워드·구조화)의 결과는 LLM에 전달되기 전에 이 모듈을 통과한다.
위치 메타데이터가 없거나 확인할 수 없는 레코드는 차단한다.
"""

from __future__ import annotations

from collections.abc import Iterable

from .models import Chunk


class SpoilerLeakError(RuntimeError):
    """허용 범위를 벗어난 근거가 생성 단계에 도달하려 할 때 발생한다."""


def is_allowed(chunk: Chunk, cutoff: int) -> bool:
    # sequence_start <= cutoff < sequence_end 인 경계 걸침 chunk도 차단된다.
    return chunk.has_position_metadata and chunk.sequence_end <= cutoff


def enforce(chunks: Iterable[Chunk], cutoff: int) -> tuple[list[Chunk], list[Chunk]]:
    allowed: list[Chunk] = []
    blocked: list[Chunk] = []
    seen: set[str] = set()
    for c in chunks:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        (allowed if is_allowed(c, cutoff) else blocked).append(c)
    return allowed, blocked


def assert_safe(chunks: Iterable[Chunk], cutoff: int) -> None:
    bad = [c.chunk_id for c in chunks if not is_allowed(c, cutoff)]
    if bad:
        raise SpoilerLeakError(f"허용 범위(sequence <= {cutoff})를 벗어난 근거: {bad}")
