"""규칙 기반 질문 유형 라우터 (초기 버전)."""

from __future__ import annotations

from enum import Enum


class QueryType(str, Enum):
    SUMMARY = "summary"  # 지금까지 줄거리 요약
    CHARACTER = "character"  # 특정 인물 소개·행적
    RELATION = "relation"  # 두 인물의 관계
    FIRST_APPEARANCE = "first_appearance"  # 처음 등장 위치
    FACT = "fact"  # 일반 사실 질문


LABELS = {
    QueryType.SUMMARY: "줄거리 요약",
    QueryType.CHARACTER: "인물",
    QueryType.RELATION: "관계",
    QueryType.FIRST_APPEARANCE: "첫 등장",
    QueryType.FACT: "사실 질문",
}

_SUMMARY = ("줄거리", "요약", "정리해", "지금까지 무슨", "지금까지 어떤 일", "무슨 일이 있었")
_FIRST = ("처음", "첫 등장", "언제 나왔", "어디서 나왔", "언제 등장", "처음 만났")
_RELATION = ("관계", "사이", "어떤 사이", "서로")


def route(question: str, entity_ids: list[str], character_ids: list[str]) -> QueryType:
    q = question.replace(" ", "")
    has = lambda keys: any(k.replace(" ", "") in q for k in keys)  # noqa: E731

    if entity_ids and has(_FIRST):
        return QueryType.FIRST_APPEARANCE
    if len(character_ids) >= 2 and has(_RELATION):
        return QueryType.RELATION
    if not entity_ids and has(_SUMMARY):
        return QueryType.SUMMARY
    # 정체가 공개된 별칭(예: "회색 기사")은 엔티티 두 개로 연결될 수 있으므로 1개 이상이면 인물 질문으로 본다.
    if character_ids:
        return QueryType.CHARACTER
    return QueryType.FACT
