"""핵심 데이터 모델.

모든 검색 가능한 레코드는 서술 순서(narrative order) 기준의 위치 정보
(`sequence_start`, `sequence_end`)를 가진다. sequence는 작품 전체에서 문단마다
1씩 증가하는 정수다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Paragraph:
    sequence: int
    book_number: int
    chapter_number: int
    text: str


@dataclass(frozen=True)
class Chapter:
    book_number: int
    chapter_number: int
    book_title: str
    chapter_title: str
    sequence_start: int
    sequence_end: int

    @property
    def label(self) -> str:
        return f"{self.book_number}권 {self.chapter_number}장"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    series_id: str
    book_number: int
    chapter_number: int
    sequence_start: int | None
    sequence_end: int | None
    # entity_id -> 독자가 이 chunk에서 해당 엔티티를 "알 수 있게 되는" sequence.
    # 보통 chunk의 sequence_end이지만, 나중에 정체가 공개되는 별칭은 공개 시점이 된다.
    mentions: dict[str, int] = field(default_factory=dict)
    content_type: str = "raw_chunk"  # raw_chunk | chapter_summary | book_summary | character_snapshot
    location: str = ""  # 파생 레코드처럼 "N권 M장"으로 표현하기 어려운 경우의 표시 이름

    @property
    def has_position_metadata(self) -> bool:
        return isinstance(self.sequence_start, int) and isinstance(self.sequence_end, int)

    @property
    def location_label(self) -> str:
        return self.location or f"{self.book_number}권 {self.chapter_number}장"

    def entity_ids_known_at(self, cutoff: int) -> set[str]:
        return {eid for eid, seq in self.mentions.items() if seq <= cutoff}


@dataclass
class Progress:
    user_id: str
    series_id: str
    book_number: int
    chapter_number: int
    current_sequence: int
    updated_at: str = ""

    @property
    def label(self) -> str:
        return f"{self.book_number}권 {self.chapter_number}장"


@dataclass
class Turn:
    """대화 기록 한 턴. 생성 당시의 cutoff를 함께 저장해 진행도 변경 시 격리한다."""

    question: str
    answer: str
    cutoff: int
    evidence_ids: list[str] = field(default_factory=list)
