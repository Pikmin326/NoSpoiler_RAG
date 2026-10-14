"""인물·장소 엔티티와 시점별 별칭 사전.

별칭 사전 자체도 스포일러가 될 수 있다. 예를 들어 "회색 기사"가 사실 "카이든"이라는
연결은 정체가 공개되는 장 이전에는 쓰이면 안 된다. 그래서 각 (엔티티, 별칭) 연결마다
독자가 그 연결을 알 수 있게 되는 sequence(`known_seq`)를 계산해 둔다.

    known_seq = max(별칭 문자열이 원문에 처음 등장한 chunk의 sequence_end,
                    revealed_at으로 지정된 장의 sequence_start)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .corpus import Corpus
from .models import Chunk


@dataclass
class Alias:
    text: str
    revealed_at: tuple[int, int] | None = None  # (book, chapter)
    known_seq: int | None = None  # 원문에 한 번도 나오지 않으면 None


@dataclass
class Entity:
    entity_id: str
    canonical_name: str
    kind: str
    aliases: list[Alias] = field(default_factory=list)

    def first_known_seq(self) -> int | None:
        seqs = [a.known_seq for a in self.aliases if a.known_seq is not None]
        return min(seqs) if seqs else None


class EntityCatalog:
    def __init__(self, entities: list[Entity]):
        self.entities = {e.entity_id: e for e in entities}

    @classmethod
    def from_dict(cls, data: dict) -> "EntityCatalog":
        entities = []
        for raw in data.get("entities", []):
            aliases = []
            for a in raw.get("aliases", []):
                if isinstance(a, str):
                    aliases.append(Alias(a))
                else:
                    rev = a.get("revealed_at")
                    aliases.append(
                        Alias(a["text"], (rev["book"], rev["chapter"]) if rev else None)
                    )
            entities.append(
                Entity(raw["entity_id"], raw["canonical_name"], raw.get("kind", "character"), aliases)
            )
        return cls(entities)

    @classmethod
    def load(cls, path: str | Path) -> tuple["EntityCatalog", dict]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data), data

    # ------------------------------------------------------------------ ingest

    def bind(self, corpus: Corpus, chunks: list[Chunk]) -> None:
        """별칭의 known_seq를 계산하고 각 chunk에 mentions를 태깅한다."""
        for entity in self.entities.values():
            for alias in entity.aliases:
                reveal_seq = 0
                if alias.revealed_at:
                    reveal_seq = corpus.chapter(*alias.revealed_at).sequence_start
                first_seen = next(
                    (c.sequence_end for c in chunks if alias.text in c.text), None
                )
                alias.known_seq = None if first_seen is None else max(first_seen, reveal_seq)

        for chunk in chunks:
            chunk.mentions = {}
            for entity in self.entities.values():
                for alias in entity.aliases:
                    if alias.text not in chunk.text:
                        continue
                    reveal_seq = (
                        corpus.chapter(*alias.revealed_at).sequence_start if alias.revealed_at else 0
                    )
                    seq = max(chunk.sequence_end, reveal_seq)
                    prev = chunk.mentions.get(entity.entity_id)
                    chunk.mentions[entity.entity_id] = seq if prev is None else min(prev, seq)

    # ------------------------------------------------------------------- query

    def find_in_query(self, question: str, cutoff: int) -> list[str]:
        """질문에 나온 엔티티 중, cutoff 시점에 그 별칭 연결을 알 수 있는 것만 반환한다."""
        found: list[str] = []
        for entity in self.entities.values():
            for alias in entity.aliases:
                if (
                    alias.text in question
                    and alias.known_seq is not None
                    and alias.known_seq <= cutoff
                ):
                    found.append(entity.entity_id)
                    break
        return found

    def future_terms(self, cutoff: int) -> set[str]:
        """cutoff 시점에 독자가 아직 접하지 않은 이름 문자열."""
        known = {
            a.text
            for e in self.entities.values()
            for a in e.aliases
            if a.known_seq is not None and a.known_seq <= cutoff
        }
        future = {
            a.text
            for e in self.entities.values()
            for a in e.aliases
            if a.known_seq is not None and a.known_seq > cutoff
        }
        # 같은 문자열이 다른 엔티티의 별칭으로 이미 알려져 있다면(예: "회색 기사") 금지어가 아니다.
        return future - known

    def name(self, entity_id: str) -> str:
        return self.entities[entity_id].canonical_name

    def kind(self, entity_id: str) -> str:
        return self.entities[entity_id].kind
