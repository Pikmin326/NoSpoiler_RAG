"""소설에서 인물·장소·별칭 사전을 자동으로 추출한다 (시점 인식).

별칭 사전에서 가장 위험한 오류는 "정체가 공개되는 시점"을 실제보다 **이르게** 적는 것이다.
그러면 공개 전에 검색이 숨은 정체로 이어져 스포일러가 샌다. 그래서 다음 원칙으로 만든다.

1. 장 순서 추출: N장을 처리할 때 모델에 주는 것은 N장 본문과, N-1장까지 만들어진 사전뿐이다.
   모델은 N장 이후의 본문을 본 적이 없으므로 미래의 정체 공개를 "알 수" 없다.
2. 배운 장부터 연결: 모든 이름·별칭에 그것을 처음 확인한 장을 `revealed_at`으로 기록한다.
   추출이 늦거나 놓치면 연결이 늦어질 뿐(정보가 적어질 뿐) 이르게 되지는 않는다.
3. 추측 차단: 이름·별칭은 그 장 본문에 글자 그대로 있어야 받는다. 근거 문장(quote)도 본문과 대조한다.
   이미 다른 대상의 호칭인 문자열을 새 대상에 연결하는 경우(정체 공개)는 별도 확인 호출에서
   "본문에 명시적으로 서술되어 있다"는 답과 본문 그대로의 근거 문장이 나와야만 받는다.
4. 확실하지 않으면 버린다. 버린 항목은 사유와 함께 `rejected`에 남긴다.

`note`(대상 설명)는 다음 장 추출의 참고 자료로만 쓰며 답변 생성에는 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from .corpus import Corpus
from .entities import EntityCatalog
from .llm import DEFAULT_MODEL, try_create_claude

PROMPT_VERSION = "entities-v1"
KINDS = ("character", "location", "group", "item")
MAX_PART_CHARS = 30000  # 아주 긴 장은 이 길이 단위로 나눠 순서대로 처리한다.

_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "「": '"', "」": '"', "『": '"', "』": '"'})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.translate(_QUOTES)).strip()


def _clean_name(s: str) -> str:
    return _norm(s).strip("\"'.,·「」[]()")


# ---------------------------------------------------------------- extractor


class EntityExtractor(Protocol):
    model: str

    def extract(self, chapter_label: str, chapter_text: str, known: list[dict]) -> dict: ...

    def confirm(self, chapter_text: str, name_a: str, name_b: str) -> tuple[bool, str]: ...


EXTRACTION_SYSTEM_PROMPT = """역할: 소설을 앞에서부터 한 장씩 읽으며 등장 대상(인물·장소·집단·물건)과 그 호칭을 기록하는 정리자

규칙:
1. <chapter> 본문과 <known_entities>만 근거로 삼는다. 작품에 대한 사전 지식은 절대 쓰지 않는다. 이 장 이후의 일을 추측하지 않는다.
2. 이름과 호칭은 본문에 적힌 글자 그대로 옮기고, 뒤에 붙은 조사(은/는/이/가/을/를/의/에게 등)는 뗀다.
3. entities: 이 장에 처음 나온 대상 중 고유한 이름이나 고유한 호칭(예: "회색 기사", "바위곶 등대")으로 불리는 것만 넣는다. known_entities에 이미 있는 대상, 대명사, "남자"·"병사들" 같은 일반 명사는 넣지 않는다. other_names에는 같은 장에서 같은 대상을 가리키는 다른 표기(예: "모르간 영주"와 "모르간")를 넣는다.
4. aliases: known_entities의 대상이 이 장에서 새로운 이름으로 불리고, 같은 대상이라는 것이 본문에서 분명할 때만 넣는다. quote에는 그 근거 문장을 본문 그대로 옮긴다.
5. reveals: 지금까지 서로 다른 대상처럼 불리던 두 호칭이 같은 대상이라는 사실이 이 장 본문에 명시적으로 서술될 때만 넣는다(예: 이름을 숨기던 인물이 본명을 밝힘). 암시, 의심, 분위기, 독자의 추측은 넣지 않는다. quote에는 그 사실을 밝히는 문장을 본문 그대로 옮긴다.
6. notes: 대상을 이 장까지 읽은 독자가 아는 범위에서 한 문장으로 설명한다. 이후의 내용은 쓰지 않는다.
7. 확실하지 않으면 넣지 않는다. 빠뜨리는 것은 괜찮지만 잘못 연결하는 것은 안 된다."""

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "other_names": {"type": "array", "items": {"type": "string"}},
                    "note": {"type": "string"},
                },
                "required": ["name", "kind", "other_names", "note"],
                "additionalProperties": False,
            },
        },
        "aliases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                    "alias": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["entity_id", "alias", "quote"],
                "additionalProperties": False,
            },
        },
        "reveals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "known_as": {"type": "string"},
                    "true_name": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["known_as", "true_name", "quote"],
                "additionalProperties": False,
            },
        },
        "notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"entity_id": {"type": "string"}, "note": {"type": "string"}},
                "required": ["entity_id", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "aliases", "reveals", "notes"],
    "additionalProperties": False,
}

CONFIRM_SYSTEM_PROMPT = """역할: 소설 본문 한 장만 보고 사실 여부를 판정하는 검토자

규칙:
1. <chapter> 본문만 근거로 삼는다. 작품에 대한 사전 지식과 추측은 쓰지 않는다.
2. 두 호칭이 같은 대상이라는 사실이 본문에 명시적으로 서술되어 있으면 explicit=true, 암시·의심·정황뿐이면 false다.
3. explicit=true이면 quote에 그 사실을 밝히는 문장을 본문 그대로 옮긴다. false이면 quote는 빈 문자열이다."""

CONFIRM_SCHEMA = {
    "type": "object",
    "properties": {"explicit": {"type": "boolean"}, "quote": {"type": "string"}},
    "required": ["explicit", "quote"],
    "additionalProperties": False,
}


class ClaudeEntityExtractor:
    def __init__(self, client, model: str = DEFAULT_MODEL, effort: str = "medium"):
        self.client = client
        self.model = model
        self.effort = effort

    def _call(self, system: str, prompt: str, schema: dict) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("모델이 요청을 거절했습니다.")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("응답이 max_tokens에서 잘렸습니다.")
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)

    def extract(self, chapter_label: str, chapter_text: str, known: list[dict]) -> dict:
        prompt = (
            f"<known_entities>\n{json.dumps(known, ensure_ascii=False, indent=1)}\n</known_entities>\n\n"
            f'<chapter label="{chapter_label}">\n{chapter_text}\n</chapter>\n\n'
            "<task>이 장에서 새로 확인되는 대상, 별칭, 정체 공개, 설명을 규칙에 따라 기록하라.</task>"
        )
        return self._call(EXTRACTION_SYSTEM_PROMPT, prompt, EXTRACTION_SCHEMA)

    def confirm(self, chapter_text: str, name_a: str, name_b: str) -> tuple[bool, str]:
        prompt = (
            f"<chapter>\n{chapter_text}\n</chapter>\n\n"
            f"<question>이 본문에 '{name_a}'와(과) '{name_b}'가 같은 대상이라는 사실이 명시적으로 서술되어 있는가?</question>"
        )
        data = self._call(CONFIRM_SYSTEM_PROMPT, prompt, CONFIRM_SCHEMA)
        return bool(data.get("explicit")), str(data.get("quote", ""))


def try_create_extractor(model: str = DEFAULT_MODEL) -> tuple[EntityExtractor | None, str]:
    generator, reason = try_create_claude(model)
    if generator is None:
        return None, reason
    return ClaudeEntityExtractor(generator.client, model), ""


# ------------------------------------------------------------------ builder


@dataclass
class _Entity:
    entity_id: str
    name: str
    kind: str
    first_seen: tuple[int, int]
    aliases: dict[str, tuple[int, int]] = field(default_factory=dict)  # 호칭 -> 연결을 배운 (권, 장)
    evidence: dict[str, str] = field(default_factory=dict)  # 호칭 -> 근거 문장
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.name,
            "kind": self.kind,
            "aliases": [
                {
                    "text": text,
                    "revealed_at": {"book": b, "chapter": c},
                    **({"evidence": self.evidence[text]} if text in self.evidence else {}),
                }
                for text, (b, c) in self.aliases.items()
            ],
            "first_seen": {"book": self.first_seen[0], "chapter": self.first_seen[1]},
            "note": self.note,  # 추출용 참고 자료. 답변 생성에는 쓰지 않는다.
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_Entity":
        fs = d.get("first_seen") or {"book": 0, "chapter": 0}
        ent = cls(d["entity_id"], d["canonical_name"], d.get("kind", "character"), (fs["book"], fs["chapter"]))
        for a in d.get("aliases", []):
            rev = a["revealed_at"]
            ent.aliases[a["text"]] = (rev["book"], rev["chapter"])
            if a.get("evidence"):
                ent.evidence[a["text"]] = a["evidence"]
        ent.note = d.get("note", "")
        return ent


@dataclass
class ExtractionReport:
    processed: int = 0
    skipped: int = 0
    added_entities: int = 0
    added_aliases: int = 0
    reveals: int = 0
    rejected: int = 0
    error: str = ""
    restarted: bool = False


class EntityExtractionBuilder:
    """장 순서대로 추출해 사전을 만든다. 장마다 저장하므로 중간에 멈춰도 이어서 할 수 있다."""

    def __init__(self, corpus: Corpus, extractor: EntityExtractor, state: dict | None = None):
        self.corpus = corpus
        self.extractor = extractor
        self.entities: dict[str, _Entity] = {}
        self.processed: list[tuple[int, int]] = []
        self.rejected: list[dict] = []
        self._hashes = self._cumulative_hashes()
        self.restarted = False
        if state:
            self._restore(state)

    # -- 상태 저장/복원
    def _cumulative_hashes(self) -> dict[tuple[int, int], str]:
        out = {}
        h = hashlib.sha1(PROMPT_VERSION.encode())
        for ch in self.corpus.chapters:
            for p in self.corpus.paragraphs:
                if ch.sequence_start <= p.sequence <= ch.sequence_end:
                    h.update(p.text.encode("utf-8"))
                    h.update(b"\0")
            out[(ch.book_number, ch.chapter_number)] = h.copy().hexdigest()[:16]
        return out

    def _restore(self, state: dict) -> None:
        meta = state.get("auto_extraction") or {}
        processed = [tuple(x) for x in meta.get("processed_chapters", [])]
        # 처리했던 마지막 장까지의 원문이 그대로이고 프롬프트 버전이 같을 때만 이어서 한다.
        if (
            not processed
            or meta.get("prompt_version") != PROMPT_VERSION
            or self._hashes.get(processed[-1]) != meta.get("source_hash")
        ):
            self.restarted = bool(processed)
            return
        self.processed = processed
        self.entities = {d["entity_id"]: _Entity.from_dict(d) for d in state.get("entities", [])}
        self.rejected = list(meta.get("rejected", []))

    def to_dict(self) -> dict:
        last = self.processed[-1] if self.processed else None
        return {
            "series_id": self.corpus.series_id,
            "series_title": self.corpus.series_title,
            "auto_extraction": {
                "model": self.extractor.model,
                "prompt_version": PROMPT_VERSION,
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "processed_chapters": [list(x) for x in self.processed],
                "complete": len(self.processed) == len(self.corpus.chapters),
                "source_hash": self._hashes.get(last) if last else None,
                "rejected": self.rejected,
            },
            "entities": [e.to_dict() for e in self.entities.values()],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, corpus: Corpus, extractor: EntityExtractor, path: str | Path) -> "EntityExtractionBuilder":
        state = None
        if Path(path).exists():
            try:
                state = json.loads(Path(path).read_text(encoding="utf-8"))
            except ValueError:
                state = None
        return cls(corpus, extractor, state)

    def catalog(self) -> EntityCatalog:
        return EntityCatalog.from_dict(self.to_dict())

    # -- 도우미
    def _alias_owners(self, text: str) -> set[str]:
        return {e.entity_id for e in self.entities.values() if text in e.aliases}

    def _resolve(self, name: str) -> _Entity | None:
        owners = self._alias_owners(name)
        exact = [self.entities[i] for i in owners if self.entities[i].name == name]
        if exact:
            return exact[0]
        return self.entities[next(iter(owners))] if len(owners) == 1 else None

    def _known_list(self) -> list[dict]:
        return [
            {"entity_id": e.entity_id, "name": e.name, "kind": e.kind, "aliases": list(e.aliases), "note": e.note}
            for e in self.entities.values()
        ]

    def _new_id(self, kind: str) -> str:
        n = sum(1 for e in self.entities.values() if e.kind == kind) + 1
        return f"{kind}_{n:03d}"

    def _reject(self, label: str, item: str, reason: str, report: ExtractionReport) -> None:
        self.rejected.append({"chapter": label, "item": item, "reason": reason})
        report.rejected += 1

    @staticmethod
    def _in_text(s: str, text: str) -> bool:
        return len(s) >= 2 and _norm(s) in _norm(text)

    @staticmethod
    def _quote_ok(quote: str, text: str) -> bool:
        return len(_norm(quote)) >= 4 and _norm(quote) in _norm(text)

    def _link_existing_name(
        self, ent: _Entity, alias: str, where: tuple[int, int], label: str, text: str, report: ExtractionReport
    ) -> None:
        """이미 다른 대상의 호칭인 문자열을 ent에 연결한다 = 정체 공개. 확인 호출을 반드시 통과해야 한다."""
        if ent.entity_id in self._alias_owners(alias):
            return
        if not (self._in_text(alias, text) and any(self._in_text(n, text) for n in ent.aliases)):
            self._reject(label, f"{alias} = {ent.name}", "두 호칭이 이 장 본문에 함께 나오지 않음", report)
            return
        explicit, quote = self.extractor.confirm(text, alias, ent.name)
        if not explicit:
            self._reject(label, f"{alias} = {ent.name}", "확인 단계: 본문에 명시적 서술 없음", report)
            return
        if not self._quote_ok(quote, text):
            self._reject(label, f"{alias} = {ent.name}", "확인 단계: 근거 문장이 본문과 일치하지 않음", report)
            return
        ent.aliases[alias] = where
        ent.evidence[alias] = _norm(quote)
        report.reveals += 1

    # -- 한 장(또는 장의 일부) 처리
    def _apply(self, result: dict, where: tuple[int, int], label: str, text: str, report: ExtractionReport) -> None:
        # 1) 새 대상
        for item in result.get("entities", []):
            name = _clean_name(item.get("name", ""))
            kind = item.get("kind") if item.get("kind") in KINDS else "character"
            if not self._in_text(name, text):
                self._reject(label, name, "본문에 없는 이름", report)
                continue
            if self._alias_owners(name):
                continue  # 이미 아는 이름
            ent = _Entity(self._new_id(kind), name, kind, where, {name: where}, {}, _norm(item.get("note", "")))
            self.entities[ent.entity_id] = ent
            report.added_entities += 1
            for other in item.get("other_names", []):
                other = _clean_name(other)
                if not self._in_text(other, text):
                    self._reject(label, f"{other} ({name})", "본문에 없는 별칭", report)
                elif self._alias_owners(other):
                    self._link_existing_name(ent, other, where, label, text, report)
                else:
                    ent.aliases[other] = where

        # 2) 아는 대상의 새 별칭
        for item in result.get("aliases", []):
            ent = self.entities.get(item.get("entity_id", ""))
            alias = _clean_name(item.get("alias", ""))
            if ent is None:
                self._reject(label, alias, "존재하지 않는 대상 ID", report)
            elif alias in ent.aliases:
                continue
            elif not self._in_text(alias, text):
                self._reject(label, f"{alias} ({ent.name})", "본문에 없는 별칭", report)
            elif self._alias_owners(alias):
                self._link_existing_name(ent, alias, where, label, text, report)
            elif not (self._quote_ok(item.get("quote", ""), text) and alias in _norm(item.get("quote", ""))):
                self._reject(label, f"{alias} ({ent.name})", "근거 문장이 본문과 일치하지 않음", report)
            else:
                ent.aliases[alias] = where
                ent.evidence[alias] = _norm(item["quote"])
                report.added_aliases += 1

        # 3) 정체 공개
        for item in result.get("reveals", []):
            known_as = _clean_name(item.get("known_as", ""))
            true_name = _clean_name(item.get("true_name", ""))
            target = self._resolve(true_name)
            if not self._alias_owners(known_as):
                self._reject(label, f"{known_as} = {true_name}", "알려지지 않은 호칭", report)
            elif target is None:
                self._reject(label, f"{known_as} = {true_name}", "대상 이름을 특정할 수 없음", report)
            else:
                self._link_existing_name(target, known_as, where, label, text, report)

        # 4) 설명 갱신 (다음 장 추출용)
        for item in result.get("notes", []):
            ent = self.entities.get(item.get("entity_id", ""))
            if ent and item.get("note"):
                ent.note = _norm(item["note"])

    def _parts(self, chapter) -> list[str]:
        paras = [p.text for p in self.corpus.paragraphs if chapter.sequence_start <= p.sequence <= chapter.sequence_end]
        parts, buf, size = [], [], 0
        for p in paras:
            if buf and size + len(p) > MAX_PART_CHARS:
                parts.append("\n\n".join(buf))
                buf, size = [], 0
            buf.append(p)
            size += len(p)
        if buf:
            parts.append("\n\n".join(buf))
        return parts

    def run(
        self,
        on_chapter: Callable[[str, "EntityExtractionBuilder"], None] | None = None,
    ) -> ExtractionReport:
        report = ExtractionReport(restarted=self.restarted)
        done = set(self.processed)
        for chapter in self.corpus.chapters:
            where = (chapter.book_number, chapter.chapter_number)
            if where in done:
                report.skipped += 1
                continue
            try:
                for text in self._parts(chapter):
                    result = self.extractor.extract(chapter.label, text, self._known_list())
                    self._apply(result, where, chapter.label, text, report)
            except Exception as exc:  # 네트워크·인증 오류: 장 순서를 지키려면 여기서 멈춰야 한다.
                report.error = f"{chapter.label}에서 중단 ({exc.__class__.__name__}: {exc})"
                break
            self.processed.append(where)
            report.processed += 1
            if on_chapter:
                on_chapter(chapter.label, self)
        return report


# --------------------------------------------------------------- comparison


@dataclass
class Comparison:
    """두 사전의 연결을 비교한다.

    연결 "x ↔ y"는 "x와 y가 같은 대상의 이름"이라는 정보이고, 그 정보를 쓸 수 있게 되는 시점은
    max(x의 known_seq, y의 known_seq)다. 대상에서 꺼낼 수 있는 이름·태그·표시 이름(`display_name`)이
    모두 시점 제한을 받으므로, 한쪽 이름이 아직 공개 전이면 대상에 닿아도 두 이름의 연결은 드러나지 않는다.
    예: "회색 기사"(1권)와 "카이든"(2권 3장 공개)을 한 대상으로 묶어도, 연결 시점은 "카이든"이 알려지는 때다.
    """

    matched: list[str] = field(default_factory=list)
    later: list[str] = field(default_factory=list)  # 자동이 더 늦게 연결: 정보가 적어질 뿐 안전
    earlier: list[str] = field(default_factory=list)  # 자동이 더 이르게 연결: 스포일러 위험
    missing: list[str] = field(default_factory=list)  # 자동이 연결하지 않음: 안전
    extra: list[str] = field(default_factory=list)  # 정답에 없는 연결: 확인 필요
    names_missing: list[str] = field(default_factory=list)
    names_extra: list[str] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        return not self.earlier and not self.extra


def _links(catalog: EntityCatalog) -> dict[tuple[str, str], int]:
    links: dict[tuple[str, str], int] = {}
    for e in catalog.entities.values():
        known = [(a.text, a.known_seq) for a in e.aliases if a.known_seq is not None]
        for x, sx in known:
            for y, sy in known:
                if x < y:
                    seq = max(sx, sy)
                    links[(x, y)] = min(links.get((x, y), seq), seq)
    return links


def _names(catalog: EntityCatalog) -> set[str]:
    return {a.text for e in catalog.entities.values() for a in e.aliases if a.known_seq is not None}


def compare_catalogs(auto: EntityCatalog, reference: EntityCatalog) -> Comparison:
    """두 사전 모두 같은 코퍼스에 bind된 상태여야 한다."""
    out = Comparison()
    a, r = _links(auto), _links(reference)
    fmt = lambda k: f"{k[0]} ↔ {k[1]}"  # noqa: E731
    for key, seq in sorted(r.items(), key=lambda kv: kv[1]):
        if key not in a:
            out.missing.append(f"{fmt(key)} (정답 {seq})")
        elif a[key] < seq:
            out.earlier.append(f"{fmt(key)} (자동 {a[key]} < 정답 {seq})")
        elif a[key] > seq:
            out.later.append(f"{fmt(key)} (자동 {a[key]} > 정답 {seq})")
        else:
            out.matched.append(f"{fmt(key)} ({seq})")
    for key, seq in sorted(a.items(), key=lambda kv: kv[1]):
        if key not in r:
            out.extra.append(f"{fmt(key)} (자동 {seq})")
    out.names_missing = sorted(_names(reference) - _names(auto))
    out.names_extra = sorted(_names(auto) - _names(reference))
    return out
