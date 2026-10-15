"""시점별 파생 데이터: 장 요약, 권 요약, 인물 상태 스냅샷 (명세서 Phase 3).

가장 위험한 ingest 패턴은 "전체 소설을 한 번에 넣고 장별 요약을 만들라"고 하는 것이다.
모델이 미래를 알고 있어 앞부분 요약에 정체 공개·후속 결과가 섞인다. 그래서 여기서는

    장을 순서대로 처리하고, 각 단계의 입력을 "그 장까지의 원문 + 이전 단계의 파생 데이터"로 제한한다.

모든 레코드는 입력으로 허용한 최대 sequence(`generation_scope_end`), 근거 chunk ID, 모델·프롬프트
버전, 원문 해시를 가진다. 원문이 바뀌면 해시가 달라져 오래된 캐시는 버려진다.
파생 레코드는 `Chunk`로 변환되어 검색 결과와 같은 firewall·검증기를 통과한다.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from . import firewall
from .corpus import Corpus
from .entities import EntityCatalog
from .llm import DEFAULT_MODEL, format_evidence, try_create_claude
from .models import Chunk
from .verifier import CITATION, verify

PROMPT_VERSION = "summary-v1"

CHAPTER = "chapter_summary"
BOOK = "book_summary"
SNAPSHOT = "character_snapshot"
_ID_PREFIX = {CHAPTER: "S", BOOK: "B", SNAPSHOT: "P"}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


# ------------------------------------------------------------------ record


@dataclass
class DerivedRecord:
    record_id: str
    kind: str
    book_number: int
    chapter_number: int  # 권 요약이면 그 권의 마지막 장
    sequence_start: int
    sequence_end: int
    text: str
    evidence_chunk_ids: list[str]
    generation_scope_end: int
    source_hash: str
    model: str
    prompt_version: str
    status: str  # verified | extractive | fallback
    created_at: str
    character_id: str | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple:
        return (self.kind, self.book_number, self.chapter_number, self.character_id)

    def location(self) -> str:
        if self.kind == BOOK:
            return f"{self.book_number}권 전체 요약"
        if self.kind == SNAPSHOT:
            return f"인물 상태 · {self.book_number}권 {self.chapter_number}장 시점"
        return f"{self.book_number}권 {self.chapter_number}장 요약"

    def as_chunk(self, series_id: str) -> Chunk:
        # 레코드가 담을 수 있는 정보의 상한은 generation_scope_end다. 이를 chunk의 sequence_end로 쓰면
        # 일반 원문 chunk와 똑같이 firewall의 `sequence_end <= cutoff` 검사를 받는다.
        return Chunk(
            chunk_id=self.record_id,
            text=self.text,
            series_id=series_id,
            book_number=self.book_number,
            chapter_number=self.chapter_number,
            sequence_start=self.sequence_start,
            sequence_end=self.generation_scope_end,
            mentions={self.character_id: self.generation_scope_end} if self.character_id else {},
            content_type=self.kind,
            location=self.location(),
        )


class SummaryStore:
    """파생 레코드 저장소 (JSON 파일 캐시)."""

    def __init__(self, records: list[DerivedRecord] | None = None):
        self._by_key: dict[tuple, DerivedRecord] = {}
        for r in records or []:
            self._by_key[r.key] = r

    @property
    def records(self) -> list[DerivedRecord]:
        return sorted(self._by_key.values(), key=lambda r: (r.generation_scope_end, r.kind, r.record_id))

    def __len__(self) -> int:
        return len(self._by_key)

    def get(self, kind: str, book: int, chapter: int, character_id: str | None = None) -> DerivedRecord | None:
        return self._by_key.get((kind, book, chapter, character_id))

    def put(self, record: DerivedRecord) -> None:
        self._by_key[record.key] = record

    def remove(self, record: DerivedRecord) -> None:
        self._by_key.pop(record.key, None)

    def next_id(self, kind: str) -> str:
        prefix = _ID_PREFIX[kind]
        used = [int(r.record_id[1:]) for r in self._by_key.values() if r.record_id.startswith(prefix)]
        return f"{prefix}{max(used, default=0) + 1:04d}"

    @classmethod
    def load(cls, path: str | Path | None) -> "SummaryStore":
        if not path or not Path(path).exists():
            return cls()
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls([DerivedRecord(**r) for r in raw.get("records", [])])
        except (ValueError, TypeError, KeyError):
            return cls()  # 깨진 캐시는 없는 것으로 보고 다시 만든다.

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "records": [asdict(r) for r in self.records]}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)


# -------------------------------------------------------------- summarizers


@dataclass
class Task:
    kind: str
    label: str  # "1권 2장" 등
    evidence: list[Chunk]
    context: list[str] = field(default_factory=list)  # 이전 시점의 파생 텍스트(참고용)
    subject: str | None = None  # 스냅샷 대상 이름
    subject_terms: list[str] = field(default_factory=list)  # 대상을 가리키는 표현(이 시점에 알려진 것만)


class Summarizer(Protocol):
    model: str

    def summarize(self, task: Task, feedback: str | None = None) -> str: ...


def _sentences(text: str) -> list[str]:
    return [s.strip() for chunk in text.split("\n") for s in _SENTENCE_SPLIT.split(chunk) if s.strip()]


def _spread_indices(n: int, limit: int) -> list[int]:
    if n <= limit:
        return list(range(n))
    step = (n - 1) / max(limit - 1, 1)
    return sorted({round(i * step) for i in range(limit)})


class ExtractiveSummarizer:
    """API 없이 쓰는 기본 요약기: 허용 범위 안의 원문 문장을 그대로 골라 근거 ID를 붙인다."""

    model = "extractive"

    def __init__(self, chapter_lines: int = 5, snapshot_lines: int = 8):
        self.chapter_lines = chapter_lines
        self.snapshot_lines = snapshot_lines

    def summarize(self, task: Task, feedback: str | None = None) -> str:
        if task.kind == CHAPTER:
            idx = _spread_indices(len(task.evidence), self.chapter_lines)
            lines = []
            for i in idx:
                c = task.evidence[i]
                if sents := _sentences(c.text):
                    lines.append(f"- {sents[0]} [{c.chunk_id}]")
            return "\n".join(lines)

        if task.kind == BOOK:
            # 근거는 장 요약이다. 각 장 요약의 첫 줄만 이어 붙인다.
            lines = []
            for c in task.evidence:
                first = next((ln for ln in c.text.splitlines() if ln.strip()), "")
                if first:
                    lines.append(first)
            return "\n".join(lines)

        # SNAPSHOT: 이전 상태 줄들 + 이번 장에서 대상이 언급된 문장
        new_lines = []
        for c in task.evidence:
            hits = [s for s in _sentences(c.text) if any(t in s for t in task.subject_terms)]
            new_lines += [f"- {s} [{c.chunk_id}]" for s in hits[:2]]
        carried = [ln for text in task.context for ln in text.splitlines() if ln.strip()]
        return "\n".join((carried + new_lines)[-self.snapshot_lines :])


SUMMARY_SYSTEM_PROMPT = """역할: 소설의 특정 시점까지의 내용을 정리하는 요약가

규칙:
1. <evidence>와 <context>에 있는 내용만 사용한다. 작품에 대한 사전 지식은 쓰지 않는다.
2. 지정된 시점 이후에 일어나는 일, 결과, 정체 공개, 사건의 훗날 의미를 암시하거나 추측하지 않는다.
3. 근거에 나오지 않은 이름과 사실을 만들지 않는다. 인물의 동기나 관계를 근거 이상으로 단정하지 않는다.
4. 원문에 명시된 사실과 추론을 구분한다. 추론은 "(추론)"이라고 표시한다.
5. 각 문장·항목 뒤에 근거 ID를 [C0001] 형식으로 붙인다. <evidence>와 <context>에 나온 ID만 쓴다.
6. "나중에 밝혀진다", "결말", "후반부" 같은 미래를 암시하는 표현을 쓰지 않는다.
7. 한국어로, 지정된 분량에 맞춰 간결하게 쓴다."""


def _build_prompt(task: Task, feedback: str | None) -> str:
    if task.kind == CHAPTER:
        goal = f"{task.label}의 내용을 3~5문장으로 요약하라. 사건의 순서를 지키고, 이 장에서 새로 알게 된 사실을 중심으로 쓴다."
        ctx_title = "이전 장 요약 (흐름 참고용, 이미 읽은 내용)"
    elif task.kind == BOOK:
        goal = f"{task.label}을 5~8문장으로 요약하라. 입력은 장 요약들이며, 큰 사건의 흐름만 남긴다."
        ctx_title = "참고"
    else:
        goal = (
            f"'{task.subject}'의 {task.label} 시점 상태를 불릿 3~6개로 정리하라. "
            "현재 처한 상황, 지금까지 겪은 일, 근거에 명시된 관계를 담는다. "
            "이전 시점 상태가 있으면 그것을 바탕으로 이번 장의 변화를 반영해 갱신한다."
        )
        ctx_title = "이전 시점의 상태 (갱신의 바탕)"

    parts = []
    if task.context:
        parts.append(f"<context title=\"{ctx_title}\">\n" + "\n\n".join(task.context) + "\n</context>")
    parts.append(f"<evidence>\n{format_evidence(task.evidence)}\n</evidence>")
    if feedback:
        parts.append(f"<revision_note>이전 초안이 검증을 통과하지 못했다: {feedback}\n규칙을 지켜 다시 작성하라.</revision_note>")
    parts.append(f"<task>{goal}</task>")
    return "\n\n".join(parts)


class ClaudeSummarizer:
    def __init__(self, client, model: str = DEFAULT_MODEL, effort: str = "low"):
        self.client = client
        self.model = model
        self.effort = effort

    def summarize(self, task: Task, feedback: str | None = None) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=SUMMARY_SYSTEM_PROMPT,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": _build_prompt(task, feedback)}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("모델이 요청을 거절했습니다.")
        return "".join(b.text for b in response.content if b.type == "text").strip()


def try_create_summarizer(model: str = DEFAULT_MODEL) -> tuple[Summarizer | None, str]:
    generator, reason = try_create_claude(model)
    if generator is None:
        return None, reason
    return ClaudeSummarizer(generator.client, model), ""


# ------------------------------------------------------------------ builder


@dataclass
class BuildReport:
    created: int = 0
    skipped: int = 0
    dropped_stale: int = 0
    by_status: dict[str, int] = field(default_factory=dict)


class DerivedBuilder:
    """장 순서대로 파생 레코드를 만든다. 이미 있는 유효한 레코드는 건너뛴다(재개 가능)."""

    def __init__(
        self,
        corpus: Corpus,
        chunks: list[Chunk],
        catalog: EntityCatalog,
        summarizer: Summarizer,
        store: SummaryStore,
        context_chapters: int = 3,
    ):
        self.corpus = corpus
        self.catalog = catalog
        self.summarizer = summarizer
        self.store = store
        self.context_chapters = context_chapters
        self.chunks = sorted((c for c in chunks if c.has_position_metadata), key=lambda c: c.sequence_start)
        self._raw = {c.chunk_id: c for c in self.chunks}
        self._hash_at = self._cumulative_hashes()

    # -- 원문 해시: 사전 지문 + scope_end까지의 모든 chunk 텍스트에 대한 누적 해시
    def _cumulative_hashes(self) -> dict[int, str]:
        out: dict[int, str] = {}
        # 사전이 바뀌면(예: 자동 추출을 다시 실행) 스냅샷 대상과 검증 기준이 달라지므로 캐시를 새로 만든다.
        h = hashlib.sha1(self.catalog.fingerprint().encode())
        it = iter(self.chunks)
        pending = next(it, None)
        for ch in self.corpus.chapters:
            while pending is not None and pending.sequence_end <= ch.sequence_end:
                h.update(pending.text.encode("utf-8"))
                h.update(b"\0")
                pending = next(it, None)
            out[ch.sequence_end] = h.copy().hexdigest()[:16]
        return out

    def _is_stale(self, r: DerivedRecord) -> bool:
        try:
            ch = self.corpus.chapter(r.book_number, r.chapter_number)
        except KeyError:
            return True
        if r.character_id and r.character_id not in self.catalog.entities:
            return True
        return (
            r.generation_scope_end != ch.sequence_end
            or self._hash_at.get(r.generation_scope_end) != r.source_hash
            or r.prompt_version != PROMPT_VERSION
        )

    def _needs(self, existing: DerivedRecord | None) -> bool:
        if existing is None:
            return True
        # 이전에 API 없이 만든 레코드는 LLM 요약기로 다시 만들 때 교체한다.
        return existing.status != "verified" and self.summarizer.model != "extractive"

    # -- 단위 생성
    def _generate(
        self, task: Task, scope_end: int, context_records: list[DerivedRecord]
    ) -> tuple[str, str, list[str]]:
        firewall.assert_safe(task.evidence, scope_end)
        if self.summarizer.model == "extractive":
            return self.summarizer.summarize(task), "extractive", []

        # 이전 단계 산출물이 인용한 ID도 허용한다(모두 더 이른 범위의 근거).
        allowed = list(task.evidence)
        for rec in context_records:
            allowed += [self._raw[i] for i in rec.evidence_chunk_ids if i in self._raw]
        feedback: str | None = None
        issues: list[str] = []
        for _ in range(2):
            try:
                text = self.summarizer.summarize(task, feedback)
            except Exception as exc:  # 네트워크·인증 오류 등
                issues = [f"{exc.__class__.__name__}: {exc}"]
                break
            result = verify(text, allowed, scope_end, "", self.catalog)
            if text.strip() and result.passed:
                return text, "verified", []
            issues = result.issues or ["빈 응답"]
            feedback = "; ".join(issues)
        return ExtractiveSummarizer().summarize(task), "fallback", issues

    def _record(
        self,
        kind: str,
        chapter,
        text: str,
        status: str,
        issues: list[str],
        evidence: list[Chunk],
        sequence_start: int,
        character_id: str | None = None,
    ) -> DerivedRecord:
        cited = [i for i in dict.fromkeys(CITATION.findall(text))]
        existing = self.store.get(kind, chapter.book_number, chapter.chapter_number, character_id)
        scope_end = chapter.sequence_end
        return DerivedRecord(
            record_id=existing.record_id if existing else self.store.next_id(kind),
            kind=kind,
            book_number=chapter.book_number,
            chapter_number=chapter.chapter_number,
            sequence_start=sequence_start,
            sequence_end=scope_end,
            text=text,
            evidence_chunk_ids=cited or [c.chunk_id for c in evidence],
            generation_scope_end=scope_end,
            source_hash=self._hash_at[scope_end],
            model=self.summarizer.model if status != "fallback" else f"{self.summarizer.model}->extractive",
            prompt_version=PROMPT_VERSION,
            status=status,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            character_id=character_id,
            issues=issues,
        )

    def run(
        self,
        on_record: Callable[[DerivedRecord], None] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> BuildReport:
        report = BuildReport()
        for r in list(self.store.records):
            if self._is_stale(r):
                self.store.remove(r)
                report.dropped_stale += 1

        characters = [e for e in self.catalog.entities.values() if e.kind == "character"]
        series_id = self.corpus.series_id

        def commit(rec: DerivedRecord) -> None:
            self.store.put(rec)
            report.created += 1
            report.by_status[rec.status] = report.by_status.get(rec.status, 0) + 1
            if on_record:
                on_record(rec)

        for chapter in self.corpus.chapters:
            scope_end = chapter.sequence_end
            in_chapter = [
                c for c in self.chunks if chapter.sequence_start <= c.sequence_start and c.sequence_end <= scope_end
            ]
            if progress:
                progress(f"{chapter.label} 처리 중")

            # 1) 장 요약
            existing = self.store.get(CHAPTER, chapter.book_number, chapter.chapter_number)
            if self._needs(existing):
                prior = [
                    r
                    for r in self.store.records
                    if r.kind == CHAPTER and r.generation_scope_end < chapter.sequence_start
                ][-self.context_chapters :]
                task = Task(CHAPTER, chapter.label, in_chapter, [r.text for r in prior])
                text, status, issues = self._generate(task, scope_end, prior)
                commit(self._record(CHAPTER, chapter, text, status, issues, in_chapter, chapter.sequence_start))
            else:
                report.skipped += 1

            # 2) 인물 스냅샷: 이 장에서 (이 시점에 알 수 있는 방식으로) 언급된 인물만
            for entity in characters:
                mentioned = [c for c in in_chapter if entity.entity_id in c.entity_ids_known_at(scope_end)]
                if not mentioned:
                    continue
                existing = self.store.get(SNAPSHOT, chapter.book_number, chapter.chapter_number, entity.entity_id)
                if not self._needs(existing):
                    report.skipped += 1
                    continue
                previous = [
                    r
                    for r in self.store.records
                    if r.kind == SNAPSHOT and r.character_id == entity.entity_id and r.generation_scope_end < scope_end
                ][-1:]
                terms = [a.text for a in entity.aliases if a.known_seq is not None and a.known_seq <= scope_end]
                task = Task(
                    SNAPSHOT,
                    chapter.label,
                    mentioned,
                    [r.text for r in previous],
                    # 정식 이름이 아직 공개 전이면 생성 프롬프트에도 넣지 않는다.
                    subject=self.catalog.display_name(entity.entity_id, scope_end) or entity.canonical_name,
                    subject_terms=terms,
                )
                text, status, issues = self._generate(task, scope_end, previous)
                commit(
                    self._record(
                        SNAPSHOT, chapter, text, status, issues, mentioned, mentioned[0].sequence_start, entity.entity_id
                    )
                )

            # 3) 권 요약: 그 권의 마지막 장을 처리한 직후, 장 요약들만을 입력으로 만든다(계층적 요약).
            last_of_book = self.corpus.chapters_of(chapter.book_number)[-1]
            if chapter == last_of_book:
                existing = self.store.get(BOOK, chapter.book_number, chapter.chapter_number)
                if self._needs(existing):
                    parts = [
                        r
                        for r in self.store.records
                        if r.kind == CHAPTER and r.book_number == chapter.book_number
                    ]
                    evidence = [r.as_chunk(series_id) for r in parts]
                    task = Task(BOOK, f"{chapter.book_number}권 전체", evidence)
                    # 입력은 이미 검증된 장 요약뿐이므로, 인용 가능한 ID는 그 장 요약 ID다.
                    text, status, issues = self._generate(task, scope_end, [])
                    start = self.corpus.chapters_of(chapter.book_number)[0].sequence_start
                    commit(self._record(BOOK, chapter, text, status, issues, evidence, start))
                else:
                    report.skipped += 1
        return report


# -------------------------------------------------------------------- index


class DerivedIndex:
    """진행도 이하의 파생 레코드만 돌려주는 조회기. 반환값은 모두 firewall을 통과하는 Chunk다."""

    def __init__(self, series_id: str, records: list[DerivedRecord]):
        self.series_id = series_id
        self._records = sorted(records, key=lambda r: r.generation_scope_end)

    def __bool__(self) -> bool:
        return bool(self._records)

    def _allowed(self, cutoff: int, kind: str) -> list[DerivedRecord]:
        return [r for r in self._records if r.kind == kind and r.generation_scope_end <= cutoff]

    def chapter_summaries(self, cutoff: int) -> list[Chunk]:
        return [r.as_chunk(self.series_id) for r in self._allowed(cutoff, CHAPTER)]

    def summary_context(self, cutoff: int, max_items: int = 24) -> list[Chunk]:
        """줄거리 요약용 근거. 장 요약이 많아지면 완료된 권은 권 요약으로 대체한다."""
        chapters = self._allowed(cutoff, CHAPTER)
        if len(chapters) <= max_items:
            return [r.as_chunk(self.series_id) for r in chapters]
        books = self._allowed(cutoff, BOOK)
        covered = {b.book_number for b in books}
        merged = sorted(
            books + [c for c in chapters if c.book_number not in covered],
            key=lambda r: (r.book_number, r.chapter_number, r.kind != BOOK),
        )
        return [r.as_chunk(self.series_id) for r in merged]

    def latest_snapshot(self, entity_id: str, cutoff: int) -> Chunk | None:
        found = [r for r in self._allowed(cutoff, SNAPSHOT) if r.character_id == entity_id]
        return found[-1].as_chunk(self.series_id) if found else None

    def snapshot_timeline(self, entity_id: str, cutoff: int) -> list[Chunk]:
        return [
            r.as_chunk(self.series_id) for r in self._allowed(cutoff, SNAPSHOT) if r.character_id == entity_id
        ]


def build_in_memory(
    corpus: Corpus,
    chunks: list[Chunk],
    catalog: EntityCatalog,
    cache_path: str | Path | None = None,
) -> DerivedIndex:
    """캐시(있으면)를 읽고, 비어 있는 부분은 API 없이 추출식으로 채워 메모리에서 쓴다. 파일은 쓰지 않는다."""
    store = SummaryStore.load(cache_path)
    DerivedBuilder(corpus, chunks, catalog, ExtractiveSummarizer(), store).run()
    return DerivedIndex(corpus.series_id, store.records)
