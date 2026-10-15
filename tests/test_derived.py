"""Phase 3: 장 요약·권 요약·인물 스냅샷의 스포일러 안전성과 재현성 테스트."""

from __future__ import annotations

import pytest

from spoiler_rag import SpoilerSafeRAG
from spoiler_rag.config import SAMPLE_ENTITIES_PATH as ENTITIES_PATH
from spoiler_rag.config import SAMPLE_NOVEL_PATH as NOVEL_PATH
from spoiler_rag.corpus import chunk_corpus, load_novel
from spoiler_rag.entities import EntityCatalog
from spoiler_rag.firewall import enforce
from spoiler_rag.models import Progress
from spoiler_rag.summaries import (
    BOOK,
    CHAPTER,
    SNAPSHOT,
    ClaudeSummarizer,
    DerivedBuilder,
    DerivedIndex,
    ExtractiveSummarizer,
    SummaryStore,
)
from spoiler_rag.router import QueryType
from tests.test_spoiler_safety import RecordingGenerator


def make_parts():
    catalog, meta = EntityCatalog.load(ENTITIES_PATH)
    corpus = load_novel(NOVEL_PATH, meta["series_id"], meta.get("series_title", ""))
    chunks = chunk_corpus(corpus)
    catalog.bind(corpus, chunks)
    return corpus, chunks, catalog


class FakeSummarizer:
    """입력을 기록하고, 근거 첫 항목을 인용하는 요약을 돌려준다."""

    model = "fake"

    def __init__(self, reply: str | None = None):
        self.tasks = []
        self.reply = reply

    def summarize(self, task, feedback=None):
        self.tasks.append((task, feedback))
        return self.reply if self.reply is not None else f"요약입니다. [{task.evidence[0].chunk_id}]"


@pytest.fixture(scope="module")
def built():
    corpus, chunks, catalog = make_parts()
    store = SummaryStore()
    DerivedBuilder(corpus, chunks, catalog, ExtractiveSummarizer(), store).run()
    return corpus, chunks, catalog, store


# ------------------------------------------------------------ 생성 입력 범위


def test_summarizer_only_sees_text_up_to_the_scope():
    corpus, chunks, catalog = make_parts()
    fake = FakeSummarizer()
    DerivedBuilder(corpus, chunks, catalog, fake, SummaryStore()).run()
    assert fake.tasks
    for task, _ in fake.tasks:
        scope_end = max(c.sequence_end for c in task.evidence)
        # 이 과제의 모든 입력은 해당 시점 이전이어야 한다.
        for c in task.evidence:
            assert c.sequence_end <= scope_end
    # 1권 1장 요약을 만드는 시점에는 이후 장의 텍스트가 한 글자도 들어가지 않는다.
    first_task, _ = fake.tasks[0]
    ch1 = corpus.chapter(1, 1)
    assert all(c.sequence_end <= ch1.sequence_end for c in first_task.evidence)
    assert first_task.context == []
    sent_text = "\n".join(c.text for c in first_task.evidence)
    for p in corpus.paragraphs:
        if p.sequence > ch1.sequence_end:
            assert p.text not in sent_text, f"이후 문단이 입력에 포함됨: {p.sequence}"
    for p in corpus.paragraphs:  # 대조군: 1장 문단은 실제로 들어 있다
        if p.sequence <= ch1.sequence_end:
            assert p.text in sent_text


def test_chapter_context_only_contains_earlier_chapters():
    corpus, chunks, catalog = make_parts()
    fake = FakeSummarizer()
    store = SummaryStore()
    DerivedBuilder(corpus, chunks, catalog, fake, store).run()
    by_label = {t.label: t for t, _ in fake.tasks if t.subject is None and "전체" not in t.label}
    # 3장 요약의 참고 문맥은 1·2장 요약 두 개뿐이다 (3장 자신이나 이후 장이 아님).
    assert len(by_label["1권 3장"].context) == 2


# -------------------------------------------------------------- 레코드 속성


def test_every_record_carries_scope_evidence_and_metadata(built):
    corpus, chunks, catalog, store = built
    raw = {c.chunk_id: c for c in chunks}
    assert len(store) > 0
    for r in store.records:
        ch = corpus.chapter(r.book_number, r.chapter_number)
        assert r.generation_scope_end == ch.sequence_end == r.sequence_end
        assert r.sequence_start <= r.sequence_end
        assert r.evidence_chunk_ids and r.model and r.prompt_version and r.created_at and r.source_hash
        for cid in r.evidence_chunk_ids:
            if cid in raw:
                assert raw[cid].sequence_end <= r.generation_scope_end


def test_records_contain_no_name_the_reader_has_not_met_yet(built):
    corpus, chunks, catalog, store = built
    for r in store.records:
        for term in catalog.future_terms(r.generation_scope_end):
            assert term not in r.text, (r.record_id, term)


def test_kaiden_snapshot_does_not_exist_before_the_reveal(built):
    corpus, chunks, catalog, store = built
    reveal = corpus.chapter(2, 3).sequence_start
    kaiden = [r for r in store.records if r.character_id == "character_kaiden"]
    assert kaiden
    assert all(r.generation_scope_end >= reveal for r in kaiden)


def test_derived_chunks_pass_the_firewall_only_inside_the_range(built):
    corpus, chunks, catalog, store = built
    index = DerivedIndex(corpus.series_id, store.records)
    cutoff = corpus.chapter(1, 4).sequence_end
    pool = [r.as_chunk(corpus.series_id) for r in store.records]
    allowed, blocked = enforce(pool, cutoff)
    assert allowed and blocked
    assert all(c.sequence_end <= cutoff for c in allowed)
    # 조회기도 같은 경계를 지킨다.
    for c in index.summary_context(cutoff) + index.snapshot_timeline("character_lian", cutoff):
        assert c.sequence_end <= cutoff
    assert index.latest_snapshot("character_kaiden", cutoff) is None


# ------------------------------------------------------------------ 검증 실패


def test_leaky_summary_is_rejected_and_replaced_with_extractive():
    corpus, chunks, catalog = make_parts()
    fake = FakeSummarizer(reply="1장 요약: 회색 기사의 정체는 카이든이다. [C0001]")
    store = SummaryStore()
    DerivedBuilder(corpus, chunks, catalog, fake, store).run()
    ch1 = store.get(CHAPTER, 1, 1)
    assert ch1.status == "fallback"
    assert "카이든" not in ch1.text
    assert any("카이든" in i for i in ch1.issues)
    # 1회 재생성 후 포기하고, 재시도 때는 실패 사유가 피드백으로 전달된다.
    first_calls = [(t, fb) for t, fb in fake.tasks if t.label == "1권 1장" and t.subject is None]
    assert len(first_calls) == 2 and first_calls[1][1]


def test_api_error_falls_back_without_aborting_the_build():
    class Broken:
        model = "broken"

        def summarize(self, task, feedback=None):
            raise ConnectionError("network down")

    corpus, chunks, catalog = make_parts()
    store = SummaryStore()
    report = DerivedBuilder(corpus, chunks, catalog, Broken(), store).run()
    assert report.by_status == {"fallback": report.created}
    assert "ConnectionError" in store.get(CHAPTER, 1, 1).issues[0]


# ------------------------------------------------------- 캐시와 재현성


def test_cache_roundtrip_and_resume(tmp_path):
    corpus, chunks, catalog = make_parts()
    path = tmp_path / "s.json"
    store = SummaryStore()
    DerivedBuilder(corpus, chunks, catalog, ExtractiveSummarizer(), store).run()
    store.save(path)

    loaded = SummaryStore.load(path)
    assert len(loaded) == len(store)
    report = DerivedBuilder(corpus, chunks, catalog, ExtractiveSummarizer(), loaded).run()
    assert report.created == 0 and report.skipped > 0  # 이어서 하기: 유효한 레코드는 건너뛴다


def test_corrupt_cache_is_ignored(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("{not json", encoding="utf-8")
    assert len(SummaryStore.load(path)) == 0


def test_edited_source_invalidates_cached_records():
    corpus, chunks, catalog = make_parts()
    store = SummaryStore()
    DerivedBuilder(corpus, chunks, catalog, ExtractiveSummarizer(), store).run()
    total = len(store)

    chunks[0].text += " 원문이 수정되었다."  # 맨 앞 chunk만 바뀌어도 이후 모든 시점의 입력이 달라진다.
    report = DerivedBuilder(corpus, chunks, catalog, ExtractiveSummarizer(), store).run()
    assert report.dropped_stale == total
    assert report.created == total


def test_extractive_records_are_upgraded_by_a_real_summarizer():
    corpus, chunks, catalog = make_parts()
    store = SummaryStore()
    DerivedBuilder(corpus, chunks, catalog, ExtractiveSummarizer(), store).run()
    n = len(store)
    fake = FakeSummarizer()
    report = DerivedBuilder(corpus, chunks, catalog, fake, store).run()
    assert report.created == n
    assert {r.status for r in store.records} == {"verified"}
    assert {r.model for r in store.records} == {"fake"}
    assert len({r.record_id for r in store.records}) == n  # ID는 유지되고 중복되지 않는다


def test_claude_prompt_contains_only_given_evidence():
    class FakeClient:
        def __init__(self):
            self.sent = []

            class _M:
                def create(inner, **kw):
                    self.sent.append(kw)

                    class R:
                        stop_reason = "end_turn"
                        content = [type("B", (), {"type": "text", "text": "요약 [C0001]"})()]

                    return R()

            self.messages = _M()

    corpus, chunks, catalog = make_parts()
    client = FakeClient()
    store = SummaryStore()
    DerivedBuilder(corpus, chunks, catalog, ClaudeSummarizer(client, "m"), store).run()
    ch1 = corpus.chapter(1, 1)
    first_prompt = client.sent[0]["messages"][0]["content"]
    future_text = [p.text for p in corpus.paragraphs if p.sequence > ch1.sequence_end]
    assert not any(t[:20] in first_prompt for t in future_text)
    assert "<evidence>" in first_prompt and "1권 1장" in first_prompt


# ------------------------------------------------------------- 파이프라인


def test_summary_question_uses_chapter_summaries_within_progress():
    gen = RecordingGenerator()
    rag = SpoilerSafeRAG.build(NOVEL_PATH, ENTITIES_PATH, generator=gen)
    progress = rag.progress_for(1, 3)
    answer = rag.ask("지금까지 줄거리 요약해 줘", progress)
    assert answer.query_type is QueryType.SUMMARY
    kinds = {c.content_type for c in answer.evidence}
    assert CHAPTER in kinds
    assert BOOK not in kinds  # 1권이 아직 끝나지 않았으므로 권 요약은 쓰지 않는다
    assert all(c.sequence_end <= progress.current_sequence for c in answer.evidence)
    labels = {c.location_label for c in answer.evidence if c.content_type == CHAPTER}
    assert labels == {"1권 1장 요약", "1권 2장 요약", "1권 3장 요약"}


def test_mid_chapter_progress_excludes_the_current_chapter_summary():
    rag = SpoilerSafeRAG.build(NOVEL_PATH, ENTITIES_PATH, generator=RecordingGenerator())
    ch3 = rag.corpus.chapter(1, 3)
    mid = Progress("u", rag.corpus.series_id, 1, 3, ch3.sequence_start)  # 3장의 첫 문단까지만 읽음
    answer = rag.ask("지금까지 줄거리 요약해 줘", mid)
    assert all(c.sequence_end <= mid.current_sequence for c in answer.evidence)
    assert not any(c.location_label == "1권 3장 요약" for c in answer.evidence)


def test_character_question_gets_the_latest_snapshot_not_a_future_one():
    gen = RecordingGenerator()
    rag = SpoilerSafeRAG.build(NOVEL_PATH, ENTITIES_PATH, generator=gen)
    early = rag.progress_for(1, 2)
    answer = rag.ask("리안은 지금까지 어떤 일을 겪었어?", early)
    snaps = [c for c in answer.evidence if c.content_type == SNAPSHOT]
    assert len(snaps) == 1
    assert snaps[0].sequence_end == early.current_sequence  # 가장 최근, 그러나 현재 이하
    late = rag.progress_for(2, 4)
    answer = rag.ask("리안은 지금까지 어떤 일을 겪었어?", late)
    snaps = [c for c in answer.evidence if c.content_type == SNAPSHOT]
    assert snaps[0].sequence_end == late.current_sequence


def test_book_summary_replaces_chapters_of_finished_books_when_context_is_long(built):
    corpus, chunks, catalog, store = built
    index = DerivedIndex(corpus.series_id, store.records)
    end = corpus.max_sequence
    ctx = index.summary_context(end, max_items=5)
    types = [(c.content_type, c.book_number) for c in ctx]
    assert (BOOK, 1) in types and (BOOK, 2) in types
    assert all(t != CHAPTER for t, _ in types)
    # 1권이 끝나기 전 시점에서는 권 요약을 쓰지 않는다.
    mid = corpus.chapter(1, 3).sequence_end
    assert all(c.content_type != BOOK for c in index.summary_context(mid, max_items=1))


def test_pipeline_without_cache_still_builds_summaries_in_memory():
    rag = SpoilerSafeRAG.build(NOVEL_PATH, ENTITIES_PATH)
    assert rag.derived
    assert rag.derived.chapter_summaries(rag.corpus.max_sequence)
