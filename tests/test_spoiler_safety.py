"""스포일러 안전성 회귀 테스트.

Phase 1 완료 기준: 읽지 않은 위치의 원문 chunk가 LLM 입력에 한 번도 포함되지 않는다.
"""

from __future__ import annotations

import pytest

from spoiler_rag import NEUTRAL_ANSWER, SpoilerSafeRAG
from spoiler_rag.config import ENTITIES_PATH, NOVEL_PATH
from spoiler_rag.corpus import chunk_corpus, parse_novel
from spoiler_rag.firewall import SpoilerLeakError, assert_safe, enforce
from spoiler_rag.llm import ClaudeGenerator
from spoiler_rag.models import Chunk, Turn
from spoiler_rag.router import QueryType
from spoiler_rag.verifier import verify

ADVERSARIAL_QUESTIONS = [
    "지금까지 줄거리 요약해 줘",
    "하벨은 어떻게 죽었어?",
    "하벨이 죽어?",
    "회색 기사의 정체는 뭐야?",
    "회색 기사는 하벨의 아들이야?",
    "카이든은 누구야?",
    "에일라는 누구야?",
    "모르간 영주가 배신해?",
    "등불이 꺼지는 일이 있어?",
    "리안은 나중에 혼자 등대지기가 돼?",
    "회색 기사가 나중에도 나와?",
    "리안과 회색 기사는 어떤 사이야?",
    "항해일지는 결국 어떻게 돼?",
    "검은 암초 아래 수로를 누가 찾아내?",
]


class RecordingGenerator:
    """LLM 대신 입력을 기록하는 가짜 생성기."""

    model = "fake"

    def __init__(self, reply: str | None = None):
        self.calls: list[dict] = []
        self.reply = reply

    def generate(self, *, question, progress_label, cutoff, evidence, history, feedback=None):
        self.calls.append({"cutoff": cutoff, "evidence": list(evidence), "history": list(history)})
        if self.reply is not None:
            return self.reply
        return f"근거에 따르면 다음과 같아요. [{evidence[0].chunk_id}]"


@pytest.fixture(scope="module")
def rag() -> SpoilerSafeRAG:
    return SpoilerSafeRAG.build(NOVEL_PATH, ENTITIES_PATH)


def all_positions(rag):
    return [(c.book_number, c.chapter_number) for c in rag.corpus.chapters]


# ------------------------------------------------------------------ ingest


def test_every_chunk_has_position_and_stays_inside_one_chapter(rag):
    for c in rag.chunks:
        assert c.has_position_metadata
        ch = rag.corpus.chapter(c.book_number, c.chapter_number)
        assert ch.sequence_start <= c.sequence_start <= c.sequence_end <= ch.sequence_end


def test_sequences_are_strictly_increasing(rag):
    seqs = [p.sequence for p in rag.corpus.paragraphs]
    assert seqs == list(range(1, len(seqs) + 1))


def test_parser_rejects_text_before_first_chapter():
    with pytest.raises(ValueError):
        parse_novel("# 1권\n본문이 장 제목보다 먼저 나옴\n\n## 1장\n문단", "s")


def test_overlapping_chunks_still_respect_chapter_boundary():
    corpus = parse_novel("# 1권\n## 1장\n가\n\n나\n\n다\n\n## 2장\n라\n\n마", "s")
    chunks = chunk_corpus(corpus, paragraphs_per_chunk=2, overlap=1)
    for c in chunks:
        assert corpus.chapter(c.book_number, c.chapter_number).sequence_end >= c.sequence_end


# --------------------------------------------------------------- retrieval


@pytest.mark.parametrize("question", ADVERSARIAL_QUESTIONS)
def test_indexes_never_return_future_chunks(rag, question):
    for book, chapter in all_positions(rag):
        cutoff = rag.progress_for(book, chapter).current_sequence
        for chunk, _ in rag.retriever.vector.search(question, cutoff, top_k=50):
            assert chunk.sequence_end <= cutoff
        for chunk, _ in rag.retriever.bm25.search(question, cutoff, top_k=50):
            assert chunk.sequence_end <= cutoff


@pytest.mark.parametrize("question", ADVERSARIAL_QUESTIONS)
def test_future_chunks_never_reach_the_llm(question):
    gen = RecordingGenerator()
    rag = SpoilerSafeRAG.build(NOVEL_PATH, ENTITIES_PATH, generator=gen)
    for book, chapter in all_positions(rag):
        progress = rag.progress_for(book, chapter)
        rag.ask(question, progress)
    assert gen.calls, "생성기가 한 번도 호출되지 않았다"
    for call in gen.calls:
        for chunk in call["evidence"]:
            assert chunk.sequence_end <= call["cutoff"], (question, chunk.chunk_id)


def test_identity_alias_is_not_linked_before_reveal(rag):
    before = rag.progress_for(1, 4).current_sequence
    after = rag.progress_for(2, 3).current_sequence
    assert "character_kaiden" not in rag.catalog.find_in_query("회색 기사는 누구야?", before)
    assert "character_kaiden" not in rag.catalog.find_in_query("카이든은 누구야?", before)
    assert "character_kaiden" in rag.catalog.find_in_query("회색 기사는 누구야?", after)
    for c in rag.retriever.by_entities(["character_kaiden"], before):
        pytest.fail(f"공개 전 카이든으로 태깅된 chunk가 반환됨: {c.chunk_id}")


def test_unknown_future_character_gets_neutral_or_past_only_evidence(rag):
    progress = rag.progress_for(1, 4)
    qtype, candidates, entity_ids = rag.retrieve("에일라는 누구야?", progress.current_sequence)
    assert "character_ayla" not in entity_ids
    assert all(c.sequence_end <= progress.current_sequence for c in candidates)


def test_normal_question_is_answered_not_refused():
    gen = RecordingGenerator()
    rag = SpoilerSafeRAG.build(NOVEL_PATH, ENTITIES_PATH, generator=gen)
    answer = rag.ask("하벨은 지금까지 어떤 일을 겪었어?", rag.progress_for(1, 4))
    assert answer.mode == "llm"
    assert answer.query_type is QueryType.CHARACTER
    assert answer.evidence


def test_first_appearance_uses_structured_lookup(rag):
    answer = rag.ask("회색 기사는 처음 어디서 나왔어?", rag.progress_for(2, 4), use_llm=False)
    assert answer.query_type is QueryType.FIRST_APPEARANCE
    assert "1권 2장" in answer.text


def test_revealed_alias_still_routes_as_character_question(rag):
    after = rag.progress_for(2, 4).current_sequence
    qtype, candidates, entity_ids = rag.retrieve("회색 기사는 누구야?", after)
    assert set(entity_ids) >= {"character_grey_knight", "character_kaiden"}
    assert qtype is QueryType.CHARACTER
    assert any("카이든" in c.text for c in candidates)


def test_relation_query_routes_and_uses_cooccurrence(rag):
    qtype, candidates, _ = rag.retrieve("리안과 세라는 어떤 사이야?", rag.progress_for(1, 4).current_sequence)
    assert qtype is QueryType.RELATION
    assert any({"character_lian", "character_sera"} <= set(c.mentions) for c in candidates)


# ---------------------------------------------------------------- firewall


def test_firewall_blocks_missing_metadata_and_boundary_crossing():
    ok = Chunk("C1", "a", "s", 1, 1, 1, 5)
    missing = Chunk("C2", "b", "s", 1, 1, None, None)
    crossing = Chunk("C3", "c", "s", 1, 1, 9, 12)
    future = Chunk("C4", "d", "s", 1, 2, 11, 12)
    allowed, blocked = enforce([ok, missing, crossing, future], cutoff=10)
    assert [c.chunk_id for c in allowed] == ["C1"]
    assert {c.chunk_id for c in blocked} == {"C2", "C3", "C4"}
    with pytest.raises(SpoilerLeakError):
        assert_safe([ok, future], cutoff=10)


def test_claude_generator_refuses_to_send_future_evidence():
    class ExplodingClient:
        class beta:  # noqa: N801
            class messages:  # noqa: N801
                @staticmethod
                def create(**_):
                    raise AssertionError("미래 근거가 있는데 API 요청이 전송됨")

    gen = ClaudeGenerator(client=ExplodingClient())
    future = Chunk("C9", "미래", "s", 2, 1, 20, 21)
    with pytest.raises(SpoilerLeakError):
        gen.generate(
            question="q", progress_label="1권 1장", cutoff=10, evidence=[future], history=[]
        )


# ---------------------------------------------------------------- verifier


def test_verifier_flags_future_names_and_fake_citations(rag):
    cutoff = rag.progress_for(1, 4).current_sequence
    evidence = [c for c in rag.chunks if c.sequence_end <= cutoff][:2]
    good = verify(f"리안은 견습 등대지기예요. [{evidence[0].chunk_id}]", evidence, cutoff, "리안은?", rag.catalog)
    assert good.passed

    bad = verify("회색 기사의 이름은 카이든이에요. [C0015]", evidence, cutoff, "회색 기사는?", rag.catalog)
    assert not bad.passed
    assert any("카이든" in i for i in bad.issues)
    assert any("C0015" in i for i in bad.issues)

    meta = verify("그건 나중에 밝혀지는 비밀이에요.", evidence, cutoff, "q", rag.catalog)
    assert not meta.passed


def test_name_written_by_user_is_not_a_leak(rag):
    cutoff = rag.progress_for(1, 4).current_sequence
    v = verify(f"'에일라'에 대해서는 {NEUTRAL_ANSWER}", [], cutoff, "에일라는 누구야?", rag.catalog)
    assert v.passed


def test_failed_verification_falls_back_to_neutral_answer():
    gen = RecordingGenerator(reply="회색 기사의 정체는 카이든이에요.")
    rag = SpoilerSafeRAG.build(NOVEL_PATH, ENTITIES_PATH, generator=gen)
    answer = rag.ask("회색 기사는 누구야?", rag.progress_for(1, 4))
    assert answer.mode == "fallback"
    assert answer.text == NEUTRAL_ANSWER
    assert len(gen.calls) == 2  # 1회 재생성 후 포기


# ------------------------------------------------------------ memory/rollback


def test_rollback_excludes_history_from_later_progress():
    gen = RecordingGenerator()
    rag = SpoilerSafeRAG.build(NOVEL_PATH, ENTITIES_PATH, generator=gen)
    late = rag.progress_for(2, 4)
    early = rag.progress_for(1, 2)
    history = [
        Turn("하벨은 어떻게 됐어?", "하벨은 숨을 거두었어요.", late.current_sequence),
        Turn("리안은 누구야?", "견습 등대지기예요.", early.current_sequence),
    ]
    rag.ask("하벨은 지금 뭐 해?", early, history=history)
    passed_history = gen.calls[-1]["history"]
    assert [t.question for t in passed_history] == ["리안은 누구야?"]
