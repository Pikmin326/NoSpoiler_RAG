"""인물·별칭 사전 자동 추출과 일반 텍스트 입력 처리 테스트.

실제 API 대신, 규칙대로 동작하는 가짜 추출기와 일부러 틀리게 동작하는 가짜 추출기를 쓴다.
"""

from __future__ import annotations

import json
import re

import pytest

from spoiler_rag import SpoilerSafeRAG
from spoiler_rag.config import SAMPLE_ENTITIES_PATH, SAMPLE_NOVEL_PATH
from spoiler_rag.corpus import chunk_corpus, load_novel, parse_novel
from spoiler_rag.entities import EntityCatalog
from spoiler_rag.entity_extraction import ClaudeEntityExtractor, EntityExtractionBuilder, compare_catalogs
from spoiler_rag.summaries import DerivedBuilder, ExtractiveSummarizer, SummaryStore

# 샘플 소설의 정답 (이름, 종류, 다른 표기). 가짜 추출기는 본문에 보이는 것만 보고한다.
TRUTH = [
    ("리안", "character", []),
    ("하벨", "character", ["하벨 영감"]),
    ("세라", "character", []),
    ("모르간 영주", "character", ["모르간"]),
    ("회색 기사", "character", []),
    ("카이든", "character", []),
    ("에일라", "character", []),
    ("엘른", "location", []),
    ("바위곶 등대", "location", ["등대"]),
    ("검은 암초", "location", []),
    ("카르벤", "location", []),
    ("항해일지", "item", ["일지"]),
]
REVEAL_MARK = "내 이름은 카이든"


def sentence_with(text: str, needle: str) -> str:
    return next(s for s in re.split(r"(?<=[.!?\"])\s+", text) if needle in s)


class RuleExtractor:
    """규칙을 지키는 가짜 추출기: 그 장 본문과 이미 아는 사전만 보고 판단한다."""

    model = "rule"

    def __init__(self):
        self.calls: list[dict] = []

    def extract(self, chapter_label, chapter_text, known):
        self.calls.append({"label": chapter_label, "text": chapter_text, "known": known})
        known_names = {a for k in known for a in k["aliases"]}
        out = {"entities": [], "aliases": [], "reveals": [], "notes": []}
        for name, kind, others in TRUTH:
            if name in chapter_text and name not in known_names:
                out["entities"].append(
                    {"name": name, "kind": kind, "other_names": [o for o in others if o in chapter_text], "note": ""}
                )
            elif name in known_names:
                ent = next(k for k in known if name in k["aliases"] and k["name"] == name)
                for o in others:
                    if o in chapter_text and o not in ent["aliases"]:
                        out["aliases"].append(
                            {"entity_id": ent["entity_id"], "alias": o, "quote": sentence_with(chapter_text, o)}
                        )
        if REVEAL_MARK in chapter_text:
            out["reveals"].append(
                {"known_as": "회색 기사", "true_name": "카이든", "quote": sentence_with(chapter_text, REVEAL_MARK)}
            )
        return out

    def confirm(self, chapter_text, name_a, name_b):
        if REVEAL_MARK in chapter_text and {name_a, name_b} == {"회색 기사", "카이든"}:
            return True, sentence_with(chapter_text, REVEAL_MARK)
        return False, ""


def load_sample():
    return load_novel(SAMPLE_NOVEL_PATH, "sample_novel", "sample_novel")


def bound(catalog: EntityCatalog, corpus):
    catalog.bind(corpus, chunk_corpus(corpus))
    return catalog


def reference(corpus):
    return bound(EntityCatalog.load(SAMPLE_ENTITIES_PATH)[0], corpus)


@pytest.fixture(scope="module")
def extracted():
    corpus = load_sample()
    extractor = RuleExtractor()
    builder = EntityExtractionBuilder(corpus, extractor)
    report = builder.run()
    return corpus, extractor, builder, report


# --------------------------------------------------------- 장 순서와 입력 범위


def test_extractor_sees_only_the_current_chapter(extracted):
    corpus, extractor, _, _ = extracted
    assert [c["label"] for c in extractor.calls] == [ch.label for ch in corpus.chapters]
    for call, chapter in zip(extractor.calls, corpus.chapters):
        for p in corpus.paragraphs:
            inside = chapter.sequence_start <= p.sequence <= chapter.sequence_end
            assert (p.text in call["text"]) == inside, (call["label"], p.sequence)


def test_known_dictionary_only_contains_earlier_chapters(extracted):
    corpus, extractor, _, _ = extracted
    for call, chapter in zip(extractor.calls, corpus.chapters):
        earlier = " ".join(p.text for p in corpus.paragraphs if p.sequence < chapter.sequence_start)
        for k in call["known"]:
            assert all(a in earlier for a in k["aliases"]), (call["label"], k)
    # 첫 장을 읽을 때 사전은 비어 있다.
    assert extractor.calls[0]["known"] == []


def test_every_alias_is_linked_from_the_chapter_it_was_learned(extracted):
    corpus, _, builder, _ = extracted
    kaiden = next(e for e in builder.entities.values() if e.name == "카이든")
    assert kaiden.aliases["회색 기사"] == (2, 3)
    assert "내 이름은 카이든이다" in kaiden.evidence["회색 기사"]
    grey = next(e for e in builder.entities.values() if e.name == "회색 기사")
    assert grey.aliases == {"회색 기사": (1, 2)}


# ----------------------------------------------------- 정답 사전과의 비교


def test_rule_following_extraction_matches_the_hand_written_dictionary(extracted):
    corpus, _, builder, report = extracted
    cmp = compare_catalogs(bound(builder.catalog(), corpus), reference(corpus))
    assert cmp.safe, (cmp.earlier, cmp.extra)
    assert not cmp.missing and not cmp.later
    assert any(m.startswith("카이든 ↔ 회색 기사") for m in cmp.matched)
    assert report.reveals == 1 and report.rejected == 0


# 두 이름이 공개 전부터 모두 알려진 작은 소설: "가면 기사"의 정체가 2장에서 "도현"으로 밝혀진다.
MINI_NOVEL = """# 1권
## 1장 가면
가면 기사가 마을에 왔다.

도윤은 사라진 형 도현을 그리워했다.

## 2장 정체
가면 기사가 투구를 벗었다. "내가 도현이다."
"""


def mini_catalog(revealed_chapter: int) -> EntityCatalog:
    return EntityCatalog.from_dict(
        {
            "entities": [
                {"entity_id": "mask", "canonical_name": "가면 기사", "aliases": ["가면 기사"]},
                {
                    "entity_id": "dohyun",
                    "canonical_name": "도현",
                    "aliases": ["도현", {"text": "가면 기사", "revealed_at": {"book": 1, "chapter": revealed_chapter}}],
                },
            ]
        }
    )


def test_comparison_flags_an_identity_linked_too_early():
    corpus = parse_novel(MINI_NOVEL, "mini")
    cmp = compare_catalogs(bound(mini_catalog(1), corpus), bound(mini_catalog(2), corpus))
    assert not cmp.safe
    assert any(x.startswith("가면 기사 ↔ 도현") for x in cmp.earlier)
    # 이 사전이 실제로 위험한 이유: 1장 독자의 "가면 기사" 질문이 도현에 대한 문단으로 이어진다.
    rag_corpus_cutoff = corpus.chapter(1, 1).sequence_end
    early = bound(mini_catalog(1), corpus)
    assert "dohyun" in early.find_in_query("가면 기사는 누구야?", rag_corpus_cutoff)


def test_merged_entity_like_the_real_model_output_is_not_a_leak():
    """실제 모델은 회색 기사와 카이든을 한 대상으로 묶고 "카이든"을 2권 3장부터 붙였다. 이는 안전하다."""
    corpus = load_sample()
    raw = json.loads(SAMPLE_ENTITIES_PATH.read_text(encoding="utf-8"))
    raw["entities"] = [e for e in raw["entities"] if e["entity_id"] != "character_kaiden"]
    grey = next(e for e in raw["entities"] if e["entity_id"] == "character_grey_knight")
    grey["aliases"] = ["회색 기사", {"text": "카이든", "revealed_at": {"book": 2, "chapter": 3}}]
    merged = bound(EntityCatalog.from_dict(raw), corpus)
    cmp = compare_catalogs(merged, reference(corpus))
    assert not cmp.earlier and not cmp.extra
    assert any(m.startswith("카이든 ↔ 회색 기사") for m in cmp.matched)
    before = corpus.chapter(1, 4).sequence_end
    assert merged.display_name("character_grey_knight", before) == "회색 기사"
    assert "카이든" in merged.future_terms(before)


# ------------------------------------------------- 정식 이름 노출 경로 차단


def kaiden_first_catalog_path(tmp_path):
    """정식 이름이 "카이든"인 대상에 "회색 기사"가 1권부터 붙은 사전 (회색 기사 별도 대상 없음)."""
    raw = json.loads(SAMPLE_ENTITIES_PATH.read_text(encoding="utf-8"))
    raw["entities"] = [e for e in raw["entities"] if e["entity_id"] != "character_grey_knight"]
    for e in raw["entities"]:
        if e["entity_id"] == "character_kaiden":
            e["aliases"] = ["카이든", "회색 기사"]
    path = tmp_path / "kaiden_first.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return path


def test_display_name_hides_a_canonical_name_not_yet_revealed(tmp_path):
    rag = SpoilerSafeRAG.build(SAMPLE_NOVEL_PATH, kaiden_first_catalog_path(tmp_path))
    before = rag.progress_for(1, 4)
    assert rag.catalog.display_name("character_kaiden", before.current_sequence) == "회색 기사"
    assert rag.catalog.display_name("character_kaiden", rag.progress_for(2, 4).current_sequence) == "카이든"
    answer = rag.ask("회색 기사는 처음 어디서 나왔어?", before, use_llm=False)
    assert "회색 기사" in answer.text and "카이든" not in answer.text


def test_snapshot_prompt_does_not_name_an_unrevealed_canonical_name(tmp_path):
    corpus = load_sample()
    chunks = chunk_corpus(corpus)
    catalog = EntityCatalog.load(kaiden_first_catalog_path(tmp_path))[0]
    catalog.bind(corpus, chunks)

    class Recorder:
        model = "rec"

        def __init__(self):
            self.subjects = []

        def summarize(self, task, feedback=None):
            if task.subject:
                self.subjects.append((task.label, task.subject))
            return f"요약 [{task.evidence[0].chunk_id}]"

    rec = Recorder()
    DerivedBuilder(corpus, chunks, catalog, rec, SummaryStore()).run()
    book1 = [s for label, s in rec.subjects if label.startswith("1권")]
    assert "회색 기사" in book1 and "카이든" not in book1


def test_auto_dictionary_keeps_the_identity_hidden_before_the_reveal(extracted):
    corpus, _, builder, _ = extracted
    catalog = bound(builder.catalog(), corpus)
    kaiden_id = next(e.entity_id for e in builder.entities.values() if e.name == "카이든")
    before = corpus.chapter(1, 4).sequence_end
    after = corpus.chapter(2, 3).sequence_end
    assert kaiden_id not in catalog.find_in_query("회색 기사는 누구야?", before)
    assert kaiden_id in catalog.find_in_query("회색 기사는 누구야?", after)
    assert "카이든" in catalog.future_terms(before)


# ----------------------------------------------------------- 추측 차단


class ScriptedExtractor(RuleExtractor):
    """특정 장에서 일부러 틀린 결과를 섞는 추출기."""

    def __init__(self, at: str, extra: dict, confirm_reply=(False, "")):
        super().__init__()
        self.at, self.extra, self.confirm_reply = at, extra, confirm_reply

    def extract(self, chapter_label, chapter_text, known):
        out = super().extract(chapter_label, chapter_text, known)
        if chapter_label == self.at:
            for key, items in self.extra.items():
                out[key] = out[key] + items
        return out

    def confirm(self, chapter_text, name_a, name_b):
        if REVEAL_MARK in chapter_text:
            return super().confirm(chapter_text, name_a, name_b)
        return self.confirm_reply


def run_with(extractor):
    builder = EntityExtractionBuilder(load_sample(), extractor)
    builder.run()
    return builder


def test_names_not_in_the_text_are_rejected():
    # 사전 지식으로 미래 인물을 미리 적는 경우
    b = run_with(ScriptedExtractor("1권 2장", {"entities": [{"name": "카이든", "kind": "character", "other_names": [], "note": ""}]}))
    kaiden = next(e for e in b.entities.values() if e.name == "카이든")
    assert kaiden.first_seen == (2, 3)
    assert any(r["chapter"] == "1권 2장" and r["reason"] == "본문에 없는 이름" for r in b.rejected)


def test_guessed_identity_with_a_future_name_is_rejected():
    b = run_with(ScriptedExtractor("1권 2장", {"reveals": [{"known_as": "회색 기사", "true_name": "카이든", "quote": "회색 기사"}]}))
    kaiden = next(e for e in b.entities.values() if e.name == "카이든")
    assert kaiden.aliases["회색 기사"] == (2, 3)
    assert any(r["chapter"] == "1권 2장" for r in b.rejected)


def test_hinted_identity_is_rejected_by_the_confirmation_step():
    # 1권 2장: 하벨이 회색 기사의 얼굴을 바라보는 '암시'만 있다. 확인 단계가 명시적 서술이 아니라고 답한다.
    quote = "회색 기사가 투구를 벗고 물을 마실 때, 하벨은 그의 얼굴을 한참 바라보다가 아무 말 없이 등대로 돌아갔다."
    b = run_with(ScriptedExtractor("1권 2장", {"reveals": [{"known_as": "회색 기사", "true_name": "하벨", "quote": quote}]}))
    havel = next(e for e in b.entities.values() if e.name == "하벨")
    assert "회색 기사" not in havel.aliases
    assert any("명시적 서술 없음" in r["reason"] for r in b.rejected)


def test_confirmation_with_a_fabricated_quote_is_rejected():
    b = run_with(
        ScriptedExtractor(
            "1권 2장",
            {"reveals": [{"known_as": "회색 기사", "true_name": "하벨", "quote": "x"}]},
            confirm_reply=(True, "회색 기사는 하벨의 아들이었다."),  # 본문에 없는 문장
        )
    )
    havel = next(e for e in b.entities.values() if e.name == "하벨")
    assert "회색 기사" not in havel.aliases
    assert any("근거 문장이 본문과 일치하지 않음" in r["reason"] for r in b.rejected)


def test_alias_with_a_quote_not_in_the_text_is_rejected():
    b = run_with(ScriptedExtractor("1권 3장", {"aliases": [{"entity_id": "character_002", "alias": "영주", "quote": "하벨이 곧 영주다."}]}))
    havel = b.entities["character_002"]
    assert havel.name == "하벨" and "영주" not in havel.aliases


# --------------------------------------------------------- 이어서 하기


class FailingAt(RuleExtractor):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def extract(self, chapter_label, chapter_text, known):
        if chapter_label == self.label:
            raise ConnectionError("network down")
        return super().extract(chapter_label, chapter_text, known)


def test_error_stops_in_order_and_resumes_from_that_chapter(tmp_path):
    corpus = load_sample()
    path = tmp_path / "entities.json"
    first = EntityExtractionBuilder(corpus, FailingAt("1권 3장"))
    report = first.run(on_chapter=lambda _l, b: b.save(path))
    first.save(path)
    assert "1권 3장" in report.error and first.processed == [(1, 1), (1, 2)]  # 실패한 장 이후로 건너뛰지 않는다

    extractor = RuleExtractor()
    second = EntityExtractionBuilder.load(corpus, extractor, path)
    report = second.run()
    assert report.skipped == 2 and not report.error
    assert extractor.calls[0]["label"] == "1권 3장"
    ids_before = {e.entity_id: e.name for e in first.entities.values()}
    assert all(second.entities[i].name == n for i, n in ids_before.items())


def test_edited_source_restarts_extraction(tmp_path, extracted):
    _, _, builder, _ = extracted
    path = tmp_path / "entities.json"
    builder.save(path)
    edited = parse_novel(SAMPLE_NOVEL_PATH.read_text(encoding="utf-8").replace("엘른은", "엘른 마을은", 1), "s")
    again = EntityExtractionBuilder.load(edited, RuleExtractor(), path)
    assert again.restarted and again.processed == [] and not again.entities


# --------------------------------------------------- 파이프라인과의 연결


def test_pipeline_runs_without_any_dictionary(tmp_path):
    rag = SpoilerSafeRAG.build(SAMPLE_NOVEL_PATH, tmp_path / "missing.json")
    assert not rag.catalog.entities
    assert rag.corpus.series_id == "sample_novel"
    answer = rag.ask("지금까지 줄거리 요약해 줘", rag.progress_for(1, 2))
    assert all(c.sequence_end <= rag.progress_for(1, 2).current_sequence for c in answer.evidence)


def test_pipeline_uses_the_auto_dictionary(tmp_path, extracted):
    _, _, builder, _ = extracted
    path = tmp_path / "entities.json"
    builder.save(path)
    rag = SpoilerSafeRAG.build(SAMPLE_NOVEL_PATH, path)
    assert len(rag.catalog.entities) == len(TRUTH)
    qtype, _, ids = rag.retrieve("회색 기사는 누구야?", rag.progress_for(1, 4).current_sequence)
    assert [rag.catalog.name(i) for i in ids] == ["회색 기사"]


def test_dictionary_change_invalidates_summary_cache(extracted):
    corpus, _, builder, _ = extracted
    chunks = chunk_corpus(corpus)
    store = SummaryStore()
    empty = EntityCatalog([])
    empty.bind(corpus, chunks)
    DerivedBuilder(corpus, chunks, empty, ExtractiveSummarizer(), store).run()
    catalog = builder.catalog()
    catalog.bind(corpus, chunks)
    report = DerivedBuilder(corpus, chunks, catalog, ExtractiveSummarizer(), store).run()
    assert report.dropped_stale > 0
    assert any(r.character_id for r in store.records)  # 사전이 생기자 인물 스냅샷도 만들어진다


def test_claude_extractor_requests_structured_json():
    class FakeClient:
        def __init__(self):
            self.sent = []
            outer = self

            class Messages:
                def create(self, **kw):
                    outer.sent.append(kw)

                    class R:
                        stop_reason = "end_turn"
                        content = [type("B", (), {"type": "text", "text": json.dumps(
                            {"entities": [], "aliases": [], "reveals": [], "notes": []})})()]

                    return R()

            self.messages = Messages()

    client = FakeClient()
    result = ClaudeEntityExtractor(client, "m").extract("1권 1장", "본문", [])
    assert result["entities"] == []
    fmt = client.sent[0]["output_config"]["format"]
    assert fmt["type"] == "json_schema" and "reveals" in fmt["schema"]["properties"]


# ------------------------------------------------------- 일반 텍스트 입력


def test_plain_text_headings_are_detected():
    corpus = parse_novel("프롤로그\n\n처음.\n\n제1장 시작\n\n가.\n\n3장의 편지를 받았다.\n\n제2장: 끝\n\n나.", "x")
    assert corpus.structure == "detected_headings"
    assert [c.chapter_title for c in corpus.chapters] == ["프롤로그", "제1장 시작", "제2장: 끝"]
    assert any("3장의 편지" in p.text for p in corpus.paragraphs)  # 본문 문장은 장 제목으로 오인하지 않는다


def test_text_without_headings_is_split_automatically():
    corpus = parse_novel("\n\n".join(f"문단 {i}." for i in range(70)), "y")
    assert corpus.structure == "auto_split"
    assert [(c.sequence_start, c.sequence_end) for c in corpus.chapters] == [(1, 30), (31, 60), (61, 70)]


def test_cp949_file_is_readable(tmp_path):
    path = tmp_path / "novel.txt"
    path.write_bytes("# 1권\n## 1장\n안녕하세요.".encode("cp949"))
    assert load_novel(path, "z").paragraphs[0].text == "안녕하세요."
