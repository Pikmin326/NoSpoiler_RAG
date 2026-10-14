"""질의 처리 파이프라인.

사용자 질문 + 진행도
  -> 엔티티 인식(시점 제한) / 질문 유형 라우팅
  -> 진행도 pre-filter가 걸린 검색 (벡터·키워드·구조화)
  -> firewall 재검증
  -> 허용된 근거만으로 답변 생성
  -> 출력 검증 (실패 시 1회 재생성, 그래도 실패하면 중립 응답)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import firewall
from .corpus import Corpus, chunk_corpus, load_novel
from .entities import EntityCatalog
from .llm import Generator
from .models import Chunk, Progress, Turn
from .retriever import SpoilerSafeRetriever
from .router import QueryType, route
from .store import QueryLogger
from .verifier import Verification, verify

NEUTRAL_ANSWER = "현재 읽은 범위의 근거만으로는 확인할 수 없어요."


@dataclass
class Answer:
    text: str
    evidence: list[Chunk]
    query_type: QueryType
    mode: str  # llm | extractive | fallback | no_evidence
    cutoff: int
    verification: Verification | None = None
    blocked_ids: list[str] = field(default_factory=list)
    note: str = ""


def _topic_josa(word: str) -> str:
    """받침 유무에 따라 '은/는'을 고른다."""
    last = word[-1] if word else ""
    if "가" <= last <= "힣":
        return "은" if (ord(last) - ord("가")) % 28 else "는"
    return "은(는)"


class SpoilerSafeRAG:
    def __init__(
        self,
        corpus: Corpus,
        chunks: list[Chunk],
        catalog: EntityCatalog,
        generator: Generator | None = None,
        logger: QueryLogger | None = None,
        history_turns: int = 4,
    ):
        self.corpus = corpus
        self.chunks = chunks
        self.catalog = catalog
        self.retriever = SpoilerSafeRetriever(chunks, catalog)
        self.generator = generator
        self.logger = logger or QueryLogger(None)
        self.history_turns = history_turns

    @classmethod
    def build(
        cls,
        novel_path: str | Path,
        entities_path: str | Path,
        paragraphs_per_chunk: int = 2,
        generator: Generator | None = None,
        log_path: str | Path | None = None,
    ) -> "SpoilerSafeRAG":
        catalog, meta = EntityCatalog.load(entities_path)
        corpus = load_novel(novel_path, meta["series_id"], meta.get("series_title", ""))
        chunks = chunk_corpus(corpus, paragraphs_per_chunk=paragraphs_per_chunk)
        catalog.bind(corpus, chunks)
        return cls(corpus, chunks, catalog, generator, QueryLogger(log_path))

    # ---------------------------------------------------------------- progress

    def progress_for(self, book: int, chapter: int, user_id: str = "local") -> Progress:
        """'book권 chapter장까지 읽음' -> 해당 장의 마지막 문단 sequence를 경계로 쓴다."""
        ch = self.corpus.chapter(book, chapter)
        return Progress(user_id, self.corpus.series_id, book, chapter, ch.sequence_end)

    # --------------------------------------------------------------- retrieval

    def retrieve(self, question: str, cutoff: int) -> tuple[QueryType, list[Chunk], list[str]]:
        entity_ids = self.catalog.find_in_query(question, cutoff)
        character_ids = [e for e in entity_ids if self.catalog.kind(e) == "character"]
        qtype = route(question, entity_ids, character_ids)
        r = self.retriever

        if qtype is QueryType.SUMMARY:
            candidates = r.timeline(cutoff)
        elif qtype is QueryType.CHARACTER:
            candidates = r.by_entities(character_ids, cutoff, limit=8) + r.hybrid(
                question, cutoff, top_k=4, boost_entities=entity_ids
            )
        elif qtype is QueryType.RELATION:
            candidates = r.by_entities(character_ids[:2], cutoff, require_all=True, limit=8) + r.hybrid(
                question, cutoff, top_k=4, boost_entities=entity_ids
            )
        elif qtype is QueryType.FIRST_APPEARANCE:
            first = r.first_mention(entity_ids[0], cutoff)
            candidates = ([first] if first else []) + r.hybrid(
                question, cutoff, top_k=3, boost_entities=entity_ids
            )
        else:
            candidates = r.hybrid(question, cutoff, top_k=6, boost_entities=entity_ids)

        return qtype, candidates, entity_ids

    # -------------------------------------------------------------------- ask

    def ask(
        self,
        question: str,
        progress: Progress,
        history: list[Turn] | None = None,
        use_llm: bool = True,
    ) -> Answer:
        cutoff = progress.current_sequence
        qtype, candidates, entity_ids = self.retrieve(question, cutoff)

        # 검색기의 필터만 믿지 않고 최종 후보를 다시 검증한다.
        ranked, blocked = firewall.enforce(candidates, cutoff)
        # LLM에는 서술 순서로, 검색 결과 표시는 관련도 순으로 전달한다.
        evidence = sorted(ranked, key=lambda c: c.sequence_start)

        if not evidence:
            answer = Answer(NEUTRAL_ANSWER, [], qtype, "no_evidence", cutoff)
        elif self.generator is None or not use_llm:
            answer = self._extractive(question, qtype, ranked, entity_ids, cutoff)
        else:
            # 진행도를 뒤로 돌렸을 때 이후 범위에서 생성된 대화는 컨텍스트에서 제외한다.
            safe_history = [t for t in (history or []) if t.cutoff <= cutoff][-self.history_turns :]
            try:
                answer = self._generate(question, progress, qtype, evidence, safe_history)
            except firewall.SpoilerLeakError:
                raise
            except Exception as exc:  # 네트워크·인증 오류 등은 검색 결과 표시로 대체
                answer = self._extractive(question, qtype, ranked, entity_ids, cutoff)
                answer.note = f"Claude 호출 실패({exc.__class__.__name__}): {exc}"

        answer.blocked_ids = [c.chunk_id for c in blocked]
        self.logger.log(
            {
                "user_id": progress.user_id,
                "series_id": progress.series_id,
                "progress": progress.label,
                "cutoff": cutoff,
                "question": question,
                "query_type": qtype.value,
                "entity_ids": entity_ids,
                "evidence_ids": [c.chunk_id for c in answer.evidence],
                "blocked_ids": answer.blocked_ids,
                "mode": answer.mode,
                "issues": answer.verification.issues if answer.verification else [],
                "note": answer.note,
            }
        )
        return answer

    def _generate(
        self,
        question: str,
        progress: Progress,
        qtype: QueryType,
        evidence: list[Chunk],
        history: list[Turn],
    ) -> Answer:
        cutoff = progress.current_sequence
        feedback: str | None = None
        verification: Verification | None = None
        for _ in range(2):
            draft = self.generator.generate(
                question=question,
                progress_label=progress.label,
                cutoff=cutoff,
                evidence=evidence,
                history=history,
                feedback=feedback,
            )
            verification = verify(draft, evidence, cutoff, question, self.catalog)
            if verification.passed:
                return Answer(draft, evidence, qtype, "llm", cutoff, verification)
            feedback = "; ".join(verification.issues)
        return Answer(
            NEUTRAL_ANSWER,
            evidence,
            qtype,
            "fallback",
            cutoff,
            verification,
            note="답변 검증에 실패해 안전한 응답으로 대체했습니다.",
        )

    def _extractive(
        self,
        question: str,
        qtype: QueryType,
        evidence: list[Chunk],
        entity_ids: list[str],
        cutoff: int,
    ) -> Answer:
        if qtype is QueryType.FIRST_APPEARANCE and entity_ids:
            name = self.catalog.name(entity_ids[0])
            first = self.retriever.first_mention(entity_ids[0], cutoff)
            if first:
                text = f"'{name}'{_topic_josa(name)} {first.location_label}에서 처음 언급돼요. [{first.chunk_id}]"
                return Answer(text, [first], qtype, "extractive", cutoff)

        lines = ["Claude가 연결되지 않아 검색된 근거 문단을 그대로 보여드려요.", ""]
        for c in evidence[:4]:
            excerpt = c.text.replace("\n\n", " ")
            if len(excerpt) > 220:
                excerpt = excerpt[:220] + "…"
            lines.append(f"- **{c.location_label}** [{c.chunk_id}] {excerpt}")
        return Answer("\n".join(lines), evidence, qtype, "extractive", cutoff)
