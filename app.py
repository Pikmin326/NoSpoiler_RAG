"""Streamlit 채팅 인터페이스.

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from spoiler_rag.config import ENTITIES_PATH, LOG_PATH, NOVEL_PATH, PATHS, PROGRESS_PATH, SUMMARIES_PATH
from spoiler_rag.llm import DEFAULT_MODEL, try_create_claude
from spoiler_rag.models import Turn
from spoiler_rag.pipeline import Answer, SpoilerSafeRAG
from spoiler_rag.router import LABELS
from spoiler_rag.store import ProgressStore

USER_ID = "local"
MODE_LABELS = {
    "llm": "Claude 답변",
    "extractive": "검색 결과",
    "fallback": "검증 실패 → 안전 응답",
    "no_evidence": "근거 없음",
}
EXAMPLES = [
    "지금까지 줄거리 요약해 줘",
    "회색 기사는 누구야?",
    "리안과 세라는 어떤 사이야?",
    "하벨은 지금까지 어떤 일을 겪었어?",
]

st.set_page_config(page_title="Spoiler-safe 소설 도우미", page_icon="📖")


@st.cache_resource
def load_rag() -> SpoilerSafeRAG:
    generator, _ = load_generator()
    return SpoilerSafeRAG.build(NOVEL_PATH, ENTITIES_PATH, generator=generator, log_path=LOG_PATH, summaries_path=SUMMARIES_PATH)


@st.cache_resource
def load_generator():
    return try_create_claude(DEFAULT_MODEL)


rag = load_rag()
generator, generator_reason = load_generator()
store = ProgressStore(PROGRESS_PATH)
corpus = rag.corpus

ss = st.session_state
ss.setdefault("history", [])  # list[dict]: {"turn": Turn, "answer": Answer}
ss.setdefault("pending", None)

first = corpus.chapters[0]
saved = store.get(USER_ID, corpus.series_id)
try:
    # 원문이 바뀌었을 수 있으므로 저장된 위치의 sequence를 현재 코퍼스 기준으로 다시 계산한다.
    saved = rag.progress_for(saved.book_number, saved.chapter_number, USER_ID) if saved else None
except KeyError:
    saved = None
saved = saved or rag.progress_for(first.book_number, first.chapter_number, USER_ID)


def is_read(book: int, chapter: int) -> bool:
    """이미 읽은 장인지. 읽지 않은 장의 제목은 스포일러가 될 수 있어 숨긴다."""
    return (book, chapter) <= (saved.book_number, saved.chapter_number)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("📖 읽은 위치")
    st.caption(corpus.series_title)

    book_titles = dict(corpus.books())
    book_numbers = list(book_titles)
    book = st.selectbox(
        "권",
        book_numbers,
        index=book_numbers.index(saved.book_number),
        format_func=lambda b: f"{b}권 · {book_titles[b]}" if b <= saved.book_number else f"{b}권",
    )
    chapters = corpus.chapters_of(book)
    chapter_numbers = [c.chapter_number for c in chapters]
    chapter_titles = {c.chapter_number: c.chapter_title for c in chapters}
    default_ch = saved.chapter_number if book == saved.book_number else chapter_numbers[0]
    chapter = st.selectbox(
        "장 (이 장까지 읽음)",
        chapter_numbers,
        index=chapter_numbers.index(default_ch) if default_ch in chapter_numbers else 0,
        format_func=lambda n: f"{n}장 · {chapter_titles[n]}" if is_read(book, n) else f"{n}장",
    )

    progress = rag.progress_for(book, chapter, USER_ID)
    if (progress.book_number, progress.chapter_number) != (saved.book_number, saved.chapter_number):
        store.set(progress)
        if progress.current_sequence < saved.current_sequence:
            # 진행도를 뒤로 돌리면 이후 범위에서 생성된 대화를 제거한다 (메모리 스포일러 방지).
            before = len(ss.history)
            ss.history = [h for h in ss.history if h["turn"].cutoff <= progress.current_sequence]
            removed = before - len(ss.history)
            if removed:
                st.toast(f"읽은 위치가 앞당겨져 이후 범위에서 나눈 대화 {removed}건을 숨겼어요.")
        st.rerun()

    st.progress(
        progress.current_sequence / corpus.max_sequence,
        text=f"허용 범위: 문단 1–{progress.current_sequence} / 전체 {corpus.max_sequence}",
    )

    n_entities = len(rag.catalog.entities)
    if PATHS.manual_entities:
        st.caption(f"인물 사전: 지정한 사전 · {n_entities}개")
    elif n_entities:
        st.caption(f"인물 사전: 소설에서 자동 추출 · {n_entities}개")
    else:
        st.warning(
            "인물 사전이 아직 없어 인물 인식·인물 상태 기능이 꺼져 있어요. "
            "`run_app.bat`으로 다시 열거나 `python -m spoiler_rag.prepare`를 실행하세요(Claude 필요).",
            icon="ℹ️",
        )

    st.divider()
    use_llm = st.toggle(
        "Claude로 답변 생성",
        value=generator is not None,
        disabled=generator is None,
        help=f"모델: {DEFAULT_MODEL}",
    )
    if generator is None:
        st.caption(f"{generator_reason} 검색된 근거 문단만 표시합니다.")

    if st.button("대화 지우기", use_container_width=True):
        ss.history = []
        st.rerun()


# --------------------------------------------------------------------- main
def render_answer(answer: Answer) -> None:
    st.markdown(answer.text)
    st.caption(
        f"유형: {LABELS[answer.query_type]} · {MODE_LABELS.get(answer.mode, answer.mode)} · "
        f"근거 {len(answer.evidence)}개"
    )
    if answer.note:
        st.caption(f"⚠️ {answer.note}")
    if answer.evidence:
        with st.expander("근거 문단 보기"):
            for c in answer.evidence:
                st.markdown(f"**{c.location_label}** `{c.chunk_id}`")
                st.write(c.text)


st.title("Spoiler-safe 소설 도우미")
st.caption(f"**{progress.label}**까지 읽은 내용만 검색하고 답해요.")

for item in ss.history:
    with st.chat_message("user"):
        st.markdown(item["turn"].question)
    with st.chat_message("assistant"):
        render_answer(item["answer"])

if not ss.history:
    st.write("예시 질문")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, use_container_width=True):
            ss.pending = example
            st.rerun()

question = st.chat_input("읽은 부분에 대해 무엇이든 물어보세요")
if not question and ss.pending:
    question, ss.pending = ss.pending, None

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("읽은 범위에서 찾는 중…"):
            answer = rag.ask(
                question,
                progress,
                history=[h["turn"] for h in ss.history],
                use_llm=use_llm,
            )
        render_answer(answer)
    ss.history.append(
        {
            "turn": Turn(question, answer.text, answer.cutoff, [c.chunk_id for c in answer.evidence]),
            "answer": answer,
        }
    )
