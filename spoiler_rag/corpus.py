"""원문 파싱과 chunking.

입력 형식 (UTF-8 텍스트):

    # 1권 책 제목
    ## 1장 장 제목
    문단...

    문단...

- `# ` 줄은 권, `## ` 줄은 장의 시작이다.
- 빈 줄로 문단을 구분한다.
- 모든 문단에 작품 전체에서 증가하는 sequence를 부여한다.
- chunk는 장 경계를 넘지 않는다. 따라서 장 단위 진행도의 경계에 걸친 chunk가 생기지 않는다.

`#`/`##` 표기가 없는 일반 텍스트도 받는다(`normalize_markup`).
- "제1장", "1장", "제3화", "Chapter 1", "프롤로그", "에필로그"처럼 생긴 줄을 장 제목으로 인식한다.
- 장 제목을 하나도 찾지 못하면 일정 문단 수마다 장을 나눈다(자동 분할).

PDF와 여러 파일로 나뉜 소설(폴더)은 `sources.load_source`가 이 형식으로 바꿔 준다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .models import Chapter, Chunk, Paragraph

if TYPE_CHECKING:
    from .sources import SourceReport

_BOOK = re.compile(r"^#\s+(?:(\d+)\s*권\s*)?(.*)$")
_CHAPTER = re.compile(r"^##\s+(?:(\d+)\s*장\s*)?(.*)$")
# 일반 텍스트의 장 제목 줄. 번호 뒤에는 공백·구두점이 오거나 줄이 끝나야 한다("3장의 편지를..." 같은 본문 제외).
_PLAIN_CHAPTER = re.compile(
    r"^(?:제\s*\d+\s*[장화편부]|\d+\s*[장화]|chapter\s+\d+|프롤로그|에필로그|prologue|epilogue)"
    r"(?:(?:\s+|\s*[.:\-–—]\s*).{0,40})?$",
    re.IGNORECASE,
)
AUTO_SPLIT_PARAGRAPHS = 30


@dataclass
class Corpus:
    series_id: str
    series_title: str
    chapters: list[Chapter]
    paragraphs: list[Paragraph]
    structure: str = "markup"  # markup | detected_headings | auto_split | pdf_dividers | mixed
    source: SourceReport | None = None  # PDF·폴더 입력일 때 파일별 처리 결과

    @property
    def max_sequence(self) -> int:
        return self.paragraphs[-1].sequence if self.paragraphs else 0

    def books(self) -> list[tuple[int, str]]:
        seen: dict[int, str] = {}
        for ch in self.chapters:
            seen.setdefault(ch.book_number, ch.book_title)
        return list(seen.items())

    def chapters_of(self, book_number: int) -> list[Chapter]:
        return [c for c in self.chapters if c.book_number == book_number]

    def chapter(self, book_number: int, chapter_number: int) -> Chapter:
        for c in self.chapters:
            if c.book_number == book_number and c.chapter_number == chapter_number:
                return c
        raise KeyError(f"{book_number}권 {chapter_number}장을 찾을 수 없습니다.")

    def chapter_at(self, sequence: int) -> Chapter | None:
        for c in self.chapters:
            if c.sequence_start <= sequence <= c.sequence_end:
                return c
        return None


def normalize_markup(text: str, paragraphs_per_chapter: int = AUTO_SPLIT_PARAGRAPHS) -> tuple[str, str]:
    """`## 장` 표기가 없으면 장 제목 줄을 찾아 표기를 붙이고, 그래도 없으면 문단 수로 나눈다."""
    lines = text.splitlines()
    if any(_CHAPTER.match(line.rstrip()) for line in lines):
        return text, "markup"

    has_book = any(_BOOK.match(line.rstrip()) for line in lines)
    out = [] if has_book else ["# 1권", ""]
    found = 0
    for line in lines:
        stripped = line.strip()
        if stripped and _PLAIN_CHAPTER.match(stripped):
            out += ["", f"## {stripped}", ""]
            found += 1
        else:
            out.append(line)
    if found:
        return "\n".join(out), "detected_headings"

    # 장 제목이 없는 텍스트: 빈 줄 기준 문단을 일정 개수씩 묶어 장으로 만든다.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = ["# 1권", ""]
    for i in range(0, len(paragraphs), paragraphs_per_chapter):
        out += [f"## {i // paragraphs_per_chapter + 1}장 (자동 분할)", ""]
        for p in paragraphs[i : i + paragraphs_per_chapter]:
            out += [p, ""]
    return "\n".join(out), "auto_split"


def parse_novel(text: str, series_id: str, series_title: str = "") -> Corpus:
    text, structure = normalize_markup(text)
    chapters: list[Chapter] = []
    paragraphs: list[Paragraph] = []

    book_no, book_title = 0, ""
    chapter_no, chapter_title = 0, ""
    chapter_start: int | None = None
    buffer: list[str] = []
    seq = 0

    def flush_paragraph() -> None:
        nonlocal seq
        body = " ".join(line.strip() for line in buffer).strip()
        buffer.clear()
        if not body:
            return
        if chapter_no == 0:
            raise ValueError("장 제목(## ...) 이전에 본문이 있습니다.")
        seq += 1
        paragraphs.append(Paragraph(seq, book_no, chapter_no, body))

    def close_chapter() -> None:
        if chapter_no and chapter_start is not None and seq >= chapter_start:
            chapters.append(
                Chapter(book_no, chapter_no, book_title, chapter_title, chapter_start, seq)
            )

    for raw in text.splitlines():
        line = raw.rstrip()
        if m := _CHAPTER.match(line):
            flush_paragraph()
            close_chapter()
            chapter_no = int(m.group(1)) if m.group(1) else chapter_no + 1
            chapter_title = m.group(2).strip()
            chapter_start = seq + 1
        elif m := _BOOK.match(line):
            flush_paragraph()
            close_chapter()
            book_no = int(m.group(1)) if m.group(1) else book_no + 1
            book_title = m.group(2).strip()
            chapter_no, chapter_start = 0, None
        elif not line.strip():
            flush_paragraph()
        else:
            buffer.append(line)
    flush_paragraph()
    close_chapter()

    if not paragraphs:
        raise ValueError("본문 문단을 찾지 못했습니다.")
    return Corpus(series_id, series_title or series_id, chapters, paragraphs, structure)


def read_text_any(path: str | Path) -> str:
    """UTF-8(BOM 포함)을 먼저 시도하고, 실패하면 한국어 Windows 기본 인코딩(CP949)으로 읽는다."""
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path}: UTF-8 또는 CP949로 읽을 수 없는 파일입니다.")


def load_novel(path: str | Path, series_id: str, series_title: str = "") -> Corpus:
    """텍스트 파일 하나, PDF 파일, 또는 나뉜 파일들이 든 폴더를 읽는다."""
    from .sources import is_multi_source, load_source

    if not is_multi_source(path):
        return parse_novel(read_text_any(path), series_id, series_title)
    text, report = load_source(path)
    corpus = parse_novel(text, series_id, series_title)
    corpus.structure = report.structure
    corpus.source = report
    return corpus


def chunk_corpus(corpus: Corpus, paragraphs_per_chunk: int = 2, overlap: int = 0) -> list[Chunk]:
    """장 경계를 넘지 않도록 연속 문단을 묶는다."""
    if paragraphs_per_chunk < 1 or not 0 <= overlap < paragraphs_per_chunk:
        raise ValueError("paragraphs_per_chunk >= 1, 0 <= overlap < paragraphs_per_chunk 이어야 합니다.")

    chunks: list[Chunk] = []
    step = paragraphs_per_chunk - overlap
    for ch in corpus.chapters:
        paras = [p for p in corpus.paragraphs if ch.sequence_start <= p.sequence <= ch.sequence_end]
        for i in range(0, len(paras), step):
            group = paras[i : i + paragraphs_per_chunk]
            chunks.append(
                Chunk(
                    chunk_id=f"C{len(chunks) + 1:04d}",
                    text="\n\n".join(p.text for p in group),
                    series_id=corpus.series_id,
                    book_number=ch.book_number,
                    chapter_number=ch.chapter_number,
                    sequence_start=group[0].sequence,
                    sequence_end=group[-1].sequence,
                )
            )
            if i + paragraphs_per_chunk >= len(paras):
                break
    return chunks
