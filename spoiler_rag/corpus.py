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
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .models import Chapter, Chunk, Paragraph

_BOOK = re.compile(r"^#\s+(?:(\d+)\s*권\s*)?(.*)$")
_CHAPTER = re.compile(r"^##\s+(?:(\d+)\s*장\s*)?(.*)$")


@dataclass
class Corpus:
    series_id: str
    series_title: str
    chapters: list[Chapter]
    paragraphs: list[Paragraph]

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


def parse_novel(text: str, series_id: str, series_title: str = "") -> Corpus:
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
    return Corpus(series_id, series_title or series_id, chapters, paragraphs)


def load_novel(path: str | Path, series_id: str, series_title: str = "") -> Corpus:
    return parse_novel(Path(path).read_text(encoding="utf-8"), series_id, series_title)


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
