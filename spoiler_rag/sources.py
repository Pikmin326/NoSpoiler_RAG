"""여러 파일·PDF로 나뉜 소설을 하나의 `#`/`##` 표기 텍스트로 합친다.

입력은 파일 하나(.txt/.md/.pdf) 또는 폴더다. 폴더면 안의 .txt/.md/.pdf 파일을 이름의 숫자 순서로
읽는다("무정1_…", "무정2_…", …, "무정10_…").

PDF 처리 (위키문헌 등에서 인쇄한 PDF를 기준으로 했다):
- 쪽마다 글자를 뽑고 NFKC로 정규화한다(⼈ 같은 호환 부수 글자 → 人).
- 여러 쪽의 맨 위·맨 아래에 반복되는 줄(머리글·바닥글, 쪽 번호)은 지운다.
- 글자·숫자 없이 기호만 있는 줄(◆, * * *, 글꼴에 없는 기호 등)을 장 구분선으로 본다.
  첫 구분선 앞의 짧은 머리말("무정/1장~20장", "< 무정")은 버린다.
- 쪽 너비에서 줄바꿈된 줄을 문단으로 다시 잇는다. 문장 끝 부호로 끝나는 짧은 줄,
  여는 따옴표로 시작하는 줄을 문단 경계로 본다(추정이므로 결과를 미리보기 파일로 확인할 것).
- 구분선이 없으면 일반 텍스트와 같이 장 제목 줄 인식 → 문단 수 자동 분할 순서로 처리한다.

여러 파일을 합칠 때:
- 모든 파일이 `#` 권 표기를 갖고 있으면 그대로 이어 붙인다.
- 파일마다 장 번호가 1부터 다시 시작하면(권별 파일) 파일 하나를 한 권으로 본다.
- 그 밖에는(한 작품을 나눈 파일) 한 권으로 보고 장 번호를 파일을 넘어 1부터 이어 매긴다.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .corpus import _BOOK, _CHAPTER, _PLAIN_CHAPTER, normalize_markup, read_text_any

SUPPORTED_SUFFIXES = (".txt", ".md", ".pdf")

# 장 제목에서 번호 부분: "제3장", "3장", "제3화", "Chapter 3"
_HEADING_NUMBER = re.compile(r"^(?:제\s*(\d+)\s*[장화편부]|(\d+)\s*[장화]|chapter\s+(\d+))\s*[.:\-–—]?\s*", re.IGNORECASE)
# 파일 이름·머리말의 장 범위: "1장~20장", "121장~126장"
_RANGE = re.compile(r"(\d+)\s*장\s*[~∼〜\-–]\s*(?:(\d+)\s*장)?")
# 문단 끝으로 볼 수 있는 줄 끝 글자
_SENTENCE_END = re.compile(r"[.!?。…\"'”’」』)\]]$")
_OPENING_QUOTES = ('"', "“", "「", "『", "'", "‘")
# 구분선 줄에만 올 수 있는 글자: 공백, 기호류, 사용자 정의 영역(글꼴에 없는 기호), 대체 문자
_DIVIDER_CHARS = re.compile(r"^[\s*＊·•◆◇■□●○★☆※◎♦◈▲△▼▽─━\-=~_#-�]+$")
# pypdf가 글자로 바꾸지 못한 글리프 이름("/g0", "/uni25C6"). 본문의 "/usr/bin" 같은 줄과 구분하려고 숫자를 요구한다.
_GLYPH_NAMES = re.compile(r"^(?:/[A-Za-z][A-Za-z._]*\d[A-Za-z0-9._]*\s*)+$")
_DIVIDER_STRONG =re.compile(r"[*＊◆◇■□●○★☆※◎♦◈▲△▼▽-�]|[─━\-=~_]{3,}")


@dataclass
class _Chapter:
    number: int | None  # 원문에 적힌 장 번호 (없으면 None)
    title: str
    heading: str  # 원문의 장 제목 줄 그대로 (없으면 "")
    paragraphs: list[str]


@dataclass
class FileReport:
    name: str
    kind: str  # txt | pdf
    structure: str  # markup | detected_headings | auto_split | pdf_dividers
    chapters: int = 0
    paragraphs: int = 0
    first_chapter: int = 0  # 합친 뒤의 장 번호
    last_chapter: int = 0
    book: int = 1
    expected_chapters: int | None = None  # 파일 이름·머리말의 "1장~20장"에서 읽은 장 수
    dropped: list[str] = field(default_factory=list)  # 버린 줄 (머리말, 머리글·바닥글)

    @property
    def warning(self) -> str | None:
        if self.expected_chapters is not None and self.expected_chapters != self.chapters:
            return f"파일 이름·머리말로는 {self.expected_chapters}장인데 {self.chapters}장으로 나눴습니다"
        if self.kind == "pdf" and self.structure == "auto_split":
            return "PDF에서 장 구분선·장 제목을 찾지 못해 문단 수로 나눴습니다"
        return None


@dataclass
class SourceReport:
    files: list[FileReport]
    layout: str  # single_file | markup_concat | file_per_book | continuous
    structure: str  # 파일들의 구조가 모두 같으면 그 값, 다르면 mixed

    @property
    def warnings(self) -> list[str]:
        return [f"{f.name}: {f.warning}" for f in self.files if f.warning]


def is_multi_source(path: str | Path) -> bool:
    """파일 하나짜리 일반 텍스트가 아니라 이 모듈로 읽어야 하는 입력인지."""
    p = Path(path)
    return p.is_dir() or p.suffix.lower() == ".pdf"


def natural_key(name: str) -> list:
    """'무정2_…'가 '무정10_…'보다 앞에 오도록 숫자는 수로 비교한다."""
    return [(0, int(t), "") if t.isdigit() else (1, 0, t.lower()) for t in re.split(r"(\d+)", name) if t]


def list_source_files(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"소설 파일이나 폴더가 없습니다: {p}")
    files = [
        f for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES and not f.name.startswith((".", "~$"))
    ]
    if not files:
        raise ValueError(f"{p}: 읽을 수 있는 소설 파일(.txt, .md, .pdf)이 없습니다.")
    return sorted(files, key=lambda f: natural_key(f.name))


# ---------------------------------------------------------------- PDF


def read_pdf_pages(path: str | Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 설치 안내
        raise RuntimeError("PDF를 읽으려면 pypdf가 필요합니다: .\\.conda\\python.exe -m pip install pypdf") from exc
    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def _nfkc(line: str) -> str:
    """NFKC 정규화(⼈ → 人, NBSP → 공백). 공백 개수는 그대로 둔다(넓은 띄어쓰기 판단에 쓴다)."""
    line = unicodedata.normalize("NFKC", line).replace("\x00", "")
    return re.sub(r"[\t\u3000]", " ", line).strip()


def _normalize_line(line: str) -> str:
    return re.sub(r" +", " ", _nfkc(line))


def is_divider(line: str) -> bool:
    s = _normalize_line(line)
    if not s:
        return False
    if _GLYPH_NAMES.match(s):  # 글꼴에 없는 기호를 pypdf가 글리프 이름으로 낸 것 (위키문헌 인쇄본: "/g0 /g0")
        return True
    return bool(_DIVIDER_CHARS.match(s)) and bool(_DIVIDER_STRONG.search(s))


# ---- 넓은 띄어쓰기: 낱말 사이 2칸, 문장부호 옆 1칸으로 뽑히는 PDF (브라우저 인쇄본)


def has_wide_spacing(lines: list[str]) -> bool:
    spaced = [line for line in lines if " " in line and not is_divider(line)]
    if len(spaced) < 5:
        return False
    return sum("  " in line for line in spaced) >= len(spaced) * 0.5


def fix_wide_spacing(line: str) -> str:
    """2칸 이상 = 낱말 사이, 1칸 = 문장부호에 붙은 틈. 1칸은 지우되 문장부호 뒤의 1칸은 띄어쓰기로 남긴다.

    "“ 여보 , 당신은  귀족이오 !”" → "“여보, 당신은 귀족이오!”"
    "선형 ( 善馨 ) 이가" → "선형(善馨)이가"
    """
    words = []
    for token in re.split(r" {2,}", line.strip()):
        token = _SPACE_BEFORE_PARTICLE.sub("", token)
        token = re.sub(r"(?<=[.,?!;:…”’]) (?=[^\s.,?!;:…”’)\]」』])", "\x01", token)
        words.append(token.replace(" ", "").replace("\x01", " "))
    return " ".join(w for w in words if w)


# 낱말 하나로 떨어져 나왔을 때 앞말에 붙여야 하는 조사·어미
PARTICLES = frozenset(
    "이 가 을 를 은 는 과 와 의 도 로 에 만 요 다 나 야 랴 며 고 서 니 지 라 냐 오 죠 께 "
    "에서 으로 에게 한테 께서 이나 이며 이요 이오 이다 이고 이라 이니 이랑 하고 부터 까지 마저 조차 처럼 보다 만큼 "
    "이라고 라고 이라는 라는 에는 에도 으로는 으로도 에서는 에서도 에게는 이었다 였다 이지 이야".split()
)
_PARTICLE_ALT = "|".join(sorted(PARTICLES, key=len, reverse=True))
# 닫는 따옴표·괄호 뒤의 1칸 다음이 조사면 붙인다: "‘좋은 자리’ 를" → "‘좋은 자리’를"
_SPACE_BEFORE_PARTICLE = re.compile(rf"(?<=[”’)\]」』]) (?=(?:{_PARTICLE_ALT})(?:[\s.,?!;:…”’]|$))")


# ---- 줄 잇기: 낱말 중간에서 줄이 바뀌는 PDF ("김\n장로의")

_HANGUL = re.compile(r"[\uac00-\ud7a3]")
_EDGE_PUNCT = ".,?!;:…\"'“”‘’()[]「」『』<>-—–~"


def _core(word: str) -> str:
    return word.strip(_EDGE_PUNCT)


class LineJoiner:
    """두 줄을 이을 때 공백을 넣을지 정한다.

    줄 안쪽 낱말(줄의 첫·끝 낱말 제외)로 어휘를 만든다. 줄 끝 낱말 A와 다음 줄 첫 낱말 B에 대해
    - A+B가 어휘에 있으면 낱말 중간에서 끊긴 것 → 붙임
    - A+B는 없고 A가 어휘에 있으면(A가 두 글자 이상이거나 B도 어휘에 있을 때) 낱말 사이에서 끊긴 것 → 공백
    - 그래도 모르면 B가 조사 하나뿐이면 붙이고, A가 세 글자·B가 두 글자 이상이면 띄우고,
      나머지는 문서 전체의 경향을 따른다(낱말 중간 끊김이 흔한 문서면 붙임, 아니면 공백).
    한글끼리 만나는 경계와 닫는 괄호·따옴표 뒤 조사에만 적용하고, 그 밖에는 항상 공백으로 잇는다.
    """

    def __init__(self, lines: list[str]) -> None:
        self.vocab: set[str] = set()
        for line in lines:
            words = line.split(" ")
            self.vocab.update(_core(w) for w in words[1:-1] if _core(w))
        votes = [self._decide(a, b) for a, b in zip(lines, lines[1:])]
        mid, gap = votes.count("join"), votes.count("space")
        # 브라우저 인쇄본처럼 아무 글자에서나 끊는 문서는 낱말 중간 끊김이 낱말 사이 끊김만큼 흔하다.
        self.default = "join" if mid and mid >= gap * 0.5 else "space"
        self.votes = (mid, gap)

    def _decide(self, prev: str, nxt: str) -> str | None:
        if not prev or not nxt or not _HANGUL.match(prev[-1]) or not _HANGUL.match(nxt[0]):
            return None
        a, b = prev.split(" ")[-1], _core(nxt.split(" ")[0])
        if a + b in self.vocab:
            return "join"
        if a in self.vocab and (len(a) >= 2 or b in self.vocab):
            return "space"
        return None

    def __call__(self, prev: str, nxt: str) -> str:
        if not prev or not nxt:
            return " "
        b = _core(nxt.split(" ")[0])
        if prev[-1] in ")”’」』" and b in PARTICLES and _HANGUL.match(nxt[0]):
            return ""
        if not _HANGUL.match(prev[-1]) or not _HANGUL.match(nxt[0]):
            return " "
        decision = self._decide(prev, nxt)
        if decision is None:
            a = prev.split(" ")[-1]
            if b in PARTICLES:
                decision = "join"
            elif len(a) >= 3 and len(b) >= 2:
                decision = "space"
            else:
                decision = self.default
        return "" if decision == "join" else " "


def _strip_running_lines(pages: list[list[str]], dropped: list[str]) -> list[list[str]]:
    """두 쪽 이상에서 맨 위·아래 두 줄 안에 반복되는 줄(숫자만 다른 것 포함)을 지운다(쪽의 60% 이상)."""
    with_text = [p for p in pages if p]
    if len(with_text) < 2:
        return pages

    def key(line: str) -> str:
        return re.sub(r"\d+", "#", line)

    counts: dict[str, int] = {}
    for p in with_text:
        for k in {key(line) for line in p[:2] + p[-2:] if not is_divider(line)}:
            counts[k] = counts.get(k, 0) + 1
    threshold = max(2, int(len(with_text) * 0.6 + 0.5))
    running = {k for k, n in counts.items() if n >= threshold}
    if not running:
        return pages

    out = []
    recorded: set[str] = set()  # 쪽 번호만 다른 줄은 한 번만 기록
    for p in pages:
        edge = set(range(min(2, len(p)))) | set(range(max(0, len(p) - 2), len(p)))
        kept = []
        for i, line in enumerate(p):
            if i in edge and key(line) in running and not is_divider(line):
                if key(line) not in recorded:
                    recorded.add(key(line))
                    dropped.append(line)
            else:
                kept.append(line)
        out.append(kept)
    return out


def full_line_width(lines: list[str]) -> int:
    """꽉 찬 줄의 길이(상위 20% 지점). 이보다 눈에 띄게 짧은 줄은 문단의 마지막 줄일 수 있다."""
    widths = sorted(len(line) for line in lines if line)
    return widths[int(len(widths) * 0.8)] if widths else 0


def rejoin_lines(lines: list[str], full: int | None = None, joiner: LineJoiner | None = None) -> list[str]:
    """쪽 너비에서 줄바꿈된 줄을 문단으로 잇는다. 빈 줄은 항상 문단 경계다."""
    if full is None:
        full = full_line_width(lines)
    paragraphs: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            text = buf[0]
            for prev, nxt in zip(buf, buf[1:]):
                text += (joiner(prev, nxt) if joiner else " ") + nxt
            paragraphs.append(text)
            buf.clear()

    for line in lines:
        if not line:
            flush()
            continue
        if _PLAIN_CHAPTER.match(line):  # 장 제목 줄은 앞뒤 문장과 잇지 않는다
            flush()
            paragraphs.append(line)
            continue
        if buf and line.startswith(_OPENING_QUOTES) and _SENTENCE_END.search(buf[-1]):
            flush()
        buf.append(line)
        # 쪽 너비에서 줄바꿈된 글에서는 문단의 마지막 줄만 짧다.
        if len(line) < full * 0.6 or (_SENTENCE_END.search(line) and len(line) < full * 0.85):
            flush()
    flush()
    return paragraphs


def _is_front_matter(lines: list[str]) -> bool:
    text = [line for line in lines if line]
    return len(text) <= 5 and sum(len(line) for line in text) <= 300


def pdf_lines(pages: list[str], dropped: list[str]) -> list[str]:
    """쪽 텍스트 → 정규화·머리글/바닥글 제거된 줄 목록 (구분선 줄 포함)."""
    page_lines = [[_nfkc(line) for line in page.splitlines()] for page in pages]
    page_lines = [[line for line in p if line] for p in page_lines]
    clean = fix_wide_spacing if has_wide_spacing([line for p in page_lines for line in p]) else _normalize_line
    page_lines = [[line if is_divider(line) else clean(line) for line in p] for p in page_lines]
    page_lines = _strip_running_lines(page_lines, dropped)
    # 쪽 경계는 문단 경계가 아니다(문장이 다음 쪽으로 이어진다).
    return [line for p in page_lines for line in p]


def pdf_chapters(
    pages: list[str], dropped: list[str], joiner: LineJoiner | None = None
) -> tuple[list[_Chapter], str]:
    """joiner를 주면 그 어휘로 줄을 잇는다(나뉜 파일 전체에서 모은 어휘를 쓰려고)."""
    lines = pdf_lines(pages, dropped)
    if joiner is None:
        joiner = LineJoiner([line for line in lines if not is_divider(line)])

    segments: list[list[str]] = [[]]
    for line in lines:
        if is_divider(line):
            segments.append([])
        else:
            segments[-1].append(line)

    full = full_line_width(lines)  # 장이 짧아도 판단이 흔들리지 않도록 파일 전체 기준
    if len(segments) == 1:
        # 구분선이 없으면 일반 텍스트처럼 장 제목 인식 → 자동 분할
        return _text_chapters("\n\n".join(rejoin_lines(lines, full, joiner)), dropped)

    front, body = segments[0], segments[1:]
    if _is_front_matter(front):
        dropped.extend(line for line in front if line)
    else:
        body.insert(0, front)
    chapters = [_Chapter(None, "", "", rejoin_lines(seg, full, joiner)) for seg in body]
    return [c for c in chapters if c.paragraphs], "pdf_dividers"


# ---------------------------------------------------------------- 텍스트


def _split_heading(heading: str) -> tuple[int | None, str]:
    """'제3장 귀환' → (3, '귀환'), '프롤로그' → (None, '프롤로그')"""
    m = _HEADING_NUMBER.match(heading)
    if not m:
        return None, heading
    number = next(int(g) for g in m.groups() if g)
    return number, heading[m.end():].strip()


def _text_chapters(text: str, dropped: list[str]) -> tuple[list[_Chapter], str]:
    marked, structure = normalize_markup(text)
    chapters: list[_Chapter] = []
    preamble: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        body = " ".join(line.strip() for line in buf).strip()
        buf.clear()
        if body:
            (chapters[-1].paragraphs if chapters else preamble).append(body)

    for raw in marked.splitlines():
        line = raw.rstrip()
        if m := _CHAPTER.match(line):
            flush()
            heading = line[2:].strip()
            number = int(m.group(1)) if m.group(1) else None
            title = m.group(2).strip()
            if number is None:
                number, title = _split_heading(title)
            chapters.append(_Chapter(number, title, heading, []))
        elif _BOOK.match(line):
            flush()
        elif not line.strip():
            flush()
        else:
            buf.append(line)
    flush()

    if preamble:
        if _is_front_matter(preamble) and chapters:
            dropped.extend(preamble)  # 작품 제목·저자 줄 등
        else:
            chapters.insert(0, _Chapter(None, "", "", preamble))
    return [c for c in chapters if c.paragraphs], structure


# ---------------------------------------------------------------- 합치기


def _expected_chapters(name: str, dropped: list[str]) -> int | None:
    for text in [*dropped, name]:
        if (m := _RANGE.search(unicodedata.normalize("NFKC", text))) and m.group(2):
            a, b = int(m.group(1)), int(m.group(2))
            if b >= a:
                return b - a + 1
    return None


def _read_file(
    path: Path, pages: list[str] | None = None, joiner: LineJoiner | None = None
) -> tuple[str, list[_Chapter], FileReport]:
    """(원문 텍스트(txt만), 장 목록, 보고)"""
    dropped: list[str] = []
    if path.suffix.lower() == ".pdf":
        text = ""
        chapters, structure = pdf_chapters(pages if pages is not None else read_pdf_pages(path), dropped, joiner)
        kind = "pdf"
    else:
        text = read_text_any(path)
        chapters, structure = _text_chapters(text, dropped)
        kind = "txt"
    report = FileReport(
        name=path.name,
        kind=kind,
        structure=structure,
        chapters=len(chapters),
        paragraphs=sum(len(c.paragraphs) for c in chapters),
        expected_chapters=_expected_chapters(path.stem, dropped),
        dropped=dropped,
    )
    return text, chapters, report


def _restarts_per_file(per_file: list[list[_Chapter]]) -> bool:
    """파일마다 장 번호가 적혀 있고 1부터 다시 시작하면 권별 파일로 본다."""
    if len(per_file) < 2:
        return False
    for chapters in per_file:
        numbers = [c.number for c in chapters]
        if not numbers or None in numbers or numbers[0] != 1 or len(set(numbers)) != len(numbers):
            return False
    return True


def _heading_line(number: int, chapter: _Chapter) -> str:
    # 원문 번호가 매긴 번호와 다르면(예: 프롤로그 뒤의 "제1장"이 2장이 됨) 원래 제목 줄을 그대로 남긴다.
    title = chapter.title if chapter.number in (None, number) else chapter.heading
    return f"## {number}장 {title}".rstrip()


_CACHE: dict[tuple, tuple[str, SourceReport]] = {}


def load_source(path: str | Path) -> tuple[str, SourceReport]:
    """파일 또는 폴더를 읽어 `#`/`##` 표기 텍스트와 처리 보고를 돌려준다."""
    files = list_source_files(path)
    cache_key = tuple((str(f.resolve()), f.stat().st_mtime_ns, f.stat().st_size) for f in files)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    # PDF 줄 잇기에 쓸 어휘는 모든 PDF 파일에서 모은다(파일 하나보다 어휘가 많아 판단이 정확해진다).
    pdf_pages = {f: read_pdf_pages(f) for f in files if f.suffix.lower() == ".pdf"}
    joiner = None
    if pdf_pages:
        all_lines = [line for pages in pdf_pages.values() for line in pdf_lines(pages, [])]
        joiner = LineJoiner([line for line in all_lines if not is_divider(line)])
    loaded = [_read_file(f, pdf_pages.get(f), joiner) for f in files]
    texts = [t for t, _, _ in loaded]
    per_file = [c for _, c, _ in loaded]
    reports = [r for _, _, r in loaded]
    structures = {r.structure for r in reports}
    structure = structures.pop() if len(structures) == 1 else "mixed"

    if all(r.kind == "txt" for r in reports) and all(
        any(_BOOK.match(line.rstrip()) for line in t.splitlines()) for t in texts
    ):
        # 모든 파일이 권 표기를 가진 경우: 사용자가 정한 구조를 그대로 쓴다.
        layout = "single_file" if len(files) == 1 else "markup_concat"
        result = "\n\n".join(texts), SourceReport(reports, layout, structure)
        _CACHE[cache_key] = result
        return result

    out: list[str] = []
    if _restarts_per_file(per_file):
        layout = "file_per_book"
        for book, (f, chapters, report) in enumerate(zip(files, per_file, reports), start=1):
            out += [f"# {book}권 {f.stem}", ""]
            for c in chapters:
                out += [_heading_line(c.number, c), ""]
                out += [line for p in c.paragraphs for line in (p, "")]
            report.book = book
            report.first_chapter, report.last_chapter = chapters[0].number, chapters[-1].number
    else:
        layout = "single_file" if len(files) == 1 else "continuous"
        title = Path(path).stem if Path(path).is_dir() else ""
        out += [f"# 1권 {title}".rstrip(), ""]
        number = 0
        for chapters, report in zip(per_file, reports):
            report.first_chapter = number + 1 if chapters else number
            for c in chapters:
                number += 1
                out += [_heading_line(number, c), ""]
                out += [line for p in c.paragraphs for line in (p, "")]
            report.last_chapter = number

    if not any(per_file):
        raise ValueError(f"{path}: 본문 문단을 찾지 못했습니다.")
    result = "\n".join(out), SourceReport(reports, layout, structure)
    _CACHE[cache_key] = result
    return result
