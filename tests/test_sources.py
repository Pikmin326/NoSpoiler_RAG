"""PDF와 여러 파일로 나뉜 소설 입력."""

from pathlib import Path

import pytest

from spoiler_rag.config import paths_for
from spoiler_rag.corpus import load_novel
from spoiler_rag.pipeline import SpoilerSafeRAG
from spoiler_rag.sources import (
    fix_wide_spacing,
    is_divider,
    list_source_files,
    load_source,
    pdf_chapters,
    rejoin_lines,
)

DIVIDER = ""  # 글꼴에 없는 기호가 사용자 정의 영역 글자로 뽑힌 경우


def make_pdf(path: Path, pages: list[list[str]]) -> Path:
    """Helvetica로 줄을 찍은 최소 PDF (ASCII만)."""

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for lines in pages:
        body = "BT /F1 11 Tf 14 TL 50 800 Td " + " ".join(f"({esc(line)}) Tj T*" for line in lines) + " ET"
        objs.append(f"<< /Length {len(body.encode())} >>\nstream\n{body}\nendstream")
        content = len(objs)
        objs.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content} 0 R >>"
        )
        kids.append(len(objs))
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


# 위키문헌에서 인쇄한 PDF와 같은 모양: 브라우저 머리글·바닥글, 문서 제목 줄, 장 구분 기호, 쪽 너비 줄바꿈
WIKISOURCE_PAGES = [
    "\n".join([
        "26. 10. 15. 오후 3:00 무정/1장~20장 - 위키문헌, 우리 모두의 도서관",
        "무정/1장~20장",
        "< 무정",
        DIVIDER,
        "경성학교 영어 교사 이형식은 오후 두 시 사년급 영어 시간을 마치고 내리쪼이는",
        "유월 볕에 땀을 흘리면서 안동 김장로의 집으로 간다.",
        '"어디로 가시오?"',
        "하고 물었다.",
        "https://ko.wikisource.org/wiki/무정 1/3",
    ]),
    "\n".join([
        "26. 10. 15. 오후 3:00 무정/1장~20장 - 위키문헌, 우리 모두의 도서관",
        "형식은 ⼈ 생각을 하며 걸었다. 그날 밤에 형식은 오래도록 잠을 이루지 못하고 이런",
        "생각 저런 생각을 하였다.",
        DIVIDER,
        "이튿날 아침이다. 형식은 일찍 일어나 세수를 하고 학교로 가는 길에 다시 그 집",
        "앞을 지났다.",
        "https://ko.wikisource.org/wiki/무정 2/3",
    ]),
    "\n".join([
        "26. 10. 15. 오후 3:00 무정/1장~20장 - 위키문헌, 우리 모두의 도서관",
        "선형은 피아노 앞에 앉아 있었다.",
        "https://ko.wikisource.org/wiki/무정 3/3",
    ]),
]


# ------------------------------------------------------------- PDF 글자 처리


def test_wikisource_style_pdf_text_is_split_at_dividers():
    dropped: list[str] = []
    chapters, structure = pdf_chapters(WIKISOURCE_PAGES, dropped)
    assert structure == "pdf_dividers"
    assert [c.paragraphs for c in chapters] == [
        [
            "경성학교 영어 교사 이형식은 오후 두 시 사년급 영어 시간을 마치고 내리쪼이는 유월 볕에 땀을 흘리면서 안동 김장로의 집으로 간다.",
            '"어디로 가시오?"',
            "하고 물었다.",
            # 쪽이 바뀌어도 문단은 이어지고, 호환 부수 글자(⼈)는 일반 한자(人)로 바뀐다
            "형식은 人 생각을 하며 걸었다. 그날 밤에 형식은 오래도록 잠을 이루지 못하고 이런 생각 저런 생각을 하였다.",
        ],
        [
            "이튿날 아침이다. 형식은 일찍 일어나 세수를 하고 학교로 가는 길에 다시 그 집 앞을 지났다.",
            "선형은 피아노 앞에 앉아 있었다.",
        ],
    ]
    # 머리말과 반복되는 머리글·바닥글은 본문에 들어가지 않는다
    assert "무정/1장~20장" in dropped and "< 무정" in dropped
    assert any(line.startswith("https://") for line in dropped)
    body = " ".join(p for c in chapters for p in c.paragraphs)
    assert "위키문헌" not in body and "wikisource" not in body


def test_divider_detection():
    for line in [DIVIDER, "* * *", "◆", "◇ ◇ ◇", "───", "��", "/g0\xa0/g0", "/g12 /uni25C6"]:
        assert is_divider(line), line
    for line in ["……", "—", "-", "1", '"……"', "가", "A", "· ·", "/usr/bin", "/g0 를"]:
        assert not is_divider(line), line


# 위키문헌 『무정』을 브라우저에서 PDF로 인쇄한 파일을 pypdf로 뽑은 모양 (실제 파일에서 확인한 특징):
# - 장 구분 기호(X 표시 네모 두 개)가 글리프 이름 "/g0 /g0"으로 나온다
# - 낱말 사이는 2칸, 문장부호·괄호 옆은 1칸이 들어간다
# - 쪽 너비에서 아무 글자에서나 줄이 바뀐다("김\n장로의"), 문단의 마지막 줄만 짧다
MUJEONG_PRINT_PAGES = [
    "\n".join([
        "무정 /1 장 ~20 장",
        "< 무정",
        "/g0\xa0/g0",
        "경성학교  영어  교사  이형식은  김장로의  집으로  가면서  안동  김",
        "장로의  딸  선형 ( 善馨 ) 이를  생각하였다 . 형식은  그러한  모양으로",
        "젊은  여자를  대하면  얼굴이  붉어진다 . 형식은  모양으로  젊은  생각을",
    ]),
    "\n".join([
        "하였다 . 이때에 ,",
        "“ 미스터  리 , 어디로  가는가 ” 하는  소리에  ‘ 좋은  자리 ’ 를  보았다 .",
        "/g0\xa0/g0",
        "이튿날이다 .",
    ]),
]


def test_mujeong_print_pdf_format():
    dropped: list[str] = []
    chapters, structure = pdf_chapters(MUJEONG_PRINT_PAGES, dropped)
    assert structure == "pdf_dividers"
    assert [c.paragraphs for c in chapters] == [
        [
            # "김|장로의"는 낱말 중간 → 붙임, "모양으로|젊은"·"생각을|하였다"는 낱말 사이 → 띄움
            "경성학교 영어 교사 이형식은 김장로의 집으로 가면서 안동 김장로의 딸 선형(善馨)이를 생각하였다. "
            "형식은 그러한 모양으로 젊은 여자를 대하면 얼굴이 붉어진다. 형식은 모양으로 젊은 생각을 하였다. 이때에,",
            # 짧은 줄("하였다. 이때에,")에서 문단이 끝나고, 닫는 따옴표 뒤 조사는 붙는다
            "“미스터 리, 어디로 가는가” 하는 소리에 ‘좋은 자리’를 보았다.",
        ],
        ["이튿날이다."],
    ]
    assert dropped == ["무정/1장~20장", "<무정"]


@pytest.mark.parametrize(
    "raw, fixed",
    [
        ("“ 여보 , 당신은  귀족이오 !”", "“여보, 당신은 귀족이오!”"),
        ("간다 . 김장로의  딸", "간다. 김장로의 딸"),
        ("이형식이올시다 ’ 이렇게  할까", "이형식이올시다’ 이렇게 할까"),
        ("‘ 좋은  자리 ’ 를  몰라보는", "‘좋은 자리’를 몰라보는"),
    ],
)
def test_wide_spacing_is_restored(raw, fixed):
    assert fix_wide_spacing(raw) == fixed


def test_dialogue_starts_new_paragraph_even_on_full_width_line():
    lines = ["그는 길고 긴 문장을 쓰다가 마침내 이렇게 끝맺었다.", '"정말이오?"', "하고 되물었다."]
    assert rejoin_lines(lines, full=20) == lines


def test_pdf_without_dividers_falls_back_to_headings():
    pages = ["제1장 시작\n첫 문장이다.\n제2장 끝\n둘째 문장이다."]
    chapters, structure = pdf_chapters(pages, [])
    assert structure == "detected_headings"
    assert [(c.number, c.title, c.paragraphs) for c in chapters] == [
        (1, "시작", ["첫 문장이다."]),
        (2, "끝", ["둘째 문장이다."]),
    ]


# ------------------------------------------------------------- 실제 PDF 파일


def _wiki_pdf(path: Path, title: str, chapters: list[list[str]]) -> Path:
    """머리글·바닥글이 있는 여러 쪽짜리 PDF. 장마다 한 쪽, 마지막은 빈 쪽(실제 인쇄본처럼)."""
    total = len(chapters) + 1
    pages = []
    for i, lines in enumerate([*chapters, []], 1):
        head = [title, "< Novel"] if i == 1 else []
        divider = ["* * *"] if lines else []
        pages.append([f"Printed 2026 {title}", *head, *divider, *lines, f"https://example.org/{title} {i}/{total}"])
    return make_pdf(path, pages)


@pytest.fixture
def split_pdfs(tmp_path) -> Path:
    folder = tmp_path / "Novel"
    folder.mkdir()
    # 이름을 사전순으로 놓으면 10이 2보다 앞에 온다. 숫자 순서로 읽어야 한다.
    _wiki_pdf(folder / "Novel10_4-4.PDF", "Novel/4-4", [["Fourth chapter: the ending."]])
    _wiki_pdf(folder / "Novel1_1-2.PDF", "Novel/1-2", [["First chapter begins here."], ["Second chapter, Mara arrives."]])
    _wiki_pdf(folder / "Novel2_3-3.pdf", "Novel/3-3", [["Third chapter: Mara is the queen."]])
    (folder / "notes.docx").write_bytes(b"ignored")
    return folder


def test_split_pdfs_are_read_in_numeric_order_as_one_book(split_pdfs):
    assert [f.name for f in list_source_files(split_pdfs)] == ["Novel1_1-2.PDF", "Novel2_3-3.pdf", "Novel10_4-4.PDF"]
    corpus = load_novel(split_pdfs, "novel")
    assert corpus.structure == "pdf_dividers" and corpus.source.layout == "continuous"
    assert [(c.book_number, c.chapter_number) for c in corpus.chapters] == [(1, 1), (1, 2), (1, 3), (1, 4)]
    assert [p.text for p in corpus.paragraphs] == [
        "First chapter begins here.",
        "Second chapter, Mara arrives.",
        "Third chapter: Mara is the queen.",
        "Fourth chapter: the ending.",
    ]
    assert [(f.first_chapter, f.last_chapter) for f in corpus.source.files] == [(1, 2), (3, 3), (4, 4)]


def test_chapter_count_mismatch_with_stated_range_is_reported(tmp_path):
    folder = tmp_path / "w"
    folder.mkdir()
    (folder / "a_1장~3장.txt").write_text("제1장\n\n가.\n\n제2장\n\n나.", encoding="utf-8")
    _, report = load_source(folder)
    assert report.warnings == ["a_1장~3장.txt: 파일 이름·머리말로는 3장인데 2장으로 나눴습니다"]


def test_pipeline_on_split_pdfs_never_returns_later_files(split_pdfs, tmp_path):
    rag = SpoilerSafeRAG.build(split_pdfs, tmp_path / "missing.json")
    assert rag.corpus.series_id == "Novel"
    progress = rag.progress_for(1, 2)
    answer = rag.ask("Who is Mara?", progress)
    assert answer.evidence and all(c.sequence_end <= progress.current_sequence for c in answer.evidence)
    assert all("queen" not in c.text for c in answer.evidence)


def test_paths_for_folder_uses_folder_name(split_pdfs):
    paths = paths_for(split_pdfs)
    assert paths.derived_dir.name == "Novel" and paths.summaries.parent.name == "Novel"


# ------------------------------------------------------------- 나뉜 텍스트 파일


def test_split_text_files_continue_chapter_numbers(tmp_path):
    folder = tmp_path / "t"
    folder.mkdir()
    (folder / "part1.txt").write_text("작품 제목\n\n제1장 시작\n\n가.\n\n제2장 길\n\n나.", encoding="utf-8")
    (folder / "part2.txt").write_bytes("제3장 끝\n\n다.".encode("cp949"))  # 파일마다 인코딩이 달라도 된다
    corpus = load_novel(folder, "t")
    assert corpus.source.layout == "continuous"
    assert [(c.chapter_number, c.chapter_title) for c in corpus.chapters] == [(1, "시작"), (2, "길"), (3, "끝")]
    assert corpus.source.files[0].dropped == ["작품 제목"]


def test_volume_files_restarting_at_chapter_one_become_books(tmp_path):
    folder = tmp_path / "v"
    folder.mkdir()
    (folder / "1권.txt").write_text("제1장 a\n\n가.\n\n제2장 b\n\n나.", encoding="utf-8")
    (folder / "2권.txt").write_text("제1장 c\n\n다.", encoding="utf-8")
    corpus = load_novel(folder, "v")
    assert corpus.source.layout == "file_per_book"
    assert [(c.book_number, c.chapter_number, c.chapter_title) for c in corpus.chapters] == [
        (1, 1, "a"), (1, 2, "b"), (2, 1, "c"),
    ]
    assert [b for b, _ in corpus.books()] == [1, 2]


def test_prologue_keeps_its_original_heading_when_renumbered(tmp_path):
    folder = tmp_path / "p"
    folder.mkdir()
    (folder / "a.txt").write_text("프롤로그\n\n시작.\n\n제1장 본편\n\n가.", encoding="utf-8")
    (folder / "b.txt").write_text("제2장 다음\n\n나.", encoding="utf-8")
    corpus = load_novel(folder, "p")
    assert [(c.chapter_number, c.chapter_title) for c in corpus.chapters] == [
        (1, "프롤로그"), (2, "제1장 본편"), (3, "제2장 다음"),
    ]


def test_markup_files_are_concatenated_as_written(tmp_path):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "1.txt").write_text("# 1권 하나\n## 1장 가\n\n가.", encoding="utf-8")
    (folder / "2.txt").write_text("# 2권 둘\n## 1장 나\n\n나.", encoding="utf-8")
    corpus = load_novel(folder, "m")
    assert corpus.source.layout == "markup_concat"
    assert [(c.book_number, c.book_title, c.chapter_number) for c in corpus.chapters] == [(1, "하나", 1), (2, "둘", 1)]


def test_mixed_text_and_pdf_files(tmp_path):
    folder = tmp_path / "x"
    folder.mkdir()
    (folder / "1.txt").write_text("Chapter 1\n\nText part.", encoding="utf-8")
    _wiki_pdf(folder / "2.pdf", "x", [["PDF part one."], ["PDF part two."]])
    corpus = load_novel(folder, "x")
    assert corpus.structure == "mixed"
    assert [p.text for p in corpus.paragraphs] == ["Text part.", "PDF part one.", "PDF part two."]
    assert [c.chapter_number for c in corpus.chapters] == [1, 2, 3]


def test_empty_folder_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        load_novel(tmp_path, "e")
