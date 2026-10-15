"""소설 하나를 질의 가능한 상태로 준비한다: 구조 인식 → 인물·별칭 사전 자동 추출 → 요약·스냅샷 생성.

    python -m spoiler_rag.prepare                                  # .env의 SPOILER_RAG_NOVEL(기본: 샘플 소설)
    python -m spoiler_rag.prepare --novel data/novels/내소설.txt    # 다른 소설
    python -m spoiler_rag.prepare --novel data/novels/무정 --no-llm  # 나뉜 파일(PDF·txt)이 든 폴더, 구조만 확인
    python -m spoiler_rag.prepare --compare data/sample_entities.json   # 자동 사전을 정답 사전과 비교
    python -m spoiler_rag.prepare --rebuild                        # 사전·요약을 처음부터 다시

이미 처리한 부분은 건너뛰므로 여러 번 실행해도 된다(run_app.bat이 앱 시작 전에 매번 실행한다).
Claude 인증 정보가 없으면 사전 추출은 건너뛰고 요약은 추출식으로 만든다.
"""

from __future__ import annotations

import argparse
import sys

from .build_summaries import build_summaries
from .config import NovelPaths, PATHS, paths_for
from .corpus import chunk_corpus, load_novel
from .entities import EntityCatalog
from .entity_extraction import EntityExtractionBuilder, compare_catalogs, try_create_extractor
from .llm import DEFAULT_MODEL
from .sources import load_source
from .summaries import ExtractiveSummarizer, Summarizer, try_create_summarizer

STRUCTURE_LABELS = {
    "markup": "# 권 / ## 장 표기",
    "detected_headings": "장 제목 줄 자동 인식",
    "auto_split": "장 제목을 찾지 못해 문단 수로 자동 분할",
    "pdf_dividers": "PDF의 장 구분 기호로 나눔",
    "mixed": "파일마다 다름 (아래 파일별 결과 참고)",
}
LAYOUT_LABELS = {
    "single_file": "파일 1개",
    "markup_concat": "파일마다 적힌 # 권 표기대로 이어 붙임",
    "file_per_book": "파일마다 장 번호가 1부터 다시 시작 → 파일 1개 = 1권",
    "continuous": "한 작품을 나눈 파일 → 한 권으로 합치고 장 번호를 이어 매김",
}


def print_source_report(corpus, paths: NovelPaths) -> None:
    """PDF·폴더 입력의 파일별 처리 결과를 보여 주고, 합친 본문을 미리보기 파일로 저장한다."""
    report = corpus.source
    if report is None:
        return
    print(f"파일 {len(report.files)}개: {LAYOUT_LABELS[report.layout]}")
    for f in report.files:
        where = f"{f.book}권 " if report.layout == "file_per_book" else ""
        span = f"{where}{f.first_chapter}~{f.last_chapter}장" if f.chapters else "장 없음"
        print(f"  - {f.name}: {span} ({f.chapters}장, 문단 {f.paragraphs}개, {STRUCTURE_LABELS[f.structure]})")
        if f.dropped:
            shown = " / ".join(f.dropped[:3]) + (f" 외 {len(f.dropped) - 3}줄" if len(f.dropped) > 3 else "")
            print(f"      본문에서 뺀 줄: {shown}")
    for w in report.warnings:
        print(f"  [확인 필요] {w}")
    text, _ = load_source(paths.novel)
    preview = paths.derived_dir / "source_preview.txt"
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_text(text, encoding="utf-8")
    print(f"합친 본문 미리보기: {preview}  (장·문단이 제대로 나뉘었는지 확인하세요)")


def extract_entities(paths: NovelPaths, model: str, rebuild: bool) -> bool:
    """자동 사전을 만든다(이어서 하기 지원). 성공적으로 끝까지 처리했으면 True."""
    extractor, reason = try_create_extractor(model)
    if extractor is None:
        print(f"[건너뜀] 인물·별칭 사전 자동 추출에는 Claude가 필요합니다: {reason}")
        print("        사전 없이도 검색·요약은 동작하지만, 인물 인식과 인물 스냅샷은 비활성화됩니다.")
        return False

    corpus = load_novel(paths.novel, paths.novel.stem, paths.novel.stem)
    builder = (
        EntityExtractionBuilder(corpus, extractor)
        if rebuild
        else EntityExtractionBuilder.load(corpus, extractor, paths.auto_entities)
    )
    if builder.restarted:
        print("[알림] 원문 또는 추출 프롬프트가 바뀌어 사전을 처음부터 다시 만듭니다.")
    print(f"사전 추출: {extractor.model} · 이미 처리한 장 {len(builder.processed)}/{len(corpus.chapters)}")

    def on_chapter(label: str, b: EntityExtractionBuilder) -> None:
        b.save(paths.auto_entities)
        print(f"  {label} 완료 · 대상 {len(b.entities)}개 · 버린 항목 누적 {len(b.rejected)}개")

    report = builder.run(on_chapter=on_chapter)
    builder.save(paths.auto_entities)
    print(
        f"사전 완료: 처리 {report.processed}장 · 건너뜀 {report.skipped}장 · 새 대상 {report.added_entities}개 · "
        f"새 별칭 {report.added_aliases}개 · 정체 공개 {report.reveals}건 · 버림 {report.rejected}건"
    )
    if report.error:
        print(f"[중단] {report.error}\n       다시 실행하면 이 장부터 이어서 진행합니다.")
        return False
    for e in builder.entities.values():
        aliases = ", ".join(
            f"{t}({b}권 {c}장~)" if (b, c) != e.first_seen else t for t, (b, c) in e.aliases.items()
        )
        print(f"  - {e.entity_id}: {aliases}")
    return True


def print_comparison(paths: NovelPaths, reference_path: str) -> None:
    ref_catalog, _ = EntityCatalog.load(reference_path)
    auto_catalog, _ = EntityCatalog.load(paths.auto_entities)
    corpus = load_novel(paths.novel, paths.novel.stem)
    chunks = chunk_corpus(corpus)
    ref_catalog.bind(corpus, chunks)
    auto_catalog.bind(corpus, chunks)
    cmp = compare_catalogs(auto_catalog, ref_catalog)

    print(f"\n[비교] 자동 사전 vs {reference_path}  (숫자는 연결을 알게 되는 sequence)")
    sections = [
        ("일치", cmp.matched),
        ("더 늦게 연결 (안전, 그 사이 정보가 적음)", cmp.later),
        ("연결 안 함 (안전, 정보가 적음)", cmp.missing),
        ("더 이르게 연결 (스포일러 위험)", cmp.earlier),
        ("정답에 없는 연결 (확인 필요)", cmp.extra),
        ("정답에만 있는 이름", cmp.names_missing),
        ("자동에만 있는 이름", cmp.names_extra),
    ]
    for title, items in sections:
        print(f"  {title}: {len(items)}")
        for it in items:
            print(f"    · {it}")
    if cmp.earlier:
        verdict = f"스포일러 위험: 정답보다 이른 연결 {len(cmp.earlier)}건"
    elif cmp.extra:
        verdict = f"이른 연결 없음. 정답에 없는 연결 {len(cmp.extra)}건은 본문에 실제로 있는 표현인지 확인하세요"
    else:
        verdict = "이른 연결·정답에 없는 연결 없음"
    print("  판정:", verdict)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="소설 준비: 사전 자동 추출 + 요약 생성")
    p.add_argument("--novel", help="소설 파일(.txt/.pdf) 또는 나뉜 파일이 든 폴더 (기본: .env의 SPOILER_RAG_NOVEL 또는 샘플)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--no-llm", action="store_true", help="Claude를 쓰지 않음 (사전 추출 생략, 요약은 추출식)")
    p.add_argument("--rebuild", action="store_true", help="사전과 요약을 처음부터 다시 생성")
    p.add_argument("--compare", metavar="JSON", help="자동 사전을 이 정답 사전과 비교해 출력")
    args = p.parse_args()

    paths = paths_for(args.novel, PATHS.manual_entities) if args.novel else PATHS
    if not paths.novel.exists():
        sys.exit(f"소설 파일이나 폴더가 없습니다: {paths.novel}")

    corpus = load_novel(paths.novel, paths.novel.stem)
    print(f"소설: {paths.novel.name} · {len(corpus.books())}권 {len(corpus.chapters)}장 · 문단 {len(corpus.paragraphs)}개")
    print(f"구조: {STRUCTURE_LABELS[corpus.structure]}")
    print_source_report(corpus, paths)
    print(f"파생 데이터 위치: {paths.derived_dir}\n")

    # 1) 인물·별칭 사전
    if paths.manual_entities:
        print(f"[사전] 지정된 사전을 사용합니다: {paths.manual_entities} (자동 추출 생략)")
    elif args.no_llm:
        print("[사전] --no-llm: 자동 추출을 건너뜁니다.")
    else:
        extract_entities(paths, args.model, args.rebuild)

    if args.compare and paths.auto_entities.exists():
        print_comparison(paths, args.compare)

    # 2) 요약·스냅샷 (사전이 바뀌었으면 사전 지문이 달라져 자동으로 다시 만든다)
    print()
    summarizer: Summarizer = ExtractiveSummarizer()
    if not args.no_llm:
        created, reason = try_create_summarizer(args.model)
        if created is None:
            print(f"[알림] 요약은 추출식으로 만듭니다: {reason}")
        else:
            summarizer = created
    entities = paths.entities if paths.entities.exists() else None
    build_summaries(paths.novel, entities, paths.summaries, summarizer, args.rebuild)
    print("\n준비 완료. 앱 실행: run_app.bat 또는 .\\.conda\\python.exe -m streamlit run app.py")
    if paths.novel != PATHS.novel:
        print(f"[참고] 앱에서 이 소설을 쓰려면 .env에 SPOILER_RAG_NOVEL={paths.novel} 를 적으세요.")


if __name__ == "__main__":
    main()
