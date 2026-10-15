"""장별 요약·권별 요약·인물 상태 스냅샷을 (재)생성한다.

    python -m spoiler_rag.build_summaries              # Claude로 생성 (인증 정보 필요)
    python -m spoiler_rag.build_summaries --no-llm     # API 없이 추출식으로 생성
    python -m spoiler_rag.build_summaries --rebuild    # 기존 캐시를 지우고 처음부터

장을 순서대로 처리하며, 각 단계의 입력은 그 장까지의 원문과 이전 단계의 산출물뿐이다.
레코드마다 파일에 저장하므로 중간에 멈춰도 다시 실행하면 이어서 진행한다.
보통은 사전 추출까지 한 번에 하는 `python -m spoiler_rag.prepare`를 쓴다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from .config import ENTITIES_PATH, NOVEL_PATH, SUMMARIES_PATH
from .llm import DEFAULT_MODEL
from .pipeline import load_sources
from .summaries import (
    BuildReport,
    DerivedBuilder,
    DerivedRecord,
    ExtractiveSummarizer,
    Summarizer,
    SummaryStore,
    try_create_summarizer,
)


def build_summaries(
    novel_path: str | Path,
    entities_path: str | Path | None,
    output: str | Path,
    summarizer: Summarizer,
    rebuild: bool = False,
    log: Callable[[str], None] = print,
) -> BuildReport:
    corpus, chunks, catalog = load_sources(novel_path, entities_path)
    store = SummaryStore() if rebuild else SummaryStore.load(output)
    log(f"요약기: {summarizer.model} · 기존 레코드 {len(store)}개 · 장 {len(corpus.chapters)}개")

    def on_record(rec: DerivedRecord) -> None:
        store.save(output)  # 레코드마다 저장해 중간에 멈춰도 이어서 할 수 있게 한다.
        flag = "" if rec.status in ("verified", "extractive") else f"  ⚠ {'; '.join(rec.issues)}"
        who = f" {rec.character_id}" if rec.character_id else ""
        log(f"  {rec.record_id} {rec.kind}{who} ({rec.book_number}권 {rec.chapter_number}장, 범위 ≤{rec.generation_scope_end}) {rec.status}{flag}")

    report = DerivedBuilder(corpus, chunks, catalog, summarizer, store).run(on_record=on_record)
    store.save(output)
    log(
        f"요약 완료: 새로 {report.created}개 · 건너뜀 {report.skipped}개 · 오래되어 삭제 {report.dropped_stale}개"
        f" · 상태 {report.by_status or '-'}"
    )
    if report.by_status.get("fallback"):
        log("[주의] 일부 요약은 검증 실패나 API 오류로 추출식으로 대체되었습니다. 위 ⚠ 항목을 확인하세요.")
    return report


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="시점별 요약·스냅샷 생성")
    p.add_argument("--no-llm", action="store_true", help="Claude 없이 추출식으로 생성")
    p.add_argument("--rebuild", action="store_true", help="기존 캐시를 무시하고 처음부터 생성")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", default=str(SUMMARIES_PATH))
    args = p.parse_args()

    summarizer: Summarizer = ExtractiveSummarizer()
    if not args.no_llm:
        created, reason = try_create_summarizer(args.model)
        if created is None:
            print(f"[알림] {reason} 추출식으로 생성합니다.\n")
        else:
            summarizer = created

    build_summaries(NOVEL_PATH, ENTITIES_PATH, args.output, summarizer, args.rebuild)
    print(f"저장: {args.output}")


if __name__ == "__main__":
    main()
