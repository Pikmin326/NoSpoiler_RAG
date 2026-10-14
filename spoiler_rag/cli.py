"""명령줄에서 빠르게 질문해 보기.

    python -m spoiler_rag.cli --book 1 --chapter 4 "회색 기사는 누구야?"
    python -m spoiler_rag.cli --book 1 --chapter 4 --no-llm "하벨은 어떻게 됐어?"
"""

from __future__ import annotations

import argparse
import sys

from .config import ENTITIES_PATH, LOG_PATH, NOVEL_PATH
from .llm import DEFAULT_MODEL, try_create_claude
from .pipeline import SpoilerSafeRAG


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Spoiler-safe 소설 질의")
    p.add_argument("question")
    p.add_argument("--book", type=int, required=True)
    p.add_argument("--chapter", type=int, required=True)
    p.add_argument("--no-llm", action="store_true", help="Claude 없이 검색 결과만 표시")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    generator = None
    if not args.no_llm:
        generator, reason = try_create_claude(args.model)
        if generator is None:
            print(f"[알림] {reason} 검색 결과만 표시합니다.\n")

    rag = SpoilerSafeRAG.build(NOVEL_PATH, ENTITIES_PATH, generator=generator, log_path=LOG_PATH)
    progress = rag.progress_for(args.book, args.chapter, user_id="cli")
    answer = rag.ask(args.question, progress)

    print(f"[진행도 {progress.label} | sequence <= {progress.current_sequence} | 유형 {answer.query_type.value} | 모드 {answer.mode}]")
    print(answer.text)
    if answer.note:
        print(f"\n[참고] {answer.note}")
    print("\n근거:", ", ".join(f"{c.chunk_id}({c.location_label})" for c in answer.evidence) or "없음")


if __name__ == "__main__":
    main()
