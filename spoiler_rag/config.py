"""경로 설정. 환경 변수로 다른 작품 파일을 지정할 수 있다."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

NOVEL_PATH = Path(os.environ.get("SPOILER_RAG_NOVEL", DATA_DIR / "sample_novel.txt"))
ENTITIES_PATH = Path(os.environ.get("SPOILER_RAG_ENTITIES", DATA_DIR / "sample_entities.json"))
PROGRESS_PATH = DATA_DIR / "progress.json"
LOG_PATH = DATA_DIR / "logs" / "queries.jsonl"
