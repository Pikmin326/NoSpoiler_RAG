"""사용자 진행도 저장소와 질의 로그 (JSON 파일 기반, 초기 버전)."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .models import Progress


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProgressStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def get(self, user_id: str, series_id: str) -> Progress | None:
        raw = self._read().get(f"{user_id}::{series_id}")
        return Progress(**raw) if raw else None

    def set(self, progress: Progress) -> Progress:
        progress.updated_at = _now()
        data = self._read()
        data[f"{progress.user_id}::{progress.series_id}"] = asdict(progress)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return progress


class QueryLogger:
    """스포일러 사고 분석·회귀 테스트용 질의 로그 (JSON Lines)."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None

    def log(self, record: dict) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _now(), **record}, ensure_ascii=False) + "\n")
