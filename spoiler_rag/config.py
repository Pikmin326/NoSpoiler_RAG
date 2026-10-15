"""경로 설정.

`.env` 또는 환경 변수로 바꿀 수 있다.

    SPOILER_RAG_NOVEL=data/novels/내소설.txt   # 사용할 소설 (기본: 샘플 소설). .pdf 파일이나 나뉜 파일이 든 폴더도 된다.
    SPOILER_RAG_ENTITIES=data/내사전.json      # (선택) 손으로 만든 인물·별칭 사전. 없으면 자동 추출 사전을 쓴다.

파생 데이터(자동 추출 사전, 요약 캐시)는 작품마다 `data/derived/<소설 파일·폴더 이름>/`에 저장된다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ENV_PATH = ROOT / ".env"


def load_env(path: Path = ENV_PATH) -> None:
    """프로젝트 루트의 .env를 읽어 환경 변수로 넣는다. 이미 설정된 환경 변수는 덮어쓰지 않는다."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and value:
            os.environ.setdefault(key, value)


# 경로를 계산하기 전에 .env를 읽어야 SPOILER_RAG_NOVEL 등이 반영된다.
load_env()

SAMPLE_NOVEL_PATH = DATA_DIR / "sample_novel.txt"
SAMPLE_ENTITIES_PATH = DATA_DIR / "sample_entities.json"  # 손으로 만든 정답 사전 (테스트·자동 추출 비교용)
PROGRESS_PATH = DATA_DIR / "progress.json"
LOG_PATH = DATA_DIR / "logs" / "queries.jsonl"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class NovelPaths:
    novel: Path
    derived_dir: Path
    auto_entities: Path  # 자동 추출 사전
    summaries: Path  # 요약·스냅샷 캐시
    manual_entities: Path | None  # 손으로 만든 사전 (지정한 경우)

    @property
    def entities(self) -> Path:
        """실제로 쓸 사전: 손으로 만든 사전을 지정했으면 그것, 아니면 자동 추출 사전."""
        return self.manual_entities or self.auto_entities


def paths_for(novel: str | Path, manual_entities: str | Path | None = None) -> NovelPaths:
    novel = _resolve(novel)
    derived = DATA_DIR / "derived" / novel.stem
    return NovelPaths(
        novel=novel,
        derived_dir=derived,
        auto_entities=derived / "entities.json",
        summaries=derived / "summaries.json",
        manual_entities=_resolve(manual_entities) if manual_entities else None,
    )


PATHS = paths_for(
    os.environ.get("SPOILER_RAG_NOVEL", SAMPLE_NOVEL_PATH),
    os.environ.get("SPOILER_RAG_ENTITIES") or None,
)
NOVEL_PATH = PATHS.novel
ENTITIES_PATH = PATHS.entities
SUMMARIES_PATH = PATHS.summaries
