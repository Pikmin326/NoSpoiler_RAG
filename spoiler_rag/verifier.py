"""답변 출력 검증 (3차 방어선).

- 인용한 근거 ID가 실제로 제공된(=허용 범위 안의) 근거인지
- 독자가 아직 접하지 않은 이름이 답변에 등장하는지 (질문에 사용자가 직접 쓴 이름은 예외)
- 미래의 존재를 암시하는 메타 표현이 있는지
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .entities import EntityCatalog
from .models import Chunk

CITATION = re.compile(r"\[(C\d{4})\]")

META_SPOILER_PHRASES = (
    "나중에 밝혀",
    "이후에 밝혀",
    "앞으로 밝혀",
    "나중에 드러",
    "이후 드러나",
    "다음 권",
    "후반부",
    "결말",
    "스포일러",
    "아직 읽지 않은",
    "중요한 역할을 하게",
)


@dataclass
class Verification:
    passed: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def verify(
    answer: str,
    evidence: list[Chunk],
    cutoff: int,
    question: str,
    catalog: EntityCatalog,
) -> Verification:
    issues: list[str] = []
    warnings: list[str] = []

    allowed_ids = {c.chunk_id for c in evidence}
    cited = set(CITATION.findall(answer))
    unknown = sorted(cited - allowed_ids)
    if unknown:
        issues.append(f"제공되지 않은 근거 ID 인용: {', '.join(unknown)}")

    for term in sorted(catalog.future_terms(cutoff)):
        if term in answer and term not in question:
            issues.append(f"독자가 아직 접하지 않은 이름 언급: {term}")

    for phrase in META_SPOILER_PHRASES:
        if phrase in answer:
            issues.append(f"미래를 암시할 수 있는 표현: '{phrase}'")

    if evidence and not cited and len(answer) > 80:
        warnings.append("근거 인용 없이 작성된 답변")

    return Verification(passed=not issues, issues=issues, warnings=warnings)
