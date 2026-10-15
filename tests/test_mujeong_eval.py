"""『무정』 질문 세트 회귀 평가 (API 없이). 소설 PDF가 data/novels/무정/에 있을 때만 실행한다."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_mujeong.py"
spec = importlib.util.spec_from_file_location("eval_mujeong", SCRIPT)
eval_mujeong = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mujeong)

pytestmark = pytest.mark.skipif(not eval_mujeong.NOVEL.exists(), reason="data/novels/무정/ 이 없음")


def test_mujeong_question_set_has_no_spoiler_leak():
    report = eval_mujeong.evaluate(use_llm=False)
    s = report["summary"]
    assert s["questions"] == 33
    assert (s["out_of_range"], s["leaked_in_evidence"], s["leaked_in_answer"]) == (0, 0, 0)
    # 보고서(docs/무정_테스트_보고서.md) 기준 11/16. 검색을 고치면 올라가야 하고 내려가면 안 된다.
    hits = int(s["evidence_hit"].split("/")[0])
    assert hits >= 11
