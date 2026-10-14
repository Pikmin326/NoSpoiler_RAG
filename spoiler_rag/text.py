"""형태소 분석기 없이 동작하는 간단한 한국어 토크나이저."""

from __future__ import annotations

import re

_WORD = re.compile(r"[0-9A-Za-z가-힣]+")

# 길이가 긴 조사부터 확인해야 "에게서"가 "서"로 잘못 잘리지 않는다.
_JOSA = sorted(
    [
        "에게서", "으로부터", "로부터", "에서", "에게", "한테", "으로", "까지", "부터",
        "처럼", "보다", "이라고", "라고", "이랑", "랑", "은", "는", "이", "가", "을",
        "를", "의", "에", "와", "과", "도", "로", "만", "야", "아",
    ],
    key=len,
    reverse=True,
)


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def strip_josa(word: str) -> str:
    for josa in _JOSA:
        if word.endswith(josa) and len(word) - len(josa) >= 2:
            return word[: -len(josa)]
    return word


def bm25_tokens(text: str) -> list[str]:
    """원형 단어와 조사를 뗀 형태를 함께 색인한다."""
    tokens: list[str] = []
    for w in words(text):
        tokens.append(w)
        stem = strip_josa(w)
        if stem != w:
            tokens.append(stem)
    return tokens


def char_ngrams(text: str, sizes: tuple[int, ...] = (2, 3)) -> list[str]:
    """단어 경계 표시를 붙인 문자 n-gram. 조사·어미 변형에 강하다."""
    grams: list[str] = []
    for w in words(text):
        padded = f"<{w}>"
        for n in sizes:
            grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return grams
