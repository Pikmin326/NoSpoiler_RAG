"""근거 기반 답변 생성 (Claude API)."""

from __future__ import annotations

from typing import Protocol

from . import firewall
from .models import Chunk, Turn

DEFAULT_MODEL = "claude-haiku-5-5"

SYSTEM_PROMPT = """역할: 독서 진행도를 준수하는 소설 도우미

규칙:
1. 사용자 메시지의 <evidence>에 있는 근거 자료의 정보만 사용해 답한다. 작품에 대한 사전 지식은 쓰지 않는다.
2. 독자가 읽은 위치 이후의 사건이나 정보는 언급하지 않는다.
3. 미래 사건을 암시하거나, 미래에 어떤 일이 일어나는지 확인해 주거나 부정하지 않는다.
   "나중에 밝혀진다", "중요한 역할을 하게 된다", "스포일러라서 말할 수 없다" 같은 표현도 쓰지 않는다.
4. 근거만으로 답할 수 없으면 "현재 읽은 범위의 근거만으로는 확인할 수 없어요."라고 짧게 말한다.
5. 인물의 관계나 동기를 사실처럼 단정하지 말고, 근거가 명시하는 범위를 지킨다.
6. 핵심 주장 뒤에는 근거 ID를 [C0001] 형식으로 붙인다. 제공되지 않은 ID는 절대 만들지 않는다.
7. 원문에 명시된 사실과 추론을 구분한다. 추론은 "(추론)"이라고 표시한다.
8. 한국어로 간결하게 답한다."""


class Generator(Protocol):
    model: str

    def generate(
        self,
        *,
        question: str,
        progress_label: str,
        cutoff: int,
        evidence: list[Chunk],
        history: list[Turn],
        feedback: str | None = None,
    ) -> str: ...


class GenerationRefused(RuntimeError):
    pass


def format_evidence(evidence: list[Chunk]) -> str:
    return "\n\n".join(
        f'<chunk id="{c.chunk_id}" location="{c.location_label}">\n{c.text}\n</chunk>'
        for c in evidence
    )


def build_user_prompt(
    question: str, progress_label: str, cutoff: int, evidence: list[Chunk], feedback: str | None
) -> str:
    parts = [
        f"<reader_progress>독자는 {progress_label}까지 읽었다. (허용 sequence <= {cutoff})</reader_progress>",
        f"<evidence>\n{format_evidence(evidence)}\n</evidence>",
    ]
    if feedback:
        parts.append(f"<revision_note>이전 초안이 검증을 통과하지 못했다: {feedback}\n위 규칙을 지켜 다시 작성하라.</revision_note>")
    parts.append(f"<question>{question}</question>")
    return "\n\n".join(parts)


class ClaudeGenerator:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium", client=None):
        import anthropic

        self.model = model
        self.effort = effort
        self.client = client or anthropic.Anthropic()

    def generate(
        self,
        *,
        question: str,
        progress_label: str,
        cutoff: int,
        evidence: list[Chunk],
        history: list[Turn],
        feedback: str | None = None,
    ) -> str:
        # 모델 호출 직전 마지막 방어선: 허용 범위 밖 근거가 있으면 요청 자체를 보내지 않는다.
        firewall.assert_safe(evidence, cutoff)

        messages: list[dict] = []
        for turn in history:
            if turn.question.strip() and turn.answer.strip():
                messages.append({"role": "user", "content": turn.question})
                messages.append({"role": "assistant", "content": turn.answer})
        messages.append(
            {
                "role": "user",
                "content": build_user_prompt(question, progress_label, cutoff, evidence, feedback),
            }
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": self.effort},
            messages=messages,
        )
        if response.stop_reason == "refusal":
            raise GenerationRefused("모델이 요청을 거절했습니다.")
        return "".join(b.text for b in response.content if b.type == "text").strip()


def try_create_claude(model: str = DEFAULT_MODEL, effort: str = "medium") -> tuple[Generator | None, str]:
    """Claude 클라이언트를 만들 수 있으면 반환하고, 아니면 (None, 사유)를 반환한다."""
    try:
        generator = ClaudeGenerator(model=model, effort=effort)
    except ImportError:
        return None, "anthropic 패키지가 설치되어 있지 않습니다."
    except Exception as exc:
        return None, f"Claude 클라이언트를 초기화하지 못했습니다: {exc}"

    client = generator.client
    # API 키, 토큰, 또는 `ant auth login` 프로필(토큰 캐시) 중 하나가 있어야 요청할 수 있다.
    if not (client.api_key or client.auth_token or getattr(client, "_token_cache", None)):
        return None, "Claude 인증 정보(ANTHROPIC_API_KEY 등)가 설정되어 있지 않습니다."
    return generator, ""
