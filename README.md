# Spoiler-safe RAG — 초기 버전

독자가 읽은 위치까지만 검색하고 답하는 소설 도우미입니다. 설계 기준은 [spoiler_safe_rag_project.md](spoiler_safe_rag_project.md)를 따릅니다.
이번 버전은 명세서의 **Phase 1**(원문 파싱·sequence 부여·진행도 필터 검색·근거 기반 답변·출처 표시)에 다음 기능을 일부 더했습니다.

- Phase 2: BM25 하이브리드 검색, 별칭 사전
- Phase 3: 인물 언급 타임라인, 첫 등장 조회
- Phase 5: 출력 검증기, 진행도 되돌리기 시 대화 격리, 질의 로그

## 실행

```powershell
# 의존성 설치 (프로젝트의 .conda 환경 기준)
.\.conda\python.exe -m pip install -r requirements.txt

# (선택) Claude 답변 생성을 쓰려면 API 키 설정
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# 웹 UI
.\.conda\python.exe -m streamlit run app.py

# 명령줄
.\.conda\python.exe -m spoiler_rag.cli --book 1 --chapter 4 "회색 기사는 누구야?"

# 테스트
.\.conda\python.exe -m pytest -q
```

API 키가 없어도 앱은 동작합니다. 이때는 Claude 대신 **검색된 근거 문단**을 그대로 보여 줍니다(첫 등장 질문은 구조화 데이터로 직접 답함).

## 인터페이스

Streamlit 채팅 UI입니다.

- 사이드바: 권·장 선택 → 진행도 저장(`data/progress.json`), 허용 범위 표시, Claude 사용 토글
- **아직 읽지 않은 권·장의 제목은 숨깁니다.** 장 제목도 스포일러가 될 수 있기 때문입니다(예: "마지막 등불").
- 답변마다 질문 유형, 생성 모드, 근거 문단(권·장, chunk ID)을 표시합니다.
- 진행도를 뒤로 돌리면 이후 범위에서 나눈 대화를 화면과 LLM 컨텍스트에서 제거합니다.

## 구조

```
app.py                     Streamlit UI
spoiler_rag/
  corpus.py                원문 파싱(# 권 / ## 장 / 빈 줄 문단), sequence 부여, 장 경계를 넘지 않는 chunking
  entities.py              엔티티·별칭 사전, 별칭별 "알게 되는 시점(known_seq)" 계산, chunk 태깅
  index.py                 문자 n-gram TF-IDF 벡터 인덱스 + BM25 인덱스 (둘 다 pre-filter)
  retriever.py             하이브리드(RRF) 검색, 엔티티 기반 조회, 첫 언급, 줄거리 타임라인
  router.py                규칙 기반 질문 유형 분류 (요약/인물/관계/첫 등장/사실)
  firewall.py              진행도 경계를 강제하는 단일 지점
  llm.py                   Claude 답변 생성 (claude-haiku-5-5)
  verifier.py              출력 검증
  pipeline.py              전체 질의 흐름
  store.py                 진행도 저장소, 질의 로그(data/logs/queries.jsonl)
data/
  sample_novel.txt         테스트용 자작 소설 (2권 8장, 저작권 문제 없음)
  sample_entities.json     인물·장소 엔티티와 별칭
tests/test_spoiler_safety.py
```

## 스포일러 방어선

| 단계 | 구현 |
|---|---|
| 색인 | 모든 chunk에 `sequence_start/end`가 있어야 하고, 위치 메타데이터가 없으면 색인 자체를 거부합니다. chunk는 장 경계를 넘지 않습니다. |
| 1차: 검색 | 벡터·BM25 모두 점수 계산 **전에** `sequence_end <= cutoff`로 후보를 자릅니다. BM25의 문서 빈도도 허용 범위 안에서만 계산합니다. |
| 1차: 별칭 | "회색 기사 → 카이든" 같은 연결은 `revealed_at` 이전에는 질의 확장·엔티티 조회에 쓰지 않습니다. |
| 1차: firewall | 모든 검색 경로의 결과를 한 번 더 검사합니다(메타데이터 없음·경계 걸침·미래 → 차단). |
| 2차: 생성 | `ClaudeGenerator`는 API 요청 직전에 다시 검사하고, 위반이 있으면 요청을 보내지 않고 예외를 냅니다. 프롬프트는 명세서 7.4를 따릅니다. |
| 3차: 검증 | 제공되지 않은 근거 ID 인용, 아직 등장하지 않은 이름, "나중에 밝혀" 같은 메타 표현을 검사합니다. 실패하면 1회 재생성하고, 다시 실패하면 중립 응답으로 대체합니다. |
| 메모리 | 대화 턴마다 생성 당시의 cutoff를 저장하고, 현재 cutoff보다 큰 턴은 컨텍스트에서 제외합니다. |

답할 수 없을 때는 미래의 존재를 암시하지 않도록 항상 *"현재 읽은 범위의 근거만으로는 확인할 수 없어요."* 라고만 답합니다.

## 다른 작품 넣기

1. 원문을 아래 형식의 UTF-8 텍스트로 준비합니다(저장·처리 권한을 먼저 확인하세요).
   ```
   # 1권 책 제목
   ## 1장 장 제목
   문단 …

   문단 …
   ```
2. `sample_entities.json` 형식으로 엔티티를 작성합니다. 나중에 정체가 밝혀지는 별칭은 `{"text": "...", "revealed_at": {"book": 2, "chapter": 3}}`로 표기합니다.
3. 환경 변수 `SPOILER_RAG_NOVEL`, `SPOILER_RAG_ENTITIES`로 경로를 지정하고 실행합니다.

## 현재 한계와 다음 단계

- **임베딩**: 외부 의존성 없이 동작하는 문자 n-gram TF-IDF를 baseline으로 씁니다. 의미 검색 품질을 높이려면 `index.Embedder` 프로토콜을 구현한 다국어 임베딩 모델로 교체하세요. 벡터 IDF는 전체 원문 통계로 계산되는데, 순위에만 영향을 줄 뿐 미래 내용이 컨텍스트에 들어가지는 않습니다.
- **줄거리 요약**: 장별 요약 레코드가 아직 없어서, 허용 범위의 원문 chunk를 장마다 고르게 최대 16개 뽑아 씁니다. 긴 작품에서는 시점별 장 요약(Phase 3)이 필요합니다.
- **라우터·검증기**는 규칙 기반입니다. 간접 스포일러는 이름·표현 수준에서만 잡아냅니다.
- **진행도 단위**는 장입니다. 문단 단위 진행도는 모델(`current_sequence`)이 이미 지원하므로 UI만 추가하면 됩니다.
- 진행도 저장소는 단일 사용자용 JSON 파일입니다.
