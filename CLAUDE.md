# 응답 언어 설정

- 모든 설명과 답변은 한국어로 작성한다.
- 오류 원인과 해결 방법은 한국어로 설명한다.
- 터미널 명령어, 코드, 파일명, 라이브러리명은 원래 표기를 유지한다.
- 사용자가 요청하지 않는 한 설명은 한국어로 제공한다.

# 대화창 마무리

- 프로젝트의 일부 목표를 완료하면, 클로드 새 세션으로 넘어가기 위해 CLAUDE.md에 완료된 목표와 다음 목표를 업데이트 해주세요. 새 클로드 세션에서 프로젝트 진행상황을 이어받을 수 있어야 합니다.
- 변경 사항이 git에 커밋 됐는지 확인해주세요.

# 프로젝트 진행 상황 (새 세션 인수인계용)

프로젝트: `spoiler_safe_rag_project.md` 명세서 기반의 Spoiler-safe RAG. 실행·구조·방어선은 `README.md` 참고.
환경: `.conda`의 Python 3.13, 실행은 `.\.conda\python.exe`. 테스트는 `.\.conda\python.exe -m pytest -q` (105개 통과, 그중 `test_mujeong_eval`은 무정 PDF가 있을 때만 돌고 약 40초). 실행은 `run_app.bat`(prepare 후 앱 시작). 데이터 흐름은 `DATA_FLOW.md`.

## 완료된 목표
- Phase 1: 원문 파싱·sequence 부여·진행도 pre-filter 검색·근거 기반 답변·출처 표시, Streamlit UI (사용자가 실제 API 키로 동작 확인함)
- Phase 2 대부분: BM25 + 벡터 RRF 하이브리드, 시점별 별칭 사전(`revealed_at`), 중복 제거
- Phase 3: 장 요약·권 요약·인물 시점별 스냅샷 (`spoiler_rag/summaries.py`, `python -m spoiler_rag.build_summaries`)
  - 장 순서대로 생성, 입력은 해당 장까지의 원문 + 이전 산출물, 레코드에 `generation_scope_end`·근거 ID·모델·프롬프트 버전·원문 해시 기록
  - 파생 레코드는 `Chunk`로 변환되어 기존 firewall/검증기를 통과 (인용 ID: C 원문, S 장 요약, B 권 요약, P 스냅샷)
- 자동화: 소설만 넣으면 동작 (`python -m spoiler_rag.prepare`)
  - 인물·별칭 사전 자동 추출 (`spoiler_rag/entity_extraction.py`): 장 순서 입력, 모든 별칭은 배운 장부터 연결(`revealed_at`), 본문에 없는 이름·근거 문장 거부, 정체 공개는 확인 호출 필수, 장 단위 저장·이어서 하기
  - 일반 텍스트 입력: 장 제목 줄 자동 인식 / 30문단 자동 분할, UTF-8·CP949
  - PDF·나뉜 파일 입력 (`spoiler_rag/sources.py`, `pypdf`): 폴더를 지정하면 .txt/.pdf를 숫자 순서로 합침. PDF는 머리글·바닥글 제거, 기호만 있는 줄 = 장 구분선, 줄 잇기. `prepare`가 파일별 결과·경고와 `source_preview.txt`를 남김
  - 작품별 파생 데이터: `data/derived/<소설 파일 이름>/entities.json`, `summaries.json`. 소설 변경은 `.env`의 `SPOILER_RAG_NOVEL`
  - 기본 사전은 자동 추출 사전. 손으로 만든 사전은 `SPOILER_RAG_ENTITIES`로 지정. `data/sample_entities.json`은 테스트·비교용 정답
- Phase 5 일부: 출력 검증기, 진행도 되돌리기 시 대화 격리, 질의 로그

## 다음 목표 (사용자 선택 대기)
- (진행 중) 위키문헌 『무정』 PDF 7개(`data/novels/무정/`, 커밋 제외 여부는 사용자 결정 대기)로 사용자가 테스트 중. `prepare --no-llm`으로 126장(파일당 20장) 분할 확인 완료. 이 PDF의 특징: 장 구분 기호가 pypdf에서 `/g0 /g0`(글리프 이름)으로 나옴, 낱말 사이 2칸·문장부호 옆 1칸 공백, 낱말 중간 줄바꿈 → `sources.fix_wide_spacing`, `LineJoiner`로 처리. 커밋 완료. `data/novels/`는 커밋 제외
- (완료 2026-10-15) 『무정』 질의 테스트 33개(사전 없음, 추출식 요약, claude-haiku-5-5) → `docs/무정_테스트_보고서.md`. 스포일러 유출 0건, 일반 질문 16개 중 정답 8·부분 3·못 함 5(지어낸 답 없음). 권장 조치: ① 무정 사전 추출(`prepare` Claude 실행) ② 라우터: "○○에서 무슨 일이 있었어"를 요약으로 보내지 않기 ③ 단권 장편 중간 요약 ④ 임베딩 교체 ⑤ 앱 예시 질문 작품별로. 재실행: `.\.conda\python.exe scripts\eval_mujeong.py [--llm]` (조치 적용 후 전후 비교용)
- 개선 후보: 권이 하나인 긴 작품은 장 요약을 대체할 상위 요약이 없음(『무정』 100장 시점 줄거리 근거 102개) → 권 안을 N장 단위로 묶는 중간 요약
0. (완료 2026-10-15) 실제 API(claude-haiku-5-5)로 샘플 사전 추출: 이른 연결 0, 정답에 없는 연결 2건(세라↔어부의 딸, 카르벤↔수도 카르벤, 모두 본문에 있는 올바른 연결), 누락 2건(모르간 단독 표기, 항해일지). 모델은 회색 기사와 카이든을 한 대상으로 묶음(안전)
1. Phase 2 마무리: 검색 평가 세트(Recall@k, MRR) → 실제 임베딩 모델 교체(`index.Embedder`), 선택적 reranker, 효과 측정
2. Phase 3 잔여: 사건 레코드(명세서 5.5), Claude로 `build_summaries` 실제 실행 후 품질 확인, UI에 인물 타임라인 표시
3. Phase 4: 관계 레코드(시점별), 선택적 Graph DB
4. Phase 5: 스포일러 유출률 평가 세트와 자동 회귀 평가

## 알아둘 점
- `data/derived/summaries.json`, `data/progress.json`, `data/logs/`는 재생성 가능해서 `.gitignore`에 있음
- 캐시가 없어도 앱은 시작 시 추출식 요약을 메모리에서 만들어 사용함. 사전 파일이 없으면 빈 사전으로 동작(인물 기능만 꺼짐)
- 테스트는 `config.SAMPLE_NOVEL_PATH`/`SAMPLE_ENTITIES_PATH`를 써야 함(`.env` 설정에 영향받지 않도록)
- 이전 위치의 `data/derived/summaries.json`은 더 이상 쓰지 않음
- 대상 이름을 사용자·모델에 보여 줄 때는 반드시 `catalog.display_name(id, cutoff)`를 쓸 것 (`catalog.name()`은 시점 무관)
- README 최상단 변경점은 구조적 요점만, 상세는 CHANGELOG.md
- 사용자가 아직 `git init`/커밋을 하지 않았을 수 있음. 세션 마무리 때 `git status`로 확인할 것
