# 프로젝트 규칙 · 개인 맞춤 도서 추천 에이전트

무엇을·왜 만드는지는 `SERVICE.md`가 기준이다. 모호하면 짐작하지 말고 사용자에게 묻는다.
작업은 `TASKS.md`에서 한 번에 하나만 한다. 끝나면 완료 확인 방법대로 실행해 보이고 체크한다.

## 기술 스택
- Python 3.11+, LangChain, LangGraph (`langchain.agents.create_agent` 또는 LangGraph StateGraph)
- LLM: Amazon Bedrock `ChatBedrockConverse`, 모델 `us.anthropic.claude-sonnet-4-5-20250929-v1:0`, region `us-east-1`, temperature 0
  - **공통 규칙 (사용자 결정 11-19 저녁, 이전 규칙 대체)**: 모든 Bedrock 호출(LLM·임베딩·평가·RAGAS)은 `.env`의 `AWS_ACCESS_KEY_ID_NEW`·`AWS_SECRET_ACCESS_KEY_NEW`·`AWS_DEFAULT_REGION_NEW`를 쓴다 (`config.bedrock_credentials()`). **LLM은 Sonnet 4.5로 고정**한다. 경로가 막히면 같은 Sonnet 4.5의 다음 경로로 넘어간다 (사용자 결정 11-19 밤, `config.model_routes()`): ① `BEDROCK_MODEL_ID`(us.)·기본 리전 → ② `BEDROCK_FALLBACK_MODEL_ID`(global.)·기본 리전 → ③ global.·`AWS_FALLBACK_REGION`(ap-northeast-2). `us.` ID는 미국 밖 리전에서 못 부르므로 ③에서 뺀다. 경로를 바꾸면 콘솔 ⚠️ + trace `model_route`. 모든 경로가 막히면 Haiku 등 다른 모델로 전환하지 않고 **멈추고 알린다** (`agent.call_model` → `ModelUnavailable`, 콘솔 ⛔ + trace `model_error`, 평가는 그 자리에서 중지). 임베딩은 Titan v2 (임베딩 전용 모델이라 Sonnet으로 바꿀 수 없음)
  - 멈춤은 개발자(사용자)가 알아차릴 수 있게 콘솔 ⛔와 trace `model_error`로 알린다. 서비스 이용자에게는 모델 이름 없이 "지금 연결할 수 없어요"만 보인다
  - 이전 결정 기록: 11-19 오전 Haiku 자동 전환 → 오후 A안(ReAct Haiku 우선) → 오후 `_NEW` 키·Sonnet 우선 → 저녁 Sonnet 고정·전환 없음 → **밤 Sonnet 고정 + Sonnet 경로 3단계(현재)**. 1차 평가와 개선 루프 1~3회 결과에는 판정 1건(19번)이 Haiku로 넘어간 기록이 남아 있다
  - 11-19 확인: 이 계정에서 Sonnet 4.5·Haiku 4.5만 호출됨 (Claude 3.5 Sonnet v2·3.7·Sonnet 4는 호출 불가)
- 임베딩: Bedrock `amazon.titan-embed-text-v2:0`, 벡터 DB: Chroma (`chroma_db/`), 키워드 검색: BM25 (kiwipiepy 형태소)
- MCP: `langchain-mcp-adapters`의 `MultiServerMCPClient`, 정보나루 원격 서버 `https://www.data4library.kr/mcp` (streamable_http), 헤더 `X-Auth-Key`
- API 서버: FastAPI
- 평가: pytest, RAGAS, LLM-as-Judge
- 참고 코드: `../day1_practice` ~ `../day7_practice` (특히 day4 MCP, day5 가드레일, day7 run_eval·llm_judge·local_tracer)
- 사용자 허락 없이 새 프레임워크·라이브러리를 추가하지 않는다.

## 폴더 독립 (사용자 결정 2026-11-19)
- `mini-pjt_seonahshin/`을 최상위 폴더로 본다. 이 폴더만 떼어 새 public 레포로 올려도 그대로 돌아가야 한다.
- 모든 경로는 이 폴더 기준으로 쓴다. `.env`도 이 폴더의 것만 읽는다 (상위 폴더의 `.env`는 읽지 않는다).
- 참고 코드는 읽고 필요한 부분을 `src/`로 옮겨 쓴다. `../day*_practice`를 import하거나 경로로 참조하지 않는다.
- 필요한 라이브러리는 이 폴더의 `requirements.txt`에 적는다.

## 폴더 구조 (제출 규약, 바꾸지 않는다)
```
src/agent.py        메인 에이전트 그래프
src/tools.py        도메인 도구 (get_reading_log, search_catalog, filter_candidates)
src/retriever.py    RAG 파이프라인 (스냅샷 → 하이브리드 검색)
src/mcp_client.py   정보나루 MCP 연결 (쓸 도구만 골라 노출)
src/ranker.py       추천 목적별 리랭킹 (코드 점수)
src/guards.py       입력·출력 가드레일
src/api.py          FastAPI POST /query, 프로필·독서 기록 입력 API, 웹 페이지 서빙
src/static/         입력·추천 웹 페이지 (HTML 한 장, 프레임워크 없이)
src/build_snapshot.py  도서 목록 스냅샷 수집 스크립트 (11-19 추가)
src/config.py      .env·비밀값·모델 ID·호출 차단선 한곳 처리 (11-19 사용자 결정 추가)
data/               profiles.json, reading_log.csv, catalog_snapshot.json, book_tags.csv
tests/              코드 규칙 pytest (11-19 추가, T5-0)
evaluation/         test_queries.csv, run_eval.py, round1_report.md, round2_report.md
demo/               시연 캡처 이미지 + README.md(이미지별 설명) (11-20 사용자 요청 추가)
```

## 주고받는 형식 (제출 규약)
- `POST /query`, Body `{"question": "..."}`
- 응답: `{"answer": str, "contexts": [{"doc_id": str, "text": str}], "trace": [{"step": str, "input": str, "output": str}]}`
- `doc_id`는 ISBN13을 쓴다.
- `contexts`에는 추천 근거가 된 책소개·키워드와 함께, 추천 이유에 쓴 독서 기록 항목도 넣는다 (`doc_id`는 `reading_log:<reader>:<title>`). RAGAS faithfulness가 "내가 읽은 책과 닮았다"는 문장의 근거를 찾을 수 있게 하기 위함이다.
- Body에 선택 필드 `reader`(프로필 별칭)를 받을 수 있다. 없으면 질문에서 독자를 찾고, 그래도 모르면 되묻는다.
- Body에 선택 필드 `purpose`(self/child/gift, 웹 화면 드롭다운 값)를 받을 수 있다. 없으면 코드가 질문 내용과 독자 프로필로 정한다 (어린이 프로필이면 child, 본인이면 self).
- 이번 제출은 로그인 없이 `owner=demo`로 고정한다.
- 프로필·독서 기록 입력은 별도 엔드포인트(`POST /profiles`, `POST /reading-log`)로 받는다. `/query` 규약은 바꾸지 않는다.

## 도메인 규칙 (반드시 코드로 지킨다)
- 학년 판정, 이미 읽은 책 제외, 목록 존재 확인은 **Python 코드로 판정**한다. LLM에게 맡기지 않는다.
- 그 밖의 계산·규칙 판정도 Python 코드가 원칙이다. 필요하면 LLM 판정을 보조로 덧붙이되, 코드 판정을 뒤집지 않는다.
- `filter_candidates`(걸러내기 `src/tools.py` + 정렬 `src/ranker.py`)는 Agent가 고르는 선택 도구가 아니라 **추천 결과를 내기 직전에 항상 실행되는 그래프 단계**로 둔다.
- 추천 목적(`purpose`: self=나에게 / child=자녀에게 / gift=선물용)을 Pydantic enum으로 받는다. 목적별 우선순위는 `src/ranker.py`의 Python 점수 함수로 계산한다 (취향 점수 + 목적 가산점 + 프로필 조건 가산점: 60대 이상이면 큰글씨도서 `book_dvsn=big`). 가드레일 걸러내기 다음에 적용한다.
- 연령 판정은 SERVICE.md 3절 표를 따른다. 어린이 프로필은 ISBN 부가기호(`addition_symbol`) 첫 자리 `7`(아동)만 허용.
- 시연 프로필: `아이` (9~11세, `from_age=9`, `to_age=11`), `나` (40대, `age=40`), `김부장` (45세, `age=45`, 선물용 시연, 11-20 사용자 결정으로 추가), 모두 등록한 이용자(`owner`)가 `demo`. 평가 문항은 `아이`·`나` 대상. (`owner`는 프로필을 등록한 이용자이고, 추천받을 독자는 `reader`다)
- 추천 권수는 `top_k`로 받는다. 기본값 5, 선택지 5/10/15/20 (웹 화면 드롭다운, 11-19 B6 반영).
- 독서 기록 입력은 `reader`, `title`(필수), `author`, `volumes`, `disliked`(선택)뿐이다. `volumes`는 시리즈를 한 줄로 적을 때 읽은 권 범위(예: `1~15`, `전권`)다. 같은 시리즈의 이미 읽은 권은 추천에서 빼고, 다음 권은 추천 후보가 될 수 있다. 목록에 있는 책은 좋았던 책으로 본다. 취향은 정보나루 키워드·KDC·책소개로 코드와 검색이 파악한다. 입력 칸을 늘리지 않는다 (서비스 원칙: 단순하고 간편하게).
- `author`는 같은 제목의 다른 책을 구분할 때만 쓴다.
- 도서 목록 스냅샷(`data/catalog_snapshot.json`)에는 책마다 서지정보, ISBN 부가기호, KDC 분류, 키워드를 저장한다. 책소개는 Agent용 4개 도구에 없어서, 수집 전용 5번째 도구 `get_book_analysis`(type=loan, 1회 1건)의 응답에서 받는다 (11-19, A안 후속. `get_book_full_profile`은 1회 6건이라 쓰지 않음). 평가를 같은 조건으로 하려고 한 번 받아 저장하고, 평가 중에는 다시 받지 않는다.
- `disliked=Y`인 책: 추천에서 제외하고, 그 책과 비슷한 책은 순위에서 감점한다.
- 추천 결과는 Pydantic 모델로 구조화한다: 책마다 `isbn13`, `title`, `author`, `reason`(독서 기록 근거), `fit`(학년·난이도 근거), `parent_check`(보호자 확인 필요 여부).
- 보호자 확인 표시: 어린이 프로필 추천의 제목·책소개·키워드를 주의 단어 목록(`data/caution_words.txt`)과 코드로 대조한다. 걸려도 빼지 않고 `parent_check=True`만 붙인다.
- 프로필은 별칭(`reader`)·연령대·등록한 이용자(`owner`)만 저장한다. 자녀·선물 상대 프로필도 질문하는 이용자가 등록한다. 실명·학교·연락처는 받지 않는다.
- 독서 기록은 질문한 이용자가 등록한 프로필(같은 `owner`)의 것만 불러온다.

## 코드 규칙
- 파일 하나에 한 가지 역할만 둔다.
- 함수와 도구에는 한국어 docstring을 쓴다. 도구 설명에는 "언제 쓰고 언제 쓰지 않는지"를 적는다.
- 비밀 값(AWS 키, `DATA4LIBRARY_AUTH_KEY`)은 `.env`에만 적고 코드·로그·답변에 적지 않는다.
- `.env`를 읽는 곳은 `src/config.py` 하나다 (11-19 사용자 결정). 다른 파일은 `os.getenv`로 키를 직접 꺼내지 않고 config 함수(`bedrock_credentials`, `data4library_key`, `daily_limit`, `secret_values`)만 쓴다. 새 키를 추가하면 `config.SECRET_KEYS`에도 넣는다 (출력 가드가 지울 수 있게).
- 정보나루 `bookImageURL`(책표지)은 저장하거나 모델에 넣지 않는다 (이용 조건).

## 정보나루 호출량 안전장치 (사용자 요청 2026-11-18)
- 정보나루: 무료지만 하루 30,000건 한도(API·MCP 합산). 모든 호출을 `logs/api_usage.jsonl`에 날짜별로 센다. **하루 30,000건(정보나루 한도)에 닿으면 코드가 호출을 멈추고** 스냅샷·캐시만 쓴다 (11-19 저녁 사용자 결정 A안: 20,000 → 30,000).
- 같은 요청은 다시 부르지 않는다: 정보나루 응답은 `data/cache/`에 저장해 재사용한다.
- 대량 수집(스냅샷 만들기)은 시작 전에 예상 호출 수를 사용자에게 알린다.
- 한도 단계 (사용자 결정 2026-11-19): 서버 IP 등록이 실제로 먹는지 확인하기 전에는 하루 한도를 **500건**으로 보고 **450건**에서 멈춘다. 등록이 확인되면 사용자 결정 후 30,000건 기준(20,000건 차단)으로 올리고 수집 범위를 넓힌다. → 11-19 등록 확인(560건 정상), `.env` `DATA4LIBRARY_DAILY_LIMIT=20000`으로 올림. → 11-19 저녁 30,000으로 올림 (차단선 = 정보나루 하루 한도, 기록·캐시는 유지).
- 호출 예산 분배: 수집 전에 카테고리별 예산을 먼저 정해 사용자에게 보여 준다 (아이용·나용·선물용·독서 기록 확인·개발 시험 예비분).
  - 한 카테고리가 예산을 다 쓰고 다른 카테고리가 비지 않도록 **카테고리를 돌아가며(라운드 로빈) 조금씩** 수집한다. 중간에 멈춰도 모든 카테고리가 고르게 채워져 있어야 한다.
  - 한 번 호출로 여러 권을 받는 목록 조회(인기 대출 등)는 한 번에 받을 수 있는 최대 권수로 부르고, 책 한 권마다 부르는 조회(책소개·키워드)에 예산을 더 준다.

## 하지 말 것
- 요청하지 않은 파일을 새로 만들지 않는다.
- 기존 파일을 통째로 다시 쓰지 않는다. 바뀐 부분만 고친다.
- 평가셋(`test_queries.csv`) 문항을 통과시키려고 문항을 지우거나 고치지 않는다. 고칠 때는 사용자에게 먼저 묻는다.
- 한 번에 여러 문제를 고치지 않는다.
