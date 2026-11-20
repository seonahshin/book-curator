"""설정·비밀값 한곳 처리 (사용자 결정 11-19). `.env`를 읽는 곳은 이 파일 하나뿐이다.

- 키 값은 `.env` 파일에만 적는다 (.gitignore로 GitHub에 올리지 않음). 코드·로그·답변에 적지 않는다.
- 다른 파일은 os.getenv로 키를 직접 꺼내지 않고 여기 함수만 쓴다:
  bedrock_credentials()   Bedrock(LLM·임베딩·평가·RAGAS) 접속 정보: *_NEW 키 (없으면 기본 AWS_* 키)
  model_routes()          Sonnet 4.5 호출 경로 순서: 기본 모델 → 대체 모델(global) → 대체 리전 (모두 Sonnet 4.5)
  data4library_key()      정보나루 MCP 인증키
  daily_limit()           정보나루 하루 호출 차단선 (상한 30,000 = 정보나루 하루 한도)
  secret_values()         답변에서 지울 비밀값 목록 (출력 가드·평가가 씀). 키 이름은 SECRET_KEYS 한곳에서 관리
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # 이 폴더의 .env만 읽는다 (상위 폴더 .env는 읽지 않음)

# 모델 (Sonnet 4.5 고정, 사용자 결정 11-19 저녁). 막히면 같은 Sonnet 4.5의 다른 호출 경로로 (11-19 밤)
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
FALLBACK_MODEL_ID = os.getenv("BEDROCK_FALLBACK_MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0")
FALLBACK_REGION = os.getenv("AWS_FALLBACK_REGION", "ap-northeast-2")
EMBED_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# 정보나루 호출량 (사용자 결정 11-19 저녁 A안: 차단선을 정보나루 하루 한도 30,000건까지)
HARD_CEILING = 30_000      # 정보나루 하루 한도(API·MCP 합산)
DEFAULT_DAILY_LIMIT = 450  # .env 값이 없을 때: 서버 IP 등록 전 기준(500건)으로 보고 450건에서 멈춤

# 답변·로그에 나오면 안 되는 비밀값의 이름 (값은 .env에만 있다)
SECRET_KEYS = ("DATA4LIBRARY_AUTH_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
               "AWS_ACCESS_KEY_ID_NEW", "AWS_SECRET_ACCESS_KEY_NEW", "NL_API_KEY")


def bedrock_credentials(region: str | None = None) -> dict:
    """Bedrock 호출 공통 접속 정보 (사용자 공통 규칙 11-19): .env의 *_NEW 키를 쓰고, 없을 때만 기본 AWS_* 키.
    LLM(agent·평가·RAGAS)과 임베딩(retriever) 모두 이 함수를 쓴다. region을 주면 그 리전으로 (대체 리전 경로용).
    값은 출력하지 않는다."""
    key, secret = os.getenv("AWS_ACCESS_KEY_ID_NEW"), os.getenv("AWS_SECRET_ACCESS_KEY_NEW")
    if key and secret:
        return {"aws_access_key_id": key, "aws_secret_access_key": secret,
                "region_name": region or os.getenv("AWS_DEFAULT_REGION_NEW") or REGION}
    return {"region_name": region or REGION}


def model_routes() -> list[tuple[str, str]]:
    """Sonnet 4.5 호출 경로를 시도할 순서대로 (사용자 결정 11-19 밤). 모두 같은 Sonnet 4.5 모델이다.
    ① 기본 모델(us. 교차 리전)·기본 리전 → ② 대체 모델(global. 교차 리전)·기본 리전 → ③ 대체 모델·대체 리전(ap-northeast-2).
    'us.' 모델 ID는 미국 밖 리전에서 부를 수 없어(11-19 확인: ValidationException) 대체 리전에서는 빼고 시도한다."""
    base_region = bedrock_credentials()["region_name"]
    routes = []
    for region in (base_region, FALLBACK_REGION):
        for model in (MODEL_ID, FALLBACK_MODEL_ID):
            if not model or not region:
                continue
            if model.startswith("us.") and not region.startswith("us-"):
                continue
            if (model, region) not in routes:
                routes.append((model, region))
    return routes


def data4library_key() -> str:
    """정보나루 MCP 인증키 (X-Auth-Key 헤더용). 없으면 오류."""
    key = os.getenv("DATA4LIBRARY_AUTH_KEY")
    if not key:
        raise RuntimeError(".env 에 DATA4LIBRARY_AUTH_KEY 가 없습니다.")
    return key


def daily_limit() -> int:
    """오늘 정보나루를 부를 수 있는 최대 횟수. .env 값이 없거나 이상하면 기본값, 상한은 30,000."""
    try:
        value = int(os.getenv("DATA4LIBRARY_DAILY_LIMIT", DEFAULT_DAILY_LIMIT))
    except ValueError:
        value = DEFAULT_DAILY_LIMIT
    return max(0, min(value, HARD_CEILING))


def secret_values() -> list[str]:
    """답변에서 지울 비밀값 (8자 이상만, 부를 때마다 환경에서 읽는다). 값 자체는 출력하지 않는다."""
    return [v for k in SECRET_KEYS if (v := os.getenv(k)) and len(v) >= 8]
