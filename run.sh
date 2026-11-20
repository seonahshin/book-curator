#!/usr/bin/env bash
# 실행 스크립트 · 읽고 싶은 책을 찾고 있나요?
#
# 사용법 (macOS/Linux 터미널, Windows는 Git Bash)
#   ./run.sh          가상환경·라이브러리 준비 후 서버 실행 → http://localhost:8000
#   ./run.sh test     코드 규칙 pytest
#   ./run.sh eval     20문항 채점 (Bedrock·정보나루 호출, 몇 분 걸림)
#   PORT=8001 ./run.sh   다른 포트로 실행
#
# 처음 한 번: .env.example 을 .env 로 복사하고 키 값을 채운다 (.env 는 저장소에 올리지 않는다)
set -euo pipefail

cd "$(dirname "$0")"
PORT="${PORT:-8000}"

# 1) 파이썬 찾기 (3.11 이상)
PY_SYS="$(command -v python3 || command -v python || true)"
if [ -z "$PY_SYS" ]; then
  echo "⛔ Python 3.11 이상을 먼저 설치해 주세요." >&2
  exit 1
fi

# 2) 가상환경 만들기 (없을 때만)
if [ ! -d .venv ]; then
  echo "가상환경(.venv)을 만듭니다..."
  "$PY_SYS" -m venv .venv
fi
if [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe   # Windows (Git Bash)
else
  PY=.venv/bin/python           # macOS/Linux
fi

# 3) 라이브러리 설치 (requirements.txt 가 바뀌었을 때만)
REQ_HASH="$(cksum requirements.txt | cut -d' ' -f1)"
if [ "$(cat .venv/.requirements_hash 2>/dev/null || true)" != "$REQ_HASH" ]; then
  echo "라이브러리를 설치합니다 (requirements.txt)..."
  "$PY" -m pip install -q -r requirements.txt
  echo "$REQ_HASH" > .venv/.requirements_hash
fi

# 4) .env 확인 (값은 출력하지 않는다)
if [ ! -f .env ]; then
  echo "⛔ .env 가 없습니다. .env.example 을 .env 로 복사하고 키 값을 채운 뒤 다시 실행해 주세요." >&2
  echo "   cp .env.example .env" >&2
  exit 1
fi

export PYTHONIOENCODING=utf-8

# 5) 실행
case "${1:-server}" in
  server)
    echo "서버를 켭니다 → http://localhost:${PORT}  (API 문서: http://localhost:${PORT}/docs, 끄기: Ctrl+C)"
    exec "$PY" -m uvicorn api:app --app-dir src --port "$PORT"
    ;;
  test)
    exec "$PY" -m pytest -q
    ;;
  eval)
    exec "$PY" evaluation/run_eval.py --round "${2:-local}"
    ;;
  *)
    echo "사용법: ./run.sh [server|test|eval [회차이름]]" >&2
    exit 1
    ;;
esac
