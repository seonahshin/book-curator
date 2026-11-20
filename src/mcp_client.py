"""정보나루 MCP 연결.

정보나루 원격 MCP 서버(16개 도구) 중 Agent에는 4개만 노출하고 (수집 전용으로 책소개용 1개 더),
모든 호출에 호출량 안전장치(하루 한도 차단)와 응답 캐시를 씌운다.

- 호출 기록: logs/api_usage.jsonl (한 줄 = 실제로 정보나루에 보낸 호출 1건)
- 응답 캐시: data/cache/<도구>/<요청 해시>.json (같은 요청은 다시 보내지 않는다)
- 하루 한도: .env 의 DATA4LIBRARY_DAILY_LIMIT (기본 450, 서버 IP 등록 확인 전 단계).
  어떤 값을 넣어도 30,000(정보나루 하루 한도)을 넘지 않는다. 한도·인증키는 config.py에서 읽는다.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

from config import daily_limit, data4library_key  # noqa: F401  (daily_limit은 build_snapshot이 여기서 가져다 씀)

ROOT = Path(__file__).resolve().parents[1]

MCP_URL = "https://www.data4library.kr/mcp"
AGENT_TOOLS = ("search_books", "get_popular_loans", "get_book_recommendations", "get_book_keywords")
# get_book_analysis: 책소개(description)를 1회 1건으로 받는 도구. 스냅샷 수집에만 쓰고 Agent에는 주지 않는다 (11-19)
USED_TOOLS = AGENT_TOOLS + ("get_book_analysis",)

USAGE_LOG = ROOT / "logs" / "api_usage.jsonl"
CACHE_DIR = ROOT / "data" / "cache"


class DailyLimitReached(RuntimeError):
    """하루 호출 한도에 닿아 정보나루 호출을 멈췄을 때 나는 오류."""


def today_usage(date: Optional[str] = None) -> int:
    """오늘(또는 지정한 날짜) 정보나루에 실제로 보낸 호출 수를 센다. 캐시로 답한 것은 세지 않는다."""
    date = date or datetime.now().strftime("%Y-%m-%d")
    if not USAGE_LOG.exists():
        return 0
    with USAGE_LOG.open(encoding="utf-8") as f:
        return sum(1 for line in f if f'"date": "{date}"' in line)


def remaining_calls() -> int:
    """오늘 남은 호출 수."""
    return max(0, daily_limit() - today_usage())


def _cache_path(tool: str, args: dict) -> Path:
    key = json.dumps(args, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / tool / f"{digest}.json"


def _record_usage(tool: str, args: dict, ok: bool) -> None:
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    row = {"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"), "tool": tool, "args": args, "ok": ok}
    with USAGE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse(raw: Any) -> Any:
    """MCP 도구 결과(문자열 또는 내용 블록 목록)에서 JSON을 꺼낸다. JSON이 아니면 글자 그대로 돌려준다."""
    if isinstance(raw, tuple):  # (content, artifact) 형태
        raw = raw[0]
    if isinstance(raw, list):
        raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _strip_cover(data: Any) -> Any:
    """책표지 주소(bookImageURL)를 지운다. 정보나루 이용 조건상 저장·AI 입력 금지."""
    if isinstance(data, dict):
        return {k: _strip_cover(v) for k, v in data.items() if k != "bookImageURL"}
    if isinstance(data, list):
        return [_strip_cover(v) for v in data]
    return data


def _looks_like_error(result: Any) -> bool:
    if isinstance(result, dict):
        return bool(result.get("error") or result.get("errMsg") or result.get("isError"))
    if isinstance(result, str):
        return "error" in result.lower()[:200]
    return False


class Data4Library:
    """정보나루 MCP 4개 도구를 캐시·호출량 안전장치와 함께 부르는 클라이언트."""

    def __init__(self) -> None:
        key = data4library_key()  # 인증키는 config 한곳에서 (.env)
        self._client = MultiServerMCPClient(
            {"biglib": {"url": MCP_URL, "transport": "streamable_http", "headers": {"X-Auth-Key": key}}}
        )
        self._tools: dict[str, Any] = {}

    async def _load_tools(self) -> dict[str, Any]:
        if not self._tools:
            tools = await self._client.get_tools()
            self._tools = {t.name: t for t in tools if t.name in USED_TOOLS}
        return self._tools

    async def tool_names(self) -> list[str]:
        """Agent에 노출하는 도구 이름 목록 (정보나루 16개 중 4개)."""
        return sorted(n for n in (await self._load_tools()).keys() if n in AGENT_TOOLS)

    async def call(self, tool: str, **args: Any) -> Any:
        """도구 하나를 부른다. 캐시에 있으면 캐시로 답하고, 없으면 한도를 확인한 뒤 정보나루에 보낸다."""
        if tool not in USED_TOOLS:
            raise ValueError(f"쓰지 않기로 한 도구입니다: {tool}")
        args = {k: v for k, v in args.items() if v is not None}
        path = _cache_path(tool, args)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        if today_usage() >= daily_limit():
            raise DailyLimitReached(
                f"오늘 정보나루 호출 {today_usage()}건으로 한도 {daily_limit()}건에 닿았습니다. 캐시·스냅샷만 씁니다."
            )

        tools = await self._load_tools()
        try:
            result = _strip_cover(_parse(await tools[tool].ainvoke(args)))
        except Exception:
            _record_usage(tool, args, ok=False)
            raise
        ok = not _looks_like_error(result)
        _record_usage(tool, args, ok=ok)
        if ok:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result


# ---- Agent에 줄 LangChain 도구 4개 (auth_key 같은 내부 인자는 숨긴다) ----

class SearchBooksArgs(BaseModel):
    title: Optional[str] = Field(None, description="책 제목")
    author: Optional[str] = Field(None, description="저자")
    keyword: Optional[str] = Field(None, description="키워드")
    isbn13: Optional[str] = Field(None, description="ISBN 13자리 숫자")
    pageSize: int = Field(10, ge=1, le=50, description="받을 권수 (최대 50)")


class PopularLoansArgs(BaseModel):
    age: Optional[str] = Field(None, description="연령 코드: 8 초등, a10 3·4학년, 20·30·40·50, 60 60세 이상")
    from_age: Optional[str] = Field(None, description="나이 범위 시작 (age와 함께 쓰지 않음)")
    to_age: Optional[str] = Field(None, description="나이 범위 끝")
    kdc: Optional[str] = Field(None, description="주제 분류 0~9 (8 문학 등)")
    book_dvsn: Optional[str] = Field(None, description="big = 큰글씨도서")
    pageSize: int = Field(20, ge=1, le=50, description="받을 권수 (최대 50)")


class IsbnArgs(BaseModel):
    isbn13: str = Field(..., description="ISBN 13자리 숫자")


def make_tools(lib: Data4Library) -> list[StructuredTool]:
    """Agent가 쓸 정보나루 도구 4개를 만든다."""

    async def search_books(**kw: Any) -> Any:
        """도서관 책 검색. 쓸 때: 이용자가 말한 책 제목이 실제로 있는지 확인하거나 ISBN을 찾을 때.
        쓰지 않을 때: 취향에 맞는 책을 고를 때 (그때는 search_catalog)."""
        return await lib.call("search_books", **kw)

    async def get_popular_loans(**kw: Any) -> Any:
        """연령·분류별 인기 대출 순위. 쓸 때: 선물용 '최근 인기' 책이나 독서 기록이 없는 사람의 후보가 필요할 때.
        쓰지 않을 때: 독서 기록이 있어 취향 검색이 가능할 때."""
        return await lib.call("get_popular_loans", **kw)

    async def get_book_recommendations(isbn13: str) -> Any:
        """이 책을 빌린 사람들이 함께 본 책(마니아 추천). 쓸 때: 좋아한 책 한 권과 비슷한 책 후보를 늘릴 때.
        쓰지 않을 때: ISBN을 모를 때 (먼저 search_books)."""
        return await lib.call("get_book_recommendations", isbn13=isbn13, type="mania")

    async def get_book_keywords(isbn13: str) -> Any:
        """책 한 권의 주요 키워드 상위 50개. 쓸 때: 읽은 책의 취향 키워드나 추천 이유를 설명할 재료가 필요할 때.
        쓰지 않을 때: 책 목록을 찾을 때."""
        return await lib.call("get_book_keywords", isbn13=isbn13)

    return [
        StructuredTool.from_function(coroutine=search_books, name="search_books", args_schema=SearchBooksArgs,
                                     description=search_books.__doc__),
        StructuredTool.from_function(coroutine=get_popular_loans, name="get_popular_loans", args_schema=PopularLoansArgs,
                                     description=get_popular_loans.__doc__),
        StructuredTool.from_function(coroutine=get_book_recommendations, name="get_book_recommendations",
                                     args_schema=IsbnArgs, description=get_book_recommendations.__doc__),
        StructuredTool.from_function(coroutine=get_book_keywords, name="get_book_keywords", args_schema=IsbnArgs,
                                     description=get_book_keywords.__doc__),
    ]


async def _demo() -> None:
    """T1 확인: 도구 이름 4개와 '드래곤 마스터' 검색 결과(ISBN·부가기호·권 번호)를 출력한다."""
    lib = Data4Library()
    print("노출 도구:", await lib.tool_names())
    print(f"오늘 호출 {today_usage()}건 / 한도 {daily_limit()}건")
    result = await lib.call("search_books", title="드래곤 마스터", pageSize=10)
    for doc in books_of(result):
        print(f"- {doc.get('bookname', '')[:30]} | isbn13={doc.get('isbn13')} | 부가기호={doc.get('addition_symbol')} | 권={doc.get('vol')}")
    print(f"오늘 호출 {today_usage()}건 / 한도 {daily_limit()}건")


def books_of(result: Any) -> list[dict]:
    """검색·순위·추천 결과에서 책 목록만 꺼낸다 (응답 모양이 조금씩 달라도 동작하게)."""
    if isinstance(result, list):
        return [d.get("doc", d.get("book", d)) for d in result if isinstance(d, dict)]
    if isinstance(result, dict):
        for key in ("result", "docs", "books", "items", "list"):
            if key in result:
                return books_of(result[key])
    return []


if __name__ == "__main__":
    asyncio.run(_demo())
