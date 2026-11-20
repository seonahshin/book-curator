"""도서 목록 스냅샷 만들기 → data/catalog_snapshot.json

평가를 같은 조건으로 하려고 정보나루에서 한 번 받아 저장한다. 다시 돌려도 캐시로 답하므로 호출이 늘지 않고,
한도가 커지면(서버 IP 등록 확인 후) 예산만 올려 다시 돌리면 키워드를 받은 책이 늘어난다.

순서 (단계마다 카테고리를 번갈아 부른다 = 라운드 로빈):
1. 목록 조회: 9~11세 인기, 40대 인기, 최근 3개월 인기, 큰글씨 인기
2. 독서 기록 책 찾기(search_books) → 읽은 책 키워드 + 좋아한 책 기반 추천 도서
3. 후보 책 키워드: 아이 · 나 · 선물 묶음을 12:10:5 비율로 번갈아

실행: .venv/Scripts/python src/build_snapshot.py [--reserve 49]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from collections import defaultdict
from datetime import date, timedelta
from itertools import cycle
from pathlib import Path
from typing import Any

from mcp_client import ROOT, DailyLimitReached, Data4Library, books_of, daily_limit, remaining_calls, today_usage

SNAPSHOT = ROOT / "data" / "catalog_snapshot.json"
READING_LOG = ROOT / "data" / "reading_log.csv"
NLCY_LIST = ROOT / "data" / "nlcy_recommended.csv"  # 국립어린이청소년도서관 사서추천도서 (웹에서 확인, 출처 포함)
BOOK_TAGS = ROOT / "data" / "book_tags.csv"

# 후보 키워드 예산 비율 (사용자 승인 11-19: 아이 120 / 나 100 / 선물·큰글씨 50)
POOL_WEIGHTS = {"child": 12, "adult": 10, "gift": 5}
MAX_VOLUMES_PER_SERIES = 2  # 같은 시리즈가 후보를 다 차지하지 않게
CHILD_TARGET = 160  # 아이 후보 150권 이상 (11-19 사용자 요청, 여유 10권)


def norm(text: str) -> str:
    """비교용 제목: 괄호 머리말·공백·문장부호를 없앤다."""
    text = re.sub(r"^\([^)]*\)\s*", "", text or "")
    return re.sub(r"[\s\W_]+", "", text).lower()


def keep_norm(text: str) -> str:
    """비교용 제목: 괄호 기호·공백·문장부호만 없애고 괄호 안 글자는 살린다."""
    return re.sub(r"[\s\W_]+", "", text or "").lower()


def search_query(title: str) -> str:
    """검색어: 앞 괄호 머리말과 ' : ' 뒤 부제를 뺀 본제목."""
    title = re.sub(r"^\([^)]*\)\s*", "", title)
    return title.split(" : ")[0].strip()


def pick_match(title: str, author: str, docs: list[dict], reader: str) -> dict | None:
    """검색 결과에서 독서 기록 제목과 가장 맞는 책 한 권을 고른다 (코드 판정)."""
    full, main = norm(title), norm(search_query(title))
    main_kept = keep_norm(title.split(" : ")[0])  # 괄호 안 글자까지 살린 본제목
    scored = []
    for d in docs:
        name = norm(d.get("bookname", ""))
        name_kept = keep_norm(d.get("bookname", "").split(":")[0])
        # 같은 책 판정: 괄호까지 살린 본제목이 같거나 포함, 또는 머리말 뺀 제목이 서로 같을 때만
        # ("(젤 크고 재밌는) 호기심 백과"가 "초등 호기심 백과"로 잘못 맞던 문제 방지)
        if not (main_kept in name_kept or name_kept == main_kept or name.startswith(main) and main):
            continue
        score = 0
        score += 3 if full in name or name == full else 0
        score += 2 if author and author.replace(" ", "") in d.get("authors", "").replace(" ", "") else 0
        score += 1 if reader == "아이" and str(d.get("addition_symbol", "")).startswith("7") else 0
        score += 1 if str(d.get("vol", "")).strip() in ("", "1") else 0
        scored.append((score, int(d.get("loan_count") or 0), d))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def keywords_of(result: Any) -> list[dict]:
    """get_book_keywords 결과에서 [{word, weight}] 목록만 꺼낸다."""
    res = result.get("result", result) if isinstance(result, dict) else {}
    items = (res.get("keywords") or {}).get("items") or res.get("items") or []
    out = []
    for it in items:
        item = it.get("item", it) if isinstance(it, dict) else {}
        if item.get("word"):
            out.append({"word": item["word"], "weight": int(item.get("weight") or 0)})
    return out


def book_record(d: dict) -> dict:
    """스냅샷에 저장할 서지정보 (표지 주소는 이미 mcp_client에서 지워짐)."""
    return {
        "isbn13": d.get("isbn13"),
        "title": d.get("bookname"),
        "authors": d.get("authors"),
        "publisher": d.get("publisher"),
        "publication_year": d.get("publication_year"),
        "addition_symbol": d.get("addition_symbol"),
        "vol": d.get("vol"),
        "class_no": d.get("class_no"),
        "class_nm": d.get("class_nm"),
        "loan_count": d.get("loan_count"),
    }


class Collector:
    """예산 안에서만 부르는 수집기. 예비분에 닿으면 더 부르지 않는다."""

    def __init__(self, lib: Data4Library, reserve: int) -> None:
        self.lib, self.reserve = lib, reserve
        self.stopped = False

    async def call(self, tool: str, **args: Any) -> Any | None:
        """캐시에 있으면 예산과 관계없이 캐시로 답하고, 없으면 예산 안에서만 정보나루에 보낸다."""
        if self._cached(tool, args):
            return await self.lib.call(tool, **args)
        if self.stopped:
            return None
        if remaining_calls() <= self.reserve:
            self.stopped = True
            print(f"  ! 예비분 {self.reserve}건에 닿아 수집을 멈춥니다 (오늘 {today_usage()}건)")
            return None
        try:
            return await self.lib.call(tool, **args)
        except DailyLimitReached as e:
            self.stopped = True
            print("  !", e)
            return None

    @staticmethod
    def _cached(tool: str, args: dict) -> bool:
        from mcp_client import _cache_path  # 캐시에 있으면 예산과 관계없이 쓴다
        return _cache_path(tool, {k: v for k, v in args.items() if v is not None}).exists()


async def main(reserve: int, new_keywords: int) -> None:
    lib = Data4Library()
    col = Collector(lib, reserve)
    start = today_usage()
    print(f"시작: 오늘 호출 {start}건 / 한도 {daily_limit()}건, 예비분 {reserve}건")

    books: dict[str, dict] = {}  # isbn13 → 기록
    pools: dict[str, dict[str, int]] = defaultdict(dict)  # pool → {isbn: 우선 점수}

    def add(d: dict, pool: str, source: str, score: int) -> None:
        isbn = d.get("isbn13")
        if not isbn:
            return
        rec = books.setdefault(isbn, {**book_record(d), "sources": [], "pools": [], "is_big": False})
        if source not in rec["sources"]:
            rec["sources"].append(source)
        if pool not in rec["pools"]:
            rec["pools"].append(pool)
        pools[pool][isbn] = pools[pool].get(isbn, 0) + score

    # 1) 목록 조회 (8건): 카테고리를 번갈아 1쪽씩
    today = date.today()
    recent = {"start_dt": (today - timedelta(days=92)).isoformat(), "end_dt": (today - timedelta(days=1)).isoformat()}
    list_calls = [
        ("child", "popular_9_11", {"from_age": "9", "to_age": "11", "page": 1}),
        ("adult", "popular_40s", {"age": "40", "page": 1}),
        ("gift", "popular_recent", {**recent, "page": 1}),
        ("gift", "big_print", {"book_dvsn": "big", "page": 1}),
        ("child", "popular_9_11", {"from_age": "9", "to_age": "11", "page": 2}),
        ("adult", "popular_40s", {"age": "40", "page": 2}),
        ("child", "popular_9_11", {"from_age": "9", "to_age": "11", "page": 3}),
        ("adult", "popular_40s", {"age": "40", "page": 3}),
        # 아이 후보 늘리기 (11-19 사용자 요청: 아이 후보 150권 이상)
        ("child", "popular_grade3_4", {"age": "a10", "page": 1}),
        ("child", "popular_grade5_6", {"age": "a12", "page": 1}),
        ("child", "popular_9_11", {"from_age": "9", "to_age": "11", "page": 4}),
        ("child", "popular_grade3_4", {"age": "a10", "page": 2}),
        ("child", "popular_9_11", {"from_age": "9", "to_age": "11", "page": 5}),
        ("child", "popular_grade5_6", {"age": "a12", "page": 2}),
        ("child", "popular_9_11", {"from_age": "9", "to_age": "11", "page": 6}),
        ("child", "popular_grade3_4", {"age": "a10", "page": 3}),
        ("child", "popular_grade5_6", {"age": "a12", "page": 3}),
        ("child", "popular_9_11", {"from_age": "9", "to_age": "11", "page": 7}),
        ("child", "popular_grade3_4", {"age": "a10", "page": 4}),
    ]
    print("1) 목록 조회")
    for pool, source, args in list_calls:
        result = await col.call("get_popular_loans", pageSize=100, **args)
        docs = books_of(result)
        for rank, d in enumerate(docs):
            add(d, pool, source, (100 - rank) // 10 + (3 - args["page"]))  # 순위가 높을수록 점수 큼
            if source == "big_print":
                books[d["isbn13"]]["is_big"] = True
        print(f"  - {source} {args.get('page')}쪽: {len(docs)}권")

    # 1-b) 사서추천도서: 웹에서 확인한 목록을 정보나루에서 찾아 아이 후보에 넣고, 태그 파일을 만든다
    tag_rows = []
    if NLCY_LIST.exists():
        with NLCY_LIST.open(encoding="utf-8-sig") as f:
            nlcy = list(csv.DictReader(f))
        print(f"1-b) 사서추천도서 {len(nlcy)}권 찾기")
        for r in nlcy:
            result = await col.call("search_books", title=search_query(r["title"]), pageSize=10)
            match = pick_match(r["title"], r.get("author", ""), books_of(result), "아이")
            if match:
                add(match, "child", "nlcy_librarian", 30)
            tag_rows.append({"isbn13": (match or {}).get("isbn13", ""), "title": r["title"],
                             "found_title": (match or {}).get("bookname", ""),
                             "addition_symbol": (match or {}).get("addition_symbol", ""),
                             "tag": "사서추천", "grade": r["grade"], "month": r["month"], "source": r["source"]})
        print(f"  - 정보나루에서 찾음 {sum(1 for t in tag_rows if t['isbn13'])}권 / {len(tag_rows)}권")

    # 2) 독서 기록: 책 찾기 → 키워드 → 추천 도서 (아이·나 번갈아)
    with READING_LOG.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_reader = defaultdict(list)
    seen = set()
    for r in rows:
        key = (r["reader"], norm(r["title"]))
        if key not in seen:
            seen.add(key)
            by_reader[r["reader"]].append(r)
    disliked_titles = {(r["reader"], norm(r["title"])) for r in rows if (r.get("disliked") or "").upper() == "Y"}
    liked_titles = {(r["reader"], norm(r["title"])) for r in rows if (r.get("disliked") or "").upper() != "Y"}

    order = []
    for pair in zip(by_reader.get("아이", []), by_reader.get("나", [])):
        order.extend(pair)
    longest = max(by_reader.values(), key=len)
    order.extend(longest[len(order) // 2:])

    read_books, unmatched = [], []
    print(f"2) 독서 기록 {len(order)}권 찾기")
    for r in order:
        result = await col.call("search_books", title=search_query(r["title"]), author=r.get("author") or None, pageSize=20)
        match = pick_match(r["title"], r.get("author", ""), books_of(result), r["reader"])
        short = " ".join(search_query(r["title"]).split()[-2:])
        if not match and short != search_query(r["title"]):
            # 도서관은 앞 수식어를 괄호로 묶기도 한다: "(너무 지혜로워서 …) 저학년 탈무드" → 끝 두 단어로 한 번 더
            result = await col.call("search_books", title=short, pageSize=20)
            match = pick_match(r["title"], r.get("author", ""), books_of(result), r["reader"])
        if match:
            read_books.append({**{k: v for k, v in book_record(match).items() if k != "loan_count"},
                               "reader": r["reader"], "log_title": r["title"], "volumes": r.get("volumes", "")})
        else:
            unmatched.append(f"{r['reader']}: {r['title']}")
    print(f"  - 찾음 {len(read_books)}권, 못 찾음 {len(unmatched)}권")

    print("   읽은 책 키워드 + 좋아한 책 추천 도서")
    for rb in read_books:
        rb["keywords"] = keywords_of(await col.call("get_book_keywords", isbn13=rb["isbn13"]))
        key = (rb["reader"], norm(rb["log_title"]))
        if key in liked_titles:  # 싫었던 줄만 있는 책은 추천 근거로 쓰지 않는다
            rec = await col.call("get_book_recommendations", isbn13=rb["isbn13"], type="mania")
            pool = "child" if rb["reader"] == "아이" else "adult"
            for d in books_of(rec):
                add(d, pool, f"rec:{rb['reader']}:{rb['log_title']}", 3)

    # 3) 후보 키워드: 이미 읽은 책·시리즈 같은 권 많이는 빼고, 묶음을 비율대로 번갈아
    read_isbns = {rb["isbn13"] for rb in read_books}
    read_names = {norm(rb["title"]) for rb in read_books} | {norm(r["title"]) for r in rows}
    queues = {}
    for pool, scores in pools.items():
        per_series = defaultdict(int)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        q = []
        for isbn, _ in ranked:
            b = books[isbn]
            name = norm(b["title"])
            if isbn in read_isbns or name in read_names:
                continue
            if pool == "child" and not str(b.get("addition_symbol") or "").startswith("7"):
                continue
            if per_series[name] >= MAX_VOLUMES_PER_SERIES:
                continue
            per_series[name] += 1
            q.append(isbn)
        queues[pool] = q
    print("3) 후보 키워드 (대기 권수):", {p: len(q) for p, q in queues.items()})

    turn = cycle([p for p, w in POOL_WEIGHTS.items() for _ in range(w)])
    got = defaultdict(int)
    done: set[str] = set()
    deferred: dict[str, list[str]] = defaultdict(list)
    new_kw = 0

    def child_first() -> bool:
        """아이 후보가 목표 권수에 닿기 전에는 새 호출을 아이 묶음에 먼저 쓴다 (11-19 사용자 요청)."""
        return got["child"] < CHILD_TARGET and bool(queues.get("child"))

    for _ in range(2):  # 1회차: 아이 우선, 2회차: 미뤄 둔 나·선물 책
        while any(queues.values()):  # 예산에 닿은 뒤에도 캐시에 있는 책은 끝까지 읽는다
            pool = next(turn)
            while queues.get(pool) and queues[pool][0] in done:
                queues[pool].pop(0)
            if not queues.get(pool):
                continue
            isbn = queues[pool].pop(0)
            cached = col._cached("get_book_keywords", {"isbn13": isbn})
            must = "nlcy_librarian" in books[isbn]["sources"]  # 사서추천도서는 키워드 예산과 관계없이 받는다
            if not cached and not must and new_kw >= new_keywords:
                continue  # 이번 실행에서 새로 받을 키워드 권수를 넘음
            if pool != "child" and child_first() and not col._cached("get_book_keywords", {"isbn13": isbn}):
                deferred[pool].append(isbn)
                continue
            kws = await col.call("get_book_keywords", isbn13=isbn)
            if kws is None:
                continue
            books[isbn]["keywords"] = keywords_of(kws)
            done.add(isbn)
            got[pool] += 1
            new_kw += 0 if cached else 1
        queues, deferred = dict(deferred), defaultdict(list)
    print("  - 키워드 받은 책:", dict(got))

    with_kw = [b for b in books.values() if b.get("keywords")]
    without_kw = [b for b in books.values() if not b.get("keywords") and b["isbn13"] not in read_isbns]
    # 4) 책소개: 독서 기록 책 먼저, 그다음 스냅샷 책 (get_book_analysis, 1권 1건)
    print(f"4) 책소개 받기: 읽은 책 {len(read_books)}권 + 스냅샷 {len(with_kw)}권")
    desc_ok = 0
    for b in read_books + with_kw:
        r = await col.call("get_book_analysis", isbn13=b["isbn13"], type="loan", period=1)
        book = (((r or {}).get("result") or {}).get("analysis") or {}).get("book") or {}
        b["description"] = (book.get("description") or "").strip()
        desc_ok += bool(b["description"])
    print(f"  - 책소개 있음 {desc_ok}권 / {len(read_books) + len(with_kw)}권")

    snapshot = {
        "created": today.isoformat(),
        "note": "정보나루 MCP 수집. 책소개는 get_book_analysis(type=loan) 응답의 description.",
        "books": sorted(with_kw, key=lambda b: (b["pools"][0], b["title"] or "")),
        "candidates_without_keywords": [{k: b[k] for k in ("isbn13", "title", "addition_symbol", "pools")} for b in without_kw],
        "read_books": read_books,
        "unmatched_reading_log": unmatched,
    }
    if tag_rows:
        with BOOK_TAGS.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tag_rows[0].keys()))
            w.writeheader()
            w.writerows(tag_rows)
        print(f"태그 파일: {BOOK_TAGS.relative_to(ROOT)} ({len(tag_rows)}줄)")
    SNAPSHOT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n== 결과 ==")
    print(f"스냅샷 책(키워드 있음): {len(with_kw)}권 → {SNAPSHOT.relative_to(ROOT)}")
    for pool in POOL_WEIGHTS:
        print(f"  - {pool}: {sum(1 for b in with_kw if pool in b['pools'])}권")
    print(f"  - 큰글씨: {sum(1 for b in with_kw if b['is_big'])}권, 아동(부가기호 7): "
          f"{sum(1 for b in with_kw if str(b.get('addition_symbol') or '').startswith('7'))}권")
    missing = [b["isbn13"] for b in with_kw if not (b.get("isbn13") and b.get("class_no") is not None)]
    print(f"  - isbn13·class_no 빠진 책: {len(missing)}권, 부가기호 빈 책: "
          f"{sum(1 for b in with_kw if not b.get('addition_symbol'))}권")
    print(f"키워드 못 받은 후보: {len(without_kw)}권 (한도가 커지면 다시 돌려 채움)")
    print(f"독서 기록: 찾음 {len(read_books)}권, 못 찾음 {len(unmatched)}권")
    print(f"이번 실행 호출: {today_usage() - start}건, 오늘 합계 {today_usage()}건 / 한도 {daily_limit()}건")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserve", type=int, default=49, help="남겨 둘 예비 호출 수")
    parser.add_argument("--new-keywords", type=int, default=0,
                        help="이번 실행에서 새로 키워드를 받을 최대 권수 (기본 0 = 캐시에 있는 책만)")
    a = parser.parse_args()
    asyncio.run(main(a.reserve, a.new_keywords))
