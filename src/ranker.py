"""추천 목적별 정렬 (리랭킹). 모두 Python 코드로 계산한다.

가드레일(걸러내기)과 다르다: 걸러내기는 "절대 안 되는 것", 정렬은 "무엇을 먼저 보여줄지"다. 걸러내기 다음에 실행한다.

점수 = 취향 점수(검색 점수를 0~1로) + 목적 가산점 + 프로필 가산점 − 싫었던 책과 비슷한 정도의 감점
- 나에게(self): 가산점 없음, 취향 순
- 자녀에게(child): 사서추천 태그(data/book_tags.csv) +0.3, 또래 인기(9~11세·3~6학년 인기 대출) +0.1
- 선물용(gift): 최근 3개월 인기 대출 +0.3, 최근 2년 안 출간 +0.15
- 60대 이상 프로필: 큰글씨도서 +0.4
- 싫었던 책: 키워드가 많이 겹치면(자카드 0.3 이상) −0.3. 단, 싫었던 책의 시리즈에 좋았던 권이 있으면 감점하지 않는다
"""

from __future__ import annotations

import csv
import re
from datetime import date
from enum import Enum
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK_TAGS = ROOT / "data" / "book_tags.csv"


class Purpose(str, Enum):
    """추천 목적 (웹 화면 드롭다운 값)."""

    self = "self"    # 나에게
    child = "child"  # 자녀에게
    gift = "gift"    # 선물용


BONUS = {
    "librarian": 0.3,
    "peer_popular": 0.1,
    "recent_popular": 0.3,
    "new_release": 0.15,
    "big_print": 0.4,
}
DISLIKE_PENALTY = 0.3
DISLIKE_JACCARD = 0.3
PEER_SOURCES = {"popular_9_11", "popular_grade3_4", "popular_grade5_6"}


def _norm(text: str) -> str:
    text = re.sub(r"^\([^)]*\)\s*", "", text or "")
    text = re.split(r"\s*[:=]\s*|\s+-\s+", text)[0]
    return re.sub(r"[\s\W_]+", "", text).lower()


@lru_cache(maxsize=1)
def load_tags() -> dict[str, list[str]]:
    """book_tags.csv → {isbn13: [태그]}. ISBN을 찾지 못한 줄은 쓰지 않는다."""
    if not BOOK_TAGS.exists():
        return {}
    tags: dict[str, list[str]] = {}
    with BOOK_TAGS.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("isbn13"):
                tags.setdefault(r["isbn13"], []).append(r["tag"])
    return tags


def is_senior(profile: dict) -> bool:
    """60대 이상 프로필인지 (연령대 이름 또는 나이)."""
    try:
        age = int(profile.get("age") or profile.get("from_age") or 0)
    except (TypeError, ValueError):
        age = 0
    return age >= 60 or "60" in str(profile.get("age_group", ""))


def is_big_print(book: dict) -> bool:
    return bool(book.get("is_big")) or bool(re.search(r"큰\s*글[자씨]", book.get("title") or ""))


def _keyword_set(book: dict, n: int = 20) -> set[str]:
    return {k["word"] for k in (book.get("keywords") or [])[:n]}


def dislike_similarity(book: dict, disliked: list[dict]) -> tuple[float, str]:
    """싫었던 책들과 키워드가 겹치는 정도(자카드)의 최댓값과 그 책 제목."""
    mine = _keyword_set(book)
    best, title = 0.0, ""
    for d in disliked:
        theirs = _keyword_set(d)
        if mine and theirs:
            j = len(mine & theirs) / len(mine | theirs)
            if j > best:
                best, title = j, d.get("title") or d.get("log_title") or ""
    return best, title


def rank(candidates: list[dict], profile: dict, purpose: Purpose | str, tags: dict | None = None,
         disliked: list[dict] | None = None, liked_series: set[str] | None = None,
         bonus_scale: float = 1.0) -> list[dict]:
    """후보를 목적별 점수로 정렬한다.

    candidates: [{"isbn13", "score"(검색 점수), "book"}] — 걸러내기를 통과한 것
    disliked: 싫었던 책 목록(키워드 포함), liked_series: 좋았던 권이 있는 시리즈 이름(정규화)
    bonus_scale: 가산점 배율. 질문에 원하는 내용이 있으면 0.3으로 줄여, 관련도가 비슷한 책 사이에서만 순서를 바꾼다
    반환: 후보마다 final_score, taste, bonuses(가산·감점 설명)를 더해 점수 높은 순
    """
    purpose = Purpose(purpose)
    tags = load_tags() if tags is None else tags
    liked_series = liked_series or set()
    penalty_sources = [d for d in (disliked or []) if _norm(d.get("title") or d.get("log_title") or "") not in liked_series]
    top_score = max((c.get("score") or 0 for c in candidates), default=0) or 1
    this_year = date.today().year

    ranked = []
    for c in candidates:
        b = c["book"]
        taste = (c.get("score") or 0) / top_score
        bonuses, total = [], taste
        sources = set(b.get("sources") or [])
        if purpose == Purpose.child:
            if "사서추천" in tags.get(c["isbn13"], []):
                total += BONUS["librarian"] * bonus_scale
                bonuses.append(f"사서추천 +{BONUS['librarian'] * bonus_scale:.2f}")
            if sources & PEER_SOURCES:
                total += BONUS["peer_popular"] * bonus_scale
                bonuses.append(f"또래 인기 +{BONUS['peer_popular'] * bonus_scale:.2f}")
        elif purpose == Purpose.gift:
            if "popular_recent" in sources:
                total += BONUS["recent_popular"] * bonus_scale
                bonuses.append(f"최근 인기 +{BONUS['recent_popular'] * bonus_scale:.2f}")
            year = re.search(r"\d{4}", str(b.get("publication_year") or ""))
            if year and int(year.group()) >= this_year - 1:
                total += BONUS["new_release"] * bonus_scale
                bonuses.append(f"최신 출간 +{BONUS['new_release'] * bonus_scale:.2f}")
        if is_senior(profile) and is_big_print(b):
            total += BONUS["big_print"] * bonus_scale
            bonuses.append(f"큰글씨도서 +{BONUS['big_print'] * bonus_scale:.2f}")
        sim, d_title = dislike_similarity(b, penalty_sources)
        if sim >= DISLIKE_JACCARD:
            total -= DISLIKE_PENALTY
            bonuses.append(f"싫었던 책 '{d_title}'과 비슷함 -{DISLIKE_PENALTY}")
        ranked.append({**c, "taste": round(taste, 3), "final_score": round(total, 3), "bonuses": bonuses})
    return sorted(ranked, key=lambda r: -r["final_score"])
