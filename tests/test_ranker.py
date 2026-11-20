"""목적별 정렬 테스트 (T5-1, 구현 전에 먼저 작성).

정렬은 "무엇을 먼저 보여줄지"다 (가드레일과 다름). 점수 = 취향 점수 + 목적 가산점 + 프로필 가산점 − 싫었던 책 감점.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ranker import Purpose, rank  # noqa: E402

CHILD = {"reader": "아이", "age_group": "어린이", "from_age": 9, "to_age": 11}
ADULT = {"reader": "나", "age_group": "성인", "age": 40}
SENIOR = {"reader": "할머니", "age_group": "60대 이상", "age": 60}


def kw(*words):
    return [{"word": w, "weight": 10} for w in words]


def cand(isbn, score, **book):
    return {"isbn13": isbn, "score": score, "book": {"isbn13": isbn, "title": isbn, "sources": [], **book}}


# A: 취향 점수 최고 / B: 사서추천 태그 / C: 최근 인기 + 최신 출간 / D: 큰글씨도서
CANDS = [
    cand("A", 0.020, publication_year="2015", keywords=kw("모험", "용")),
    cand("B", 0.017, publication_year="2016", keywords=kw("우정")),
    cand("C", 0.016, publication_year="2026", sources=["popular_recent"], keywords=kw("요리")),
    cand("D", 0.015, publication_year="2017", is_big=True, keywords=kw("여행")),
]
TAGS = {"B": ["사서추천"]}


def order(result):
    return [r["isbn13"] for r in result]


def test_self_is_taste_order():
    assert order(rank(CANDS, ADULT, Purpose.self, tags=TAGS))[0] == "A"


def test_child_puts_tagged_book_first():
    assert order(rank(CANDS, CHILD, Purpose.child, tags=TAGS))[0] == "B"


def test_gift_puts_recent_popular_first():
    assert order(rank(CANDS, ADULT, Purpose.gift, tags=TAGS))[0] == "C"


def test_purposes_give_different_orders():
    orders = {p: tuple(order(rank(CANDS, ADULT if p != Purpose.child else CHILD, p, tags=TAGS))) for p in Purpose}
    assert len(set(orders.values())) == 3


def test_senior_profile_puts_big_print_first():
    assert order(rank(CANDS, SENIOR, Purpose.self, tags=TAGS))[0] == "D"
    assert order(rank(CANDS, ADULT, Purpose.self, tags=TAGS))[0] != "D"


def test_similar_to_disliked_book_goes_down():
    disliked = [{"title": "무서운 모험", "keywords": kw("모험", "용")}]
    result = order(rank(CANDS, ADULT, Purpose.self, tags=TAGS, disliked=disliked))
    assert result.index("A") > 0


def test_no_penalty_when_disliked_series_has_liked_volume():
    disliked = [{"title": "무서운 모험", "keywords": kw("모험", "용")}]
    result = order(rank(CANDS, ADULT, Purpose.self, tags=TAGS, disliked=disliked, liked_series={"무서운모험"}))
    assert result[0] == "A"


def test_bonus_reasons_are_recorded():
    top = rank(CANDS, CHILD, Purpose.child, tags=TAGS)[0]
    assert any("사서추천" in b for b in top["bonuses"])
