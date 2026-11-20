"""코드 규칙 테스트 (T5-0, 구현 전에 먼저 작성).

SERVICE.md 4절 가드레일과 CLAUDE.md 도메인 규칙 중 Python 코드가 판정하는 것만 검사한다.
LLM이 들어간 지표(인-아웃 통과율, faithfulness)는 여기 넣지 않고 evaluation/에서 잰다.

실제 시연 데이터(data/profiles.json, data/reading_log.csv, 스냅샷의 독서 기록 매칭)를 쓰고,
후보 책은 테스트 안에서 만든다. catalog 인자로 "도서 목록에 있는 책"을 테스트용으로 지정한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools import filter_candidates, get_reading_log_data, parse_volumes  # noqa: E402


def book(isbn, title, symbol="74810", vol=""):
    return {"isbn13": isbn, "title": title, "addition_symbol": symbol, "vol": vol}


def run(reader, cands, **kw):
    catalog = {c["isbn13"] for c in cands}
    return filter_candidates(cands, reader, catalog=catalog, **kw)


def kept_isbns(result):
    return [b["isbn13"] for b in result["kept"]]


def reason_of(result, isbn):
    return next(r["reason"] for r in result["removed"] if r["isbn13"] == isbn)


# ---- 읽은 권 범위 해석 ----

def test_parse_volumes():
    assert parse_volumes("1~15")["volumes"] == list(range(1, 16))
    assert parse_volumes("3")["volumes"] == [3]
    assert parse_volumes("전권")["all"] is True
    assert parse_volumes("") == {"all": False, "volumes": []}


# ---- 연령 판정 (SERVICE.md 3-4) ----

def test_child_gets_only_children_books():
    r = run("아이", [
        book("9000000000001", "아동 소설", "74810"),
        book("9000000000002", "초등 학습참고서", "67400"),
        book("9000000000003", "성인 교양서", "03810"),
        book("9000000000004", "부가기호 없는 책", ""),
    ])
    assert kept_isbns(r) == ["9000000000001"]
    assert "연령" in reason_of(r, "9000000000002")
    assert "부가기호" in reason_of(r, "9000000000004")


def test_adult_has_no_age_limit():
    r = run("나", [book("9000000000003", "성인 교양서", "03810"), book("9000000000004", "부가기호 없는 책", "")])
    assert kept_isbns(r) == ["9000000000003", "9000000000004"]


# ---- 이미 읽은 책 제외 (가드레일 3) ----

def test_read_book_same_isbn_excluded():
    # 드래곤 마스터 1권 (독서 기록 매칭 ISBN)
    r = run("아이", [book("9791130696812", "드래곤 마스터", "74840", "1")])
    assert kept_isbns(r) == []
    assert "읽은" in reason_of(r, "9791130696812")


def test_read_single_book_other_edition_excluded_by_title():
    r = run("아이", [book("9000000000010", "할머니의 여름휴가", "77810")])
    assert kept_isbns(r) == []


def test_series_read_volumes_excluded_next_volume_kept():
    r = run("아이", [
        book("9000000000015", "드래곤 마스터", "74840", "15"),
        book("9000000000016", "드래곤 마스터", "74840", "16"),
    ])
    assert kept_isbns(r) == ["9000000000016"]


def test_series_without_volume_number_is_excluded():
    # 권 번호를 모르면 읽은 권인지 알 수 없으므로 안전하게 뺀다
    r = run("아이", [book("9000000000020", "드래곤 마스터", "74840", "")])
    assert kept_isbns(r) == []


def test_reread_request_keeps_read_book():
    r = run("아이", [book("9791130696812", "드래곤 마스터", "74840", "1")], allow_reread=True)
    assert kept_isbns(r) == ["9791130696812"]


# ---- 싫었던 책 ----

def test_disliked_book_excluded_even_when_reread_allowed():
    r = run("아이", [book("9000000000030", "위기 탈출 도감", "74400", "3")], allow_reread=True)
    assert kept_isbns(r) == []
    assert "싫었던" in reason_of(r, "9000000000030")


# ---- 같은 시리즈는 1권만 ----

def test_one_book_per_series_prefers_lowest_volume():
    r = run("나", [
        book("9000000000041", "(제로니모의) 환상모험", "74880", "3"),
        book("9000000000040", "(제로니모의) 환상모험", "74880", "1"),
        book("9000000000050", "다른 책", "03810"),
    ])
    assert kept_isbns(r) == ["9000000000040", "9000000000050"]


# ---- 지어내지 않기 (가드레일 1) ----

def test_book_not_in_catalog_removed():
    cands = [book("9000000000060", "목록에 있는 책", "03810"), book("9000000000061", "지어낸 책", "03810")]
    r = filter_candidates(cands, "나", catalog={"9000000000060"})
    assert kept_isbns(r) == ["9000000000060"]
    assert "목록" in reason_of(r, "9000000000061")


# ---- 다른 이용자 프로필 (가드레일 4) ----

def test_other_owner_profile_is_not_loaded():
    assert get_reading_log_data("아이", owner="other")["found"] is False


def test_filter_for_other_owner_returns_nothing():
    r = run("아이", [book("9000000000001", "아동 소설", "74810")], owner="other")
    assert r["kept"] == [] and "error" in r


# ---- 부가기호 풀이 (T12 1회차: LLM이 74840을 "청소년용"으로 잘못 풀이함 → 코드가 풀어 준다) ----

def test_addition_symbol_meaning_child_literature():
    from tools import addition_symbol_meaning

    m = addition_symbol_meaning("74840")
    assert "아동" in m and "문학" in m and "청소년" not in m


def test_addition_symbol_meaning_adult_and_empty():
    from tools import addition_symbol_meaning

    assert "교양" in addition_symbol_meaning("03810")
    assert addition_symbol_meaning("") == "부가기호 없음"


# ---- 기록 조회 질문 (T12 2회차: "아이가 좋아하는 요리책" → LLM이 기록에 없는 요리책을 지어내 추천함) ----

def test_log_topic_absent_returns_empty():
    from tools import log_books_about

    assert log_books_about("아이", "요리") == []


def test_log_topic_present_returns_logged_books_only():
    from tools import log_books_about

    titles = [b["log_title"] for b in log_books_about("아이", "과학")]
    assert titles and any("과학" in t for t in titles)


# ---- 요청 성격 뽑기 (T12-2 4회: 5번 "웃기면서 모험"에서 추천 이유가 '웃김'을 설명하지 않음 → 코드가 뽑아 넘긴다) ----

def test_requested_traits_from_question():
    from tools import requested_traits

    assert requested_traits("아이한테 웃기면서 모험 이야기 나오는 책 추천해줘") == ["웃김", "모험"]
    assert requested_traits("아이한테 오싹하고 무서운 이야기 책 3권") == ["오싹함"]


def test_no_requested_traits_for_plain_request():
    from tools import requested_traits

    assert requested_traits("아이가 다음에 읽을 책 5권 추천해줘") == []
