"""도메인 도구: get_reading_log (T3), search_catalog (T4), filter_candidates (T5).

판정(누구의 기록인지, 읽은 권 범위, 싫었던 책)은 모두 Python 코드로 한다. LLM에게 맡기지 않는다.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "data" / "profiles.json"
READING_LOG = ROOT / "data" / "reading_log.csv"
SNAPSHOT = ROOT / "data" / "catalog_snapshot.json"
NEW_MATCHES = ROOT / "data" / "reading_log_matches.json"  # 웹에서 새로 입력한 책의 매칭 결과 (T7)

DEMO_OWNER = "demo"  # 이번 제출은 로그인 없이 owner=demo 고정 (SERVICE.md 4절)

# 취향을 설명하지 못하는 흔한 키워드 (정보나루 키워드에 자주 섞여 나옴)
GENERIC_WORDS = {
    "독자", "베스트셀러", "시리즈", "아마존", "이야기", "책", "작가", "출간", "작품", "세계", "사람", "생각",
    "어린이", "아이", "아이들", "친구", "그림", "글", "권", "편", "판", "출판사", "국내", "해외", "전권",
}


def norm_title(text: str) -> str:
    """비교용 제목: 앞 괄호 머리말·공백·문장부호를 없앤다."""
    text = re.sub(r"^\([^)]*\)\s*", "", text or "")
    return re.sub(r"[\s\W_]+", "", text).lower()


def parse_volumes(text: str) -> dict:
    """읽은 권 범위를 해석한다. '1~15' → {1..15}, '3' → {3}, '전권' → 전부, 빈칸 → 단권 책.

    반환: {"all": bool, "volumes": [정수 목록]}
    """
    text = (text or "").strip()
    if not text:
        return {"all": False, "volumes": []}
    if text in ("전권", "전체", "all"):
        return {"all": True, "volumes": []}
    vols: set[int] = set()
    for part in re.split(r"[,\s]+", text):
        m = re.fullmatch(r"(\d+)\s*[~\-]\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            vols.update(range(min(a, b), max(a, b) + 1))
        elif part.isdigit():
            vols.add(int(part))
    return {"all": False, "volumes": sorted(vols)}


def load_profiles() -> list[dict]:
    """프로필 목록 (별칭·연령대·owner만 저장)."""
    return json.loads(PROFILES.read_text(encoding="utf-8"))


def find_profile(reader: str, owner: str = DEMO_OWNER) -> dict | None:
    """같은 owner가 등록한 프로필만 찾는다. 다른 이용자의 프로필은 이름이 같아도 돌려주지 않는다."""
    for p in load_profiles():
        if p.get("reader") == reader and p.get("owner") == owner:
            return p
    return None


@lru_cache(maxsize=1)
def _snapshot_read_books() -> dict[tuple[str, str], dict]:
    """스냅샷에 저장된 독서 기록 매칭 결과: (reader, 정규화 제목) → 도서관 책 정보."""
    if not SNAPSHOT.exists():
        return {}
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return {(b["reader"], norm_title(b["log_title"])): b for b in data.get("read_books", [])}


@lru_cache(maxsize=1)
def _publishers() -> frozenset[str]:
    """스냅샷에 나온 출판사·저자 이름 (키워드에 섞여 나오는 이름을 거르기 위해)."""
    if not SNAPSHOT.exists():
        return frozenset()
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    names = set()
    for b in data.get("books", []) + data.get("read_books", []):
        names.add(norm_title(b.get("publisher") or ""))
        # 저자 이름도 취향이 아니다: "글: 하라 유타카 ;옮김: 오용택" → 하라유타카, 오용택
        for part in re.split(r"[;,/]", b.get("authors") or ""):
            names.add(norm_title(re.sub(r"^.*?:", "", part)))
    return frozenset(n for n in names if len(n) >= 2)


def is_taste_word(word: str, title: str = "") -> bool:
    """취향을 설명하는 키워드인지 코드로 판정한다.

    빼는 것: 흔한 단어, 한 글자, 영어·숫자만 있는 단어(DAY, Lesson), 띄어쓰기 3번 이상인 긴 구절(다른 책 제목),
    출판사·저자 이름(북이십일, 하라 유타카), 이 책 제목 글자 그대로인 단어(드래곤 마스터).
    """
    if word in GENERIC_WORDS or len(word) < 2:
        return False
    if re.fullmatch(r"[A-Za-z0-9\s.&'-]+", word):
        return False
    if word.count(" ") >= 2:
        return False
    n = norm_title(word)
    if n in _publishers() or (title and n in title):
        return False
    return True


@lru_cache(maxsize=1)
def _keyword_idf() -> tuple[dict[str, float], float]:
    """스냅샷 전체에서 키워드가 몇 권에 나오는지로 희귀도(IDF)를 계산한다. 흔한 단어("다양", "방법")일수록 낮다."""
    import math

    if not SNAPSHOT.exists():
        return {}, 1.0
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    docs = data.get("books", []) + data.get("read_books", [])
    df: dict[str, int] = defaultdict(int)
    for b in docs:
        for w in {k["word"] for k in b.get("keywords") or []}:
            df[w] += 1
    n = len(docs) or 1
    idf = {w: math.log((n + 1) / (c + 1)) for w, c in df.items()}
    return idf, math.log(n + 1)


def _new_matches() -> dict[tuple[str, str], dict]:
    """웹에서 새로 입력한 책의 매칭 결과 (reading_log_matches.json)."""
    if not NEW_MATCHES.exists():
        return {}
    data = json.loads(NEW_MATCHES.read_text(encoding="utf-8"))
    return {(b["reader"], norm_title(b["log_title"])): b for b in data}


async def match_new_book(reader: str, title: str, author: str = "") -> dict | None:
    """새로 입력한 책을 정보나루에서 찾아 키워드·책소개와 함께 저장한다 (책 1권에 호출 약 3건, 캐시 재사용).

    이미 맞춘 책이면 다시 부르지 않는다. 못 찾으면 None (이미 읽은 책 제외에는 제목으로 쓰고, 취향에는 쓰지 않음).
    """
    from build_snapshot import keywords_of, pick_match, search_query
    from mcp_client import Data4Library, books_of

    key = (reader, norm_title(title))
    if key in _snapshot_read_books() or key in _new_matches():
        return {**_snapshot_read_books(), **_new_matches()}[key]
    lib = Data4Library()
    result = await lib.call("search_books", title=search_query(title), author=author or None, pageSize=20)
    match = pick_match(title, author, books_of(result), reader)
    if not match:
        short = " ".join(search_query(title).split()[-2:])
        if short != search_query(title):
            result = await lib.call("search_books", title=short, pageSize=20)
            match = pick_match(title, author, books_of(result), reader)
    if not match:
        return None
    kws = keywords_of(await lib.call("get_book_keywords", isbn13=match["isbn13"]))
    analysis = await lib.call("get_book_analysis", isbn13=match["isbn13"], type="loan", period=1)
    book = (((analysis or {}).get("result") or {}).get("analysis") or {}).get("book") or {}
    rec = {"isbn13": match["isbn13"], "title": match.get("bookname"), "authors": match.get("authors"),
           "publisher": match.get("publisher"), "publication_year": match.get("publication_year"),
           "addition_symbol": match.get("addition_symbol"), "vol": match.get("vol"),
           "class_no": match.get("class_no"), "class_nm": match.get("class_nm"),
           "reader": reader, "log_title": title, "keywords": kws, "description": (book.get("description") or "").strip()}
    data = json.loads(NEW_MATCHES.read_text(encoding="utf-8")) if NEW_MATCHES.exists() else []
    data.append(rec)
    NEW_MATCHES.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def taste_keywords(books: list[dict], top: int = 15) -> list[dict]:
    """좋아한 책들의 키워드로 취향 키워드를 뽑는다 (코드 계산).

    책마다 키워드 가중치를 그 책의 최댓값으로 나눠 0~1로 맞춘 뒤 더한다.
    한 책이 키워드를 많이 가졌다고 취향을 독차지하지 않게, 여러 책에 걸친 단어가 위로 온다.
    """
    score: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for b in books:
        kws = b.get("keywords") or []
        if not kws:
            continue
        top_w = max(k["weight"] for k in kws) or 1
        title = norm_title(b.get("title", ""))
        for k in kws:
            word = k["word"].strip()
            if not is_taste_word(word, title):
                continue
            idf, idf_max = _keyword_idf()
            score[word] += (k["weight"] / top_w) * idf.get(word, idf_max)
            count[word] += 1
    # 여러 책에 함께 나온 단어일수록 취향일 가능성이 크다: 책 수만큼 가중
    ranked = sorted(score.items(), key=lambda x: -(x[1] * (1 + 0.5 * (count[x[0]] - 1))))
    return [{"word": w, "score": round(s, 2), "books": count[w]} for w, s in ranked[:top]]


def get_reading_log_data(reader: str, owner: str = DEMO_OWNER) -> dict:
    """프로필 + 독서 기록 + 취향 키워드를 모은다. 같은 owner의 프로필이 아니면 빈 결과와 이유를 돌려준다."""
    profile = find_profile(reader, owner)
    if profile is None:
        return {"reader": reader, "found": False, "reason": "등록한 프로필이 아닙니다."}

    matches = {**_snapshot_read_books(), **_new_matches()}
    with READING_LOG.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["reader"] == reader]

    books, unmatched = [], []
    for r in rows:
        m = matches.get((reader, norm_title(r["title"])))
        disliked = (r.get("disliked") or "").strip().upper() == "Y"
        entry = {
            "log_title": r["title"],
            "author": r.get("author") or (m or {}).get("authors", ""),
            "volumes": parse_volumes(r.get("volumes", "")),
            "volumes_text": r.get("volumes", ""),
            "disliked": disliked,
            "matched": m is not None,
        }
        if m:
            entry.update({k: m.get(k) for k in ("isbn13", "title", "addition_symbol", "class_no", "class_nm",
                                                 "keywords", "description")})
        else:
            unmatched.append(r["title"])
        books.append(entry)

    liked = [b for b in books if b["matched"] and not b["disliked"]]
    return {
        "reader": reader,
        "found": True,
        "profile": {k: v for k, v in profile.items() if k != "owner"},
        "books": books,
        "read_count": len(books),
        "disliked_count": sum(b["disliked"] for b in books),
        "unmatched": unmatched,  # 이미 읽은 책 제외에는 쓰고, 취향 키워드에는 쓰지 않는다
        "taste_keywords": taste_keywords(liked),
    }


TOPIC_WORDS = {"요리": ["요리", "음식", "식생활", "레시피", "쿠킹", "요리법"],
               "과학": ["과학", "실험"], "역사": ["역사", "한국사", "세계사"], "수학": ["수학"],
               "미술": ["미술", "그림"], "음악": ["음악"], "동물": ["동물"], "공룡": ["공룡"]}


def log_books_about(reader: str, topic: str, owner: str = DEMO_OWNER) -> list[dict]:
    """독서 기록에서 주제에 맞는 책만 고른다 (코드 판정, LLM 없음). 제목·분류·키워드(상위 10개)에서 주제어를 찾는다.
    T12 2회차: "아이가 좋아하는 요리책"처럼 기록을 묻는 질문에 LLM이 기록에 없는 책을 지어내지 않게 한다."""
    log = get_reading_log_data(reader, owner)
    if not log.get("found"):
        return []
    words = TOPIC_WORDS.get(topic, [topic])
    hits = []
    for b in log["books"]:
        text = " ".join([b["log_title"], b.get("class_nm") or "",
                         " ".join(k["word"] for k in (b.get("keywords") or [])[:10])])
        if any(w in text for w in words):
            hits.append(b)
    return hits


@tool
def get_reading_log(reader: str) -> str:
    """독자 프로필과 독서 기록(읽은 책, 시리즈에서 읽은 권, 싫었던 책)과 취향 키워드를 불러온다.

    쓸 때: 누군가에게 책을 추천하기 전에 항상 먼저 부른다. 추천 이유("읽은 ○○와 닮았다")의 근거가 된다.
    쓰지 않을 때: 책 정보 자체를 찾을 때 (그때는 search_catalog). 이용자가 등록하지 않은 사람의 기록은 불러올 수 없다.
    reader: 프로필 별칭 (예: "아이", "나")
    """
    data = get_reading_log_data(reader, DEMO_OWNER)
    if not data["found"]:
        return json.dumps({"reader": reader, "error": data["reason"]}, ensure_ascii=False)
    slim = {
        **{k: data[k] for k in ("reader", "profile", "read_count", "disliked_count", "taste_keywords")},
        "books": [{"title": b["log_title"], "volumes": b["volumes_text"], "disliked": b["disliked"],
                   "class_nm": b.get("class_nm")} for b in data["books"]],
    }
    return json.dumps(slim, ensure_ascii=False)


# 질문에서 "원하는 책의 내용"이 아닌 말 (이것만 있으면 취향 키워드로 검색한다)
REQUEST_STOPWORDS = {
    "아이", "나", "내", "저", "책", "권", "추천", "다음", "읽", "하", "주", "있", "좋", "골라", "고르", "알리", "알려",
    "해주", "만", "것", "수", "한테", "에게", "또", "더", "좀", "요즘", "우리", "자녀", "엄마", "아빠", "줄", "보",
    "분위기", "비슷", "같", "취향", "기록", "맞", "싫", "어떤", "무슨", "새", "새로",
    "싫어하", "빼", "하나", "같이", "함께", "이랑", "랑", "나오", "다시", "시리즈",
}


def clean_query(question: str) -> str:
    """검색어에서 요청 문구를 지운다: '3권', '추천해줘', '아이한테' 같은 말이 BM25에서 엉뚱한 책('3학년')과 맞지 않게."""
    q = re.sub(r"\d+\s*권(만|씩)?", " ", question or "")
    q = re.sub(r"(추천|골라|알려|찾아|보여)\s*(해)?\s*(줘|주세요|줄래|줄 수 있어)?[?.!]*", " ", q)
    names = "|".join(re.escape(p["reader"]) for p in load_profiles()) or "아이"
    q = re.sub(rf"({names}|내|나|우리 아이)\s*(가|이|한테|에게|랑|와|의|도)?\s", " ", q + " ")
    q = re.sub(r"(다음에|다음|읽을|책들?|좀|요즘)\s", " ", q + " ")
    return re.sub(r"\s+", " ", q).strip()


TRAIT_PATTERNS = [  # (요청 성격 이름, 질문에서 찾을 표현) — 순서대로 돌려준다
    ("웃김", r"웃기|웃긴|웃음|유머|재밌|재미있|유쾌"),
    ("모험", r"모험|탐험"),
    ("오싹함", r"무서운|무섭|오싹|공포|으스스"),
    ("감동", r"감동|뭉클|따뜻한"),
    ("추리", r"추리|탐정|미스터리|수수께끼"),
    ("판타지", r"판타지|마법|환상"),
]


def requested_traits(question: str) -> list[str]:
    """질문이 요청한 책의 성격을 코드로 뽑는다 (T12-2 4회). 추천 이유 작성(compose)에 넘겨 이유마다 반영하게 한다.
    예: '웃기면서 모험 이야기' → ['웃김', '모험']"""
    return [name for name, pat in TRAIT_PATTERNS if re.search(pat, question or "")]


def has_content_request(query: str) -> bool:
    """질문에 원하는 책의 내용(예: '무서운', '모험', '공룡')이 들어 있는지 코드로 판정한다."""
    from retriever import korean_tokenizer

    return any(t not in REQUEST_STOPWORDS and not t.isdigit() and len(t) >= 2 for t in korean_tokenizer(query or ""))


def taste_query(reader: str, query: str = "", n_words: int | None = None) -> str:
    """검색어 만들기: 질문 + 독자 취향 키워드 (취향 키워드는 코드가 계산한 것).

    질문에 원하는 내용이 있으면 질문이 주인공이 되도록 취향 키워드는 2개만, 없으면 8개를 붙인다.
    """
    if n_words is None:
        n_words = 2 if has_content_request(query) else 8
    data = get_reading_log_data(reader, DEMO_OWNER) if reader else {"found": False}
    words = [k["word"] for k in data.get("taste_keywords", [])[:n_words]] if data.get("found") else []
    return " ".join(p for p in [query.strip(), " ".join(words)] if p)


def search_catalog_data(query: str, reader: str = "", k: int = 20) -> list[dict]:
    """도서 목록 스냅샷에서 하이브리드 검색(BM25+벡터)으로 후보를 찾는다. 걸러내기는 하지 않는다 (filter_candidates 몫)."""
    from retriever import hybrid_search

    return hybrid_search(taste_query(reader, query), k=k)


@tool
def search_catalog(query: str, reader: str = "") -> str:
    """도서 목록에서 질문·취향에 맞는 책 후보를 찾는다 (RAG 하이브리드 검색).

    쓸 때: 추천할 책 후보가 필요할 때. reader를 주면 그 독자의 취향 키워드를 검색어에 더한다.
    쓰지 않을 때: 특정 책 제목이 도서관에 있는지 확인할 때 (그때는 search_books).
    결과에는 이미 읽은 책·연령에 안 맞는 책이 섞여 있을 수 있다. 추천 전에 항상 걸러내기 단계를 거친다.
    query: 원하는 책의 종류 (예: "웃긴 모험 이야기"), reader: 프로필 별칭 (선택)
    """
    hits = search_catalog_data(query, reader, k=20)
    return json.dumps([
        {"isbn13": h["isbn13"], "title": h["title"], "class_nm": h["book"].get("class_nm"),
         "description": (h["book"].get("description") or "")[:200]}
        for h in hits
    ], ensure_ascii=False)


# ---- filter_candidates: 추천 직전에 항상 실행되는 걸러내기 단계 (Agent가 고르는 도구가 아님) ----

def series_key(title: str) -> str:
    """시리즈 비교용 이름: 앞 괄호 머리말·부제(: = - 뒤)를 뺀 본제목. '(제로니모의) 환상모험' → '환상모험'."""
    main = re.split(r"\s*[:=]\s*|\s+-\s+", title or "")[0]
    return norm_title(main)


AGE_RULES = {
    "어린이": {"7"},             # 아동만
    "청소년": {"4", "7", "0"},   # 청소년·아동·교양
}


READER_TARGET = {"0": "교양(일반)", "1": "실용", "2": "여성", "4": "청소년", "5": "중·고등 학습참고서",
                 "6": "초등 학습참고서", "7": "아동", "9": "전문"}
PUB_FORM = {"0": "문고본", "1": "사전", "2": "신서판", "3": "단행본", "4": "전집·총서·시리즈",
            "6": "전자출판물", "7": "도록", "8": "그림책·만화", "9": "혼합자료"}
KDC_MAIN = {"0": "총류", "1": "철학", "2": "종교", "3": "사회과학", "4": "자연과학", "5": "기술과학",
            "6": "예술", "7": "언어", "8": "문학", "9": "역사"}


def addition_symbol_meaning(symbol: str) -> str:
    """ISBN 부가기호 5자리를 코드가 풀어 쓴다 (첫 자리 독자대상 · 둘째 자리 발행형태 · 끝 세 자리 KDC 내용분류).
    LLM이 부가기호를 직접 풀이하다 틀리지 않게 (T12 1회차: 74840을 '청소년용'으로 잘못 설명). 예: '74840' → '대상 아동 · 형태 전집·총서·시리즈 · 내용 문학'"""
    s = str(symbol or "").strip()
    if not re.fullmatch(r"\d{5}", s):
        return "부가기호 없음"
    return (f"대상 {READER_TARGET.get(s[0], '기타')} · 형태 {PUB_FORM.get(s[1], '기타')} · "
            f"내용 {KDC_MAIN.get(s[2], '기타')}")


def age_allowed(book: dict, profile: dict) -> tuple[bool, str]:
    """ISBN 부가기호 첫 자리(독자대상)로 연령 판정 (SERVICE.md 3-4). 성인·60대는 제한 없음."""
    allowed = AGE_RULES.get(profile.get("age_group", ""))
    if allowed is None:
        return True, ""
    symbol = str(book.get("addition_symbol") or "").strip()
    if not symbol:
        return False, "부가기호가 없어 독자대상을 확인할 수 없음"
    if symbol[0] not in allowed:
        return False, f"연령에 맞지 않음 (부가기호 {symbol}, {profile['age_group']} 허용 {'/'.join(sorted(allowed))})"
    return True, ""


def _vol_number(book: dict) -> int | None:
    m = re.search(r"\d+", str(book.get("vol") or ""))
    return int(m.group()) if m else None


def _read_index(log: dict) -> dict:
    """독서 기록을 걸러내기용으로 정리: 읽은 ISBN, 시리즈별 읽은 권, 단권 제목, 싫었던 것."""
    idx = {"isbns": set(), "disliked_isbns": set(), "series": defaultdict(lambda: {"all": False, "vols": set()}),
           "single": set(), "disliked_series": defaultdict(set), "disliked_single": set()}
    for b in log["books"]:
        keys = {series_key(b["log_title"])} | ({series_key(b["title"])} if b.get("title") else set())
        vols = b["volumes"]
        if b.get("isbn13") and not vols["volumes"] and not vols["all"]:
            # 시리즈 줄의 매칭 ISBN은 대표 1권일 뿐이라, 단권 책일 때만 ISBN으로 판정한다
            (idx["disliked_isbns"] if b["disliked"] else idx["isbns"]).add(b["isbn13"])
        for k in keys:
            if vols["all"] or vols["volumes"]:
                idx["series"][k]["all"] |= vols["all"]
                idx["series"][k]["vols"].update(vols["volumes"])
                if b["disliked"]:
                    idx["disliked_series"][k].update(vols["volumes"] or ["all"])
            else:
                idx["single"].add(k)
                if b["disliked"]:
                    idx["disliked_single"].add(k)
    return idx


def _read_reason(book: dict, idx: dict) -> tuple[str, bool]:
    """읽은 책이면 (이유, 싫었던 책 여부)를, 아니면 ("", False)를 돌려준다."""
    isbn, key, vol = book.get("isbn13"), series_key(book.get("title", "")), _vol_number(book)
    if isbn in idx["disliked_isbns"] or key in idx["disliked_single"]:
        return "싫었던 책", True
    if key in idx["disliked_series"]:
        bad = idx["disliked_series"][key]
        if "all" in bad or vol is None or vol in bad:
            return f"싫었던 책 ({book.get('title')} {vol or ''}권)", True
    if isbn in idx["isbns"] or key in idx["single"]:
        return "이미 읽은 책", False
    if key in idx["series"]:
        s = idx["series"][key]
        if s["all"] or vol is None:
            return "이미 읽은 시리즈 (권 번호로 읽지 않은 권인지 확인할 수 없음)" if vol is None else "이미 읽은 시리즈 (전권)", False
        if vol in s["vols"]:
            return f"이미 읽은 권 ({vol}권)", False
    return "", False


@lru_cache(maxsize=1)
def catalog_isbns() -> frozenset[str]:
    """도서 목록 스냅샷에 있는 ISBN (지어낸 책을 막는 기준)."""
    if not SNAPSHOT.exists():
        return frozenset()
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return frozenset(b["isbn13"] for b in data.get("books", []))


def filter_candidates(candidates: list[dict], reader: str, owner: str = DEMO_OWNER,
                      allow_reread: bool = False, catalog: set | frozenset | None = None) -> dict:
    """추천 후보를 코드 규칙으로 걸러낸다 (LLM 판정 없음). 순서는 들어온 순서(검색 순위)를 지킨다.

    규칙: ① 도서 목록에 있는 책만 ② 연령 판정(부가기호) ③ 싫었던 책 제외(다시 읽기 요청이어도)
         ④ 이미 읽은 책·시리즈의 읽은 권 제외(다시 읽기 요청이면 허용) ⑤ 같은 시리즈는 1권만(가장 앞 권)
    반환: {"kept": [책], "removed": [{isbn13, title, reason}]}
    """
    log = get_reading_log_data(reader, owner)
    if not log["found"]:
        return {"kept": [], "removed": [], "error": log["reason"]}
    profile, idx = log["profile"], _read_index(log)
    catalog = catalog_isbns() if catalog is None else catalog

    removed, survivors = [], []
    for b in candidates:
        reason = ""
        if b.get("isbn13") not in catalog:
            reason = "도서 목록에 없는 책"
        if not reason:
            ok, why = age_allowed(b, profile)
            reason = "" if ok else why
        if not reason:
            why, disliked = _read_reason(b, idx)
            if why and (disliked or not allow_reread):
                reason = why
        if reason:
            removed.append({"isbn13": b.get("isbn13"), "title": b.get("title"), "reason": reason})
        else:
            survivors.append(b)

    # ⑤ 같은 시리즈는 1권만: 시리즈 묶음의 첫 등장 위치에, 그 묶음에서 가장 앞 권을 둔다
    groups: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for b in survivors:
        k = series_key(b.get("title", "")) or b.get("isbn13")
        if k not in groups:
            order.append(k)
        groups[k].append(b)
    kept = []
    for k in order:
        members = sorted(groups[k], key=lambda b: (_vol_number(b) is None, _vol_number(b) or 0))
        kept.append(members[0])
        for other in members[1:]:
            removed.append({"isbn13": other.get("isbn13"), "title": other.get("title"),
                            "reason": f"같은 시리즈는 1권만 ({members[0].get('vol') or '-'}권을 남김)"})
    return {"kept": kept, "removed": removed}


if __name__ == "__main__":
    # T3 확인: 아이·나 각각 권수, 싫었던 책 수, 취향 키워드 상위 10개, 못 찾은 책, 애매한 매칭
    for who in ("아이", "나"):
        d = get_reading_log_data(who)
        print(f"\n[{who}] {d['profile']}")
        print(f"  읽은 책 {d['read_count']}줄, 싫었던 책 {d['disliked_count']}권, "
              f"도서관 책과 맞춘 것 {sum(b['matched'] for b in d['books'])}줄")
        print("  취향 키워드 상위 10:", ", ".join(f"{k['word']}({k['books']}권)" for k in d["taste_keywords"][:10]))
        print("  못 찾은 책:", d["unmatched"] or "없음")
        series = [f"{b['log_title'][:12]} {b['volumes_text']}" for b in d["books"] if b["volumes_text"]]
        print("  시리즈 권 범위:", ", ".join(series) or "없음")
    print("\n[다른 이용자 프로필 차단] owner='other'로 '아이' 요청 →", get_reading_log_data("아이", owner="other"))
    print("\n[애매한 매칭]")
    for b in get_reading_log_data("아이")["books"]:
        if "그리스" in b["log_title"] or "테오" in b["log_title"]:
            print(f"  {b['log_title']} → {b.get('title')} (ISBN {b.get('isbn13')}, 부가기호 {b.get('addition_symbol')})")
