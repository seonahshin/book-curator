"""FastAPI 서버: POST /query, 프로필·독서 기록 입력, 웹 페이지 서빙.

실행: .venv/Scripts/python -m uvicorn api:app --app-dir src --port 8000
- 로그인 없이 owner=demo 고정 (SERVICE.md 4절)
- /query 규약: Body {"question"} (+ 선택 reader, purpose, top_k) → {"answer", "contexts", "trace"} (+ recommendations)
- 입력 검사는 코드로 한다: 별칭 중복·형식, 연령대 값, 빈 제목, 권 범위 형식
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tools  # noqa: E402
from agent import answer_question  # noqa: E402
from ranker import Purpose  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
OWNER = tools.DEMO_OWNER

# 연령대별 나이 범위 (웹 화면 입력 범위와 같게, 11-20 사용자 결정)
AGE_GROUPS = {"어린이": (0, 13), "청소년": (14, 19), "성인": (20, 59), "60대 이상": (60, 150)}
CHILD_DEFAULT_AGE = 10  # 어린이 프로필에 나이를 안 적으면 9~11세로 본다

app = FastAPI(title="개인 맞춤 도서 추천 에이전트")


# ---------------- 요청 모델 ----------------

class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    reader: Optional[str] = None
    purpose: Optional[Purpose] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class ProfileIn(BaseModel):
    reader: str = Field(min_length=1, max_length=20, description="별칭 (실명·학교·연락처 대신)")
    age_group: Literal["어린이", "청소년", "성인", "60대 이상"]
    age: Optional[int] = Field(default=None, ge=0, le=150, description="나이 (연령대 범위 안, 예: 40)")

    @field_validator("reader")
    @classmethod
    def no_contact_info(cls, v: str) -> str:
        v = v.strip()
        if re.search(r"\d{2,3}-?\d{3,4}-?\d{4}|@", v):
            raise ValueError("별칭에 전화번호·이메일은 넣을 수 없어요")
        return v

    @model_validator(mode="after")
    def age_in_group(self) -> "ProfileIn":
        """나이가 고른 연령대 범위 안인지 코드로 확인한다 (예: 17세를 어린이로 넣으면 거절)."""
        lo, hi = AGE_GROUPS[self.age_group]
        if self.age is not None and not lo <= self.age <= hi:
            raise ValueError(f"{self.age_group} 나이는 {lo}~{hi}세 사이로 적어 주세요")
        return self


class BookIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    author: str = ""
    volumes: str = Field(default="", description="시리즈에서 읽은 권 범위 (예: 1~15, 3, 전권)")
    disliked: bool = False

    @field_validator("volumes")
    @classmethod
    def volumes_format(cls, v: str) -> str:
        v = v.strip()
        if v and not re.fullmatch(r"(전권|전체|(\d+(\s*[~\-]\s*\d+)?)([,\s]+\d+(\s*[~\-]\s*\d+)?)*)", v):
            raise ValueError("권 범위는 1~15, 3, 전권처럼 적어 주세요")
        return v


class ReadingLogIn(BaseModel):
    reader: str
    books: list[BookIn] = Field(min_length=1, max_length=50)


# ---------------- 엔드포인트 ----------------

@app.post("/query")
async def query(body: QueryIn) -> dict:
    """질문 → 추천. 응답 키: answer, contexts, trace (+ recommendations)."""
    if body.reader and tools.find_profile(body.reader, OWNER) is None:
        return {"answer": f"'{body.reader}' 프로필이 없어요. 먼저 프로필을 만들어 주세요.", "contexts": [],
                "trace": [{"step": "check_profile", "input": body.reader, "output": "등록한 프로필이 아님"}]}
    return await answer_question(body.question, reader=body.reader,
                                 purpose=body.purpose.value if body.purpose else None, top_k=body.top_k)


@app.get("/profiles")
def list_profiles() -> list[dict]:
    """내가(owner=demo) 등록한 프로필 목록."""
    return [{k: v for k, v in p.items() if k != "owner"} for p in tools.load_profiles() if p.get("owner") == OWNER]


@app.post("/profiles")
def add_profile(body: ProfileIn) -> dict:
    """프로필 등록: 별칭·연령대만 저장한다."""
    profiles = tools.load_profiles()
    if any(p["reader"] == body.reader and p.get("owner") == OWNER for p in profiles):
        raise HTTPException(409, f"'{body.reader}' 별칭은 이미 있어요")
    lo, hi = AGE_GROUPS[body.age_group]
    rec = {"reader": body.reader, "age_group": body.age_group}
    if body.age_group == "어린이":
        # 나이±1을 어린이 범위 안으로 자른다. 나이가 범위 안이라 항상 from_age ≤ to_age (0세도 0으로 저장)
        age = CHILD_DEFAULT_AGE if body.age is None else body.age
        rec.update({"from_age": max(lo, age - 1), "to_age": min(hi, age + 1)})
    else:
        rec["age"] = lo if body.age is None else body.age
    rec["owner"] = OWNER
    profiles.append(rec)
    tools.PROFILES.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    return {k: v for k, v in rec.items() if k != "owner"}


@app.post("/reading-log")
async def add_reading_log(body: ReadingLogIn) -> dict:
    """읽은 책 입력 (여러 권 한 번에). 정보나루에서 실제 책을 찾아 키워드·책소개를 저장한다."""
    if tools.find_profile(body.reader, OWNER) is None:
        raise HTTPException(404, f"'{body.reader}' 프로필이 없어요. 먼저 프로필을 만들어 주세요.")
    with tools.READING_LOG.open(encoding="utf-8-sig") as f:
        existing = {(r["reader"], tools.norm_title(r["title"])) for r in csv.DictReader(f)}
    added, skipped, results = [], [], []
    with tools.READING_LOG.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for b in body.books:
            key = (body.reader, tools.norm_title(b.title))
            if key in existing:
                skipped.append(b.title)
                continue
            w.writerow([body.reader, b.title.strip(), b.author.strip(), b.volumes, "Y" if b.disliked else ""])
            existing.add(key)
            added.append(b)
    for b in added:
        try:
            m = await tools.match_new_book(body.reader, b.title.strip(), b.author.strip())
        except Exception as e:  # 정보나루가 막혀도 기록은 남긴다
            m, err = None, f"{type(e).__name__}"
        else:
            err = ""
        results.append({"title": b.title, "matched": bool(m), "found_title": (m or {}).get("title"),
                        "isbn13": (m or {}).get("isbn13"), "error": err})
    tools._snapshot_read_books.cache_clear()
    log = tools.get_reading_log_data(body.reader, OWNER)
    return {"reader": body.reader, "added": results, "skipped_duplicates": skipped,
            "read_count": log["read_count"], "taste_keywords": log["taste_keywords"][:8]}


@app.delete("/reading-log/{reader}")
def delete_book(reader: str, title: str) -> dict:
    """읽은 책 한 줄 지우기 (제목 비교는 띄어쓰기·문장부호 무시). 취향은 다음 조회 때 남은 기록으로 다시 계산된다.

    정보나루 매칭 결과(reading_log_matches.json)는 캐시라 남겨 둔다. 같은 책을 다시 넣으면 호출 없이 재사용한다.
    """
    if tools.find_profile(reader, OWNER) is None:
        raise HTTPException(404, f"'{reader}' 프로필이 없어요.")
    with tools.READING_LOG.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    target = tools.norm_title(title)
    keep = [r for r in body if not (r and r[0] == reader and tools.norm_title(r[1]) == target)]
    if len(keep) == len(body):
        raise HTTPException(404, f"'{title}'은(는) 읽은 책 목록에 없어요.")
    with tools.READING_LOG.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows([header, *keep])  # 원본 파일과 같은 줄바꿈
    tools._snapshot_read_books.cache_clear()
    log = tools.get_reading_log_data(reader, OWNER)
    return {"reader": reader, "deleted": title, "read_count": log["read_count"]}


@app.get("/reading-log/{reader}")
def get_log(reader: str) -> dict:
    """프로필의 독서 기록 (내가 등록한 프로필만)."""
    log = tools.get_reading_log_data(reader, OWNER)
    if not log["found"]:
        raise HTTPException(404, log["reason"])
    return {"reader": reader, "read_count": log["read_count"], "disliked_count": log["disliked_count"],
            "books": [{"title": b["log_title"], "volumes": b["volumes_text"], "disliked": b["disliked"],
                       "matched": b["matched"], "found_title": b.get("title") or "",
                       "has_keywords": bool(b.get("keywords"))} for b in log["books"]],
            "taste_keywords": log["taste_keywords"][:10]}


@app.get("/")
def index():
    page = STATIC / "index.html"
    if page.exists():
        return FileResponse(page)
    return JSONResponse({"message": "웹 화면은 준비 중입니다 (T12-1). POST /query 로 질문하세요."})


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
