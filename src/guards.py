"""입력·출력 가드레일. 모두 Python 코드 판정이다 (LLM 판정 없음).

입력 가드 (그래프 실행 전)
- 개인정보 가리기: 전화번호·이메일·주민등록번호 → [전화번호] 등. LLM·trace·로그에는 가린 질문만 간다
- 비밀 요청(시스템 프롬프트, 인증키 등): 코드가 정해진 문장으로 거절, LLM을 부르지 않는다
- 등록하지 않은 사람의 독서 기록 요청: 거절하고 프로필 만들기를 안내한다
- 인젝션("이전 지시는 무시하고"): 막지 않고 표시만 한다. 연령·읽은 책 규칙은 코드가 지키므로 결과가 바뀌지 않는다

출력 가드 (그래프 실행 후)
- 비밀값(.env의 키 값)이 답변에 섞이면 지운다
- 어린이 프로필 추천에 아동 도서(부가기호 7) 아닌 책이 있으면 뺀다 (filter_candidates의 이중 확인)
- 보호자 확인 표시: caution_check (주의 단어 목록 data/caution_words.txt)
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from config import secret_values

ROOT = Path(__file__).resolve().parents[1]
CAUTION_WORDS = ROOT / "data" / "caution_words.txt"

PII_PATTERNS = [
    ("전화번호", re.compile(r"(?<!\d)0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)")),
    ("이메일", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("주민등록번호", re.compile(r"(?<!\d)\d{6}-[1-4]\d{6}(?!\d)")),  # 하이픈 필수: ISBN 13자리와 헷갈리지 않게
]
INJECTION = re.compile(r"(이전|앞의|위의|모든)\s*(지시|명령|규칙|설정).{0,6}(무시|잊)|ignore\s+(all|previous|prior)|"
                       r"너는\s*이제|규칙을?\s*(무시|어기)|탈옥|jailbreak", re.I)
SECRET = re.compile(r"시스템\s*프롬프트|system\s*prompt|인증\s*키|api\s*key|비밀\s*번호|환경\s*변수|\.env|"
                    r"aws.{0,6}(키|key)|secret|토큰", re.I)
RECORD_REQUEST = re.compile(r"([가-힣]{2,5})(?:\s*\([^)]*\))?\s*(?:이|가|의|님)\s*(?:지금까지\s*)?"
                            r"(?:읽은|읽었던|독서\s*기록|기록)")
SELF_WORDS = {"아이", "우리", "우리 아이", "내", "제", "나", "저", "자녀", "애"}

SECRET_REFUSAL = ("시스템 설정과 인증키는 알려 드릴 수 없어요. 대신 책 추천은 언제든 도와드릴게요. "
                  "예) '아이가 다음에 읽을 책 5권 추천해줘'")


def mask_pii(text: str) -> tuple[str, list[str]]:
    """개인정보를 가린다. 반환: (가린 글, 찾은 종류 목록)."""
    found = []
    for name, pat in PII_PATTERNS:
        if pat.search(text):
            found.append(name)
            text = pat.sub(f"[{name}]", text)
    return text, found


def _profiles() -> list[dict]:
    from tools import DEMO_OWNER, load_profiles

    return [p for p in load_profiles() if p.get("owner") == DEMO_OWNER]


def input_guard(question: str) -> dict:
    """질문을 검사한다. 반환: {"question": 가린 질문, "block": 거절 문장 또는 None, "flags": [...], "masked": [...]}"""
    masked, found = mask_pii(question)
    flags: list[str] = []
    block = None
    if found:
        flags.append("pii_masked")
    if INJECTION.search(masked):
        flags.append("injection")
    if SECRET.search(masked):
        flags.append("secret_request")
        block = SECRET_REFUSAL
    if block is None:
        names = {p["reader"] for p in _profiles()}
        for m in RECORD_REQUEST.finditer(masked):
            who = m.group(1)
            if who not in names and who not in SELF_WORDS and not any(n in who for n in names):
                flags.append("other_person_record")
                block = (f"'{who}'님은 등록한 프로필이 아니라서 독서 기록을 보여 드릴 수 없어요. "
                         "다른 사람의 기록은 본인이 직접 등록해야 볼 수 있어요. "
                         "책을 추천하고 싶다면 화면에서 프로필(별칭·연령대)을 만들고 읽은 책을 넣어 주세요.")
                if found:
                    block += " (질문에 있던 연락처는 가리고 저장하지 않았어요.)"
                break
    return {"question": masked, "block": block, "flags": flags, "masked": found}


@lru_cache(maxsize=1)
def caution_words() -> tuple[str, ...]:
    if not CAUTION_WORDS.exists():
        return ()
    lines = CAUTION_WORDS.read_text(encoding="utf-8").splitlines()
    return tuple(w.strip() for w in lines if w.strip() and not w.startswith("#"))


def caution_check(book: dict) -> tuple[bool, list[str]]:
    """제목·책소개·키워드에 주의 단어가 있는지. 반환: (표시 여부, 걸린 단어)."""
    text = " ".join([book.get("title") or "", book.get("description") or "",
                     " ".join(k.get("word", "") for k in (book.get("keywords") or []))])
    hits = [w for w in caution_words() if w in text]
    return bool(hits), hits


def output_guard(answer: str, recommendations: list[dict], profiles: list[dict]) -> dict:
    """답변을 검사한다. 반환: {"answer", "recommendations", "flags"}"""
    flags = []
    for v in secret_values():  # 비밀값 목록은 config 한곳 (SECRET_KEYS)
        if v in answer:
            answer = answer.replace(v, "[비밀값]")
            flags.append("secret_leak")
    answer, found = mask_pii(answer)
    if found:
        flags.append("pii_in_answer")
    if any(p.get("age_group") == "어린이" for p in profiles):
        kept = [r for r in recommendations if str(r.get("addition_symbol") or "").startswith("7")]
        if len(kept) != len(recommendations):
            flags.append("age_violation")
        recommendations = kept
    return {"answer": answer, "recommendations": recommendations, "flags": flags}
