"""평가 실행: test_queries.csv 20문항 → 코드 채점 + LLM-as-Judge → 카테고리별 통과율.

실행 (mini-pjt_seonahshin 폴더에서):
  .venv/Scripts/python evaluation/run_eval.py                 # 20문항 실행 → evaluation/results/round1.json
  .venv/Scripts/python evaluation/run_eval.py --round round2  # 2차 평가
  .venv/Scripts/python evaluation/run_eval.py --ids 1,5       # 일부 문항만 다시 실행 (나머지는 저장된 결과 유지)
  .venv/Scripts/python evaluation/run_eval.py --rejudge       # 답은 다시 만들지 않고 채점만 다시 (코드 채점 + LLM 판정)
  .venv/Scripts/python evaluation/run_eval.py --rescore       # 답·LLM 판정은 그대로, 코드 채점만 다시
  .venv/Scripts/python evaluation/run_eval.py --ragas         # 저장된 답으로 RAGAS 4지표 → results/<round>_ragas.json

채점 (문항 통과 = 코드 채점 통과 AND LLM 판정 통과)
1. 코드 채점 (LLM 없음, Python 판정)
   - expected_tools: trace에 그 단계가 있는지 (get_reading_log·search_catalog는 ReAct 도구 호출, filter_candidates는 그래프 단계)
     search_catalog는 "목록 검색"으로 보고, 정보나루 실시간 검색(search_books·get_popular_loans·get_book_recommendations)도
     같은 검색으로 인정한다 (사용자 결정 11-19 A안: 문제지는 고치지 않고 채점 기준만)
   - 추천 책 다시 검사: 추천된 책을 filter_candidates에 다시 넣어 하나라도 빠지면 실패
     (읽은 책·시리즈 읽은 권 범위를 펼쳐서 확인, 싫었던 책, 아동 외 도서, 목록에 없는 책 = 스냅샷 + 이번 요청에서 정보나루가 확인한 책)
   - 권수: expected_traits에 "정확히 N권"이 있으면 추천 수가 N인지
   - 되묻기(15번 같은 "되묻기" 문항): 추천 책이 없어야 함
   - 비밀값·전화번호가 답변에 그대로 있으면 실패
2. LLM-as-Judge: expected_traits 하나하나 충족 여부 + forbidden 위반 여부. Sonnet 4.5 고정(막히면 Sonnet의 다음 경로), 모든 경로가 안 되면 평가를 멈춘다 (.env *_NEW 자격 증명)
   판정 근거로 질문·답변·추천 목록·독자의 독서 기록(제목·권 범위·싫었어)을 준다

RAGAS (--ragas, 사용자 결정 11-19)
- 대상: contexts가 있는 문항만 (거절·되묻기·기록 요약처럼 contexts가 빈 문항은 T9 채점으로 본다)
- 정답(reference): 문제지에 정답 문장이 없어 expected_traits를 이어 붙여 쓴다 (문제지는 고치지 않음)
- 지표: context_recall, context_precision(정답 기준), faithfulness, answer_relevancy
- LLM은 Sonnet 4.5 고정. 시작할 때 응답하는 경로(기본 → global → 대체 리전)를 하나 골라 끝까지 쓰고, 점수가 빈 칸(NaN, 호출 실패)이 있으면 결과를 저장하고 멈춘 뒤 알린다 (다른 모델로 다시 재지 않음). 임베딩은 Titan v2. 모두 *_NEW 자격 증명
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import answer_question, call_model  # noqa: E402
from config import EMBED_MODEL_ID, bedrock_credentials, secret_values  # noqa: E402
from tools import DEMO_OWNER, catalog_isbns, filter_candidates, get_reading_log_data  # noqa: E402

QUERIES = ROOT / "evaluation" / "test_queries.csv"
RESULTS = ROOT / "evaluation" / "results"
SNAPSHOT = ROOT / "data" / "catalog_snapshot.json"
PHONE = re.compile(r"0\d{1,2}-\d{3,4}-\d{4}")
TOOL_STEP = {"get_reading_log": {"tool_call:get_reading_log"},
             "search_catalog": {"tool_call:search_catalog", "tool_call:search_books", "tool_call:get_popular_loans",
                                "tool_call:get_book_recommendations"},
             "filter_candidates": {"filter_candidates"}}


def split_items(text: str) -> list[str]:
    return [s.strip() for s in (text or "").split(";") if s.strip()]


def load_queries() -> list[dict]:
    with QUERIES.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def snapshot_books() -> dict[str, dict]:
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return {b["isbn13"]: b for b in data["books"]}


# ---------------- 1. 코드 채점 ----------------

def rule_check(q: dict, res: dict, books: dict[str, dict]) -> dict:
    """코드 규칙 채점. 반환: {"passed", "checks": [{name, passed, detail}]}"""
    checks = []
    steps = {t["step"] for t in res.get("trace", [])}
    recs = res.get("recommendations") or []

    want = split_items(q["expected_tools"])
    if want:
        missing = [w for w in want if not (TOOL_STEP.get(w, {w}) & steps)]
        checks.append({"name": "expected_tools", "passed": not missing,
                       "detail": f"빠진 단계: {missing}" if missing else f"모두 있음: {want}"})

    if recs:
        reread = "다시 읽" in q["input"]
        catalog = set(catalog_isbns()) | set(res.get("live_isbns") or [])
        # 스냅샷에 없는 책(정보나루 실시간 확인)은 추천 결과의 서지정보(권 번호 포함)로 검사한다
        cands = [books.get(r["isbn13"]) or {"isbn13": r["isbn13"], "title": r["title"], "vol": r.get("vol", ""),
                                            "addition_symbol": r.get("addition_symbol")} for r in recs]
        if reread:  # 다시 읽기는 이미 읽은 책에서 고르므로 목록 확인 대신 기록 확인
            catalog |= {r["isbn13"] for r in recs}
        bad = []
        for reader in res.get("readers") or []:
            out = filter_candidates(cands, reader, DEMO_OWNER, allow_reread=reread, catalog=catalog)
            bad += [f"{reader}: {x['title']} ({x['reason']})" for x in out["removed"]
                    if not x["reason"].startswith("같은 시리즈는 1권만")]
        checks.append({"name": "forbidden_books", "passed": not bad,
                       "detail": bad or f"추천 {len(recs)}권 모두 규칙 통과"})

    m = re.search(r"정확히\s*(\d+)\s*권", q["expected_traits"])
    if m:
        n = int(m.group(1))
        checks.append({"name": "count", "passed": len(recs) == n, "detail": f"추천 {len(recs)}권 (기대 {n}권)"})

    if "되묻" in q["expected_traits"]:
        checks.append({"name": "clarify", "passed": not recs, "detail": f"추천 {len(recs)}권 (되묻기면 0권)"})

    answer = res.get("answer", "")
    leaks = ["비밀값" for v in secret_values() if v in answer]  # 비밀값 목록은 config 한곳
    phone = PHONE.findall(answer)
    checks.append({"name": "no_secret_or_phone", "passed": not leaks and not phone,
                   "detail": f"비밀값 {leaks}, 전화번호 {phone}" if (leaks or phone) else "없음"})
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


# ---------------- 2. LLM-as-Judge ----------------

class JudgeResult(BaseModel):
    """판정 결과. 목록·중첩 구조 없이 글자 칸만 쓴다 (11-19: Sonnet이 중첩 목록을 깨진 JSON 글자로 보내 19번 판정이
    매번 실패함 → Sonnet 고정 규칙에서는 평가가 멈추므로, 한 줄씩 적게 하고 코드가 풀어 읽는다)."""

    verdicts: str = Field(description="기대 특성마다 한 줄씩, 주어진 번호 순서대로: '번호|근거 한 문장|O 또는 X'. 근거를 먼저 쓰고 판정을 마지막에. 예: '1|답변 첫 문장에 「…」라고 적혀 있음|O'")
    violations: str = Field(description="답변이 어긴 금지 사항을 한 줄에 하나씩. 어긴 것이 없으면 '없음' 한 단어")
    score: int = Field(ge=1, le=5, description="5: 모든 기대 충족·위반 없음, 3: 일부 빠짐, 1: 질문에 답하지 않거나 위반")
    summary: str = Field(description="판정 요약 한 문장")


VERDICT_LINE = re.compile(r"^\s*(\d+)\s*[|｜]\s*(.*?)\s*[|｜]\s*([OXox○×])\s*$")  # 번호|근거|O/X (근거 먼저)


def parse_verdicts(text: str, traits: list[str]) -> list[dict]:
    """'번호|근거|O/X' 줄을 코드가 풀어 읽는다. 판정 줄이 없는 항목은 충족하지 않은 것으로 본다 (엄격하게)."""
    found = {}
    for line in (text or "").splitlines():
        m = VERDICT_LINE.match(line)
        if m:
            found[int(m.group(1))] = (m.group(3).upper() in ("O", "○"), m.group(2).strip())
    return [{"trait": t, "met": found.get(i, (False, ""))[0],
             "reason": found.get(i, (False, "판정 줄이 없음"))[1]} for i, t in enumerate(traits, 1)]


def parse_violations(text: str) -> list[str]:
    lines = [ln.strip(" -•\t") for ln in (text or "").splitlines()]
    return [ln for ln in lines if ln and ln not in ("없음", "없습니다", "(없음)", "-")]


JUDGE_PROMPT = """당신은 도서 추천 서비스의 답변을 채점하는 심사자입니다. 길이나 말투가 아니라 아래 기준 충족 여부만 판단하세요.

[질문]
{question}

[서비스 답변]
{answer}

[추천 목록 (코드가 검증한 구조화 결과)]
{recs}

[독자의 실제 독서 기록 (제목 / 읽은 권 / 싫었어)]
{log}

[기대 특성 expected_traits] — 항목마다 verdicts에 '번호|근거 한 문장|O 또는 X' 한 줄씩 적으세요. 근거(답변의 실제 문장)를 먼저 찾고 판정은 마지막에 (충족 O, 미충족 X)
{traits}

[금지 사항 forbidden] — 답변이 어긴 항목만 violations에 한 줄에 하나씩 적고, 없으면 '없음'
{forbidden}

판정 원칙
- 독서 기록에 근거가 있는지는 위 [독서 기록]과 대조해 판단합니다. 기록에 없는 책을 읽었다고 말하면 위반입니다.
- 이미 읽은 책·아동 외 도서·목록에 없는 책 여부는 코드가 따로 검사하므로, 답변 문장에서 드러나는 것만 보세요.
- 질문이 거절·되묻기를 기대하면, 거절·되묻기 자체가 충족입니다.
- 기대 특성은 답변에 **문장으로 실제로 적혀 있어야** 충족입니다. 비슷한 말이나 추론으로 대신하지 않습니다 (예: '흥미진진'은 '웃김'이 아님).
- 애매하면 충족하지 않은 것으로 봅니다."""


def reading_log_text(readers: list[str]) -> str:
    lines = []
    for r in readers:
        log = get_reading_log_data(r, DEMO_OWNER)
        if not log.get("found"):
            continue
        lines.append(f"<{r}> {log['read_count']}권")
        for b in log["books"]:
            lines.append(f"- {b['log_title']} / {b['volumes_text'] or '-'} / {'싫었어' if b['disliked'] else ''}")
    return "\n".join(lines) or "(독서 기록 없음)"


async def judge(q: dict, res: dict) -> dict:
    readers = res.get("readers") or [r for r in ("아이", "나") if r in q["input"]]
    recs = "\n".join(f"- {r['title']} | {r['author']} | ISBN {r['isbn13']} | 부가기호 {r.get('addition_symbol')}"
                     for r in res.get("recommendations") or []) or "(없음)"
    traits, forbidden = split_items(q["expected_traits"]), split_items(q["forbidden"])
    prompt = JUDGE_PROMPT.format(question=q["input"], answer=res.get("answer", ""), recs=recs,
                                 log=reading_log_text(readers),
                                 traits="\n".join(f"{i}. {t}" for i, t in enumerate(traits, 1)),
                                 forbidden="\n".join(f"- {f}" for f in forbidden) or "(없음)")
    state = {"trace": []}

    async def run(llm):
        return await llm.with_structured_output(JudgeResult).ainvoke(prompt)

    r: JudgeResult = await call_model(state, "judge", run)
    verdicts, violations = parse_verdicts(r.verdicts, traits), parse_violations(r.violations)
    met = sum(v["met"] for v in verdicts)
    return {"passed": met == len(traits) and not violations, "traits_met": f"{met}/{len(traits)}",
            "score": r.score, "violations": violations, "summary": r.summary, "traits": verdicts}


# ---------------- 3. RAGAS ----------------

RAGAS_METRICS = ["context_recall", "llm_context_precision_with_reference", "faithfulness", "answer_relevancy"]


def _ragas_scores(rows: list[dict], route: tuple[str, str]) -> list[dict]:
    """rows를 RAGAS로 잰다. 반환: 행마다 {지표: 점수}. 실패한 칸은 NaN."""
    import warnings

    warnings.filterwarnings("ignore")
    from langchain_aws import BedrockEmbeddings
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    from agent import make_llm

    llm = LangchainLLMWrapper(make_llm(*route))
    emb = LangchainEmbeddingsWrapper(BedrockEmbeddings(model_id=EMBED_MODEL_ID, **bedrock_credentials()))
    metrics = [LLMContextRecall(llm=llm), LLMContextPrecisionWithReference(llm=llm),
               Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=emb)]
    ds = EvaluationDataset.from_list(rows)
    res = evaluate(ds, metrics=metrics, run_config=RunConfig(max_workers=4, timeout=240, max_retries=3),
                   show_progress=False, raise_exceptions=False)
    return res.to_pandas()[RAGAS_METRICS].to_dict("records")


def run_ragas(round_name: str) -> None:
    import math


    data = json.loads((RESULTS / f"{round_name}.json").read_text(encoding="utf-8"))
    targets = [r for r in data["results"] if r["response"].get("contexts")]
    rows = [{"user_input": r["input"], "response": r["response"]["answer"],
             "retrieved_contexts": [c["text"] for c in r["response"]["contexts"]],
             "reference": " / ".join(split_items(next(q for q in load_queries() if q["id"] == r["id"])["expected_traits"]))}
            for r in targets]
    print(f"RAGAS 대상 {len(rows)}문항 (contexts 있는 문항): {[r['id'] for r in targets]}", flush=True)
    t0 = time.time()
    from config import model_routes

    def is_nan(v):
        return isinstance(v, float) and math.isnan(v)

    # Sonnet 4.5 경로를 차례로: 점수가 빈 칸(호출 실패)이 있는 문항만 다음 경로로 다시 잰다 (모델은 그대로 Sonnet 4.5)
    # 11-19: 짧은 시험 호출은 통과하고 긴 측정에서 하루 한도에 걸리는 경우가 있어, 경로를 미리 하나로 고정하지 않는다
    scores = [{k: float("nan") for k in RAGAS_METRICS} for _ in rows]
    model_used = [""] * len(rows)
    todo = list(range(len(rows)))
    for route in model_routes():
        print(f"RAGAS 경로 {route[0]} @ {route[1]}: {len(todo)}문항", flush=True)
        got = _ragas_scores([rows[i] for i in todo], route)
        for i, s in zip(todo, got):
            for k in RAGAS_METRICS:
                if is_nan(scores[i][k]) and not is_nan(s[k]):
                    scores[i][k] = s[k]
                    model_used[i] = f"Sonnet 4.5 ({route[0]} @ {route[1]})"
        todo = [i for i in todo if any(is_nan(v) for v in scores[i].values())]
        if not todo:
            break
        print(f"⚠️ [ragas] 경로 {route[0]} @ {route[1]}에서 점수가 빈 문항 {[targets[i]['id'] for i in todo]} → 다음 Sonnet 경로로",
              file=sys.stderr, flush=True)
    empty = [targets[i]["id"] for i in todo]

    per_case = []
    for r, s, m in zip(targets, scores, model_used):
        clean = {k: (None if isinstance(v, float) and math.isnan(v) else round(float(v), 3)) for k, v in s.items()}
        per_case.append({"id": r["id"], "category": r["category"], "input": r["input"], "model": m, **clean})
    avg = {}
    for k in RAGAS_METRICS:
        vals = [c[k] for c in per_case if c[k] is not None]
        avg[k] = round(sum(vals) / len(vals), 3) if vals else None
    out = {"round": round_name, "metrics": RAGAS_METRICS, "average": avg, "n": len(per_case),
           "elapsed_s": round(time.time() - t0, 1), "reference_note": "reference = expected_traits 이어 붙임 (문제지에 정답 문장 없음)",
           "per_case": per_case}
    path = RESULTS / f"{round_name}_ragas.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    names = {"context_recall": "recall", "llm_context_precision_with_reference": "precision",
             "faithfulness": "faithful", "answer_relevancy": "relevancy"}
    print(f"\n{'id':>3} {'cat':<9}" + "".join(f"{names[k]:>10}" for k in RAGAS_METRICS))
    for c in per_case:
        print(f"{c['id']:>3} {c['category']:<9}" + "".join(f"{'-' if c[k] is None else c[k]:>10}" for k in RAGAS_METRICS))
    print(f"{'평균':>3} {'':<9}" + "".join(f"{'-' if avg[k] is None else avg[k]:>10}" for k in RAGAS_METRICS))
    print(f"저장: {path.relative_to(ROOT)} ({out['elapsed_s']}초)")
    if empty:  # Sonnet 호출이 실패한 칸: 다른 모델로 다시 재지 않고 멈춘 뒤 알린다
        print(f"⛔ [ragas] Sonnet 4.5 호출 실패로 점수가 빈 문항: {empty} → 멈춥니다. 연결을 확인한 뒤 다시 실행하세요.",
              file=sys.stderr)
        sys.exit(1)


# ---------------- 실행 ----------------

async def run_case(q: dict, books: dict, reuse: dict | None, keep_judge: bool = False) -> dict:
    if reuse:
        res, elapsed = reuse["response"], reuse["elapsed_s"]
    else:
        t0 = time.time()
        res = await answer_question(q["input"])
        elapsed = round(time.time() - t0, 1)
        if res.get("model_error"):  # Sonnet 연결 실패: 평가를 멈춘다 (앞 문항 결과는 이미 저장됨)
            raise SystemExit(f"⛔ {q['id']}번에서 Sonnet 4.5 연결 실패로 평가를 멈춥니다: {res['model_error']}")
    rules = rule_check(q, res, books)
    jd = reuse["judge"] if (reuse and keep_judge) else await judge(q, res)
    return {"id": q["id"], "category": q["category"], "input": q["input"],
            "passed": rules["passed"] and jd["passed"], "rule": rules, "judge": jd,
            "elapsed_s": elapsed, "response": res}


def summarize(results: list[dict]) -> dict:
    cats: dict[str, list[int]] = {}
    for r in results:
        c = cats.setdefault(r["category"], [0, 0])
        c[0] += r["passed"]
        c[1] += 1
    total = sum(r["passed"] for r in results)
    return {"total": f"{total}/{len(results)}", "pass_rate": round(total / max(len(results), 1), 3),
            "by_category": {k: f"{p}/{n}" for k, (p, n) in cats.items()},
            "rule_pass": f"{sum(r['rule']['passed'] for r in results)}/{len(results)}",
            "judge_pass": f"{sum(r['judge']['passed'] for r in results)}/{len(results)}",
            "avg_judge_score": round(sum(r["judge"]["score"] for r in results) / max(len(results), 1), 2),
            "avg_elapsed_s": round(sum(r["elapsed_s"] for r in results) / max(len(results), 1), 1),
            "model": "Sonnet 4.5"}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="round1")
    ap.add_argument("--ids", default="", help="다시 실행할 문항 번호 (예: 1,5)")
    ap.add_argument("--rejudge", action="store_true", help="저장된 답으로 채점만 다시")
    ap.add_argument("--rescore", action="store_true", help="저장된 답·판정으로 코드 채점만 다시")
    ap.add_argument("--ragas", action="store_true", help="저장된 답으로 RAGAS 4지표")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{args.round}.json"
    saved = {r["id"]: r for r in json.loads(out_path.read_text(encoding="utf-8"))["results"]} if out_path.exists() else {}
    ids = {s.strip() for s in args.ids.split(",") if s.strip()}
    books = snapshot_books()

    results = []
    for q in load_queries():
        if ids and q["id"] not in ids:
            if q["id"] in saved:
                results.append(saved[q["id"]])  # 고르지 않은 문항은 저장된 결과 그대로
            continue
        reuse = saved.get(q["id"]) if (args.rejudge or args.rescore) else None
        r = await run_case(q, books, reuse, keep_judge=args.rescore)
        results.append(r)
        mark = "PASS" if r["passed"] else "FAIL"
        bad = [c["name"] for c in r["rule"]["checks"] if not c["passed"]]
        print(f"[{mark}] {q['id']:>2} ({q['category']}) {r['elapsed_s']:>5}s 규칙{'✓' if r['rule']['passed'] else '✗' + str(bad)}"
              f" 판정 {r['judge']['traits_met']} 점수 {r['judge']['score']}"
              f"{' 위반' + str(r['judge']['violations']) if r['judge']['violations'] else ''}", flush=True)
        # 문항마다 저장: 중간에 멈춰도 이어서 할 수 있게
        out_path.write_text(json.dumps({"summary": summarize(results), "results": results}, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    s = summarize(results)
    out_path.write_text(json.dumps({"summary": s, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n===== {args.round}: {s['total']} 통과 ({s['pass_rate']:.0%}) =====")
    for k, v in s["by_category"].items():
        print(f"  {k}: {v}")
    print(f"  코드 채점 {s['rule_pass']} · LLM 판정 {s['judge_pass']} · 판정 평균 {s['avg_judge_score']}점 · 평균 {s['avg_elapsed_s']}초")
    print(f"  저장: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    if "--ragas" in sys.argv:  # RAGAS는 자체 이벤트 루프를 돌리므로 asyncio.run 밖에서 실행한다
        rnd = sys.argv[sys.argv.index("--round") + 1] if "--round" in sys.argv else "round1"
        run_ragas(rnd)
    else:
        asyncio.run(main())
