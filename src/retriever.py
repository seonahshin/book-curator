"""RAG 파이프라인: 도서 목록 스냅샷 → 하이브리드 검색 (BM25 + 벡터).

- 문서 1개 = 책 1권: 제목 + 분류명 + 키워드 + 책소개
- BM25: kiwipiepy 형태소(명사·동사·형용사 등)로 낱말이 겹치는 책을 찾는다 (Day 2 토크나이저)
- 벡터: Bedrock Titan 임베딩 + Chroma(chroma_db/)로 뜻이 비슷한 책을 찾는다
- 섞기: 두 순위를 RRF(순위 역수 합)로 합친다. 가중치 BM25 0.3 : 벡터 0.7 (Day 2 hybrid.py와 같은 비율)
  Day 2는 EnsembleRetriever(langchain_classic)를 썼지만, 새 라이브러리를 늘리지 않으려고 같은 계산을 직접 한다.

임베딩은 스냅샷이 바뀌었을 때만 다시 만든다 (스냅샷 해시를 컬렉션 메타데이터에 저장).
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import EMBED_MODEL_ID, bedrock_credentials  # noqa: F401  (.env·접속 정보는 config 한곳)

ROOT = Path(__file__).resolve().parents[1]

SNAPSHOT = ROOT / "data" / "catalog_snapshot.json"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "catalog"


WEIGHTS = {"bm25": 0.3, "vector": 0.7}
RRF_K = 60  # RRF 상수 (순위가 조금 다를 때 점수 차이를 완만하게)
POOL = 60   # 각 검색기에서 가져올 후보 수

_KEEP_TAGS = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "SL", "SN"}


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def korean_tokenizer(text: str) -> list[str]:
    """한국어 형태소 단위 토크나이저 (Day 2 검증본). 명사·어간·숫자·외래어만 남긴다."""
    return [t.form for t in _kiwi().tokenize(text) if t.tag in _KEEP_TAGS]


def book_text(b: dict, max_keywords: int = 20) -> str:
    """검색용 문서: 제목·분류·키워드·책소개를 한 덩어리로."""
    words = ", ".join(k["word"] for k in (b.get("keywords") or [])[:max_keywords])
    parts = [
        f"제목: {b.get('title') or ''}",
        f"분류: {b.get('class_nm') or ''}",
        f"키워드: {words}",
        f"소개: {b.get('description') or ''}",
    ]
    return "\n".join(parts)


@lru_cache(maxsize=1)
def load_books() -> tuple[dict, ...]:
    """스냅샷의 책 목록 (키워드를 받은 책만)."""
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return tuple(data["books"])


def _snapshot_hash() -> str:
    return hashlib.sha1(SNAPSHOT.read_bytes()).hexdigest()[:12]


@lru_cache(maxsize=1)
def _embeddings():
    from langchain_aws import BedrockEmbeddings

    return BedrockEmbeddings(model_id=EMBED_MODEL_ID, **bedrock_credentials())


@lru_cache(maxsize=1)
def vector_store():
    """Chroma 벡터 DB. 스냅샷이 바뀌었으면 지우고 다시 만든다."""
    from langchain_chroma import Chroma

    digest = _snapshot_hash()
    store = Chroma(collection_name=COLLECTION, embedding_function=_embeddings(),
                   persist_directory=str(CHROMA_DIR), collection_metadata={"snapshot": digest})
    meta = store._collection.metadata or {}
    if meta.get("snapshot") == digest and store._collection.count() == len(load_books()):
        return store

    print(f"[retriever] 임베딩 만들기: {len(load_books())}권 (스냅샷 {digest})")
    store.delete_collection()
    store = Chroma(collection_name=COLLECTION, embedding_function=_embeddings(),
                   persist_directory=str(CHROMA_DIR), collection_metadata={"snapshot": digest})
    books = load_books()
    store.add_texts(
        texts=[book_text(b) for b in books],
        metadatas=[{"isbn13": b["isbn13"]} for b in books],
        ids=[b["isbn13"] for b in books],
    )
    return store


@lru_cache(maxsize=1)
def _bm25():
    from rank_bm25 import BM25Okapi

    return BM25Okapi([korean_tokenizer(book_text(b)) for b in load_books()])


def bm25_search(query: str, n: int = POOL) -> list[str]:
    """BM25 상위 n권의 ISBN (점수 0인 책은 뺀다)."""
    books = load_books()
    scores = _bm25().get_scores(korean_tokenizer(query))
    order = sorted(range(len(books)), key=lambda i: -scores[i])
    return [books[i]["isbn13"] for i in order[:n] if scores[i] > 0]


def vector_search(query: str, n: int = POOL) -> list[str]:
    """벡터 검색 상위 n권의 ISBN."""
    return [d.metadata["isbn13"] for d in vector_store().similarity_search(query, k=n)]


def hybrid_search(query: str, k: int = 10) -> list[dict[str, Any]]:
    """BM25와 벡터 순위를 RRF로 섞어 상위 k권을 돌려준다.

    반환: [{isbn13, title, score, bm25_rank, vector_rank, text, book}]
    """
    by_isbn = {b["isbn13"]: b for b in load_books()}
    ranks = {"bm25": bm25_search(query), "vector": vector_search(query)}
    score: dict[str, float] = {}
    pos: dict[str, dict[str, int]] = {}
    for name, isbns in ranks.items():
        for r, isbn in enumerate(isbns, start=1):
            score[isbn] = score.get(isbn, 0.0) + WEIGHTS[name] / (RRF_K + r)
            pos.setdefault(isbn, {})[name] = r
    top = sorted(score, key=lambda i: -score[i])[:k]
    return [
        {
            "isbn13": i,
            "title": by_isbn[i]["title"],
            "score": round(score[i], 5),
            "bm25_rank": pos[i].get("bm25"),
            "vector_rank": pos[i].get("vector"),
            "text": book_text(by_isbn[i]),
            "book": by_isbn[i],
        }
        for i in top
    ]


if __name__ == "__main__":
    q = "웃긴 모험 이야기"
    print(f"질문: {q}")
    for h in hybrid_search(q, k=5):
        b = h["book"]
        print(f"  - {h['title'][:30]} | 부가기호 {b.get('addition_symbol')} | {b.get('class_nm')} "
              f"| BM25 {h['bm25_rank']}위, 벡터 {h['vector_rank']}위")
