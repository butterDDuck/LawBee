"""Hybrid(BM25 + FAISS + RRF) 검색"""
import json
from functools import lru_cache

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import settings


@lru_cache(maxsize=1)
def _load_docs() -> list[Document]:
    """regulations.jsonl → Document 리스트"""
    docs = []
    with open(settings.data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            docs.append(
                Document(
                    page_content=rec["text"],
                    metadata={
                        "id": rec["id"],
                        "category": rec["category"],
                        "law": rec["law"],
                        "article": rec["article"],
                        "media": ",".join(rec["media"]),
                        "compliance_check": rec["compliance_check"],
                        "source_law": rec["source_law"],
                        "source_url": rec["source_url"],
                    },
                )
            )
    return docs


@lru_cache(maxsize=1)
def _kiwi() -> Kiwi:
    return Kiwi()


def _tokenize(text: str) -> list[str]:
    """형태소 분석 기반 토크나이징 — 명사·동사·형용사·부사만 추출"""
    return [
        token.form
        for token in _kiwi().tokenize(text)
        if token.tag[:2] in ("NN", "VV", "VA", "MA", "XR")
    ]


@lru_cache(maxsize=1)
def _get_bm25() -> BM25Okapi:
    """BM25 인덱스 — 형태소 분석 기반 토크나이징"""
    docs = _load_docs()
    return BM25Okapi([_tokenize(d.page_content) for d in docs])


@lru_cache(maxsize=1)
def _get_faiss() -> FAISS:
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embed_model,
        api_key=settings.openai_api_key,
    )
    return FAISS.load_local(
        settings.vectorstore_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion — 여러 순위 리스트를 하나로 합산"""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


def search(
    query: str,
    k: int = 5,
    media: str | None = None,
) -> list[dict]:
    """Hybrid(BM25+FAISS) → RRF → media 필터 → top-k 반환

    media: '텍스트' | '영상' | 'UI' — 해당 매체 또는 '공통' 조항만 포함
    None 이면 필터 없이 전체 검색
    """
    fetch_k = k * 6
    docs = _load_docs()

    # 1) BM25 검색 — 형태소 분석으로 쿼리 토크나이징
    bm25_scores = _get_bm25().get_scores(_tokenize(query))
    bm25_ranking = sorted(range(len(docs)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_k]

    # 2) FAISS 검색
    faiss_hits = _get_faiss().similarity_search_with_score(query, k=fetch_k)
    doc_text_to_idx = {d.page_content: i for i, d in enumerate(docs)}
    faiss_ranking = [
        doc_text_to_idx[doc.page_content]
        for doc, _ in faiss_hits
        if doc.page_content in doc_text_to_idx
    ]

    # 3) RRF 결합 (FAISS 비중 높게 — 순위 리스트 2회 포함)
    combined = _rrf([bm25_ranking, faiss_ranking, faiss_ranking])

    # 4) media 필터 및 결과 구성
    results = []
    for idx in combined:
        doc = docs[idx]
        doc_media = doc.metadata.get("media", "")
        if media and media not in doc_media and "공통" not in doc_media:
            continue
        results.append(
            {
                "id": doc.metadata.get("id"),
                "category": doc.metadata.get("category"),
                "law": doc.metadata.get("law"),
                "article": doc.metadata.get("article"),
                "media": doc_media,
                "content": doc.page_content,
                "compliance_check": doc.metadata.get("compliance_check"),
                "source_url": doc.metadata.get("source_url"),
                "score": 0.0,
            }
        )
        if len(results) >= k:
            break

    return results
