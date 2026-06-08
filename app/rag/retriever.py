"""Hybrid(BM25 + FAISS) 검색 + Cohere Rerank"""
import json
from functools import lru_cache

import cohere
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import settings


# --- 인덱스 초기화 (앱 시작 시 1회) ---

@lru_cache(maxsize=1)
def _load_docs() -> list[Document]:
    """regulations.jsonl → Document 리스트 (BM25·FAISS 공유)"""
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
def _get_bm25() -> BM25Okapi:
    """BM25 인덱스 — 공백 토크나이징"""
    docs = _load_docs()
    tokenized = [d.page_content.split() for d in docs]
    return BM25Okapi(tokenized)


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


@lru_cache(maxsize=1)
def _cohere_client() -> cohere.Client:
    return cohere.Client(api_key=settings.cohere_api_key)


# --- RRF 결합 ---

def _rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion — 여러 순위 리스트를 하나로 합산"""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


# --- 공개 인터페이스 ---

def search(
    query: str,
    k: int = 5,
    media: str | None = None,
) -> list[dict]:
    """Hybrid(BM25+FAISS) → Cohere Rerank → media 필터 → top-k 반환

    media: '텍스트' | '영상' | 'UI' — 해당 매체 또는 '공통' 조항만 포함
    None 이면 필터 없이 전체 검색
    """
    fetch_k = k * 6
    docs = _load_docs()

    # 1) BM25 검색
    bm25_scores = _get_bm25().get_scores(query.split())
    bm25_ranking = sorted(range(len(docs)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_k]

    # 2) FAISS 검색
    faiss_hits = _get_faiss().similarity_search_with_score(query, k=fetch_k)
    # FAISS 결과를 docs 인덱스로 변환
    doc_text_to_idx = {d.page_content: i for i, d in enumerate(docs)}
    faiss_ranking = [
        doc_text_to_idx[doc.page_content]
        for doc, _ in faiss_hits
        if doc.page_content in doc_text_to_idx
    ]

    # 3) RRF 결합 (BM25 0.4 : FAISS 0.6 비중 반영 — FAISS 순위를 더 많이 복제)
    combined = _rrf([bm25_ranking, faiss_ranking, faiss_ranking])

    # 4) Cohere Rerank
    candidates = [docs[i] for i in combined[:fetch_k]]
    if candidates and settings.cohere_api_key:
        resp = _cohere_client().rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=[d.page_content for d in candidates],
            top_n=min(len(candidates), k * 2),
        )
        candidates = [candidates[r.index] for r in resp.results]

    # 5) media 필터 및 결과 구성
    results = []
    for doc in candidates:
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
