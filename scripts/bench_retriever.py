"""Dense 단독 vs Hybrid vs Hybrid+Rerank 검색 성능 비교"""
import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.config import settings
from app.rag.retriever import search, _rrf


def _load_all_docs():
    docs = []
    with open(settings.data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            docs.append(Document(page_content=rec["text"], metadata={"id": rec["id"]}))
    return docs


def dense_search(query: str, k: int = 10) -> list[str]:
    emb = OpenAIEmbeddings(model=settings.openai_embed_model, api_key=settings.openai_api_key)
    store = FAISS.load_local(settings.vectorstore_dir, emb, allow_dangerous_deserialization=True)
    hits = store.similarity_search_with_score(query, k=k)
    return [doc.metadata.get("id") for doc, _ in hits]


def hybrid_search(query: str, k: int = 10) -> list[str]:
    """BM25 + FAISS + RRF — Cohere 없이"""
    fetch_k = k * 6
    docs = _load_all_docs()
    doc_text_to_idx = {d.page_content: i for i, d in enumerate(docs)}

    bm25 = BM25Okapi([d.page_content.split() for d in docs])
    bm25_scores = bm25.get_scores(query.split())
    bm25_ranking = sorted(range(len(docs)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_k]

    emb = OpenAIEmbeddings(model=settings.openai_embed_model, api_key=settings.openai_api_key)
    store = FAISS.load_local(settings.vectorstore_dir, emb, allow_dangerous_deserialization=True)
    faiss_hits = store.similarity_search_with_score(query, k=fetch_k)
    faiss_ranking = [
        doc_text_to_idx[doc.page_content]
        for doc, _ in faiss_hits
        if doc.page_content in doc_text_to_idx
    ]

    combined = _rrf([bm25_ranking, faiss_ranking, faiss_ranking])
    return [docs[i].metadata["id"] for i in combined[:k]]


TEST_CASES = [
    ("수신거부 안내 없는 SMS 마케팅 발송",      "ICTNA-50"),
    ("원금 보장 확정수익 펀드 투자광고",         "FCPA-21-단정"),
    ("대출 이자율 범위 표시 의무",               "FAG-17-대출필수표기"),
    ("타사 대비 최저금리 근거 없는 비교 광고",   "FAD-3"),
    ("마케팅 동의 강제 선택 불가 UI 다크패턴",   "PIPA-22-5"),
    ("카드론 할부금융 연이율 표시 누락",         "ASSOC-CREFIA-01"),
    ("팝업 반복 노출 거부 후 재요구 반복간섭",   "DARK-11"),
]

K = 10
header = f"{'쿼리':<38} {'기대 ID':<25} {'Dense':>8} {'Hybrid':>8} {'Hybrid+Rerank':>15}"
print(header)
print("-" * len(header))

d_hits = h_hits = hr_hits = 0

for query, expected in TEST_CASES:
    d_ids  = dense_search(query, k=K)
    hy_ids = hybrid_search(query, k=K)
    hr_ids = [r["id"] for r in search(query, k=K)]

    d_rank  = str(d_ids.index(expected)  + 1) if expected in d_ids  else f">{K}"
    h_rank  = str(hy_ids.index(expected) + 1) if expected in hy_ids else f">{K}"
    hr_rank = str(hr_ids.index(expected) + 1) if expected in hr_ids else f">{K}"

    if expected in d_ids:  d_hits  += 1
    if expected in hy_ids: h_hits  += 1
    if expected in hr_ids: hr_hits += 1

    print(f"{query:<38} {expected:<25} {d_rank:>8} {h_rank:>8} {hr_rank:>15}")

print("-" * len(header))
print(f"{'Recall@10':<71} {d_hits}/{len(TEST_CASES)} {h_hits}/{len(TEST_CASES)}    {hr_hits}/{len(TEST_CASES)}")
