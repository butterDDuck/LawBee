"""Dense 단독 vs Hybrid+Rerank 검색 성능 비교"""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import settings
from app.rag.retriever import search


def dense_search(query: str, k: int = 10) -> list[str]:
    emb = OpenAIEmbeddings(
        model=settings.openai_embed_model,
        api_key=settings.openai_api_key,
    )
    store = FAISS.load_local(
        settings.vectorstore_dir, emb, allow_dangerous_deserialization=True
    )
    hits = store.similarity_search_with_score(query, k=k)
    return [doc.metadata.get("id") for doc, _ in hits]


# 쿼리 / 기대 상위 id
TEST_CASES = [
    ("수신거부 안내 없는 SMS 마케팅 발송",          "ICTNA-50"),
    ("원금 보장 확정수익 펀드 투자광고",             "FSCMA-57"),
    ("대출 이자율 범위 표시 의무",                   "FAG-17-대출필수표기"),
    ("비교 광고 근거 없는 타사 대비 최저금리",       "FAG-04-비교광고"),
    ("마케팅 동의 강제 선택 불가 UI 다크패턴",       "PIPA-22-5"),
]

K = 10
header = f"{'쿼리':<38} {'기대 ID':<25} {'Dense':>8} {'Hybrid+Rerank':>15}"
print(header)
print("-" * len(header))

dense_hits = 0
hybrid_hits = 0

for query, expected in TEST_CASES:
    d_ids = dense_search(query, k=K)
    h_ids = [r["id"] for r in search(query, k=K)]

    d_rank = str(d_ids.index(expected) + 1) if expected in d_ids else f">{K}"
    h_rank = str(h_ids.index(expected) + 1) if expected in h_ids else f">{K}"

    if expected in d_ids:
        dense_hits += 1
    if expected in h_ids:
        hybrid_hits += 1

    print(f"{query:<38} {expected:<25} {d_rank:>8} {h_rank:>15}")

print("-" * len(header))
print(f"{'Recall@10 (기대 조항 포함률)':<63} {dense_hits}/{len(TEST_CASES)}    {hybrid_hits}/{len(TEST_CASES)}")
