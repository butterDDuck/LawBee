"""저장된 FAISS 벡터스토어 로드 및 규제 조항 검색"""
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import settings


@lru_cache(maxsize=1)
def get_store() -> FAISS:
    """벡터스토어를 1회 로드 후 캐시"""
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embed_model,
        api_key=settings.openai_api_key,
    )
    return FAISS.load_local(
        settings.vectorstore_dir,
        embeddings,
        allow_dangerous_deserialization=True,  # 우리가 직접 생성한 신뢰된 파일
    )


def search(query: str, k: int = 5, media: str | None = None) -> list[dict]:
    """질의와 가장 관련 있는 규제 조항 top-k 반환

    media: '텍스트' | '영상' | 'UI' 등으로 콘텐츠 유형 필터링
           해당 매체 또는 '공통' 조항만 남김, None 이면 전체 검색
    """
    store = get_store()
    # 필터가 있으면 후보를 넉넉히 뽑은 뒤 파이썬에서 매체 매칭
    fetch_k = k * 4 if media else k
    hits = store.similarity_search_with_score(query, k=fetch_k)

    results = []
    for doc, score in hits:
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
                "score": float(score),
            }
        )
        if len(results) >= k:
            break
    return results
