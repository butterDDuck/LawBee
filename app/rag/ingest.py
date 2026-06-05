"""규제 데이터 적재 파이프라인

data/regulations.jsonl  →  OpenAI 임베딩  →  FAISS 벡터스토어 저장

실행:
    python -m app.rag.ingest
"""
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import settings


def load_chunks(path: str) -> list[Document]:
    """jsonl 각 줄을 LangChain Document 로 변환

    page_content 는 임베딩 대상인 `text` 필드를 사용하고
    나머지 필드는 metadata 로 보존해 검색 후 근거 제시·필터링에 활용
    """
    docs: list[Document] = []
    with open(path, encoding="utf-8") as f:
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
                        # FAISS metadata 필터는 스칼라가 안전하므로 콤마 문자열로 보관
                        "media": ",".join(rec["media"]),
                        "compliance_check": rec["compliance_check"],
                        "source_law": rec["source_law"],
                        "source_url": rec["source_url"],
                    },
                )
            )
    return docs


def build() -> None:
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY 가 설정되지 않았습니다. .env 를 확인하세요.")

    docs = load_chunks(settings.data_path)
    print(f"[ingest] {len(docs)}개 청크 로드 완료 → 임베딩 시작...")

    embeddings = OpenAIEmbeddings(
        model=settings.openai_embed_model,
        api_key=settings.openai_api_key,
    )
    store = FAISS.from_documents(docs, embeddings)

    out = Path(settings.vectorstore_dir)
    out.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out))
    print(f"[ingest] 벡터스토어 저장 완료 → {out}/ (index.faiss, index.pkl)")


if __name__ == "__main__":
    build()
