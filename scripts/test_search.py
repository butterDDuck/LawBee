"""RAG 검색 동작 확인용 스크립트

사전 조건: .env 에 OPENAI_API_KEY 설정 + `python -m app.ingest` 로 벡터스토어 생성

실행:
    python -m scripts.test_search "원금 보장 100% 확정 수익 보장하는 펀드"
"""
import sys

from app.retriever import search


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "누구나 무조건 대출 가능, 업계 최저 금리"
    media = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\n질의: {query}")
    if media:
        print(f"매체 필터: {media}")
    print("=" * 70)

    for i, r in enumerate(search(query, k=5, media=media), 1):
        print(f"\n[{i}] ({r['score']:.3f}) {r['id']} | {r['category']}")
        print(f"    {r['law']} {r['article']}")
        print(f"    심의 포인트: {r['compliance_check'][:120]}...")


if __name__ == "__main__":
    main()
