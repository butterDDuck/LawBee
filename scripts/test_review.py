"""심의 파이프라인 동작 확인용 스크립트

사전 조건: .env 에 OPENAI_API_KEY 설정 + `python -m app.ingest` 로 벡터스토어 생성

실행:
    python -m scripts.test_review "원금 100% 보장! 누구나 확정 수익 받는 펀드"
    python -m scripts.test_review "가입은 한 번에, 해지는 고객센터 전화로만" UI
"""
import sys

from app.graph import run_review


def main() -> None:
    content = sys.argv[1] if len(sys.argv) > 1 else "원금 100% 보장! 업계 1위 확정 수익 펀드, 누구나 가입 가능"
    media = sys.argv[2] if len(sys.argv) > 2 else None

    result = run_review(content, media=media)

    print(f"\n[심의 대상] {content}")
    print("=" * 70)
    print(f"판정: {result.status}")
    print(f"요약: {result.summary}")

    if result.rule_hits:
        print("\n[룰 엔진 탐지]")
        for h in result.rule_hits:
            print(f"  - ({h.severity}) {h.category}: '{h.term}'")

    if result.violations:
        print("\n[위반 항목]")
        for v in result.violations:
            print(f"  - {v.type} (근거: {', '.join(v.citation_ids) or '없음'})")
            print(f"    {v.reason}")

    if result.citations:
        print("\n[근거 조항]")
        for c in result.citations:
            print(f"  - [{c.id}] {c.law} {c.article}")

    if result.alternative_text:
        print("\n[대안 문구]")
        print(f"  {result.alternative_text}")


if __name__ == "__main__":
    main()
