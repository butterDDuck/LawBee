"""lexicon 패턴 검증 스크립트"""
import sys
sys.path.insert(0, ".")
from app.rules.lexicon import RULES, ABSENCE_RULES


def run_test(text: str, expected: list[str]) -> bool:
    found = []
    for rule in RULES:
        if rule.pattern.search(text):
            found.append(rule.category)
    for rule in ABSENCE_RULES:
        if rule.context_pattern.search(text) and not rule.required_pattern.search(text):
            found.append(rule.category)
    ok = all(e in found for e in expected)
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {text}")
    print(f"       탐지: {found}")
    return ok


cases = [
    ("원금 보장 확정수익 펀드",           ["원금·손실 단정", "수익 보장 단정"]),
    ("연 8% 수익 확정 상품",              ["수익률 단정"]),
    ("지금이 기회입니다 지금 사야 합니다", ["단정적 투자 권유"]),
    ("100명이 지금 신청 중",              ["소비자 활동 알림"]),
    ("이 기회를 놓치면 후회하게 됩니다",   ["감정적 언어"]),
    ("마감 30분 후",                      ["인위적 시간·수량 제한"]),
    ("특별 이벤트 가입하시면 혜택 제공",   ["수신거부 안내 누락"]),
    ("ETF 투자 상품 고수익",              ["투자 위험 미고지"]),
    ("무손실 대박 투자 상품",             ["단정적 수익 표현"]),
]

results = [run_test(text, expected) for text, expected in cases]
print(f"\n결과: {sum(results)}/{len(results)} 통과")
