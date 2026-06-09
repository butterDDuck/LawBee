"""결정론적 룰 엔진

lexicon 의 패턴으로 콘텐츠에서 정형 위반 표현을 1차 탐지
RAG 검색만으로는 변별이 약한 단정·최상급·보편적용 표현을 확실히 잡는 역할
"""
from app.domain.schema import RuleHit
from app.rules.lexicon import ABSENCE_RULES, RULES


def apply_rules(content: str) -> list[RuleHit]:
    """콘텐츠에서 룰에 매칭되는 표현을 모두 탐지하여 반환"""
    hits: list[RuleHit] = []
    seen: set[tuple[str, str]] = set()

    for rule in RULES:
        for m in rule.pattern.finditer(content):
            term = m.group(0).strip()
            key = (term, rule.category)
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                RuleHit(
                    term=term,
                    category=rule.category,
                    severity=rule.severity,
                    message=rule.message,
                )
            )

    for rule in ABSENCE_RULES:
        if rule.context_pattern.search(content) and not rule.required_pattern.search(content):
            key = (rule.category, rule.category)
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                RuleHit(
                    term=f"[{rule.category}]",
                    category=rule.category,
                    severity=rule.severity,
                    message=rule.message,
                )
            )

    return hits
