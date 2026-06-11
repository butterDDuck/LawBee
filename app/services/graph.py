"""LangGraph 기반 준법심의 파이프라인

흐름: 룰 엔진 → 규제 조항 검색 → LLM 판단 → (위반·주의 시) 대안 문구 생성
노드 흐름은 LangGraph 가 제어하고, 노드 내부는 LangChain 컴포넌트를 사용
"""
from functools import lru_cache
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from app.config import settings
from app.rules import apply_rules
from app.rag.retriever import search
from app.domain.schema import (
    Citation,
    Judgment,
    ReviewResult,
    RuleHit,
)


class State(TypedDict, total=False):
    content: str
    media: str | None
    rule_hits: list[RuleHit]
    retrieved: list[dict]
    judgment: Judgment
    alternative: str | None


@lru_cache(maxsize=1)
def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_retries=6,
    )


# --- 노드 정의 ---

def rule_node(state: State) -> State:
    """룰 엔진으로 정형 위반 표현 1차 탐지"""
    return {"rule_hits": apply_rules(state["content"])}


def retrieve_node(state: State) -> State:
    """콘텐츠와 룰 탐지 유형을 합쳐 관련 규제 조항 검색"""
    hit_terms = " ".join(h.category for h in state.get("rule_hits", []))
    query = f"{state['content']} {hit_terms}".strip()
    chunks = search(query, k=5, media=state.get("media"))
    return {"retrieved": chunks}


def _format_context(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        lines.append(
            f"[{c['id']}] {c['law']} {c['article']}\n"
            f"  규정: {c['content'][:300]}\n"
            f"  심의 포인트: {c['compliance_check'][:200]}"
        )
    return "\n\n".join(lines)


def judge_node(state: State) -> State:
    """규제 조항과 룰 탐지 결과를 근거로 LLM 이 위반 여부 판단"""
    chunks = state.get("retrieved", [])
    rule_summary = "\n".join(
        f"- ({h.severity}) {h.category}: '{h.term}' — {h.message}"
        for h in state.get("rule_hits", [])
    ) or "(룰 탐지 없음)"

    system = (
        "당신은 금융회사의 마케팅 콘텐츠를 심의하는 준법 심의역입니다. "
        "아래 제공된 규제 조항과 룰 엔진 탐지 결과만을 근거로 콘텐츠의 위반 여부를 판단하세요. "
        "근거 조항은 반드시 제공된 조항 id 중에서만 인용하고, 제공되지 않은 사실을 지어내지 마세요.\n\n"

        "【판정 기준】\n"
        "- 위반: 법령·감독 규정에 명백히 저촉되는 표현이 존재하는 경우\n"
        "- 주의: 위반 소지가 있으나 전체 맥락에 따라 달라질 수 있는 경우, 또는 필수 고지 누락이 의심되는 경우\n"
        "- 통과: 규제 조항에 저촉되는 표현이 없고 필수 고지가 적절히 포함된 경우\n\n"

        "【위반 유형명 예시 — type 필드에 사용】\n"
        "수익 보장 단정 / 원금보장 표현 / 수익률 단정 / 근거 없는 최상급 표현 / 부당 비교 / "
        "보편적용 오인 / 투자 위험 미고지 / 필수 고지 누락 / 압박형 다크패턴 / 감정적 언어 사용 / "
        "허위·과장 광고 / 부당권유\n\n"

        "【few-shot 판정 예시】\n"
        "예1) 콘텐츠: '연 8% 확정 수익 보장 적금'\n"
        "→ 판정: 위반 / type: 수익 보장 단정 / 이유: '확정 수익 보장'은 불확실한 사항에 대한 단정적 표현으로 "
        "금소법 제21조 제1호 위반\n\n"
        "예2) 콘텐츠: '업계 최저 수수료 ETF — 수익률은 시장 상황에 따라 변동될 수 있으며 원금 손실이 발생할 수 있습니다'\n"
        "→ 판정: 주의 / type: 근거 없는 최상급 표현 / 이유: '업계 최저 수수료' 표현에 비교 기준·출처 없음. "
        "단, 위험 고지는 적절히 포함됨\n\n"
        "예3) 콘텐츠: '이 적금은 연 3.5% 금리(기본 2.5% + 우대 1.0%, 급여이체 시)이며 예금자보호법에 따라 "
        "1인당 최고 5천만원까지 보호됩니다'\n"
        "→ 판정: 통과 / 이유: 기본·우대금리 구분, 우대 조건 명시, 예금자보호 한도 고지 모두 충족\n\n"

        "각 위반 항목의 type 에는 심각도가 아니라 위반 유형명을 적으세요."
    )
    human = (
        f"[심의 대상 콘텐츠]\n{state['content']}\n\n"
        f"[룰 엔진 탐지 결과]\n{rule_summary}\n\n"
        f"[관련 규제 조항]\n{_format_context(chunks)}"
    )

    judgment = _llm().with_structured_output(Judgment).invoke(
        [("system", system), ("human", human)]
    )
    return {"judgment": judgment}


def alternative_node(state: State) -> State:
    """위반·주의 콘텐츠에 대해 준법 통과 가능한 대안 문구 생성"""
    judgment = state["judgment"]
    violation_summary = "; ".join(f"{v.type}: {v.reason}" for v in judgment.violations)
    system = (
        "당신은 금융 마케팅 카피라이터이자 준법 전문가입니다. "
        "원문의 마케팅 의도는 살리되, 지적된 위반 사유를 모두 해소한 대안 문구를 한국어로 작성하세요. "
        "과장·단정·보편적용 표현을 제거하고 필요한 고지를 반영하세요. 대안 문구만 출력하세요."
    )
    import re as _re
    # 음성 자막 텍스트만 추출 — [화면 분석] 섹션 및 내부 레이블 제거
    raw = state["content"]
    voice = _re.search(r'\[음성 자막\](.*?)(?=\[화면 분석\]|$)', raw, _re.S)
    clean_content = voice.group(1).strip() if voice else raw
    human = (
        f"[원문]\n{clean_content}\n\n"
        f"[해소해야 할 위반 사유]\n{violation_summary or judgment.summary}"
    )
    resp = _llm().invoke([("system", system), ("human", human)])
    return {"alternative": resp.content.strip()}


def _route_after_judge(state: State) -> str:
    """통과면 종료, 그 외엔 대안 문구 생성으로 분기"""
    return "end" if state["judgment"].status == "통과" else "alternative"


# --- 그래프 구성 ---

@lru_cache(maxsize=1)
def _compiled():
    g = StateGraph(State)
    g.add_node("rule", rule_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("judge", judge_node)
    g.add_node("alternative", alternative_node)

    g.set_entry_point("rule")
    g.add_edge("rule", "retrieve")
    g.add_edge("retrieve", "judge")
    g.add_conditional_edges("judge", _route_after_judge, {"alternative": "alternative", "end": END})
    g.add_edge("alternative", END)
    return g.compile()


def run_review(content: str, media: str | None = None) -> ReviewResult:
    """심의 파이프라인 실행 후 구조화된 결과 반환"""
    final: State = _compiled().invoke({"content": content, "media": media})

    judgment = final["judgment"]
    # 판단이 인용한 조항만 근거로 정리
    cited_ids = {cid for v in judgment.violations for cid in v.citation_ids}
    citations = [
        Citation(id=c["id"], law=c["law"], article=c["article"], source_url=c.get("source_url"))
        for c in final.get("retrieved", [])
        if c["id"] in cited_ids
    ]

    return ReviewResult(
        status=judgment.status,
        summary=judgment.summary,
        rule_hits=final.get("rule_hits", []),
        violations=judgment.violations,
        citations=citations,
        alternative_text=final.get("alternative"),
    )
