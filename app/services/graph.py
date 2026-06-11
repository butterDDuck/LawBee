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
        "근거 조항은 반드시 제공된 조항 id 중에서만 인용하고, 제공되지 않은 사실을 지어내지 마세요. "
        "판정은 위반·주의·통과 중 하나로 하며, 명백한 법령 저촉은 '위반', 소지가 있으나 맥락 확인이 "
        "필요하면 '주의', 문제없으면 '통과'로 분류하세요. "
        "각 위반 항목의 type 에는 심각도가 아니라 위반 유형명(예: 수익 보장 단정, 부당 비교)을 적으세요."
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
