"""LangGraph 기반 준법심의 파이프라인

흐름:
  룰 엔진 → 규제 조항 검색 → LLM 판단 → [reflect: 근거 충분성 검증]
  → (통과) END
  → (위반·주의) 대안 문구 생성 → [re_judge: 대안 재심의]
      → (통과) END
      → (실패, 최대 3회) 이전 피드백 포함 대안 재생성 → 루프
      → (3회 초과) auto_fix_failed 플래그로 END

에이전트 루프: 판단(judge) → 행동(alternative) → 검증(re_judge) → 개선(alternative 재시도)
"""
from __future__ import annotations

import re as _re
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
    ReviewMode,
    ReviewResult,
    RuleHit,
)

_MAX_ALT_RETRIES = 3  # 대안 문구 재생성 최대 횟수


_MODE_GUIDE: dict[str, str] = {
    "강화": (
        "【심의 강도: 강화】 법령 저촉이 명백하지 않더라도 소지가 있으면 '위반'으로 분류하세요. "
        "주의 수준의 표현도 위반으로 상향 판정하고, 작은 불명확성도 놓치지 마세요. "
        "사전 예방 목적의 엄격한 심의입니다."
    ),
    "표준": (
        "【심의 강도: 표준】 명백한 법령 저촉은 '위반', 소지가 있으나 맥락에 따라 달라질 수 있으면 '주의', "
        "문제없으면 '통과'로 분류하세요."
    ),
    "완화": (
        "【심의 강도: 완화】 명백한 법령 저촉이 확인된 경우에만 '위반'으로 분류하세요. "
        "단순 소지나 해석 여지가 있는 경우는 '주의' 또는 '통과'로 처리하고, "
        "실무 맥락상 통용되는 표현은 관대하게 판단하세요."
    ),
    "AI추천": (
        "【심의 강도: AI 자동 결정】 콘텐츠의 금융상품 유형, 매체, 위험도를 먼저 분석한 뒤 "
        "적절한 심의 강도를 스스로 결정하세요. 고위험 투자상품·영상 광고는 강화, "
        "단순 예금·텍스트 안내는 표준~완화를 적용하고, 결정한 강도를 summary 첫 줄에 명시하세요."
    ),
}


class State(TypedDict, total=False):
    content: str
    media: str | None
    review_mode: str
    rule_hits: list[RuleHit]
    retrieved: list[dict]
    judgment: Judgment
    # reflect 노드가 판단 근거 불충분을 감지하면 True → retrieve 재실행
    needs_reretrieval: bool
    alternative: str | None
    # 대안 재심의 결과
    alt_judgment: Judgment | None
    # 대안 재생성 시 이전 실패 이유 누적
    alt_feedback: list[str]
    # 대안 재생성 시도 횟수
    alt_retries: int
    # 3회 초과 후에도 통과 못하면 True
    auto_fix_failed: bool


@lru_cache(maxsize=1)
def _llm() -> ChatOpenAI:
    """대안 문구 생성용 — 비용 우선 모델"""
    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_retries=6,
    )


@lru_cache(maxsize=1)
def _judge_llm() -> ChatOpenAI:
    """심의·재심의 판단용 — 정확도 우선 모델"""
    return ChatOpenAI(
        model=settings.openai_judge_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_retries=6,
    )


# --- 노드 정의 ---

def rule_node(state: State) -> State:
    """룰 엔진으로 정형 위반 표현 1차 탐지"""
    return {"rule_hits": apply_rules(state["content"])}


def _dedup_by_id(chunks: list[dict], cap: int = 14) -> list[dict]:
    """id 기준 중복 제거 — 앞쪽(집중 검색 결과) 우선, 최대 cap 개"""
    seen: set = set()
    out: list[dict] = []
    for c in chunks:
        cid = c.get("id")
        if cid in seen:
            continue
        seen.add(cid)
        out.append(c)
        if len(out) >= cap:
            break
    return out


def retrieve_node(state: State) -> State:
    """위반 유형별 집중 검색 + 콘텐츠 전반 검색을 병합
    단일 쿼리로 여러 위반의 정답 조항을 담기엔 신호가 희석되므로,
    룰 탐지 유형마다 별도 검색해 각 위반의 정답 조항을 top 으로 확보"""
    rule_hits = state.get("rule_hits", [])
    media = state.get("media")
    chunks: list[dict] = []
    seen_cats: set = set()
    for h in rule_hits:
        if h.category in seen_cats or len(seen_cats) >= 6:
            continue
        seen_cats.add(h.category)
        chunks += search(f"{h.category} {h.term} {h.message}", k=2, media=media,
                         boost_terms=[h.category, h.term])
    # 콘텐츠 전반 검색 — 룰이 못 잡은 시각·해석형 위반 대비
    chunks += search(state["content"], k=6, media=media)
    return {"retrieved": _dedup_by_id(chunks), "needs_reretrieval": False}


def _format_context(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        lines.append(
            f"[{c['id']}] {c['law']} {c['article']}\n"
            f"  규정: {c['content'][:300]}\n"
            f"  심의 포인트: {c['compliance_check'][:200]}"
        )
    return "\n\n".join(lines)


def _judge_prompt(mode: str) -> str:
    mode_guide = _MODE_GUIDE.get(mode, _MODE_GUIDE["표준"])
    return (
        "당신은 금융회사의 마케팅 콘텐츠를 심의하는 준법 심의역입니다. "
        "아래 제공된 규제 조항과 룰 엔진 탐지 결과만을 근거로 콘텐츠의 위반 여부를 판단하세요. "
        "근거 조항은 반드시 제공된 조항 id 중에서만 인용하고, 제공되지 않은 사실을 지어내지 마세요.\n\n"
        f"{mode_guide}\n\n"
        "【조항 인용 원칙 — 반드시 준수】\n"
        "- 가장 직접적이고 구체적인 하위규정을 1순위로 인용: 감독규정 > 시행령 > 지정고시·가이드라인 > 상위법\n"
        "- '누구나', '누구든지', '묻지마' 등 보편적용 오인 표현의 직접 근거는 반드시 감독규정 제19조(FCPA-감독규정19). "
        "FSCPA-21-4는 비교광고 오인 조항이므로 보편적용 오인에는 인용하지 말 것\n"
        "- '업계 1위', '최고', '유일' 등 최상급의 직접 근거는 공정위 지정고시(FTC-지정고시)\n"
        "- '무조건', '반드시', '확실히' 등 단정적 표현의 직접 근거는 금소법 제21조(FCPA-21-단정)\n"
        "- '선착순', '한정', '마감 임박' 등 희소성·긴박성 표현의 근거는 다크패턴 압박형(DARK-10~14)이며, "
        "DARK-15(순차공개 가격책정)와 혼동하지 말 것\n"
        "- 상위법(금소법 제22조 등)은 하위규정 인용 후 추가 병기하거나, 하위규정이 없을 때만 단독 인용\n\n"
        "【판정 기준】\n"
        "- 위반: 법령·감독 규정에 명백히 저촉되는 표현이 존재하는 경우\n"
        "- 주의: 위반 소지가 있으나 전체 맥락에 따라 달라질 수 있는 경우, 또는 필수 고지 누락이 의심되는 경우\n"
        "- 통과: 규제 조항에 저촉되는 표현이 없고 필수 고지가 적절히 포함된 경우\n\n"
        "【위반 유형명 예시 — type 필드에 사용】\n"
        "수익 보장 단정 / 원금보장 표현 / 수익률 단정 / 근거 없는 최상급 표현 / 부당 비교 / "
        "보편적용 오인 / 세전·세후 미구분 / 투자 위험 미고지 / 필수 고지 누락 / "
        "희소성·긴박성 조장 / 압박형 다크패턴 / 감정적 언어 사용 / 허위·과장 광고 / 부당권유\n\n"
        "【few-shot 판정 예시】\n"
        "예1) 콘텐츠: '연 8% 확정 수익 보장 적금'\n"
        "→ 판정: 위반 / type: 수익 보장 단정 / 이유: '확정 수익 보장'은 불확실한 사항에 대한 단정적 표현 "
        "— 금소법 제21조 제1항 제1호(FCPA-21-단정) 위반\n\n"
        "예2) 콘텐츠: '누구나 연 5% 우대금리 적용'\n"
        "→ 판정: 위반 / type: 보편적용 오인 / 이유: 우대금리는 조건 충족 시에만 적용되는데 '누구나' 표현으로 "
        "보편 적용되는 것처럼 오인 — 감독규정 제19조(FCPA-감독규정19) ①항 위반\n\n"
        "예3) 콘텐츠: '업계 1위 금리 제공'\n"
        "→ 판정: 위반 / type: 근거 없는 최상급 표현 / 이유: 공인 조사기관 근거 없는 순위 표현 "
        "— 공정위 지정고시(FTC-지정고시) 위반\n\n"
        "예4) 콘텐츠: '이 적금은 연 3.5% 금리(기본 2.5% + 우대 1.0%, 급여이체 시)이며 예금자보호법에 따라 "
        "1인당 최고 5천만원까지 보호됩니다'\n"
        "→ 판정: 통과 / 이유: 기본·우대금리 구분, 우대 조건 명시, 예금자보호 한도 고지 모두 충족\n\n"
        "각 위반 항목의 type 에는 심각도가 아니라 위반 유형명을 적으세요."
    )


def judge_node(state: State) -> State:
    """[판단] 규제 조항과 룰 탐지 결과를 근거로 LLM 이 위반 여부 판단"""
    chunks = state.get("retrieved", [])
    rule_summary = "\n".join(
        f"- ({h.severity}) {h.category}: '{h.term}' — {h.message}"
        for h in state.get("rule_hits", [])
    ) or "(룰 탐지 없음)"

    mode = state.get("review_mode") or "표준"
    human = (
        f"[심의 대상 콘텐츠]\n{state['content']}\n\n"
        f"[룰 엔진 탐지 결과]\n{rule_summary}\n\n"
        f"[관련 규제 조항]\n{_format_context(chunks)}"
    )
    judgment = _judge_llm().with_structured_output(Judgment).invoke(
        [("system", _judge_prompt(mode)), ("human", human)]
    )
    return {"judgment": judgment}


def reflect_node(state: State) -> State:
    """[검증] judge 판단의 근거 충분성을 자기 검증 — 조항 미인용 위반 항목이 있으면 재검색 요청"""
    judgment = state["judgment"]
    # 위반 항목 중 citation_ids 가 비어 있는 항목 수 계산
    uncited = [v for v in judgment.violations if not v.citation_ids]
    # 위반인데 인용 조항이 전혀 없으면 근거 불충분으로 판단해 재검색 요청
    needs = (
        judgment.status != "통과"
        and len(uncited) > 0
        and len(uncited) == len(judgment.violations)
        and not state.get("needs_reretrieval")  # 무한 루프 방지: 재검색은 1회만
    )
    return {"needs_reretrieval": needs}


def reretrieval_node(state: State) -> State:
    """[개선] 위반 유형 키워드로 규제 조항을 보강 검색"""
    judgment = state["judgment"]
    media = state.get("media")
    chunks: list[dict] = []
    seen: set = set()
    for v in judgment.violations:
        if v.type in seen or len(seen) >= 6:
            continue
        seen.add(v.type)
        chunks += search(f"{v.type} {v.reason}", k=3, media=media, boost_terms=[v.type])
    chunks += search(state["content"], k=4, media=media)
    return {"retrieved": _dedup_by_id(chunks)}


def alternative_node(state: State) -> State:
    """[행동] 위반·주의 콘텐츠에 대해 준법 통과 가능한 대안 문구 생성
    이전 재심의 실패 피드백이 있으면 반영해 개선된 문구 재생성
    """
    judgment = state["judgment"]
    violation_summary = "; ".join(f"{v.type}: {v.reason}" for v in judgment.violations)

    # 이전 시도 피드백 누적
    feedback_list: list[str] = state.get("alt_feedback") or []
    feedback_section = ""
    if feedback_list:
        feedback_section = (
            "\n\n【이전 시도 실패 피드백 — 반드시 반영하세요】\n"
            + "\n".join(f"시도 {i+1}: {fb}" for i, fb in enumerate(feedback_list))
        )

    system = (
        "당신은 금융 마케팅 카피라이터이자 준법 전문가입니다. "
        "원문의 마케팅 의도는 살리되, 지적된 위반 사유를 모두 해소한 대안 문구를 한국어로 작성하세요. "
        "과장·단정·보편적용 표현을 제거하고 필요한 고지를 반영하세요. 대안 문구만 출력하세요.\n\n"
        "【필수 준수 실무 기준】\n"
        "① 수익률·이자율 표시 시 세전(稅前) 또는 세후(稅後) 구분을 명시하세요 (감독규정 제19조).\n"
        "② 우대금리·이벤트금리 등 조건부 혜택은 반드시 달성 조건을 병기하세요 (예: '급여이체 시 우대금리 포함').\n"
        "③ '누구나', '무조건', '반드시' 등 단정·보편 표현 대신 조건·가능성을 명확히 서술하세요.\n"
        "④ '업계 1위', '최고', '최저' 등 최상급 표현은 삭제하거나 객관적 근거와 함께 표기하세요.\n"
        "⑤ 적금·예금은 투자성 상품이 아니므로 '수익률 목표', '기대수익' 등 투자성 오인 표현을 사용하지 마세요.\n"
        "⑥ '선착순', '한정' 등 긴박성·희소성 표현은 실제 조건이 있을 때만 허용되며, 없으면 삭제하세요.\n"
        "⑦ 유리한 조건(금리·혜택)을 부각할 때는 상응하는 불리한 조건(달성 조건·제한·중도해지 불이익 등)을 "
        "동등한 비중으로 병기하세요. 유리한 조건만 강조하고 불리한 조건을 축소·누락하면 부당광고입니다 (감독규정 제19조).\n"
        "⑧ '타사 대비', '경쟁사보다', '업계 평균보다 낮은', '경쟁력 있는' 등 비교·우위 표현은 "
        "비교대상·기준·출처(조사기관·기준일)를 명시할 수 있을 때만 쓰고, 없으면 삭제하세요 "
        "(표시광고법 제3조, 금소법 제21조 제3호).\n\n"
        "【대안 문구 예시】\n"
        "원문: '누구나 연 5.0% 우대금리 적용'\n"
        "→ '연 5.0%(세전, 우대금리 포함 / 기본금리 연 3.5% + 우대금리 연 1.5%, "
        "우대금리는 급여이체 등록 시 적용)이며, 실제 적용 금리는 가입 조건에 따라 달라질 수 있습니다.'\n\n"
        "원문: '확정 수익 연 8% 적금'\n"
        "→ '연 최고 8%(세전) / 우대금리 조건 충족 시 적용되며, 중도 해지 시 금리가 달라질 수 있습니다.'\n\n"
        "원문: '타사 대비 수수료 50% 저렴, 업계 평균보다 낮은 금리'\n"
        "→ '타행 이체 수수료 면제 혜택을 제공합니다(면제 조건: 월 1회 이상 급여이체). "
        "적용 금리는 가입 기간·조건에 따라 달라질 수 있습니다.' (객관적 출처 없는 비교·우위 표현은 제외)"
        + feedback_section
    )

    voice = _re.search(r'\[음성 자막\](.*?)(?=\[화면 분석\]|$)', state["content"], _re.S)
    clean_content = voice.group(1).strip() if voice else state["content"]

    human = (
        f"[원문]\n{clean_content}\n\n"
        f"[해소해야 할 위반 사유]\n{violation_summary or judgment.summary}"
    )
    resp = _llm().invoke([("system", system), ("human", human)])
    return {
        "alternative": resp.content.strip(),
        "alt_retries": (state.get("alt_retries") or 0) + 1,
    }


def re_judge_node(state: State) -> State:
    """[검증] 대안 문구가 규정을 통과하는지 재심의"""
    chunks = state.get("retrieved", [])
    mode = state.get("review_mode") or "표준"
    human = (
        f"[심의 대상 콘텐츠 — AI가 수정 제안한 대안 문구]\n{state['alternative']}\n\n"
        f"[관련 규제 조항]\n{_format_context(chunks)}"
    )
    alt_judgment = _judge_llm().with_structured_output(Judgment).invoke(
        [("system", _judge_prompt(mode)), ("human", human)]
    )
    return {"alt_judgment": alt_judgment}


# --- 라우팅 ---

def _route_after_reflect(state: State) -> str:
    """근거 불충분이면 보강 검색 후 재판단, 아니면 다음 단계로"""
    if state.get("needs_reretrieval"):
        return "reretrieval"
    return "end" if state["judgment"].status == "통과" else "alternative"


def _route_after_rejudge(state: State) -> str:
    """대안 재심의 결과에 따라 종료 또는 재생성"""
    alt_j = state.get("alt_judgment")
    retries = state.get("alt_retries") or 0

    if alt_j and alt_j.status == "통과":
        return "end"
    if retries >= _MAX_ALT_RETRIES:
        return "give_up"
    return "retry_alternative"


def _collect_feedback(state: State) -> State:
    """재생성 전 이전 실패 이유를 feedback 리스트에 추가"""
    alt_j = state.get("alt_judgment")
    feedback = state.get("alt_feedback") or []
    if alt_j:
        reasons = "; ".join(f"{v.type}: {v.reason}" for v in alt_j.violations) or alt_j.summary
        feedback = feedback + [reasons]
    return {"alt_feedback": feedback}


def give_up_node(state: State) -> State:
    """최대 재시도 초과 — 자동 수정 불가 플래그 설정"""
    return {"auto_fix_failed": True}


# --- 그래프 구성 ---

@lru_cache(maxsize=1)
def _compiled():
    g = StateGraph(State)

    g.add_node("rule",         rule_node)
    g.add_node("retrieve",     retrieve_node)
    g.add_node("judge",        judge_node)
    g.add_node("reflect",      reflect_node)
    g.add_node("reretrieval",  reretrieval_node)
    g.add_node("alternative",  alternative_node)
    g.add_node("re_judge",     re_judge_node)
    g.add_node("collect_fb",   _collect_feedback)
    g.add_node("give_up",      give_up_node)

    g.set_entry_point("rule")
    g.add_edge("rule",        "retrieve")
    g.add_edge("retrieve",    "judge")
    g.add_edge("judge",       "reflect")

    # reflect: 근거 불충분 → 보강 검색 후 재판단 / 통과 → END / 위반 → 대안 생성
    g.add_conditional_edges(
        "reflect", _route_after_reflect,
        {"reretrieval": "reretrieval", "end": END, "alternative": "alternative"},
    )
    g.add_edge("reretrieval", "judge")  # 보강 검색 후 재판단

    # 대안 생성 후 재심의
    g.add_edge("alternative", "re_judge")

    # 재심의: 통과 → END / 실패·재시도 → 피드백 수집 후 대안 재생성 / 포기 → give_up
    g.add_conditional_edges(
        "re_judge", _route_after_rejudge,
        {"end": END, "retry_alternative": "collect_fb", "give_up": "give_up"},
    )
    g.add_edge("collect_fb", "alternative")  # 피드백 반영 재생성
    g.add_edge("give_up",    END)

    return g.compile()


def run_review(content: str, media: str | None = None, review_mode: str = "표준") -> ReviewResult:
    """심의 파이프라인 실행 후 구조화된 결과 반환"""
    final: State = _compiled().invoke(
        {
            "content": content,
            "media": media,
            "review_mode": review_mode,
            "alt_retries": 0,
            "alt_feedback": [],
            "auto_fix_failed": False,
        }
    )

    judgment = final["judgment"]
    cited_ids = {cid for v in judgment.violations for cid in v.citation_ids}
    citations = [
        Citation(id=c["id"], law=c["law"], article=c["article"], source_url=c.get("source_url"))
        for c in final.get("retrieved", [])
        if c["id"] in cited_ids
    ]

    # 대안 문구: 재심의 통과한 버전 우선, 없으면 마지막 생성본
    alternative_text = final.get("alternative")
    auto_fix_failed = final.get("auto_fix_failed", False)

    return ReviewResult(
        status=judgment.status,
        summary=judgment.summary,
        rule_hits=final.get("rule_hits", []),
        violations=judgment.violations,
        citations=citations,
        alternative_text=alternative_text,
        auto_fix_failed=auto_fix_failed,
    )
