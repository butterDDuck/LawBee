"""심의 파이프라인 입출력 및 LLM 구조화 출력 스키마"""
from typing import Literal

from pydantic import BaseModel, Field

# AI 심의 판정 상태
Status = Literal["위반", "주의", "통과"]

# 준법관리자 결재 상태
DecisionStatus = Literal["대기", "승인", "조건부승인", "반려"]


class ReviewRequest(BaseModel):
    """심의 요청"""

    content: str = Field(..., description="심의 대상 마케팅 콘텐츠 원문")
    media: str | None = Field(None, description="콘텐츠 매체: 텍스트 | 영상 | UI")


class RuleHit(BaseModel):
    """룰 엔진이 탐지한 위반 의심 표현"""

    term: str = Field(..., description="탐지된 표현")
    category: str = Field(..., description="위반 의심 유형")
    severity: Literal["high", "medium"] = Field(..., description="심각도")
    message: str = Field(..., description="탐지 사유")


class Citation(BaseModel):
    """판단 근거가 된 규제 조항"""

    id: str
    law: str
    article: str
    source_url: str | None = None


class Violation(BaseModel):
    """LLM이 판단한 개별 위반 항목"""

    type: str = Field(..., description="위반 유형명 (예: 수익 보장 단정, 부당 비교, 보편적용 오인)")
    reason: str = Field(..., description="위반으로 본 근거 설명")
    citation_ids: list[str] = Field(default_factory=list, description="근거 조항 id 목록")


class Judgment(BaseModel):
    """LLM 심의 판단 결과 (구조화 출력)"""

    status: Status = Field(..., description="종합 판정")
    summary: str = Field(..., description="판정 요약")
    violations: list[Violation] = Field(default_factory=list)


class ReviewResult(BaseModel):
    """심의 파이프라인 최종 반환값"""

    status: Status
    summary: str
    rule_hits: list[RuleHit] = Field(default_factory=list)
    violations: list[Violation] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    alternative_text: str | None = Field(None, description="위반 시 제안하는 대안 문구")


class DecisionRequest(BaseModel):
    """준법관리자 결재 요청"""

    decision: DecisionStatus = Field(..., description="결재 결과: 승인 | 조건부승인 | 반려")
    comment: str = Field("", description="준법관리자 코멘트")
    reviewer: str = Field("준법관리자", description="결재자")


class ReviewRecord(BaseModel):
    """저장된 심의 건 (AI 심의 결과 + 준법관리자 결재 상태)"""

    id: int
    content: str
    media: str | None = None
    decision_status: DecisionStatus = "대기"
    ai_result: ReviewResult
    comment: str = ""
    reviewer: str | None = None
    created_at: str
    decided_at: str | None = None
