"""심의 파이프라인 입출력 및 LLM 구조화 출력 스키마"""
from typing import Literal

from pydantic import BaseModel, Field

# AI 심의 판정 상태
Status = Literal["위반", "주의", "통과"]

# 준법관리자 결재 상태 ("처리중"은 영상 분석 완료 전 임시 상태)
DecisionStatus = Literal["대기", "승인", "조건부승인", "반려", "처리중"]

# 심의 강도
ReviewMode = Literal["강화", "표준", "완화", "AI추천"]


class ReviewRequest(BaseModel):
    """심의 요청"""

    content: str = Field(..., description="심의 대상 마케팅 콘텐츠 원문")
    media: str | None = Field(None, description="콘텐츠 매체: 텍스트 | 영상 | UI")
    title: str | None = Field(None, description="콘텐츠 제목")
    review_mode: ReviewMode = Field("표준", description="심의 강도: 강화 | 표준 | 완화 | AI추천")


class RuleHit(BaseModel):
    """룰 엔진이 탐지한 위반 의심 표현"""

    term: str = Field(..., description="탐지된 표현")
    category: str = Field(..., description="위반 의심 유형")
    severity: Literal["high", "medium"] = Field(..., description="심각도")
    message: str = Field(..., description="탐지 사유")
    basis: str = Field("", description="해당 룰의 근거 법령·규정")


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


class RelevanceGrade(BaseModel):
    """CRAG 검색 결과 관련성 채점 (구조화 출력)"""

    relevant_ids: list[str] = Field(
        default_factory=list,
        description="콘텐츠 심의의 근거가 될 만큼 관련 있는 조항 id 목록",
    )


class TimelineSegment(BaseModel):
    """영상 자막 구간 (위반 문구 타임스탬프 매핑)"""

    start: float = Field(..., description="시작 시각(초)")
    end: float = Field(..., description="종료 시각(초)")
    text: str = Field(..., description="구간 자막 또는 화면 텍스트")
    flagged: bool = Field(False, description="위반 문구 포함 여부")
    terms: list[str] = Field(default_factory=list, description="구간에서 탐지된 위반 문구")
    kind: str = Field("음성", description="음성 | 화면")
    severity: str = Field("", description="high | medium | 빈값(통과)")


class ReviewResult(BaseModel):
    """심의 파이프라인 최종 반환값"""

    status: Status
    summary: str
    rule_hits: list[RuleHit] = Field(default_factory=list)
    violations: list[Violation] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    alternative_text: str | None = Field(None, description="위반 시 제안하는 대안 문구")
    auto_fix_failed: bool = Field(False, description="대안 문구 자동 수정 최대 재시도 초과 여부")
    timeline: list[TimelineSegment] = Field(default_factory=list, description="영상 구간별 위반 타임라인")


class DecisionRequest(BaseModel):
    """준법관리자 결재 요청"""

    decision: DecisionStatus = Field(..., description="결재 결과: 승인 | 조건부승인 | 반려")
    comment: str = Field("", description="준법관리자 코멘트")
    reviewer: str = Field("준법관리자", description="결재자")


class ReviewRecord(BaseModel):
    """저장된 심의 건 (AI 심의 결과 + 준법관리자 결재 상태)"""

    id: int
    content: str
    title: str | None = None
    media: str | None = None
    review_mode: ReviewMode = "표준"
    decision_status: DecisionStatus = "대기"
    ai_result: ReviewResult
    comment: str = ""
    reviewer: str | None = None
    created_at: str
    decided_at: str | None = None
