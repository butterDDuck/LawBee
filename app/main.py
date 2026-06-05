"""FastAPI 심의 API

실행:
    uvicorn app.main:app --reload
"""
from fastapi import FastAPI, HTTPException

from app import store
from app.schema import (
    DecisionRequest,
    DecisionStatus,
    ReviewRecord,
    ReviewRequest,
    ReviewResult,
)
from app.graph import run_review

app = FastAPI(
    title="LawBee API",
    description="사내 마케팅/영업 콘텐츠 준법심의 자동화 API",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    """서버 상태 확인"""
    return {"status": "ok"}


@app.post("/review", response_model=ReviewResult)
def review(req: ReviewRequest) -> ReviewResult:
    """콘텐츠를 즉시 심의하여 결과만 반환 (저장하지 않음)"""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="content 가 비어 있습니다")
    return run_review(req.content, media=req.media)


# --- 준법관리자 결재 워크플로우 ---

@app.post("/reviews", response_model=ReviewRecord)
def create_review(req: ReviewRequest) -> ReviewRecord:
    """콘텐츠 제출 → AI 1차 심의 자동 실행 → 대기 상태로 저장"""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="content 가 비어 있습니다")
    return store.create_review(req.content, media=req.media)


@app.get("/reviews", response_model=list[ReviewRecord])
def list_reviews(status: DecisionStatus | None = None) -> list[ReviewRecord]:
    """검수 큐 조회 (결재 상태 필터 가능)"""
    return store.list_reviews(status)


@app.get("/reviews/{review_id}", response_model=ReviewRecord)
def get_review(review_id: int) -> ReviewRecord:
    """심의 건 상세 조회"""
    rec = store.get_review(review_id)
    if not rec:
        raise HTTPException(status_code=404, detail="심의 건을 찾을 수 없습니다")
    return rec


@app.post("/reviews/{review_id}/decision", response_model=ReviewRecord)
def decide_review(review_id: int, req: DecisionRequest) -> ReviewRecord:
    """준법관리자 최종 결재 (승인·조건부승인·반려 + 코멘트)"""
    if req.decision == "대기":
        raise HTTPException(status_code=400, detail="결재 결과는 승인·조건부승인·반려 중 하나여야 합니다")
    rec = store.decide(review_id, req.decision, req.comment, req.reviewer)
    if not rec:
        raise HTTPException(status_code=404, detail="심의 건을 찾을 수 없습니다")
    return rec
