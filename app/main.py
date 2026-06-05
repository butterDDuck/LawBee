"""FastAPI 심의 API

실행:
    uvicorn app.main:app --reload
"""
from fastapi import FastAPI, HTTPException

from app.schema import ReviewRequest, ReviewResult
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
    """마케팅 콘텐츠를 심의하여 위반 여부·근거·대안 문구 반환"""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="content 가 비어 있습니다")
    return run_review(req.content, media=req.media)
