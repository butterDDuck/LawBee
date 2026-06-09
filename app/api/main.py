"""FastAPI 심의 API

실행:
    uvicorn app.api.main:app --reload
"""
from fastapi import FastAPI, File, HTTPException, UploadFile

from app import store
from app.domain.schema import (
    DecisionRequest,
    DecisionStatus,
    ReviewRecord,
    ReviewRequest,
    ReviewResult,
    TimelineSegment,
)
from app.preprocess.image import extract_content
from app.preprocess.video import transcribe
from app.services.graph import run_review

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


# --- 멀티모달 (이미지·영상) ---

def _build_timeline(segments, rule_hits) -> list[TimelineSegment]:
    """자막 구간에 위반 문구(룰 탐지어)를 매핑하여 타임라인 생성"""
    terms = [h.term for h in rule_hits]
    out = []
    for s in segments:
        hit = [t for t in terms if t in s.text]
        out.append(TimelineSegment(start=s.start, end=s.end, text=s.text, flagged=bool(hit), terms=hit))
    return out


@app.post("/reviews/image", response_model=ReviewRecord)
async def create_review_image(file: UploadFile = File(...)) -> ReviewRecord:
    """이미지 업로드 → Vision 텍스트 추출 → AI 심의 → 대기 저장"""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="이미지 파일이 비어 있습니다")
    text = extract_content(data)
    return store.create_review(text, media="이미지")


@app.post("/reviews/video", response_model=ReviewRecord)
async def create_review_video(file: UploadFile = File(...)) -> ReviewRecord:
    """영상 업로드 → Whisper 자막 추출 → AI 심의 → 타임라인 매핑 → 대기 저장"""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="영상 파일이 비어 있습니다")
    segments = transcribe(data, filename=file.filename or "upload.mp4")
    transcript = "\n".join(s.text for s in segments).strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="영상에서 자막을 추출하지 못했습니다")
    result = run_review(transcript, media="영상")
    result.timeline = _build_timeline(segments, result.rule_hits)
    return store.create_with_result(transcript, "영상", result)


@app.post("/reviews/{review_id}/decision", response_model=ReviewRecord)
def decide_review(review_id: int, req: DecisionRequest) -> ReviewRecord:
    """준법관리자 최종 결재 (승인·조건부승인·반려 + 코멘트)"""
    if req.decision == "대기":
        raise HTTPException(status_code=400, detail="결재 결과는 승인·조건부승인·반려 중 하나여야 합니다")
    rec = store.decide(review_id, req.decision, req.comment, req.reviewer)
    if not rec:
        raise HTTPException(status_code=404, detail="심의 건을 찾을 수 없습니다")
    return rec
