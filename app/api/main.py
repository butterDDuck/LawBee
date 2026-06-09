"""FastAPI 심의 API

실행:
    uvicorn app.api.main:app --reload
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app import store
from app.config import settings
from app.domain.schema import (
    DecisionRequest,
    DecisionStatus,
    ReviewRecord,
    ReviewRequest,
    ReviewResult,
    TimelineSegment,
)
from app.preprocess.image import extract_content
from app.preprocess.video import extract_frames, transcribe
from app.services.graph import run_review

app = FastAPI(
    title="LawBee API",
    description="사내 마케팅/영업 콘텐츠 준법심의 자동화 API",
    version="0.1.0",
)
# 브라우저에서 원본 미디어를 표시할 수 있도록 CORS 허용 (로컬 데모)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_UPLOADS = Path(settings.uploads_dir)


def _save_media(review_id: int, data: bytes, filename: str | None) -> None:
    _UPLOADS.mkdir(parents=True, exist_ok=True)
    ext = Path(filename or "").suffix or ".bin"
    (_UPLOADS / f"{review_id}{ext}").write_bytes(data)


def _media_path(review_id: int) -> Path | None:
    matches = sorted(_UPLOADS.glob(f"{review_id}.*"))
    return matches[0] if matches else None


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
    return store.create_review(req.content, media=req.media, title=req.title)


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

def _seg(start, end, text, terms, kind):
    """텍스트에 위반 문구(룰 탐지어)를 매핑한 타임라인 구간 생성"""
    one = " ".join(text.split())
    hit = [t for t in terms if t in text]
    disp = (one[:70] + "…") if len(one) > 70 else one
    return TimelineSegment(start=start, end=end, text=disp, flagged=bool(hit), terms=hit, kind=kind)


@app.post("/reviews/image", response_model=ReviewRecord)
async def create_review_image(file: UploadFile = File(...), title: str = Form("")) -> ReviewRecord:
    """이미지 업로드 → Vision 텍스트 추출 → AI 심의 → 대기 저장"""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="이미지 파일이 비어 있습니다")
    text = extract_content(data)
    rec = store.create_review(text, media="이미지", title=title or None)
    _save_media(rec.id, data, file.filename)
    return rec


@app.post("/reviews/video", response_model=ReviewRecord)
async def create_review_video(file: UploadFile = File(...), title: str = Form("")) -> ReviewRecord:
    """영상 업로드 → 음성 자막(Whisper) + 화면 프레임(Vision) 추출 → 통합 심의 → 타임라인 저장"""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="영상 파일이 비어 있습니다")
    fname = file.filename or "upload.mp4"

    # 음성 자막 (없는 영상도 허용)
    try:
        segments = transcribe(data, filename=fname)
    except Exception:
        segments = []
    transcript = "\n".join(s.text for s in segments).strip()

    # 화면 프레임 → 프레임별 Vision 분석 (병렬)
    frames = extract_frames(data, filename=fname)
    frame_texts: list[tuple[float, str]] = []
    if frames:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(extract_content, jpg): t for t, jpg in frames}
            for fut in as_completed(futs):
                try:
                    frame_texts.append((futs[fut], fut.result()))
                except Exception:
                    pass
        frame_texts.sort()

    if not transcript and not frame_texts:
        raise HTTPException(status_code=422, detail="영상에서 자막·화면 텍스트를 추출하지 못했습니다")

    parts = []
    if transcript:
        parts.append("[음성 자막]\n" + transcript)
    if frame_texts:
        parts.append("[화면 텍스트]\n" + "\n".join(t for _, t in frame_texts))
    combined = "\n\n".join(parts)

    result = run_review(combined, media="영상")
    terms = [h.term for h in result.rule_hits]
    audio_tl = [_seg(s.start, s.end, s.text, terms, "음성") for s in segments]
    visual_tl = [_seg(t, t + 4.0, txt, terms, "화면") for t, txt in frame_texts]
    result.timeline = sorted(audio_tl + visual_tl, key=lambda x: x.start)

    rec = store.create_with_result(combined, "영상", result, title=title or None)
    _save_media(rec.id, data, file.filename)
    return rec


@app.get("/reviews/{review_id}/media")
def get_media(review_id: int):
    """업로드된 원본 미디어(이미지·영상) 반환"""
    p = _media_path(review_id)
    if not p:
        raise HTTPException(status_code=404, detail="원본 미디어가 없습니다")
    return FileResponse(p)


@app.delete("/reviews/{review_id}")
def delete_review(review_id: int) -> dict:
    """심의 건 및 원본 미디어 삭제"""
    if not store.delete_review(review_id):
        raise HTTPException(status_code=404, detail="심의 건을 찾을 수 없습니다")
    p = _media_path(review_id)
    if p:
        p.unlink(missing_ok=True)
    return {"deleted": review_id}


@app.post("/reviews/{review_id}/decision", response_model=ReviewRecord)
def decide_review(review_id: int, req: DecisionRequest) -> ReviewRecord:
    """준법관리자 최종 결재 (승인·조건부승인·반려 + 코멘트)"""
    if req.decision == "대기":
        raise HTTPException(status_code=400, detail="결재 결과는 승인·조건부승인·반려 중 하나여야 합니다")
    rec = store.decide(review_id, req.decision, req.comment, req.reviewer)
    if not rec:
        raise HTTPException(status_code=404, detail="심의 건을 찾을 수 없습니다")
    return rec
