"""FastAPI 심의 API

실행:
    uvicorn app.api.main:app --reload
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app import store
from app.config import settings
from app.domain.schema import (
    DecisionRequest,
    DecisionStatus,
    ReviewMode,
    ReviewRecord,
    ReviewRequest,
    ReviewResult,
    TimelineSegment,
)
from app.preprocess.image import analyze_frame, extract_content
from app.preprocess.video import check_disclosure_duration, extract_frames, transcribe
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
    return run_review(req.content, media=req.media, review_mode=req.review_mode)


# --- 준법관리자 결재 워크플로우 ---

@app.post("/reviews", response_model=ReviewRecord)
def create_review(req: ReviewRequest) -> ReviewRecord:
    """콘텐츠 제출 → AI 1차 심의 자동 실행 → 대기 상태로 저장"""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="content 가 비어 있습니다")
    return store.create_review(req.content, media=req.media, title=req.title, review_mode=req.review_mode)


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

def _seg(start, end, text, term_severity, kind):
    """텍스트에 위반 문구(룰 탐지어)를 매핑한 타임라인 구간 생성
    term_severity: {term: 'high'|'medium'} 딕셔너리
    """
    one = " ".join(text.split())
    hit = [t for t in term_severity if t in text]
    disp = (one[:70] + "…") if len(one) > 70 else one
    sev = "high" if any(term_severity[t] == "high" for t in hit) else ("medium" if hit else "")
    return TimelineSegment(start=start, end=end, text=disp, flagged=bool(hit), terms=hit, kind=kind, severity=sev)


@app.post("/reviews/image", response_model=ReviewRecord)
async def create_review_image(
    file: UploadFile = File(...),
    title: str = Form(""),
    review_mode: ReviewMode = Form("표준"),
) -> ReviewRecord:
    """이미지 업로드 → Vision 텍스트 추출 → AI 심의 → 대기 저장"""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="이미지 파일이 비어 있습니다")
    text = extract_content(data)
    rec = store.create_review(text, media="이미지", title=title or None, review_mode=review_mode)
    _save_media(rec.id, data, file.filename)
    return rec


def _merge_frame_runs(
    frame_data: list[tuple[float, str, list]],
    frame_interval: float = 4.0,
) -> list[tuple[float, float, str, list]]:
    """연속된 동일 화면(동일 시각 위반 집합) 프레임을 하나의 구간으로 병합

    같은 장면이 여러 프레임에 걸쳐 이어질 때 동일 시각 위반이 프레임마다 중복
    출력되는 것을 방지. 위반이 있는 프레임은 위반 카테고리 집합을, 없는 프레임은
    화면 텍스트를 병합 기준으로 삼음 (OCR 노이즈로 텍스트가 미세하게 달라도 같은
    위반이면 한 구간으로 묶임).

    Args:
        frame_data: (시각(초), 화면 텍스트, finds) 목록 — 시각 오름차순 정렬 가정

    Returns:
        (start, end, 대표 텍스트, 대표 finds) 목록 — start==end 면 단일 프레임
    """
    def _key(txt: str, finds: list):
        cats = frozenset(f.get("category") for f in finds if f.get("category"))
        if cats:
            return ("v", cats)                       # 위반 프레임: 카테고리 집합 기준
        return ("t", " ".join(txt.split())[:24])     # 무위반 프레임: 화면 텍스트 기준

    runs: list[dict] = []
    for t, txt, finds in frame_data:
        k = _key(txt, finds)
        if runs and runs[-1]["key"] == k:
            runs[-1]["frames"].append((t, txt, finds))
            runs[-1]["end"] = t
        else:
            runs.append({"start": t, "end": t, "key": k, "frames": [(t, txt, finds)]})

    out: list[tuple[float, float, str, list]] = []
    for r in runs:
        # 대표 텍스트는 구간 내 가장 정보가 많은(긴) 프레임 텍스트
        best_txt = max((f[1] for f in r["frames"]), key=lambda s: len(s.strip()), default="")
        # 대표 finds 는 위반이 있는 첫 프레임의 것 (카테고리 집합이 동일해 대표성 있음)
        rep_finds = next((f[2] for f in r["frames"] if f[2]), [])
        out.append((r["start"], r["end"], best_txt, rep_finds))
    return out


def _process_video(review_id: int, data: bytes, fname: str, review_mode: str = "표준") -> None:
    """백그라운드 영상 분석 — Whisper 포함 전 과정 처리 후 DB 갱신"""
    import time as _time
    import traceback as _tb
    try:
        # 음성 자막 추출
        try:
            segments_raw = transcribe(data, filename=fname)
        except Exception:
            segments_raw = []
        transcript = "\n".join(s.text for s in segments_raw).strip()
        frames = extract_frames(data, filename=fname)
        frame_data: list[tuple[float, str, list]] = []
        if frames:
            with ThreadPoolExecutor(max_workers=2) as ex:
                futs = {ex.submit(analyze_frame, jpg): t for t, jpg in frames}
                for fut in as_completed(futs):
                    try:
                        fa = fut.result()
                        frame_data.append((futs[fut], fa["text"], fa["findings"]))
                    except Exception:
                        pass
            frame_data.sort()

        # 연속 동일 화면을 구간으로 병합 — 같은 위반의 프레임별 중복 출력 방지
        frame_runs = _merge_frame_runs(frame_data)

        # 고지띠 노출 시간 검사 — 필수 고지자막 3초 미만 탐지 (병합 전 프레임 단위로 측정)
        frame_texts = [(t, txt) for t, txt, _ in frame_data]
        disclosure_violations = check_disclosure_duration(frame_texts)

        parts = []
        if transcript:
            parts.append("[음성 자막]\n" + transcript)
        if frame_runs:
            lines = []
            for start, end, txt, finds in frame_runs:
                label = f"{int(start)}초" if start == end else f"{int(start)}~{int(end)}초"
                line = f"{label}: {txt}".strip()
                for f in finds:
                    line += f" (시각 위반: {f.get('category')} - {f.get('detail', '')})"
                lines.append(line)
            parts.append("[화면 분석]\n" + "\n".join(lines))
        if disclosure_violations:
            dv_lines = []
            for dv in disclosure_violations:
                dv_lines.append(
                    f"- \"{dv.text}\" — {dv.start}초~{dv.end}초 ({dv.duration}초 노출, "
                    f"기준 {dv.required}초 미달)"
                )
            parts.append("[고지띠 노출 시간 위반]\n" + "\n".join(dv_lines))
        combined = "\n\n".join(parts)

        if not combined:
            combined = "(내용을 추출하지 못했습니다)"

        # Vision 호출 직후 TPM 소진 방지를 위해 대기 후 심의 실행
        _time.sleep(10)

        # 429 대비 재시도 (최대 3회, 지수 백오프)
        result = None
        for attempt in range(3):
            try:
                result = run_review(combined, media="영상", review_mode=review_mode)
                break
            except Exception as e:
                if attempt < 2:
                    _time.sleep(20 * (attempt + 1))
                else:
                    raise

        term_severity = {h.term: h.severity for h in result.rule_hits}
        audio_tl = [_seg(s.start, s.end, s.text, term_severity, "음성") for s in segments_raw]
        visual_tl = []
        for start, end, txt, finds in frame_runs:
            cats = [f.get("category") for f in finds if f.get("category")]
            one = " ".join(txt.split())
            disp = (one[:64] + "…") if len(one) > 64 else (one or "(화면)")
            sev = "high" if any(f.get("severity") == "high" for f in finds) else ("medium" if finds else "")
            visual_tl.append(TimelineSegment(start=start, end=end + 4.0, text=disp,
                                             flagged=bool(finds), terms=cats, kind="화면", severity=sev))
        result.timeline = sorted(audio_tl + visual_tl, key=lambda x: x.start)

        # content 업데이트 후 ai_result 갱신
        with store._conn() as c:
            c.execute("UPDATE reviews SET content = ? WHERE id = ?", (combined, review_id))
        store.update_ai_result(review_id, result)
    except Exception as e:
        print(f"[ERROR] _process_video rid={review_id}: {e}")
        _tb.print_exc()


@app.post("/reviews/video", response_model=ReviewRecord)
async def create_review_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(""),
    review_mode: ReviewMode = Form("표준"),
) -> ReviewRecord:
    """영상 업로드 → 즉시 레코드 반환 후 백그라운드에서 전 과정 분석 진행"""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="영상 파일이 비어 있습니다")
    fname = file.filename or "upload.mp4"

    # 처리중 레코드 즉시 생성 후 반환 — Whisper 포함 모든 분석은 백그라운드 처리
    rec = store.create_pending("(분석 중…)", "영상", title=title or None, review_mode=review_mode)
    _save_media(rec.id, data, fname)

    background_tasks.add_task(_process_video, rec.id, data, fname, review_mode)
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
