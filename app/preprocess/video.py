"""영상 전처리 — Whisper STT(음성) + 프레임 추출(화면)

음성 자막과 화면 프레임을 각각 추출하여 음성·화면 위반을 통합 심사
"""
import io
import os
import tempfile
from dataclasses import dataclass

import cv2
from openai import OpenAI

from app.config import settings


@dataclass
class Segment:
    start: float  # 시작 시각(초)
    end: float    # 종료 시각(초)
    text: str     # 해당 구간 발화·자막


def transcribe(video_bytes: bytes, filename: str = "upload.mp4") -> list[Segment]:
    """영상에서 타임스탬프 자막 구간 목록을 추출

    Args:
        video_bytes: 영상(또는 음성) 원본 바이트
        filename: 확장자 판별용 파일명

    Returns:
        시작·종료 시각과 텍스트를 가진 Segment 목록
    """
    client = OpenAI(api_key=settings.openai_api_key)
    buf = io.BytesIO(video_bytes)
    buf.name = filename

    resp = client.audio.transcriptions.create(
        model=settings.openai_stt_model,
        file=buf,
        language="ko",
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )
    segments = getattr(resp, "segments", None) or []
    return [Segment(start=float(s.start), end=float(s.end), text=s.text.strip()) for s in segments]


def extract_frames(video_bytes: bytes, every: float = 4.0, max_frames: int = 15,
                   filename: str = "upload.mp4") -> list[tuple[float, bytes]]:
    """영상에서 일정 간격으로 프레임을 추출하여 (시각(초), JPG 바이트) 목록 반환"""
    suffix = os.path.splitext(filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(video_bytes)
        path = tf.name
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = (total / fps) if fps else 0.0
        frames: list[tuple[float, bytes]] = []
        t = 0.0
        while (duration == 0 or t <= duration) and len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            ok2, buf = cv2.imencode(".jpg", frame)
            if ok2:
                frames.append((round(t, 1), buf.tobytes()))
            t += every
        cap.release()
        return frames
    finally:
        os.unlink(path)
