"""영상 전처리 — OpenAI Whisper STT

심의 대상 영상에서 타임스탬프가 포함된 자막을 추출
각 구간(Segment)을 심의 엔진에 넣어 타임라인 위반 매핑에 사용
"""
import io
from dataclasses import dataclass

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
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )
    segments = getattr(resp, "segments", None) or []
    return [Segment(start=float(s.start), end=float(s.end), text=s.text.strip()) for s in segments]
