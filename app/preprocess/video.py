"""영상 전처리 — Whisper STT(음성) + 프레임 추출(화면)

음성 자막과 화면 프레임을 각각 추출하여 음성·화면 위반을 통합 심사
"""
import io
import os
import re
import tempfile
from dataclasses import dataclass, field

import cv2
from openai import OpenAI

from app.config import settings

# 필수 고지 키워드 — 이 표현을 포함한 자막은 노출 시간을 측정
_DISCLOSURE_KEYWORDS = re.compile(
    r"원금\s*손실|손실\s*(위험|가능)|투자\s*위험|과거\s*수익률.*미래|예금자\s*보호|"
    r"보험료|해약\s*환급|면책|지급\s*제한|수수료|위험\s*등급|"
    r"금융소비자|준법감시|준법심의|투자\s*원금|원금.*보장.*안"
)

# 고지띠 최소 노출 시간 기준(초) — 금감원 가이드라인
_MIN_DISCLOSURE_SEC = 3.0


@dataclass
class Segment:
    start: float  # 시작 시각(초)
    end: float    # 종료 시각(초)
    text: str     # 해당 구간 발화·자막


@dataclass
class DisclosureViolation:
    """고지띠 노출 시간 부족 위반 항목"""
    text: str          # 고지 문구
    start: float       # 첫 등장 시각(초)
    end: float         # 마지막 등장 시각(초)
    duration: float    # 실제 노출 시간(초)
    required: float = field(default=_MIN_DISCLOSURE_SEC)


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


def check_disclosure_duration(
    frame_texts: list[tuple[float, str]],
) -> list[DisclosureViolation]:
    """프레임별 텍스트에서 고지띠 노출 시간을 측정하고 기준 미달 항목을 반환

    연속된 프레임에 같은 고지 문구가 등장하는 구간을 하나의 노출로 간주.
    마지막 프레임 시각 + 프레임 간격만큼을 노출 종료 시각으로 추정.

    Args:
        frame_texts: (시각(초), OCR 텍스트) 목록 — 시각 오름차순 정렬 가정

    Returns:
        3초 미만으로 노출된 필수 고지 항목 목록
    """
    if not frame_texts:
        return []

    # 프레임 간격 추정 (중간값 사용, 단일 프레임이면 4.0초로 간주)
    timestamps = [t for t, _ in frame_texts]
    if len(timestamps) >= 2:
        gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        frame_interval = sorted(gaps)[len(gaps) // 2]
    else:
        frame_interval = 4.0

    # 고지 키워드를 포함하는 프레임만 추출
    disclosure_frames: list[tuple[float, str]] = [
        (t, txt) for t, txt in frame_texts
        if _DISCLOSURE_KEYWORDS.search(txt)
    ]

    if not disclosure_frames:
        return []

    # 문구별로 그룹화 — 유사 문구를 같은 고지로 묶기 위해 앞 20자를 키로 사용
    def _key(txt: str) -> str:
        return re.sub(r"\s+", " ", txt.strip())[:20]

    # 연속 구간 탐지: 이전 프레임과 같은 키이고 시각 간격이 frame_interval * 1.5 이내면 같은 구간
    groups: list[list[tuple[float, str]]] = []
    for t, txt in sorted(disclosure_frames):
        k = _key(txt)
        if groups and _key(groups[-1][-1][1]) == k and (t - groups[-1][-1][0]) <= frame_interval * 1.5:
            groups[-1].append((t, txt))
        else:
            groups.append([(t, txt)])

    violations = []
    for grp in groups:
        start = grp[0][0]
        end = grp[-1][0] + frame_interval  # 마지막 프레임 이후 한 간격까지 표시된다고 추정
        duration = round(end - start, 1)
        if duration < _MIN_DISCLOSURE_SEC:
            violations.append(DisclosureViolation(
                text=grp[0][1][:60],
                start=start,
                end=round(end, 1),
                duration=duration,
            ))
    return violations


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
