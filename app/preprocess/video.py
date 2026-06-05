"""영상 전처리 인터페이스

심의 대상 영상에서 타임스탬프가 포함된 자막을 추출
각 구간(Segment)을 심의 엔진에 넣어 타임라인 위반 매핑에 사용

구현 예정: 팀원 핸드오프 이슈 (멀티모달 전처리)
권장 방식: Whisper STT 로 구간별 (시작·종료 시각, 발화 텍스트) 추출
"""
from dataclasses import dataclass


@dataclass
class Segment:
    start: float  # 시작 시각(초)
    end: float    # 종료 시각(초)
    text: str     # 해당 구간 발화·자막


def transcribe(video_bytes: bytes) -> list[Segment]:
    """영상에서 타임스탬프 자막 구간 목록을 추출

    Args:
        video_bytes: 영상 원본 바이트

    Returns:
        시작·종료 시각과 텍스트를 가진 Segment 목록
    """
    raise NotImplementedError("멀티모달 전처리 이슈에서 구현 예정")
