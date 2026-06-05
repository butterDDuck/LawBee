"""이미지 전처리 인터페이스

심의 대상 이미지(배너·캡처 등)에서 문구와 시각 구성을 텍스트로 추출
추출 결과는 심의 엔진(`run_review`)에 그대로 전달 가능한 텍스트여야 함

구현 예정: 팀원 핸드오프 이슈 (멀티모달 전처리)
권장 방식: OpenAI Vision 으로 이미지 내 문구 + 레이아웃(버튼 색·크기 대비 등) 설명 생성
"""


def extract_content(image_bytes: bytes) -> str:
    """이미지에서 심의 대상 텍스트·시각 구성 설명을 추출

    Args:
        image_bytes: 이미지 원본 바이트

    Returns:
        심의 엔진에 입력 가능한 텍스트 (문구 + 시각 구성 설명)
    """
    raise NotImplementedError("멀티모달 전처리 이슈에서 구현 예정")
