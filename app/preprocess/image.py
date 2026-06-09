"""이미지 전처리 — OpenAI Vision

심의 대상 이미지(배너·캡처 등)에서 문구와 시각 구성을 텍스트로 추출
추출 결과는 심의 엔진(`run_review`)에 그대로 전달 가능한 텍스트
"""
import base64

from openai import OpenAI

from app.config import settings

_PROMPT = (
    "이미지에 보이는 모든 텍스트를 그대로 읽어서 적어라. "
    "이어서 어떤 문구가 크게/강조되어 있고 어떤 문구가 작거나 흐리게 표시되는지 한 줄로 덧붙여라. "
    "텍스트와 표시 방식만 객관적으로 옮기고, 인물 등 다른 묘사는 하지 마라."
)


def _mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def extract_content(image_bytes: bytes) -> str:
    """이미지에서 심의 대상 텍스트·시각 구성 설명을 추출

    Args:
        image_bytes: 이미지 원본 바이트

    Returns:
        심의 엔진에 입력 가능한 텍스트 (문구 + 시각 구성 설명)
    """
    client = OpenAI(api_key=settings.openai_api_key)
    b64 = base64.b64encode(image_bytes).decode()
    data_uri = f"data:{_mime(image_bytes)};base64,{b64}"

    resp = client.chat.completions.create(
        model=settings.openai_vision_model,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }],
    )
    return resp.choices[0].message.content.strip()
