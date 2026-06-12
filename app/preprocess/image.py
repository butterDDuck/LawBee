"""이미지 전처리 — OpenAI Vision

심의 대상 이미지(배너·캡처 등)에서 문구와 시각 구성을 텍스트로 추출
영상 프레임은 시각 위반까지 구조화 판단 (analyze_frame)
"""
import base64
import json

import cv2
import numpy as np
from openai import OpenAI

from app.config import settings

_PROMPT = (
    "이미지에 보이는 모든 텍스트를 그대로 읽어서 적어라. "
    "이어서 어떤 문구가 크게/강조되어 있고 어떤 문구가 작거나 흐리게 표시되는지 한 줄로 덧붙여라. "
    "텍스트와 표시 방식만 객관적으로 옮기고, 인물 등 다른 묘사는 하지 마라."
)

_FRAME_PROMPT = (
    "이 금융 광고 영상의 한 프레임을 준법 심의 관점에서 분석하라. "
    "첫 번째 이미지는 전체 화면이고, 두 번째 이미지는 하단 고지영역을 확대한 것이다. "
    "JSON 형식으로만 답하라.\n"
    '{"text": "화면의 모든 텍스트를 위치 순서대로 나열. '
    "대형 슬로건·빌보드·제품명·수치·출처 표기·자막·고지문 등 크기와 무관하게 전부 포함. "
    '없으면 빈 문자열", '
    '"findings": [{"category": "위반 유형", "severity": "high 또는 mid", "detail": "무엇이 왜 문제인지 한 문장"}]}\n'
    "다음 시각적 위반을 평가하라: "
    "① 위험·필수 고지문이 지나치게 작거나 흐려 가독성이 떨어짐(위험 고지 가독성) "
    "② 혜택·수익만 크게 강조하고 위험·불이익은 작게/흐리게 표시(혜택·불이익 불균형) "
    "③ 단정적·과장·최상급 문구를 시각적으로 크게 강조(단정·과장 강조) "
    "④ 선택을 오인시키는 시각 구성(오인 유발 구성). "
    "가독성 판단은 확대본이 아니라 첫 번째 전체 화면의 실제 크기를 기준으로 하라. "
    "문제가 없으면 findings 를 빈 배열로 두어라. 인물·배경 묘사는 하지 마라."
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
    client = OpenAI(api_key=settings.openai_api_key, max_retries=6)
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


def _lower_third_zoom(image_bytes: bytes) -> str | None:
    """프레임 하단 고지영역을 크롭·확대해 작은 고지문 가독성을 높인 data URI 반환"""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h = img.shape[0]
    crop = img[int(h * 0.78):h, :]
    crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def analyze_frame(image_bytes: bytes) -> dict:
    """영상 프레임을 시각 위반까지 구조화 분석

    전체 화면과 하단 고지영역 확대본을 함께 분석해 작은 필수 표기까지 탐지

    Returns:
        {"text": 화면 문구, "findings": [{"category","severity","detail"}, ...]}
    """
    # 프레임 실패는 드롭되므로 재시도를 낮게 두어 토큰 폭주를 방지
    client = OpenAI(api_key=settings.openai_api_key, max_retries=2)
    b64 = base64.b64encode(image_bytes).decode()
    data_uri = f"data:{_mime(image_bytes)};base64,{b64}"

    content = [
        {"type": "text", "text": _FRAME_PROMPT},
        {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
    ]
    zoom_uri = _lower_third_zoom(image_bytes)
    if zoom_uri:
        content.append({"type": "image_url", "image_url": {"url": zoom_uri, "detail": "high"}})

    resp = client.chat.completions.create(
        model=settings.openai_vision_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": content}],
    )
    try:
        data = json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return {"text": "", "findings": []}
    findings = [f for f in data.get("findings", []) if isinstance(f, dict) and f.get("category")]
    text = str(data.get("text", "")).strip()
    # 모델이 JSON 스키마 설명문을 화면 문구로 잘못 반환하는 경우 제거
    if "화면에 보이는 모든 문구" in text:
        text = ""
    return {"text": text, "findings": findings}
