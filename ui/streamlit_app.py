"""LawBee 준법심의 콘솔 (Streamlit)

실행:
    streamlit run ui/streamlit_app.py

API_BASE_URL 환경변수로 API 주소를 지정 (기본 http://127.0.0.1:8000)
디자인 레퍼런스(React 콘솔)를 Streamlit 으로 재현
"""
import html
import json
import os

import requests
import streamlit as st
import streamlit.components.v1 as components

API = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
PUBLIC_API = os.environ.get("PUBLIC_API_URL", API)

# 디자인 색 시스템
TONE = {
    "green":  {"fg": "#15803d", "bg": "#e9f7ef", "bd": "#bbe7cb", "dot": "#1ca25b"},
    "amber":  {"fg": "#b45309", "bg": "#fdf3e2", "bd": "#f3dcae", "dot": "#e08600"},
    "red":    {"fg": "#c0322b", "bg": "#fdecea", "bd": "#f6cfc9", "dot": "#dc4338"},
    "blue":   {"fg": "#1d4ed8", "bg": "#e8eefe", "bd": "#c9d8fb", "dot": "#2563eb"},
    "violet": {"fg": "#6d28d9", "bg": "#f0eafc", "bd": "#dccdf5", "dot": "#7c3aed"},
    "slate":  {"fg": "#475569", "bg": "#eef1f6", "bd": "#dde3ec", "dot": "#94a3b8"},
}
# 디자인 tone → Streamlit 네이티브 뱃지 색
STCOLOR = {"green": "green", "amber": "orange", "red": "red", "blue": "blue", "violet": "violet", "slate": "gray"}

VERDICT = {"통과": ("적합", "green"), "주의": ("조건부 적합", "amber"), "위반": ("부적합", "red")}
DSTATUS = {
    "대기": ("검토 대기", "blue"),
    "승인": ("승인 완료", "green"),
    "조건부승인": ("조건부 승인", "amber"),
    "반려": ("반려", "red"),
    "처리중": ("분석 중", "slate"),
}
DECISIONS = [("승인", "원안 그대로 게재", "green"),
             ("조건부승인", "수정 후 게재 허용", "amber"),
             ("반려", "게재 불가 · 반려", "red")]
MEDIA_OPTIONS = ["텍스트", "이미지", "영상", "UI"]
_CHANNEL_OPTIONS = ["웹/배너", "앱 푸시", "유튜브/영상", "인스타·SNS", "이메일", "지면(인쇄)"]
_CHANNEL_MEDIA: dict[str, str] = {
    "웹/배너": "이미지", "앱 푸시": "텍스트", "유튜브/영상": "영상",
    "인스타·SNS": "이미지", "이메일": "텍스트", "지면(인쇄)": "이미지",
}


# --- API 클라이언트 ---

def api_list():
    return requests.get(f"{API}/reviews", timeout=60).json()


def api_get(rid):
    return requests.get(f"{API}/reviews/{rid}", timeout=60).json()


def api_create(content, media, title=None, review_mode="표준"):
    return requests.post(f"{API}/reviews", json={"content": content, "media": media, "title": title,
                                                  "review_mode": review_mode}, timeout=180).json()


def api_create_image(file_tuple, title="", review_mode="표준"):
    return requests.post(f"{API}/reviews/image", files={"file": file_tuple},
                         data={"title": title, "review_mode": review_mode}, timeout=180).json()


def api_create_video(file_tuple, title="", review_mode="표준"):
    return requests.post(f"{API}/reviews/video", files={"file": file_tuple},
                         data={"title": title, "review_mode": review_mode}, timeout=600).json()


def api_decide(rid, decision, comment, reviewer):
    return requests.post(
        f"{API}/reviews/{rid}/decision",
        json={"decision": decision, "comment": comment, "reviewer": reviewer},
        timeout=60,
    ).json()


def api_delete(rid):
    return requests.delete(f"{API}/reviews/{rid}", timeout=30)


# --- 데이터 파생 (실데이터 → 디자인 모델) ---

def ai_score(ai):
    hits = ai["rule_hits"]
    high = sum(1 for h in hits if h["severity"] == "high")
    mid = sum(1 for h in hits if h["severity"] == "medium")
    base = {"통과": 96, "주의": 80, "위반": 58}[ai["status"]]
    return max(35, min(99, base - high * 7 - mid * 3 - max(0, len(ai["violations"]) - high - mid) * 2))


def build_findings(ai):
    """룰 탐지 + LLM 위반을 카테고리별로 병합한 finding 목록으로"""
    rank = {"high": 2, "mid": 1}
    by_cat: dict[str, dict] = {}

    def ensure(cat):
        if cat not in by_cat:
            by_cat[cat] = {"cat": cat, "sev": "mid", "phrases": [], "issue": "", "rule": ""}
        return by_cat[cat]

    for h in ai["rule_hits"]:
        f = ensure(h["category"])
        sev = "high" if h["severity"] == "high" else "mid"
        if rank[sev] > rank[f["sev"]]:
            f["sev"] = sev
        if h["term"] not in f["phrases"]:
            f["phrases"].append(h["term"])
        f["issue"] = f["issue"] or h["message"]
        f["rule"] = f["rule"] or "룰 엔진 1차 탐지"

    cmap = {c["id"]: c for c in ai["citations"]}
    vsev = "high" if ai["status"] == "위반" else "mid"
    for v in ai["violations"]:
        f = ensure(v["type"])
        if rank[vsev] > rank[f["sev"]]:
            f["sev"] = vsev
        f["issue"] = v["reason"]  # LLM 설명을 우선
        rule = "; ".join(f"{cmap[i]['law']} {cmap[i]['article']}" for i in v["citation_ids"] if i in cmap)
        if rule:
            f["rule"] = rule

    return list(by_cat.values())


def counts_of(findings):
    return (sum(1 for f in findings if f["sev"] == "high"),
            sum(1 for f in findings if f["sev"] == "mid"))


def title_of(rec):
    if rec.get("title"):
        return rec["title"]
    t = rec["content"].strip().splitlines()[0]
    return (t[:80] + "…") if len(t) > 80 else t


def highlight(content, rule_hits):
    out = html.escape(content)
    for h in rule_hits:
        term = html.escape(h["term"])
        c = TONE["red"] if h["severity"] == "high" else TONE["amber"]
        mark = (f'<mark style="background:{c["bg"]};color:{c["fg"]};'
                f'border-bottom:2px solid {c["dot"]};padding:0 2px;border-radius:3px;">{term}</mark>')
        out = out.replace(term, mark, 1)
    return out.replace("\n", "<br>")


# --- HTML 조각 ---

def badge(text, tone, dot=False, sm=False):
    c = TONE[tone]
    pad, fs, ds = ("4px 10px", "13px", "5px") if sm else ("3px 10px", "12px", "6px")
    d = (f'<span style="width:{ds}px;height:{ds}px;border-radius:999px;background:{c["dot"]};'
         f'margin-right:5px;display:inline-block;"></span>') if dot else ""
    return (f'<span style="display:inline-flex;align-items:center;padding:{pad};border-radius:999px;'
            f'font-size:{fs};font-weight:650;color:{c["fg"]};background:{c["bg"]};border:1px solid {c["bd"]};'
            f'white-space:nowrap;">{d}{html.escape(text)}</span>')


def dot_span(tone, size=7):
    return (f'<span style="display:inline-block;width:{size}px;height:{size}px;border-radius:999px;'
            f'background:{TONE[tone]["dot"]};"></span>')


MEDIA_ICON = {"텍스트": "▭", "이미지": "▧", "영상": "▷", "UI": "▢"}


def media_tag(m):
    icon = MEDIA_ICON.get(m, "▭")
    return (f'<span style="display:inline-flex;align-items:center;gap:6px;font-size:14.5px;font-weight:600;color:#405572;">'
            f'<span style="display:inline-grid;place-items:center;width:18px;height:18px;border-radius:5px;'
            f'background:#eef2f8;color:#64748b;font-size:10px;">{icon}</span>{html.escape(m or "-")}</span>')


def score_ring(value, size=46, stroke=5):
    r = (size - stroke) / 2
    circ = 2 * 3.14159 * r
    off = circ * (1 - value / 100)
    col = "#1ca25b" if value >= 90 else "#e08600" if value >= 70 else "#dc4338"
    return (
        f'<div style="position:relative;width:{size}px;height:{size}px;display:inline-block;">'
        f'<svg width="{size}" height="{size}" style="transform:rotate(-90deg);">'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="#eef1f6" stroke-width="{stroke}"/>'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{col}" stroke-width="{stroke}" '
        f'stroke-dasharray="{circ}" stroke-dashoffset="{off}" stroke-linecap="round" '
        f'style="transition:stroke-dashoffset .9s ease;"/></svg>'
        f'<div style="position:absolute;inset:0;display:grid;place-items:center;font-size:{int(size*0.26)}px;'
        f'font-weight:800;color:#0f1b2d;">{value}</div></div>'
    )


def score_arc(value, size=64, stroke=7):
    """숫자 없이 아크만 그리는 링"""
    r = (size - stroke) / 2
    circ = 2 * 3.14159 * r
    off = circ * (1 - value / 100)
    col = "#1ca25b" if value >= 90 else "#e08600" if value >= 70 else "#dc4338"
    return (
        f'<svg width="{size}" height="{size}" style="transform:rotate(-90deg);display:block;">'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="#eef1f6" stroke-width="{stroke}"/>'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{col}" stroke-width="{stroke}" '
        f'stroke-dasharray="{circ}" stroke-dashoffset="{off}" stroke-linecap="round" '
        f'style="transition:stroke-dashoffset .9s ease;"/></svg>'
    )


CSS = """
<style>
/* ── 요약 더보기 토글 ── */
details.lb-summary-toggle > summary { list-style:none; }
details.lb-summary-toggle > summary::-webkit-details-marker { display:none; }
details.lb-summary-toggle[open] > summary { display:none; }

/* ── 레이아웃 ── */
header[data-testid="stHeader"]{display:none;}
#MainMenu,footer{display:none;}
.stApp{background:#f4f6fa;}
.block-container{padding:0.4rem 1.6rem 3.5rem;max-width:820px;animation:fadeIn .35s ease;}

/* ── 사이드바 ── */
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #eaeef5;}
[data-testid="stSidebarHeader"]{display:none !important;}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding-top:1rem !important;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.25rem;}
[data-testid="stSidebar"] .stButton>button{
  background:transparent;border:none;color:#52617a;
  justify-content:flex-start !important;text-align:left !important;
  font-weight:600;font-size:13.5px;border-radius:10px;width:100%;
  padding:9px 12px;transition:all .13s ease;}
[data-testid="stSidebar"] .stButton>button>div{justify-content:flex-start !important;width:100%;}
[data-testid="stSidebar"] .stButton>button [data-testid="stMarkdownContainer"]{width:100%;text-align:left !important;}
[data-testid="stSidebar"] .stButton>button [data-testid="stMarkdownContainer"] p{text-align:left !important;margin:0;}
[data-testid="stSidebar"] .stButton>button:hover{background:#f0f4ff;color:#1d4ed8;}
[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:#eff4ff !important;color:#1d4ed8 !important;border:none !important;font-weight:700;}
[data-testid="stSidebar"] .stButton>button[kind="primary"]:hover{background:#dce9ff !important;}

/* ── 글로벌 버튼 ── */
.stButton>button{border-radius:10px;font-weight:700;transition:all .13s ease;}
.stButton>button[kind="primary"]{background:#2563eb;border:none;color:#fff !important;font-weight:800;}
.stButton>button[kind="primary"]:hover{background:#1d4ed8;transform:translateY(-1px);box-shadow:0 6px 18px rgba(37,99,235,.28);}


/* ── 새 검수 — 상단/히어로 ── */
[class*="st-key-nr-topbar"]{
  position:relative;left:auto;transform:none;
  width:calc(100vw - 244px);box-sizing:border-box;
  margin-left:calc(50% - 50vw + 122px);
  padding:0 44px 0;margin-bottom:44px;}
[class*="st-key-nr-topbar"] [data-testid="stHorizontalBlock"]{
  max-width:none !important;}
.nr-topbar-rule{
  height:1px;background:#e8edf5;margin:18px 0 0;}
.nr-ai-chip{
  position:fixed;right:24px;top:32px;z-index:120;
  background:#eaf2ff;color:#1263ff;border-radius:999px;padding:9px 18px;
  font-size:12px;font-weight:800;white-space:nowrap;}
.nr-ai-chip::before{
  content:"";display:inline-block;width:7px;height:7px;border-radius:999px;
  background:#1263ff;margin-right:7px;vertical-align:1px;}
[class*="st-key-nr-back"] button{
  width:38px !important;height:38px !important;padding:0 !important;
  border-radius:12px !important;background:#fff !important;border:1px solid #dce3ee !important;
  color:#0f172a !important;box-shadow:none !important;}
[class*="st-key-nr-back"] button:hover{
  background:#f7faff !important;border-color:#bdd4ff !important;color:#1263ff !important;}
.nr-hero-art{
  position:relative;height:104px;overflow:hidden;
  display:flex;align-items:center;justify-content:center;}
.nr-hero-art::before{
  content:"";position:absolute;inset:24px 6px 18px 0;
  background:linear-gradient(140deg,transparent 18%,rgba(18,99,255,.14) 18.5%,transparent 19.5%),
             linear-gradient(150deg,transparent 38%,rgba(18,99,255,.12) 38.5%,transparent 39.5%),
             linear-gradient(330deg,transparent 52%,rgba(18,99,255,.10) 52.5%,transparent 53.5%);
  transform:skewX(-16deg);}
.nr-spark{
  position:absolute;width:8px;height:8px;background:#7da8ff;transform:rotate(45deg);border-radius:2px;}
.nr-spark.s1{left:30px;top:54px;}
.nr-spark.s2{right:38px;top:76px;width:6px;height:6px;}
.nr-spark.s3{right:78px;top:34px;width:12px;height:12px;border-radius:999px;background:#d7e5ff;}
.nr-shield-orb{
  position:relative;width:76px;height:76px;border-radius:999px;background:#fff;
  display:grid;place-items:center;box-shadow:0 14px 34px rgba(18,99,255,.14),0 0 0 15px rgba(238,245,255,.88);}

/* ── 새 검수 — 섹션 카드 ── */
[class*="st-key-nr-sec-"]{
  margin-bottom:18px;background:#fff !important;border-radius:18px !important;
  padding:22px 24px !important;border:1px solid #edf0f5 !important;
  box-shadow:0 10px 26px rgba(15,23,42,.04) !important;}
[class*="st-key-nr-sec-"] [data-testid="stVerticalBlock"]{gap:0.9rem;}

/* ── 새 검수 — 유형 카드 버튼 ── */
[class*="st-key-nr-sec-1"]{
  padding:38px 42px 44px !important;border-radius:26px !important;
  box-shadow:0 22px 48px rgba(15,23,42,.07) !important;}
[class*="st-key-type-text"],
[class*="st-key-type-image"],
[class*="st-key-type-video"]{position:relative !important;}
[class*="st-key-type-text"] button,
[class*="st-key-type-image"] button,
[class*="st-key-type-video"] button{
  position:relative !important;display:flex !important;flex-direction:column !important;
  align-items:center !important;justify-content:center !important;gap:14px !important;
  background:#fff !important;border:1.5px solid #e0e6ef !important;
  color:#111827 !important;border-radius:24px !important;
  padding:32px 24px 28px !important;min-height:250px !important;
  box-shadow:none !important;text-align:center !important;
  transition:border-color .15s,background .15s,box-shadow .15s !important;}
[class*="st-key-type-text"] button p,
[class*="st-key-type-image"] button p,
[class*="st-key-type-video"] button p{
  font-size:18px !important;font-weight:850 !important;color:#111827 !important;
  line-height:1.15 !important;margin:0 !important;letter-spacing:-.025em !important;}
[class*="st-key-type-text"] button::before,
[class*="st-key-type-image"] button::before,
[class*="st-key-type-video"] button::before{
  content:"";display:block;width:96px;height:96px;margin-bottom:8px;border-radius:999px;
  background-color:#f1f5f9;background-repeat:no-repeat;background-position:center;background-size:44px 44px;}
[class*="st-key-type-text"] button::before{
  background-image:url("data:image/svg+xml,%3Csvg width='44' height='44' viewBox='0 0 44 44' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M15 17h14M15 22h14M15 27h10' stroke='%235a6b83' stroke-width='3.2' stroke-linecap='round'/%3E%3C/svg%3E");}
[class*="st-key-type-image"] button::before{
  background-image:url("data:image/svg+xml,%3Csvg width='44' height='44' viewBox='0 0 44 44' xmlns='http://www.w3.org/2000/svg'%3E%3Crect x='11' y='12' width='22' height='20' rx='4' fill='none' stroke='%235a6b83' stroke-width='3'/%3E%3Ccircle cx='18' cy='18' r='2.5' fill='%235a6b83'/%3E%3Cpath d='M14 29l6.2-6.3 4.1 4 3-3.1L33 29' fill='none' stroke='%235a6b83' stroke-width='2.8' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");}
[class*="st-key-type-video"] button::before{
  background-image:url("data:image/svg+xml,%3Csvg width='44' height='44' viewBox='0 0 44 44' xmlns='http://www.w3.org/2000/svg'%3E%3Crect x='10' y='15' width='20' height='14' rx='4' fill='none' stroke='%235a6b83' stroke-width='3'/%3E%3Cpath d='M31 19l6-3.5v13L31 25z' fill='none' stroke='%235a6b83' stroke-width='3' stroke-linejoin='round'/%3E%3C/svg%3E");}
[class*="st-key-type-text"] button::after,
[class*="st-key-type-image"] button::after,
[class*="st-key-type-video"] button::after{
  display:block;font-size:10px;font-weight:550;line-height:1.45;color:#69778e;margin-top:2px;}
[class*="st-key-type-text"] button::after{content:"문서, 문구, 문장 등 텍스트 콘텐츠";}
[class*="st-key-type-image"] button::after{content:"사진, 배너, 포스터 등 이미지 콘텐츠";}
[class*="st-key-type-video"] button::after{content:"동영상, TVC, 유튜브 등 영상 콘텐츠";}
[class*="st-key-type-text"] button[kind="primary"],
[class*="st-key-type-image"] button[kind="primary"],
[class*="st-key-type-video"] button[kind="primary"]{
  background:#fbfdff !important;border:2px solid #1263ff !important;color:#1263ff !important;
  box-shadow:0 14px 34px rgba(18,99,255,.06) !important;}
[class*="st-key-type-text"] button[kind="primary"] p,
[class*="st-key-type-image"] button[kind="primary"] p,
[class*="st-key-type-video"] button[kind="primary"] p{color:#1263ff !important;}
[class*="st-key-type-text"] button[kind="primary"]::before,
[class*="st-key-type-image"] button[kind="primary"]::before,
[class*="st-key-type-video"] button[kind="primary"]::before{
  background-color:#eef5ff;}
[class*="st-key-type-text"] button[kind="primary"]::before{
  background-image:url("data:image/svg+xml,%3Csvg width='44' height='44' viewBox='0 0 44 44' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M15 17h14M15 22h14M15 27h10' stroke='%231263ff' stroke-width='3.2' stroke-linecap='round'/%3E%3C/svg%3E");}
[class*="st-key-type-image"] button[kind="primary"]::before{
  background-image:url("data:image/svg+xml,%3Csvg width='44' height='44' viewBox='0 0 44 44' xmlns='http://www.w3.org/2000/svg'%3E%3Crect x='11' y='12' width='22' height='20' rx='4' fill='none' stroke='%231263ff' stroke-width='3'/%3E%3Ccircle cx='18' cy='18' r='2.5' fill='%231263ff'/%3E%3Cpath d='M14 29l6.2-6.3 4.1 4 3-3.1L33 29' fill='none' stroke='%231263ff' stroke-width='2.8' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");}
[class*="st-key-type-video"] button[kind="primary"]::before{
  background-image:url("data:image/svg+xml,%3Csvg width='44' height='44' viewBox='0 0 44 44' xmlns='http://www.w3.org/2000/svg'%3E%3Crect x='10' y='15' width='20' height='14' rx='4' fill='none' stroke='%231263ff' stroke-width='3'/%3E%3Cpath d='M31 19l6-3.5v13L31 25z' fill='none' stroke='%231263ff' stroke-width='3' stroke-linejoin='round'/%3E%3C/svg%3E");}
[class*="st-key-type-text"] button:hover:not([kind="primary"]),
[class*="st-key-type-image"] button:hover:not([kind="primary"]),
[class*="st-key-type-video"] button:hover:not([kind="primary"]){
  border-color:#9bbcff !important;background:#fbfdff !important;}

[class*="st-key-type-text"]:has(button[kind="primary"])::after,
[class*="st-key-type-image"]:has(button[kind="primary"])::after,
[class*="st-key-type-video"]:has(button[kind="primary"])::after{
  content:'';position:absolute;top:32px;right:32px;width:34px;height:34px;
  border-radius:50%;pointer-events:none;z-index:10;
  background:#1263ff url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 12'%3E%3Cpath d='M2 6.5L4.5 9 10 3.5' stroke='white' stroke-width='1.9' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") center/16px no-repeat;
  box-shadow:0 8px 18px rgba(18,99,255,.22);}

/* ── 새 검수 — 심의 강도 카드 버튼 ── */
[class*="st-key-rm-strict"],
[class*="st-key-rm-standard"],
[class*="st-key-rm-relaxed"],
[class*="st-key-rm-ai"]{position:relative !important;}

[class*="st-key-rm-strict"] button,
[class*="st-key-rm-standard"] button,
[class*="st-key-rm-relaxed"] button,
[class*="st-key-rm-ai"] button{
  position:relative !important;display:flex !important;flex-direction:column !important;
  align-items:center !important;justify-content:center !important;gap:8px !important;
  background:#fff !important;border:1.5px solid #e0e6ef !important;
  color:#111827 !important;border-radius:16px !important;
  padding:16px 10px !important;min-height:102px !important;text-align:center !important;
  box-shadow:none !important;transition:border-color .15s,background .15s !important;}
[class*="st-key-rm-strict"] button p,[class*="st-key-rm-standard"] button p,
[class*="st-key-rm-relaxed"] button p,[class*="st-key-rm-ai"] button p{
  font-size:0 !important;margin:0 !important;line-height:0 !important;}
[class*="st-key-rm-strict"] button::before,
[class*="st-key-rm-standard"] button::before,
[class*="st-key-rm-relaxed"] button::before,
[class*="st-key-rm-ai"] button::before{
  content:"";display:block;width:46px;height:30px;margin-bottom:2px;
  background-repeat:no-repeat;background-position:center;background-size:46px 30px;}
[class*="st-key-rm-strict"] button::before{
  background-image:url("data:image/svg+xml,%3Csvg width='46' height='30' viewBox='0 0 46 30' xmlns='http://www.w3.org/2000/svg'%3E%3Crect x='5' y='17' width='9' height='8' rx='2' fill='%2360a5fa'/%3E%3Crect x='18.5' y='10' width='9' height='15' rx='2' fill='%233b82f6'/%3E%3Crect x='32' y='4' width='9' height='21' rx='2' fill='%232563eb'/%3E%3C/svg%3E");}
[class*="st-key-rm-standard"] button::before{
  background-image:url("data:image/svg+xml,%3Csvg width='46' height='30' viewBox='0 0 46 30' xmlns='http://www.w3.org/2000/svg'%3E%3Crect x='5' y='17' width='9' height='8' rx='2' fill='%2393c5fd'/%3E%3Crect x='18.5' y='10' width='9' height='15' rx='2' fill='%2360a5fa'/%3E%3Crect x='32' y='5' width='9' height='20' rx='2' fill='%233b82f6'/%3E%3C/svg%3E");}
[class*="st-key-rm-relaxed"] button::before{
  background-image:url("data:image/svg+xml,%3Csvg width='46' height='30' viewBox='0 0 46 30' xmlns='http://www.w3.org/2000/svg'%3E%3Crect x='5' y='18' width='9' height='7' rx='2' fill='%23cbd5e1'/%3E%3Crect x='18.5' y='14' width='9' height='11' rx='2' fill='%2394a3b8'/%3E%3Crect x='32' y='9' width='9' height='16' rx='2' fill='%236b7a90'/%3E%3C/svg%3E");}
[class*="st-key-rm-ai"] button::before{
  background-image:url("data:image/svg+xml,%3Csvg width='46' height='30' viewBox='0 0 46 30' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M23 3l3.6 8.4L35 15l-8.4 3.6L23 27l-3.6-8.4L11 15l8.4-3.6L23 3z' fill='none' stroke='%233b82f6' stroke-width='3' stroke-linejoin='round'/%3E%3Ccircle cx='36.5' cy='6.5' r='2.5' fill='%2393c5fd'/%3E%3C/svg%3E");}
[class*="st-key-rm-strict"] button::after,
[class*="st-key-rm-standard"] button::after,
[class*="st-key-rm-relaxed"] button::after,
[class*="st-key-rm-ai"] button::after{
  display:block;white-space:pre;line-height:1.55;font-size:12px;font-weight:650;color:#7a8596;}
[class*="st-key-rm-strict"] button::after{content:"강화\\A 엄격한 심의 기준";}
[class*="st-key-rm-standard"] button::after{content:"표준   기본\\A 일반적인 심의 기준";}
[class*="st-key-rm-relaxed"] button::after{content:"완화\\A 완화된 심의 기준";}
[class*="st-key-rm-ai"] button::after{content:"AI 추천\\A AI가 상황에 맞춰 추천";}
[class*="st-key-rm-strict"] button[kind="primary"],
[class*="st-key-rm-standard"] button[kind="primary"],
[class*="st-key-rm-relaxed"] button[kind="primary"],
[class*="st-key-rm-ai"] button[kind="primary"]{
  background:#fbfdff !important;border:2px solid #1263ff !important;
  color:#1263ff !important;font-weight:700 !important;box-shadow:none !important;}
[class*="st-key-rm-strict"] button[kind="primary"]::after,
[class*="st-key-rm-standard"] button[kind="primary"]::after,
[class*="st-key-rm-relaxed"] button[kind="primary"]::after,
[class*="st-key-rm-ai"] button[kind="primary"]::after{
  color:#1263ff !important;}
[class*="st-key-rm-strict"] button:hover:not([kind="primary"]),
[class*="st-key-rm-standard"] button:hover:not([kind="primary"]),
[class*="st-key-rm-relaxed"] button:hover:not([kind="primary"]),
[class*="st-key-rm-ai"] button:hover:not([kind="primary"]){
  border-color:#9bbcff !important;background:#fbfdff !important;}
[class*="st-key-rm-strict"]:has(button[kind="primary"])::after,
[class*="st-key-rm-standard"]:has(button[kind="primary"])::after,
[class*="st-key-rm-relaxed"]:has(button[kind="primary"])::after,
[class*="st-key-rm-ai"]:has(button[kind="primary"])::after{
  content:'';position:absolute;top:12px;right:12px;width:20px;height:20px;
  border-radius:50%;pointer-events:none;z-index:10;
  background:#1263ff url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 12'%3E%3Cpath d='M2 6.5L4.5 9 10 3.5' stroke='white' stroke-width='1.9' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") center/11px no-repeat;}

[class*="st-key-nr-sec-3"]{
  padding-bottom:32px !important;}

/* ── 새 검수 — 매체 채널 필 버튼 ── */
[class*="st-key-nr-sec-2"]{
  padding:36px 42px 46px !important;border-radius:26px !important;
  box-shadow:0 22px 48px rgba(15,23,42,.07) !important;}
[class*="st-key-ch-"] button{
  background:#fff !important;border:1.5px solid #dfe6f0 !important;
  color:#26384f !important;border-radius:13px !important;
  padding:0 6px !important;height:48px !important;min-height:48px !important;box-shadow:none !important;
  transition:background .13s,color .13s,border .13s !important;}
[class*="st-key-ch-"] button p{
  white-space:nowrap !important;font-size:12.5px !important;line-height:1 !important;margin:0 !important;
  font-weight:650 !important;}
[class*="st-key-ch-"] button[kind="primary"]{
  background:#fff !important;border:1.5px solid #1263ff !important;
  color:#1263ff !important;font-weight:700 !important;
  box-shadow:0 8px 18px rgba(18,99,255,.08) !important;}
[class*="st-key-ch-"] button:hover:not([kind="primary"]){
  background:#fbfdff !important;color:#1263ff !important;border-color:#9bbcff !important;}
[class*="st-key-nr-sample"] button{
  width:108px !important;min-width:108px !important;height:38px !important;min-height:38px !important;
  border-radius:12px !important;background:#fff !important;
  border:1.5px solid #dfe6f0 !important;color:#26384f !important;
  padding:0 10px !important;box-sizing:border-box !important;
  box-shadow:none !important;}
[class*="st-key-nr-sample"] button p{
  white-space:nowrap !important;font-size:12.5px !important;line-height:1 !important;margin:0 !important;
  font-weight:650 !important;}
[class*="st-key-nr-sample"]{
  transform:translateX(-34px);}
[class*="st-key-nr-sample"] button:hover{
  border-color:#9bbcff !important;color:#1263ff !important;background:#fbfdff !important;}

/* ── 새 검수 — 하단 고정 바 ── */
[class*="st-key-nr-actions"]{
  position:fixed;left:244px;right:0;bottom:0;
  background:rgba(255,255,255,.93);backdrop-filter:blur(10px);
  border-top:1px solid #e5e7eb;padding:18px 48px;z-index:100;
  box-shadow:0 -18px 45px rgba(15,23,42,.04);}
[class*="st-key-nr-actions"] > div:first-child{
  max-width:620px;margin:0 auto;}
[class*="st-key-nr-actions"] [data-testid="stHorizontalBlock"]{gap:20px;}

/* ── 새 검수 — 하단 저장/제출 버튼 ── */
[class*="st-key-nr-save"] button{
  background:#f6f8fb !important;border:none !important;
  color:#111827 !important;border-radius:18px !important;
  font-size:15px !important;font-weight:850 !important;padding:16px 0 14px !important;
  min-height:68px !important;
  box-shadow:none !important;transition:background .13s !important;}
[class*="st-key-nr-save"] button p{line-height:1.55 !important;margin:0 !important;}
[class*="st-key-nr-save"] button p strong{font-size:15px !important;font-weight:850 !important;color:#111827 !important;}
[class*="st-key-nr-save"] button p em{font-style:normal !important;font-size:12px !important;font-weight:550 !important;color:#8b95a1 !important;}
[class*="st-key-nr-save"] button:hover{background:#edf0f5 !important;}
[class*="st-key-nr-submit"] button{
  background:linear-gradient(135deg,#1677ff,#0057ff) !important;
  border:none !important;border-radius:18px !important;
  font-size:16px !important;font-weight:850 !important;padding:16px 0 14px !important;
  min-height:68px !important;
  box-shadow:0 14px 30px rgba(18,99,255,.25) !important;
  letter-spacing:0 !important;}
[class*="st-key-nr-submit"] button p{line-height:1.55 !important;margin:0 !important;color:#fff !important;}
[class*="st-key-nr-submit"] button p strong{font-size:16px !important;font-weight:900 !important;color:#fff !important;}
[class*="st-key-nr-submit"] button p em{font-style:normal !important;font-size:12px !important;font-weight:600 !important;color:#dbeafe !important;}
[class*="st-key-nr-submit"] button:hover{
  box-shadow:0 18px 36px rgba(18,99,255,.35) !important;transform:translateY(-1px) !important;}
[class*="st-key-nr-submit"] button:disabled{
  background:#c7d8f8 !important;box-shadow:none !important;transform:none !important;}

@media(max-width:900px){
  .block-container{padding-left:1rem !important;padding-right:1rem !important;}
  [class*="st-key-nr-topbar"]{
    left:auto;transform:none;width:100%;padding:0;margin-bottom:32px;}
  .nr-ai-chip{right:16px;top:18px;}
  [class*="st-key-nr-actions"]{left:0;padding:14px 16px;}
  [class*="st-key-nr-actions"] [data-testid="stHorizontalBlock"]{display:block;}
  [class*="st-key-nr-save"]{margin-bottom:10px;}
  .nr-hero-art{display:none;}
}

/* ── 결재 버튼 ── */
[class*="st-key-dec-"] button p{font-size:11px !important;line-height:1.5;margin:0;}
[class*="st-key-dec-"] button p strong{font-size:14.5px !important;font-weight:800;}

/* ── 입력 박스 ── */
[data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input{
  border:1.5px solid #e2e8f0 !important;background:#fff !important;
  border-radius:10px !important;font-size:14px !important;min-height:46px !important;}
[class*="st-key-nr-sec-2"] [data-testid="stTextInput"] input{
  height:52px !important;min-height:52px !important;border-radius:13px !important;
  font-size:14px !important;line-height:1.4 !important;color:#27364e !important;
  padding:0 18px !important;box-sizing:border-box !important;box-shadow:none !important;}
[class*="st-key-nr-sec-2"] [data-baseweb="input"]{
  height:52px !important;min-height:52px !important;border-radius:13px !important;
  display:flex !important;align-items:center !important;box-shadow:none !important;}
[class*="st-key-nr-sec-2"] [data-baseweb="textarea"]{
  box-shadow:none !important;border-radius:13px !important;}
[class*="st-key-nr-sec-2"] [data-testid="stTextArea"] textarea{
  border-radius:13px !important;font-size:14px !important;height:238px !important;min-height:238px !important;
  padding:18px 20px !important;line-height:1.65 !important;color:#27364e !important;box-shadow:none !important;}
[class*="st-key-nr-sec-2"] [data-testid="stTextInput"] input:focus,
[class*="st-key-nr-sec-2"] [data-testid="stTextArea"] textarea:focus{
  box-shadow:none !important;}
[data-testid="stTextArea"] textarea:focus,[data-testid="stTextInput"] input:focus{
  border-color:#2563eb !important;box-shadow:0 0 0 3px rgba(37,99,235,.1) !important;}

/* ── 파일 업로더 ── */
[data-testid="stFileUploaderDropzone"]{
  position:relative !important;flex-direction:column !important;gap:10px !important;
  border:2px dashed #dde4ee !important;border-radius:14px !important;
  background:#fafbfd !important;padding:28px 24px 22px !important;
  min-height:150px !important;display:flex !important;align-items:center !important;
  justify-content:center !important;}
[data-testid="stFileUploaderDropzone"]:hover{
  border-color:#2563eb !important;background:#f0f5ff !important;}
[data-testid="stFileUploaderDropzone"] button{
  width:44px !important;height:44px !important;min-height:44px !important;padding:0 !important;
  border:none !important;border-radius:13px !important;background:#fff !important;
  box-shadow:0 6px 18px rgba(15,23,42,.08) !important;font-size:0 !important;
  display:grid !important;place-items:center !important;}
[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stFileUploaderDropzone"] button svg,
[data-testid="stFileUploaderDropzone"] button span{
  display:none !important;}
[data-testid="stFileUploaderDropzone"] button::before{
  content:"";display:block;width:24px;height:24px;background-repeat:no-repeat;
  background-position:center;background-size:24px 24px;
  background-image:url("data:image/svg+xml,%3Csvg width='34' height='34' viewBox='0 0 34 34' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M17 22V8m0 0l-6 6m6-6l6 6' fill='none' stroke='%238b95a1' stroke-width='2.7' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M9 21v4.5A2.5 2.5 0 0011.5 28h11a2.5 2.5 0 002.5-2.5V21' fill='none' stroke='%238b95a1' stroke-width='2.7' stroke-linecap='round'/%3E%3C/svg%3E");}
[data-testid="stFileUploaderDropzone"]::after{
  content:"파일을 끌어다 놓거나 클릭해서 업로드\\A MP4 · MOV · WEBM · 음성파일 · 최대 200MB";
  white-space:pre;text-align:center;line-height:1.6;
  color:#8b95a1;font-size:12px;font-weight:600;}
[data-testid="stFileUploaderDropzone"]::first-line{
  color:#374151;font-size:13px;font-weight:750;}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p,
[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploaderDropzone"] [class*="Instructions"],
[data-testid="stFileUploaderDropzone"] [class*="instructions"]{
  display:none !important;}
.nr-field-label{
  display:block;font-size:17px;font-weight:850;color:#111827;margin:4px 0 12px;}
.nr-field-label .hint{
  color:#9aa3af;font-size:14px;font-weight:750;margin-left:5px;}
.nr-field-help{
  color:#8b95a1;font-size:13px;font-weight:650;margin:0 0 14px;}

h1,h2,h3{color:#0f1b2d;}
mark{text-decoration:none;}

/* ── 대시보드 ── */
[class*="st-key-listcard"]{background:#fff !important;border:1px solid #e6eaf1 !important;
  border-radius:18px !important;box-shadow:0 14px 32px rgba(16,24,40,.045);padding:12px 26px 16px !important;}
a.lb-row{color:inherit;transition:background .12s;animation:fadeIn .3s ease;border-radius:10px;display:block;}
a.lb-row:hover{background:#f8fafe;}
[class*="st-key-listcard"] [data-testid="stVerticalBlock"]{gap:0 !important;}
[class*="st-key-listcard"] [data-testid="stHorizontalBlock"]{gap:.35rem !important;margin:0 !important;}
[class*="st-key-listcard"] [data-testid="stElementContainer"]{margin:0 !important;}
[class*="st-key-listcard"] [data-testid="stCheckbox"]{display:flex;align-items:center;height:64px;transform:translateY(-5px);}
[class*="st-key-dash-filter-"] button{
  min-height:40px !important;border-radius:11px !important;padding:8px 18px !important;
  font-size:13px !important;font-weight:850 !important;box-shadow:0 5px 14px rgba(16,24,40,.04) !important;
  white-space:nowrap !important;}
[class*="st-key-dash-filter-"] button[kind="secondary"]{
  background:#fff !important;border:1px solid #e6eaf1 !important;color:#344055 !important;}
[class*="st-key-dash-filter-"] button[kind="primary"]{
  background:#2f7df6 !important;color:#fff !important;border:0 !important;
  box-shadow:0 8px 18px rgba(47,125,246,.26) !important;}
[class*="st-key-new-review-btn"] button{
  min-height:44px !important;border-radius:13px !important;
  background:#2f7df6 !important;color:#fff !important;border:0 !important;
  font-size:13.5px !important;font-weight:850 !important;
  box-shadow:0 10px 22px rgba(47,125,246,.26) !important;}
[class*="st-key-new-review-btn"] button:hover{
  background:#256ee8 !important;transform:translateY(-1px);
  box-shadow:0 12px 26px rgba(47,125,246,.3) !important;}
[class*="st-key-bulk-del-btn"] button{
  min-height:44px !important;border-radius:13px !important;
  background:#fff !important;color:#d1433c !important;border:1px solid #f0d3d0 !important;
  font-size:13.5px !important;font-weight:850 !important;
  box-shadow:0 8px 18px rgba(16,24,40,.04) !important;}
[class*="st-key-bulk-del-btn"] button:hover{background:#fff5f4 !important;border-color:#efb4ae !important;}
[class*="st-key-bulk-del-btn"] button:disabled{
  background:#fff !important;color:#cbd3df !important;border-color:#e7ecf3 !important;
  box-shadow:none !important;}
[class*="st-key-del-confirm"] button{background:#c0322b !important;border:none !important;color:#fff !important;font-weight:800;}

@keyframes fadeIn{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:none;}}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.35;}}
.lb-anim{animation:fadeIn .35s ease;}
.lb-pulse{animation:pulse 1.7s infinite;}
.lb-kpi{transition:all .15s ease;}
.lb-kpi:hover{transform:translateY(-1px);box-shadow:0 10px 22px rgba(16,24,40,.06);border-color:#d7e2f2 !important;}
.lb-finding{transition:all .15s ease;}
.lb-finding:hover{transform:translateY(-1px);box-shadow:0 5px 16px rgba(16,24,40,.08);}
</style>
"""

LAW_BEE_MARK = (
    '<svg width="24" height="24" viewBox="0 0 32 32" fill="none" aria-hidden="true">'
    '<path d="M16 3.5 26.8 9.7v12.6L16 28.5 5.2 22.3V9.7L16 3.5Z" '
    'fill="rgba(255,255,255,.16)" stroke="white" stroke-width="1.8" stroke-linejoin="round"/>'
    '<path d="M10.7 16.2 14.2 19.7 21.6 12.4" stroke="white" stroke-width="2.4" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M23.7 6.2 25.2 3.6M7 7.3 4.7 5.6M26.2 24.8 28.7 26.2" '
    'stroke="rgba(255,255,255,.72)" stroke-width="1.6" stroke-linecap="round"/>'
    '<circle cx="24.6" cy="6.3" r="1.4" fill="white" fill-opacity=".85"/>'
    '</svg>'
)


# --- 네비게이션 ---

def go(view, rid=None):
    st.query_params.clear()
    st.session_state["view"] = view
    st.session_state["rid"] = rid
    st.session_state.pop("decision", None)
    st.markdown("<script>window.parent.scrollTo({top:0,behavior:'instant'});</script>",
                unsafe_allow_html=True)
    st.rerun()


def sidebar():
    with st.sidebar:
        # 로고
        st.markdown(
            '<a href="?view=dashboard" target="_self" style="text-decoration:none;display:block;">'
            '<div style="display:flex;align-items:center;gap:10px;padding:6px 2px 18px;">'
            f'<div style="width:40px;height:40px;border-radius:14px;background:linear-gradient(135deg,#1d4ed8 0%,#2563eb 48%,#06b6d4 100%);'
            f'display:grid;place-items:center;box-shadow:0 10px 24px rgba(37,99,235,.28);flex-shrink:0;">{LAW_BEE_MARK}</div>'
            '<div style="line-height:1.15;">'
            '<div style="font-weight:900;font-size:20px;color:#0f1b2d;letter-spacing:-.03em;">LawBee</div>'
            '</div></div></a>',
            unsafe_allow_html=True)

        # 사용자 카드
        st.markdown(
            '<div style="background:#f6f8fc;border:1px solid #eaeef5;border-radius:11px;padding:10px 12px;'
            'display:flex;align-items:center;gap:10px;margin-bottom:22px;">'
            '<div style="width:32px;height:32px;border-radius:999px;background:linear-gradient(135deg,#3b82f6,#1d4ed8);'
            'display:grid;place-items:center;font-size:13px;font-weight:800;color:#fff;flex-shrink:0;">김</div>'
            '<div><div style="font-size:13px;font-weight:750;color:#1e2d40;">김준기</div>'
            '<div style="font-size:11px;color:#94a3b8;font-weight:550;margin-top:1px;">준법관리자 · 결재권자</div></div>'
            '</div>',
            unsafe_allow_html=True)

        # 네비게이션
        st.markdown(
            '<div style="font-size:10px;font-weight:750;color:#b0bac8;letter-spacing:.1em;padding:0 4px 8px;"</div>',
            unsafe_allow_html=True)

        view = st.session_state.get("view", "dashboard")
        try:
            pending = sum(1 for r in api_list() if r["decision_status"] == "대기")
        except Exception:
            pending = 0

        dash_label = ":material/grid_view: 검수 대시보드"
        if pending:
            dash_label += f"&nbsp; :blue-badge[{pending}]"
        if st.button(dash_label, use_container_width=True, key="nav-dash",
                     type="primary" if view in ("dashboard", "detail", "new") else "secondary"):
            go("dashboard")
        for icon, label in [("menu_book", "심의 규정"), ("history", "결재 이력"), ("bar_chart", "통계·리포트")]:
            if st.button(f":material/{icon}: {label}", use_container_width=True, key=f"nav-{label}"):
                st.toast("준비 중인 메뉴입니다")

        # AI 엔진 상태
        st.markdown(
            '<div style="margin-top:20px;background:#f6f8fc;border:1px solid #eaeef5;border-radius:11px;padding:12px 13px;">'
            '<div style="display:flex;align-items:center;gap:7px;margin-bottom:5px;">'
            '<span class="lb-pulse" style="width:7px;height:7px;border-radius:999px;background:#16934f;'
            'box-shadow:0 0 0 3px rgba(22,147,79,.15);display:inline-block;"></span>'
            '<span style="font-size:11.5px;font-weight:700;color:#3a4a63;">AI 심의엔진</span>'
            '<span style="margin-left:auto;font-size:10.5px;color:#16934f;font-weight:750;">정상</span></div>'
            '<div style="font-size:10.5px;color:#94a3b8;line-height:1.5;">규정셋 v4.2 · 2026.05.30</div>'
            '</div>'
            '<div style="font-size:10.5px;color:#b8c2d0;line-height:1.7;padding:14px 4px 4px;">'
            'JB금융그룹 준법감시부<br>심의 콘솔 v1.0</div>',
            unsafe_allow_html=True)


def topbar(crumb, title, right_html=""):
    st.markdown(
        f'<div class="lb-anim" style="display:flex;align-items:center;padding:0 2px 12px;border-bottom:1px solid #e6eaf1;margin-bottom:18px;">'
        f'<div><div style="font-size:11px;font-weight:700;color:#2563eb;margin-bottom:3px;letter-spacing:.03em;">{html.escape(crumb)}</div>'
        f'<div style="font-size:20px;font-weight:800;color:#0f1b2d;letter-spacing:-.02em;">{html.escape(title)}</div></div>'
        f'<div style="margin-left:auto;">{right_html}</div></div>',
        unsafe_allow_html=True)


def live_dot(text):
    return (f'<span style="font-size:12.5px;color:#64748b;font-weight:600;display:inline-flex;align-items:center;gap:6px;">'
            f'<span class="lb-pulse" style="width:7px;height:7px;border-radius:999px;background:#34d399;display:inline-block;"></span>{text}</span>')


# --- 화면: 대시보드 ---

@st.dialog("선택 항목 삭제")
def _confirm_delete(ids):
    st.markdown(
        f'<div style="font-size:14.5px;color:#27364e;line-height:1.6;">선택한 <b>{len(ids)}건</b>을 삭제하시겠습니까?</div>'
        f'<div style="font-size:12.5px;color:#94a3b8;margin-top:6px;">'
        + ", ".join(f"RV-{i:04d}" for i in ids) + ' · 되돌릴 수 없습니다</div>',
        unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    if c1.button("취소", use_container_width=True, key="del-cancel"):
        st.session_state.pop("pending_bulk", None)
        st.rerun()
    if c2.button("삭제", type="primary", use_container_width=True, key="del-confirm"):
        for i in ids:
            api_delete(i)
            st.session_state.pop(f"sel-{i}", None)
        st.session_state.pop("pending_bulk", None)
        st.toast(f"{len(ids)}건 삭제됨")
        st.rerun()


def dashboard():
    st.markdown("<style>.block-container{max-width:1500px !important;padding:1.4rem 3.2rem 3.5rem !important;}</style>",
                unsafe_allow_html=True)
    topbar("준법심의 콘솔", "검수 대시보드", live_dot("AI 1차 심의 자동 적용 중"))

    if st.session_state.get("pending_bulk"):
        _confirm_delete(sorted(st.session_state["pending_bulk"]))

    try:
        reviews = api_list()
    except requests.RequestException:
        st.error(f"API 서버에 연결할 수 없습니다 ({API})")
        return

    c = {"전체": len(reviews), "대기": 0, "승인": 0, "조건부승인": 0, "반려": 0}
    for r in reviews:
        c[r["decision_status"]] = c.get(r["decision_status"], 0) + 1

    tiles = [
        ("전체 검수", c["전체"], "누적 요청", True),
        ("검토 대기", c["대기"], "결재 필요", False),
        ("승인 완료", c["승인"], "원안 승인", False),
        ("조건부 승인", c["조건부승인"], "수정 후", False),
        ("반려", c["반려"], "재작업", False),
    ]
    for col, (label, val, sub, active) in zip(st.columns(5, gap="small"), tiles):
        border = "#2f7df6" if active else "#e7ecf3"
        bg = "#f8fbff" if active else "#fff"
        num_color = "#2563eb" if active else "#101828"
        col.markdown(
            f'<div class="lb-anim lb-kpi" style="height:96px;background:{bg};'
            f'border:1.5px solid {border};border-radius:16px;'
            f'padding:20px 22px;box-shadow:0 8px 22px rgba(16,24,40,.03);'
            f'display:flex;flex-direction:column;justify-content:center;">'
            f'<div style="font-size:17px;font-weight:900;color:#3f4d63;letter-spacing:-.025em;line-height:1.1;">'
            f'{label}</div>'
            f'<div style="display:flex;align-items:flex-end;gap:8px;margin-top:14px;width:100%;">'
            f'<span style="font-size:31px;font-weight:900;color:{num_color};line-height:.86;letter-spacing:-.035em;">{val}</span>'
            f'<span style="font-size:13px;font-weight:800;color:#8895a8;padding-bottom:2px;">건</span>'
            f'<span style="margin-left:auto;text-align:right;font-size:13px;font-weight:800;color:#a2adbd;padding-bottom:2px;">{sub}</span>'
            f'</div></div>',
            unsafe_allow_html=True)

    st.write("")
    filter_map = {"전체": "전체", "대기": "대기", "승인": "승인", "조건부승인": "조건부승인", "반려": "반려"}
    flt = st.session_state.get("dash_filter", "전체")
    fcols = st.columns([0.5, 0.5, 0.5, 0.9, 0.5, 4.2], gap="small")
    for col, label in zip(fcols[:5], filter_map):
        if col.button(label, use_container_width=True, key=f"dash-filter-{label}",
                      type="primary" if flt == filter_map[label] else "secondary"):
            st.session_state["dash_filter"] = filter_map[label]
            st.rerun()
    flt = st.session_state.get("dash_filter", "전체")

    tc1, tc2, tc3, tc4 = st.columns([6.0, 0.95, 1.15, 0.78], gap="small", vertical_alignment="center")
    q = tc1.text_input("검색", placeholder="제목·번호 검색", label_visibility="collapsed")
    media = tc2.selectbox("매체", ["모든 매체"] + MEDIA_OPTIONS, label_visibility="collapsed")
    if tc3.button(":material/add: 새 검수 요청", type="primary", use_container_width=True, key="new-review-btn"):
        go("new")
    rows = [r for r in reviews
            if (flt in (None, "전체") or r["decision_status"] == flt)
            and (media == "모든 매체" or r["media"] == media)
            and (not q or q in title_of(r) or q in f"RV-{r['id']:04d}")]
    selected = [r["id"] for r in rows if st.session_state.get(f"sel-{r['id']}")]
    delete_label = f":material/delete: 삭제 {len(selected)}" if selected else ":material/delete: 삭제"
    if tc4.button(delete_label, use_container_width=True,
                  key="bulk-del-btn", disabled=not selected, help="선택 항목 삭제"):
        st.session_state["pending_bulk"] = set(selected)
        st.rerun()

    with st.container(border=True, key="listcard"):
        st.markdown(
            f'<div style="font-weight:900;font-size:17px;color:#0f1b2d;letter-spacing:-.02em;'
            f'padding-bottom:12px;padding-top:4px;">검수 목록 &nbsp;'
            f'<span style="font-size:12px;font-weight:800;color:#7b8798;background:#eef2f7;'
            f'border-radius:999px;padding:3px 10px;vertical-align:middle;">{len(rows)}건</span></div>',
            unsafe_allow_html=True)

        grid = "minmax(300px,3.4fr) .72fr .88fr .9fr .78fr .48fr"
        heads = ["콘텐츠 / 요청", "매체", "AI 심의결과", "발견 항목", "처리 상태", ""]
        hc = st.columns([0.032, 0.968], gap="small", vertical_alignment="center")
        hc[1].markdown(
            f'<div style="display:grid;grid-template-columns:{grid};gap:18px;padding:2px 8px 12px;'
            f'font-size:13px;font-weight:850;color:#9aa6b8;border-bottom:1px solid #edf1f6;">'
            + "".join(f"<div>{h}</div>" for h in heads) + "</div>", unsafe_allow_html=True)
        if not rows:
            st.info("조건에 맞는 검수 건이 없습니다.")
            return

        for r in rows:
            ai = r["ai_result"]
            is_processing = r["decision_status"] == "처리중"
            high, mid = (0, 0) if is_processing else counts_of(build_findings(ai))
            vlabel, vtone = ("분석 중", "slate") if is_processing else VERDICT[ai["status"]]
            slabel, stone = DSTATUS[r["decision_status"]]
            score = 0 if is_processing else ai_score(ai)
            fb = '<span style="font-size:14.5px;font-weight:500;color:#9aa8bb;">분석 중</span>' if is_processing else (
                (f'<span style="font-size:14.5px;font-weight:500;color:#50647f;">위반 {high}</span> ' if high else "")
                + (f'<span style="font-size:14.5px;font-weight:500;color:#7a8aa0;">주의 {mid}</span>' if mid else ""))
            if not is_processing and not high and not mid:
                fb = '<span style="font-size:14.5px;font-weight:500;color:#a5b0bf;">이슈 없음</span>'
            verdict_color = "#50647f" if vtone == "red" else "#6d7f96" if vtone == "amber" else "#8997a8" if vtone == "green" else "#9aa8bb"
            verdict_html = (
                f'<span style="display:inline-flex;align-items:center;gap:6px;font-size:14.5px;font-weight:500;'
                f'color:{verdict_color};">'
                f'{html.escape(vlabel)}</span>'
            )
            status_html = (
                f'<span style="display:inline-flex;align-items:center;justify-content:center;min-width:76px;'
                f'height:30px;padding:0 12px;border-radius:999px;background:#eef5ff;'
                f'color:#2563eb;font-size:14.5px;font-weight:500;">{html.escape(slabel)}</span>'
            )
            trail = ('<span style="font-size:15px;font-weight:600;color:#2563eb;">결재 ›</span>'
                     if r["decision_status"] == "대기"
                     else '<span style="font-size:14.5px;font-weight:500;color:#a9b5c5;">완료 ›</span>')
            rc = st.columns([0.032, 0.968], gap="small", vertical_alignment="center")
            rc[0].checkbox("선택", key=f"sel-{r['id']}", label_visibility="collapsed")
            rc[1].markdown(
                f'<a class="lb-row" href="?rid={r["id"]}" target="_self" style="text-decoration:none;display:block;">'
                f'<div style="display:grid;grid-template-columns:{grid};gap:18px;align-items:center;'
                f'height:64px;padding:0 8px;border-bottom:1px solid #f1f3f8;box-sizing:border-box;">'
                f'<div style="display:flex;align-items:center;gap:15px;min-width:0;transform:translateY(-5px);">{score_ring(score, 38, 4)}'
                f'<div style="min-width:0;display:flex;flex-direction:column;justify-content:center;">'
                f'<div style="font-size:15.5px;font-weight:600;color:#152238;letter-spacing:-.01em;line-height:1.25;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{html.escape(title_of(r))}</div>'
                f'<div style="font-size:13.5px;color:#8d9aaf;font-weight:500;line-height:1.25;margin-top:4px;">RV-{r["id"]:04d} · {html.escape(r["created_at"][:10])}</div></div></div>'
                f'<div style="display:flex;align-items:center;height:100%;transform:translateY(-5px);">{media_tag(r["media"])}</div>'
                f'<div style="display:flex;align-items:center;height:100%;transform:translateY(-5px);">{verdict_html}</div>'
                f'<div style="display:flex;align-items:center;height:100%;transform:translateY(-5px);">{fb}</div>'
                f'<div style="display:flex;align-items:center;height:100%;transform:translateY(-5px);">{status_html}</div>'
                f'<div style="display:flex;align-items:center;justify-content:flex-end;height:100%;text-align:right;transform:translateY(-5px);">{trail}</div></div></a>',
                unsafe_allow_html=True)


# --- 화면: 새 검수 요청 ---

def _ai_guide(msg="AI가 본문을 분석해 위반·주의 항목과 수정 제안을 자동 생성합니다. 결과는 준법관리자가 검토·결재합니다."):
    st.markdown(
        '<div class="lb-anim" style="padding:12px 14px;border-radius:12px;background:#0e1626;color:#9fb0d0;font-size:11.8px;line-height:1.6;">'
        f'<div style="font-weight:700;color:#fff;margin-bottom:5px;">AI 심의 안내</div>{msg}</div>',
        unsafe_allow_html=True)


_MODE_META = {
    "완화":   {"desc": "명백한 법령 저촉만 위반으로 처리합니다."},
    "표준":   {"desc": "명백한 저촉은 위반, 해석 소지는 주의로 분류합니다."},
    "강화":   {"desc": "소지만 있어도 위반으로 판정합니다."},
    "AI추천": {"desc": "콘텐츠 유형·매체에 맞춰 강도를 자동으로 적용합니다."},
}
_MODE_KEY = {"강화": "strict", "표준": "standard", "완화": "relaxed", "AI추천": "ai"}
_TYPE_KEY = {"텍스트": "text", "이미지": "image", "영상": "video"}


def _review_mode_selector() -> str:
    """심의 강도 선택 — 레퍼런스 level-card 스타일"""
    selected = st.session_state.get("review_mode", "표준")
    cols = st.columns(4, gap="small")
    for col, (mode_key, meta) in zip(cols, _MODE_META.items()):
        with col:
            if st.button(mode_key, key=f"rm-{_MODE_KEY[mode_key]}", use_container_width=True,
                         type="primary" if selected == mode_key else "secondary"):
                st.session_state["review_mode"] = mode_key
                st.rerun()
    meta = _MODE_META[selected]
    st.markdown(
        f'<div style="margin-top:20px;background:#f1f6ff;border-radius:12px;'
        f'padding:13px 16px;color:#5b6472;font-size:14px;margin-bottom:10px;">'
        f'<b style="color:#1263ff;margin-right:8px;">ⓘ {html.escape(selected)}</b>'
        f'{html.escape(meta["desc"])}</div>',
        unsafe_allow_html=True)
    return selected


_AI_DESC = {
    "텍스트": "본문의 광고 문구를 분석해 위반·주의 항목과 수정 제안을 자동 생성합니다. 결과는 준법관리자가 검토·결재합니다.",
    "이미지": "이미지 속 문구와 시각요소를 인식해 과장·필수고지 누락 등을 영역별로 표시합니다. 결과는 준법감시인이 검토·결재합니다.",
    "영상":   "영상의 음성과 화면을 구간별로 분석해 위반 지점을 타임라인에 표시하고, 수정 제안을 함께 제시합니다. 결과는 작성자와 이해 관계자가 공유할 수 있습니다.",
}


def new_review():
    # ── 영상 업로드 진행 중 ──
    if st.session_state.get("_video_uploading"):
        st.markdown(
            '<div style="text-align:center;padding:100px 0;">'
            '<div class="lb-pulse" style="display:inline-block;width:10px;height:10px;border-radius:999px;'
            'background:#34d399;margin-right:10px;vertical-align:middle;"></div>'
            '<span style="font-size:15px;font-weight:700;color:#52617a;vertical-align:middle;">'
            '영상을 업로드하고 있습니다…</span></div>',
            unsafe_allow_html=True)
        up_data = st.session_state.pop("_video_upload_data")
        api_create_video(up_data["file"], up_data["title"], up_data["review_mode"])
        st.session_state.pop("_video_uploading", None)
        go("dashboard")
        return

    # ── 상단 바 ──
    with st.container(border=False, key="nr-topbar"):
        top_l, top_c = st.columns([0.07, 0.93], vertical_alignment="center")
        with top_l:
            if st.button(":material/chevron_left:", key="nr-back", help="대시보드로 이동"):
                go("dashboard")
        with top_c:
            st.markdown(
                '<div style="padding:2px 0 0;">'
                '<div style="color:#1263ff;font-size:12px;font-weight:800;margin-bottom:3px;">준법심의 요청</div>'
                '<div style="font-size:18px;font-weight:850;color:#111827;letter-spacing:-.02em;">새 검수 요청</div>'
                '</div>',
                unsafe_allow_html=True)
        st.markdown('<div class="nr-ai-chip">AI 1차 검토</div>', unsafe_allow_html=True)
        st.markdown('<div class="nr-topbar-rule"></div>', unsafe_allow_html=True)

    # ── 히어로 ──
    hc1, hc2 = st.columns([3, 1])
    with hc1:
        st.markdown(
            '<div style="margin-bottom:42px;">'
            '<h1 style="font-size:24px;line-height:1.3;margin:0 0 12px;font-weight:900;color:#111827;">'
            '어떤 콘텐츠를<br>심의할까요?</h1>'
            '<p style="color:#7b8494;font-size:13px;line-height:1.7;margin:0;">'
            '유형을 선택하면 AI가 알맞게 분석하고,<br>위반·주의 항목을 자동으로 찾아드립니다.</p>'
            '</div>', unsafe_allow_html=True)
    with hc2:
        st.markdown(
            '<div class="nr-hero-art">'
            '<span class="nr-spark s1"></span><span class="nr-spark s2"></span><span class="nr-spark s3"></span>'
            '<div class="nr-shield-orb">'
            '<svg width="40" height="40" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
            '<path d="M12 2.5 4.5 5.6v5.8c0 4.2 2.9 7.7 7.5 9.1 4.6-1.4 7.5-4.9 7.5-9.1V5.6L12 2.5Z" '
            'fill="#1263ff"/>'
            '<path d="m8.2 12.1 2.4 2.4 5.1-5.3" stroke="#fff" stroke-width="1.8" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
            '</svg></div></div>',
            unsafe_allow_html=True)

    # ── 섹션 1: 콘텐츠 유형 선택 ──
    with st.container(border=False, key="nr-sec-1"):
        st.markdown(
            '<div style="display:flex;align-items:flex-start;gap:18px;margin-bottom:34px;">'
            '<div style="width:48px;height:48px;border-radius:999px;background:#eef5ff;color:#1263ff;'
            'display:grid;place-items:center;font-size:25px;font-weight:900;line-height:1;">1</div>'
            '<div><div style="font-size:21px;font-weight:900;color:#111827;letter-spacing:-.035em;line-height:1.15;">'
            '콘텐츠 유형 선택</div>'
            '<div style="font-size:10px;font-weight:500;color:#657389;margin-top:14px;">'
            '심의할 콘텐츠의 유형을 선택해 주세요.</div></div></div>',
            unsafe_allow_html=True)
        mode = st.session_state.get("_new_mode", "텍스트")
        tc1, tc2, tc3 = st.columns(3, gap="medium")
        for col, m in zip([tc1, tc2, tc3], ["텍스트", "이미지", "영상"]):
            with col:
                if st.button(m, key=f"type-{_TYPE_KEY[m]}",
                             use_container_width=True,
                             type="primary" if mode == m else "secondary"):
                    st.session_state["_new_mode"] = m
                    st.rerun()
        mode = st.session_state.get("_new_mode", "텍스트")

    # ── 섹션 2: 기본 정보 입력 ──
    with st.container(border=False, key="nr-sec-2"):
        st.markdown(
            '<div style="display:flex;align-items:flex-start;gap:18px;margin-bottom:38px;">'
            '<div style="width:48px;height:48px;border-radius:999px;background:#eef5ff;color:#1263ff;'
            'display:grid;place-items:center;font-size:21px;font-weight:900;line-height:1;">2</div>'
            '<div><div style="font-size:21px;font-weight:900;color:#111827;letter-spacing:-.035em;line-height:1.15;">'
            '기본 정보 입력</div>'
            '<div style="font-size:10px;font-weight:500;color:#657389;margin-top:14px;">'
            '검수에 필요한 기본 정보를 입력해 주세요.</div></div></div>',
            unsafe_allow_html=True)
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown('<label class="nr-field-label">제목</label>',
                        unsafe_allow_html=True)
            title = st.text_input("제목", label_visibility="collapsed",
                                  placeholder="예) 신규 정기적금 ‘드림플러스’ 출시 안내")

            st.markdown(
                '<div style="height:34px;"></div>'
                '<label class="nr-field-label">매체 <span class="hint">콘텐츠가 노출될 채널</span></label>',
                unsafe_allow_html=True)
            channel = st.session_state.get("_new_channel", "웹/배너")
            ch_icons = {
                "웹/배너": "desktop_windows",
                "앱 푸시": "smartphone",
                "유튜브/영상": "smart_display",
                "인스타·SNS": "photo_camera",
                "이메일": "mail",
                "지면(인쇄)": "description",
            }
            ch_r1 = st.columns(3, gap="small")
            ch_r2 = st.columns(3, gap="small")
            for col_c, ch in zip(ch_r1, _CHANNEL_OPTIONS[:3]):
                with col_c:
                    if st.button(f":material/{ch_icons.get(ch, 'campaign')}: {ch}", key=f"ch-{ch.replace('/', '_').replace('·', '_')}",
                                 use_container_width=True,
                                 type="primary" if channel == ch else "secondary"):
                        st.session_state["_new_channel"] = ch
                        st.rerun()
            for col_c, ch in zip(ch_r2, _CHANNEL_OPTIONS[3:]):
                with col_c:
                    if st.button(f":material/{ch_icons.get(ch, 'campaign')}: {ch}", key=f"ch-{ch.replace('/', '_').replace('·', '_')}",
                                 use_container_width=True,
                                 type="primary" if channel == ch else "secondary"):
                        st.session_state["_new_channel"] = ch
                        st.rerun()

        with right:
            up = None
            body = ""
            if mode == "텍스트":
                sc1, sc2 = st.columns([2.25, 0.8], gap="small", vertical_alignment="center")
                sc1.markdown('<label class="nr-field-label">광고/콘텐츠 본문</label>',
                             unsafe_allow_html=True)
                if sc2.button("샘플 불러오기", key="nr-sample"):
                    st.session_state["draft"] = (
                        "내 통장이 매달 불어나는 가장 확실한 방법, 원금 손실 걱정 없이 누구나 "
                        "최고 연 5.0% 수익을 무조건 보장해 드립니다. 업계 1위 드림은행 「드림플러스」 "
                        "적금으로 지금 바로 시작하세요. 선착순 1만 좌 한정!")
                body = st.text_area("콘텐츠 본문", value=st.session_state.get("draft", ""),
                                    height=238, label_visibility="collapsed",
                                    placeholder="광고 카피 본문 텍스트를 입력하십시오")
            elif mode == "이미지":
                hint_img = '<label class="nr-field-label">이미지 첨부 <span class="hint">음성·화면을 함께 분석합니다</span></label>'
                if st.session_state.get("_up_img_name"):
                    st.markdown(hint_img, unsafe_allow_html=True)
                    fn = st.session_state["_up_img_name"]
                    sz = st.session_state.get("_up_img_size", 0)
                    sz_str = f"{sz / 1024:.0f}KB" if sz < 1024 * 1024 else f"{sz / 1024 / 1024:.1f}MB"
                    col_f, col_d = st.columns([1, 0.18], vertical_alignment="center")
                    col_f.markdown(
                        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
                        f'padding:10px 14px;font-size:13px;color:#334155;font-weight:600;'
                        f'display:flex;align-items:center;gap:8px;">'
                        f'<span style="font-size:18px;">🖼</span>'
                        f'<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{fn}</span>'
                        f'<span style="color:#94a3b8;font-size:12px;font-weight:400;">{sz_str}</span>'
                        f'</div>', unsafe_allow_html=True)
                    if col_d.button("✕", key="del-img", help="파일 삭제"):
                        for k in ("_up_img_name", "_up_img_size", "_up_img_data", "_up_img_type"):
                            st.session_state.pop(k, None)
                        st.rerun()
                    up = type("F", (), {
                        "name": st.session_state["_up_img_name"],
                        "getvalue": lambda self: st.session_state["_up_img_data"],
                        "type": st.session_state.get("_up_img_type", "image/png"),
                    })()
                else:
                    st.markdown(hint_img, unsafe_allow_html=True)
                    raw = st.file_uploader("이미지 첨부", type=["png", "jpg", "jpeg", "webp"],
                                           label_visibility="collapsed", key="up_img")
                    if raw:
                        st.session_state["_up_img_name"] = raw.name
                        st.session_state["_up_img_size"] = raw.size
                        st.session_state["_up_img_data"] = raw.getvalue()
                        st.session_state["_up_img_type"] = raw.type
                        st.rerun()
            else:
                hint = '<label class="nr-field-label">영상 첨부 <span class="hint">음성·화면을 함께 분석합니다</span></label>'
                # 파일이 이미 세션에 캐시돼 있으면 미리보기 + 삭제 버튼 표시
                if st.session_state.get("_up_video_name"):
                    st.markdown(hint, unsafe_allow_html=True)
                    fn = st.session_state["_up_video_name"]
                    sz = st.session_state.get("_up_video_size", 0)
                    sz_str = f"{sz / 1024 / 1024:.1f}MB" if sz else ""
                    col_f, col_d = st.columns([1, 0.18], vertical_alignment="center")
                    col_f.markdown(
                        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
                        f'padding:10px 14px;font-size:13px;color:#334155;font-weight:600;'
                        f'display:flex;align-items:center;gap:8px;">'
                        f'<span style="font-size:18px;">🎬</span>'
                        f'<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{fn}</span>'
                        f'<span style="color:#94a3b8;font-size:12px;font-weight:400;">{sz_str}</span>'
                        f'</div>', unsafe_allow_html=True)
                    if col_d.button("✕", key="del-video", help="파일 삭제"):
                        for k in ("_up_video_name", "_up_video_size", "_up_video_data", "_up_video_type"):
                            st.session_state.pop(k, None)
                        st.rerun()
                    up = type("F", (), {
                        "name": st.session_state["_up_video_name"],
                        "getvalue": lambda self: st.session_state["_up_video_data"],
                        "type": st.session_state.get("_up_video_type", "video/mp4"),
                    })()
                else:
                    st.markdown(hint, unsafe_allow_html=True)
                    raw = st.file_uploader("영상 첨부", type=["mp4", "mov", "webm", "m4a", "mp3", "wav"],
                                           label_visibility="collapsed", key="up_vid")
                    if raw:
                        st.session_state["_up_video_name"] = raw.name
                        st.session_state["_up_video_size"] = raw.size
                        st.session_state["_up_video_data"] = raw.getvalue()
                        st.session_state["_up_video_type"] = raw.type
                        st.rerun()

    # ── 섹션 3: 심의 강도 선택 ──
    with st.container(border=False, key="nr-sec-3"):
        st.markdown(
            '<div style="display:flex;align-items:flex-start;gap:18px;margin-bottom:26px;">'
            '<div style="width:48px;height:48px;border-radius:999px;background:#eef5ff;color:#1263ff;'
            'display:grid;place-items:center;font-size:21px;font-weight:900;line-height:1;">3</div>'
            '<div><div style="font-size:21px;font-weight:900;color:#111827;letter-spacing:-.035em;line-height:1.15;padding-top:10px;">'
            '심의 강도 선택</div>'
            '<div style="font-size:10px;font-weight:500;color:#657389;margin-top:14px;">'
            '활용 기준의 엄격도를 선택하세요.</div></div></div>',
            unsafe_allow_html=True)
        review_mode = _review_mode_selector()

    # ── AI 심의 안내 ──
    st.markdown(
        f'<div style="display:flex;gap:18px;align-items:flex-start;background:#f5f9ff;'
        f'border:1.5px solid #b8d3ff;border-radius:16px;padding:20px 24px;margin:18px 0 28px;">'
        f'<div style="width:46px;height:46px;flex-shrink:0;border-radius:14px;background:#fff;'
        f'box-shadow:0 8px 20px rgba(18,99,255,.12);display:flex;align-items:center;'
        f'justify-content:center;color:#1263ff;font-size:20px;">✣</div>'
        f'<div><div style="font-size:15px;font-weight:850;color:#111827;margin-bottom:6px;">'
        f'AI 1차 심의는 이렇게 진행돼요</div>'
        f'<div style="color:#5b6472;line-height:1.6;font-size:13px;">{html.escape(_AI_DESC[mode])}</div>'
        f'</div></div>',
        unsafe_allow_html=True)

    # ── 하단 액션 바 ──
    can_submit = bool(title.strip()) and (
        (mode == "텍스트" and body.strip()) or
        (mode in ("이미지", "영상") and up is not None)
    )
    media_val = mode

    st.markdown('<div style="height:90px;"></div>', unsafe_allow_html=True)
    with st.container(key="nr-actions"):
        ba1, ba2 = st.columns([1, 2], gap="small")
        with ba1:
            st.button(
                "**임시 저장**\n\n*나중에 이어서 작성할 수 있습니다.*",
                use_container_width=True, disabled=True, key="nr-save")
        with ba2:
            if mode == "텍스트":
                if st.button(
                    "**AI 심의 요청하기**\n\n*AI 1차 심의를 시작합니다.*",
                    type="primary", use_container_width=True,
                    disabled=not can_submit, key="nr-submit"):
                    with st.spinner("AI 1차 심의 진행 중…"):
                        rec = api_create(body, media_val, title.strip(), review_mode)
                    st.session_state.pop("draft", None)
                    go("detail", rec["id"])
            elif mode == "이미지":
                if st.button(
                    "**AI 심의 요청하기**\n\n*AI 1차 심의를 시작합니다.*",
                    type="primary", use_container_width=True,
                    disabled=not can_submit, key="nr-submit"):
                    with st.spinner("이미지에서 문구 추출 + 심의 중…"):
                        rec = api_create_image((up.name, up.getvalue(), up.type), title.strip(), review_mode)
                    for k in ("_up_img_name", "_up_img_size", "_up_img_data", "_up_img_type"):
                        st.session_state.pop(k, None)
                    go("detail", rec["id"])
            else:
                if st.button(
                    "**AI 심의 요청하기**\n\n*AI 1차 심의를 시작합니다.*",
                    type="primary", use_container_width=True,
                    disabled=not can_submit, key="nr-submit"):
                    st.session_state["_video_uploading"] = True
                    st.session_state["_video_upload_data"] = {
                        "file": (up.name, up.getvalue(), up.type),
                        "title": title.strip(),
                        "review_mode": review_mode,
                    }
                    # 캐시 정리
                    for k in ("_up_video_name", "_up_video_size", "_up_video_data", "_up_video_type"):
                        st.session_state.pop(k, None)
                    st.rerun()


# --- 화면: 검수·결재 상세 ---

_VIDEO_TPL = """
<style>
@keyframes lb-fadein {
  from { opacity:0; transform:translateY(5px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes lb-pop {
  0%   { opacity:0; transform:translateY(6px) scale(.97); }
  60%  { transform:translateY(-1px) scale(1.01); }
  100% { opacity:1; transform:translateY(0) scale(1); }
}
.lb-seg-pass { animation: lb-fadein .25s ease both; }
.lb-seg-flag { animation: lb-pop .38s cubic-bezier(.22,.68,0,1.2) both; }
</style>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:16px;font-family:-apple-system,'Apple SD Gothic Neo',sans-serif;align-items:start;">
  <div style="border-radius:14px;background:#020617;overflow:hidden;box-shadow:0 14px 36px rgba(15,23,42,.10);display:flex;align-items:center;">
    <video id="vid" src="__URL__" controls autoplay muted playsinline style="width:100%;display:block;background:#000;max-height:420px;"></video>
  </div>
  <div style="display:flex;flex-direction:column;border:1px solid #e6eaf1;border-radius:14px;background:#fff;overflow:hidden;height:420px;box-shadow:0 8px 22px rgba(15,23,42,.05);">
    <div style="padding:10px 14px;border-bottom:1px solid #eef1f6;font-size:13px;font-weight:800;color:#0f1b2d;display:flex;align-items:center;gap:7px;">
      <span style="width:7px;height:7px;border-radius:9px;background:#34d399;display:inline-block;flex-shrink:0;"></span>실시간 위반 감지
      <div style="margin-left:auto;display:flex;align-items:center;gap:5px;">
        <span id="cnt-high" style="display:none;align-items:center;gap:4px;background:#fef2f2;color:#dc2626;
          font-size:11px;font-weight:700;padding:2px 8px;border-radius:99px;border:1px solid #fecaca;">
          위반 <span id="cnt-high-n">0</span>
        </span>
        <span id="cnt-mid" style="display:none;align-items:center;gap:4px;background:#fffbeb;color:#d97706;
          font-size:11px;font-weight:700;padding:2px 8px;border-radius:99px;border:1px solid #fde68a;">
          주의 <span id="cnt-mid-n">0</span>
        </span>
        <span id="scan-counter" style="font-size:11px;color:#94a3b8;font-weight:600;font-family:ui-monospace,monospace;"></span>
      </div>
    </div>
    <div id="segs" style="flex:1;overflow:auto;padding:4px 0;">__FEED_INIT__</div>
  </div>
</div>
<script>
const SEGS = __SEGS__;
const IS_PROCESSING = __IS_PROCESSING__;
const vid = document.getElementById('vid');
const wrap = document.getElementById('segs');
const counter = document.getElementById('scan-counter');
const cntHighEl = document.getElementById('cnt-high');
const cntHighN  = document.getElementById('cnt-high-n');
const cntMidEl  = document.getElementById('cnt-mid');
const cntMidN   = document.getElementById('cnt-mid-n');
var shown = new Array(SEGS.length).fill(false);
var scanned = 0;
var highCount = 0;
var midCount  = 0;

function tryAutoplay() {
  if (!vid) return;
  vid.muted = true;
  var p = vid.play();
  if (p && typeof p.catch === 'function') p.catch(function(){});
}

vid.addEventListener('canplay', tryAutoplay, { once: true });
setTimeout(tryAutoplay, 250);

function mmss(t){var m=Math.floor(t/60),s=Math.floor(t%60);return (''+m).padStart(2,'0')+':'+(''+s).padStart(2,'0');}

function updateCounter() {
  if (scanned > 0) {
    if (counter) counter.textContent = scanned + ' / ' + SEGS.length + ' 구간';
  } else {
    if (counter) counter.textContent = '';
  }
  if (cntHighEl) {
    cntHighN.textContent = highCount;
    cntHighEl.style.display = highCount > 0 ? 'inline-flex' : 'none';
  }
  if (cntMidEl) {
    cntMidN.textContent = midCount;
    cntMidEl.style.display = midCount > 0 ? 'inline-flex' : 'none';
  }
}

function highlightTerms(text, terms, bg, color) {
  bg = bg || '#fff1e6'; color = color || '#c2410c';
  if (!terms || !terms.length) return escHtml(text);
  var result = escHtml(text);
  terms.forEach(function(term) {
    var escaped = escHtml(term);
    result = result.split(escaped).join(
      '<mark style="background:' + bg + ';color:' + color + ';border-radius:3px;padding:0 2px;font-weight:700;">' + escaped + '</mark>'
    );
  });
  return result;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function makePassEl(s, i) {
  var d = document.createElement('div');
  d.id = 'seg'+i;
  d.className = 'lb-seg-pass';
  d.style.cssText = 'padding:6px 14px 6px 14px;border-bottom:1px solid #f3f5f8;cursor:pointer;display:flex;align-items:center;gap:8px;opacity:.55;transition:opacity .15s;';
  var kindColor = s.kind === '화면' ? '#7c3aed' : '#64748b';
  d.innerHTML = '<span style="font-family:ui-monospace,monospace;font-size:10.5px;color:#94a3b8;font-weight:700;flex-shrink:0;">' + mmss(s.start) + '</span>'
    + '<span style="font-size:10px;font-weight:700;color:' + kindColor + ';flex-shrink:0;">' + s.kind + '</span>'
    + '<span style="font-size:11.5px;color:#94a3b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + escHtml(s.text) + '</span>'
    + '<span style="margin-left:auto;font-size:10px;color:#16a34a;font-weight:800;font-size:12px;flex-shrink:0;">✓</span>';
  d.onmouseenter = function(){ d.style.opacity = '.85'; };
  d.onmouseleave = function(){ d.style.opacity = '.55'; };
  d.onclick = function(){ vid.currentTime = s.start; vid.play(); };
  return d;
}

function makeFlagEl(s, i) {
  var isHigh = s.severity === 'high';
  var bgColor  = isHigh ? '#fef2f2' : '#fff8f1';
  var bgHover  = isHigh ? '#fee2e2' : '#ffedd5';
  var markBg   = isHigh ? '#fecaca' : '#fed7aa';
  var markColor= isHigh ? '#b91c1c' : '#c2410c';
  var d = document.createElement('div');
  d.id = 'seg'+i;
  d.className = 'lb-seg-flag';
  d.style.cssText = 'position:relative;padding:10px 14px;border-bottom:1px solid #eef2f7;cursor:pointer;background:' + bgColor + ';transition:background .15s;';
  var kindColor = s.kind === '화면' ? '#7c3aed' : '#64748b';
  var termBadges = s.terms.map(function(t){
    return '<span style="color:#94a3b8;font-size:10.5px;font-weight:600;">' + escHtml(t) + '</span>';
  }).join('<span style="color:#d1d5db;margin:0 2px;">·</span>');
  d.innerHTML = '<div style="display:flex;align-items:center;gap:7px;margin-bottom:4px;">'
    + '<span style="font-family:ui-monospace,monospace;font-size:11px;color:#64748b;font-weight:700;">' + mmss(s.start) + '</span>'
    + '<span style="font-size:10.5px;font-weight:700;color:' + kindColor + ';">' + s.kind + '</span>'
    + '<span style="margin-left:auto;">' + termBadges + '</span>'
    + '</div>'
    + '<div style="font-size:12.5px;color:#1e293b;font-weight:500;line-height:1.6;">' + highlightTerms(s.text, s.terms, markBg, markColor) + '</div>';
  d.onmouseenter = function(){ d.style.background = bgHover; };
  d.onmouseleave = function(){ d.style.background = bgColor; };
  d.onclick = function(){ vid.currentTime = s.start; vid.play(); };
  return d;
}

if (!IS_PROCESSING) {
  wrap.innerHTML = '<div style="height:100%;display:grid;place-items:center;text-align:center;color:#c4cdd9;font-size:12.5px;line-height:1.8;padding:22px;">재생하면 구간별 분석 결과가<br>실시간으로 표시됩니다.</div>';
}

function syncFeed(t) {
  SEGS.forEach(function(s, i) {
    if (shown[i] || t < s.start) return;
    if (scanned === 0) wrap.innerHTML = '';
    var el = s.flagged ? makeFlagEl(s, i) : makePassEl(s, i);
    wrap.appendChild(el);
    el.scrollIntoView({behavior:'smooth', block:'nearest'});
    shown[i] = true;
    scanned++;
    if (s.flagged) {
      if (s.severity === 'high') highCount++; else midCount++;
      revealNextFinding();
    }
    updateCounter();
  });
}

vid.addEventListener('seeked', function(){
  var t = vid.currentTime;
  SEGS.forEach(function(s, i) {
    if (shown[i] || t < s.start) return;
    if (scanned === 0) wrap.innerHTML = '';
    var el = s.flagged ? makeFlagEl(s, i) : makePassEl(s, i);
    wrap.appendChild(el);
    shown[i] = true;
    scanned++;
    if (s.flagged) {
      if (s.severity === 'high') highCount++; else midCount++;
      revealNextFinding();
    }
  });
  updateCounter();
});
vid.addEventListener('timeupdate', function(){ syncFeed(vid.currentTime); });


</script>
"""


_SKELETON_HTML = """
<style>
@keyframes lb-spin { to { transform: rotate(360deg); } }
@keyframes lb-bar  {
  0%   { width: 15%; }
  50%  { width: 80%; }
  100% { width: 15%; }
}
@keyframes lb-fade { 0%,100%{opacity:.35} 50%{opacity:1} }
.lb-spin-ring {
  width: 36px; height: 36px; border-radius: 50%;
  border: 3px solid #e2e8f0;
  border-top-color: #3b82f6;
  animation: lb-spin .9s linear infinite;
}
.lb-progress-bar {
  height: 3px; border-radius: 99px; background: #3b82f6;
  animation: lb-bar 1.8s ease-in-out infinite;
}
.lb-step { animation: lb-fade 2s ease-in-out infinite; }
.lb-step:nth-child(2) { animation-delay: .6s; }
.lb-step:nth-child(3) { animation-delay: 1.2s; }
</style>
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;padding:32px 20px;gap:20px;">
  <div class="lb-spin-ring"></div>
  <div style="width:100%;background:#f1f5f9;border-radius:99px;overflow:hidden;">
    <div class="lb-progress-bar"></div>
  </div>
  <div style="font-size:13px;font-weight:700;color:#334155;text-align:center;">AI 심의 분석 중</div>
  <div style="display:flex;flex-direction:column;gap:6px;width:100%;">
    <div class="lb-step" style="display:flex;align-items:center;gap:8px;font-size:11.5px;color:#64748b;">
      <span style="width:18px;height:18px;border-radius:5px;background:#eff6ff;display:flex;align-items:center;justify-content:center;font-size:10px;">🎙</span>음성 자막 추출 중…
    </div>
    <div class="lb-step" style="display:flex;align-items:center;gap:8px;font-size:11.5px;color:#64748b;">
      <span style="width:18px;height:18px;border-radius:5px;background:#eff6ff;display:flex;align-items:center;justify-content:center;font-size:10px;">🖼</span>화면 프레임 분석 중…
    </div>
    <div class="lb-step" style="display:flex;align-items:center;gap:8px;font-size:11.5px;color:#64748b;">
      <span style="width:18px;height:18px;border-radius:5px;background:#eff6ff;display:flex;align-items:center;justify-content:center;font-size:10px;">⚖️</span>준법 심의 판단 중…
    </div>
  </div>
</div>
"""


def _video_review(rid, timeline, findings=None, is_processing=False):
    """영상 재생 + 재생 시각 동기화 실시간 위반 피드 (클라이언트 사이드)"""
    segs = json.dumps(
        [{"start": s["start"], "end": s["end"], "text": s["text"], "flagged": s["flagged"],
          "terms": s["terms"], "kind": s.get("kind", "음성"), "severity": s.get("severity", "")}
         for s in timeline], ensure_ascii=False)
    feed_content = _SKELETON_HTML if is_processing else ""
    out = (_VIDEO_TPL
           .replace("__URL__", f"{PUBLIC_API}/reviews/{rid}/media")
           .replace("__SEGS__", segs)
           .replace("__IS_PROCESSING__", "true" if is_processing else "false")
           .replace("__FEED_INIT__", feed_content))
    components.html(out, height=440)


def detail(rid):
    import time as _time
    st.markdown(
        "<style>.block-container{max-width:1180px !important;}</style>",
        unsafe_allow_html=True,
    )
    rec = api_get(rid)
    is_processing = rec.get("decision_status") == "처리중"
    ai = rec["ai_result"]
    findings = [] if is_processing else build_findings(ai)
    high, mid = counts_of(findings)
    vlabel, vtone = ("분석 중", "slate") if is_processing else VERDICT[ai["status"]]

    media = rec["media"]
    right_badge = (live_dot("AI 분석 중…") if is_processing else badge(vlabel, vtone, dot=True))
    topbar(f"준법심의 · RV-{rec['id']:04d}", title_of(rec), right_badge)

    # 영상 처리 중이면 로딩 화면만 단독 렌더링
    if media == "영상" and is_processing:
        st.markdown("""
<style>
@keyframes lb-spin2  { to { transform:rotate(360deg); } }
@keyframes lb-bar2   { 0%{width:0%} 60%{width:72%} 100%{width:92%} }
@keyframes lb-stepin { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:none} }
</style>
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:72vh;gap:28px;">

  <div style="position:relative;width:64px;height:64px;">
    <svg width="64" height="64" style="position:absolute;inset:0;transform:rotate(-90deg);">
      <circle cx="32" cy="32" r="26" fill="none" stroke="#e2e8f0" stroke-width="4"/>
      <circle cx="32" cy="32" r="26" fill="none" stroke="#3b82f6" stroke-width="4"
        stroke-dasharray="163" stroke-dashoffset="40" stroke-linecap="round"
        style="animation:lb-spin2 1.1s linear infinite;transform-origin:32px 32px;"/>
    </svg>
    <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
      font-size:11px;font-weight:800;color:#3b82f6;">AI</div>
  </div>

  <div style="text-align:center;">
    <div style="font-size:18px;font-weight:850;color:#0f172a;letter-spacing:-.02em;margin-bottom:8px;">심의 분석 중</div>
    <div style="font-size:13.5px;color:#94a3b8;font-weight:500;">완료까지 1~2분 소요됩니다</div>
  </div>

  <div style="width:200px;height:2px;background:#e2e8f0;border-radius:99px;overflow:hidden;">
    <div style="height:2px;background:linear-gradient(90deg,#3b82f6,#6366f1);border-radius:99px;
      animation:lb-bar2 8s cubic-bezier(.4,0,.2,1) forwards;"></div>
  </div>

  <div style="display:flex;flex-direction:column;gap:10px;width:220px;">
    <div style="display:flex;align-items:center;gap:12px;animation:lb-stepin .4s ease both;">
      <div style="width:6px;height:6px;border-radius:99px;background:#3b82f6;flex-shrink:0;"></div>
      <span style="font-size:13px;color:#475569;font-weight:600;">음성 자막 추출</span>
    </div>
    <div style="display:flex;align-items:center;gap:12px;animation:lb-stepin .4s ease .5s both;">
      <div style="width:6px;height:6px;border-radius:99px;background:#6366f1;flex-shrink:0;"></div>
      <span style="font-size:13px;color:#475569;font-weight:600;">화면 프레임 분석</span>
    </div>
    <div style="display:flex;align-items:center;gap:12px;animation:lb-stepin .4s ease 1s both;">
      <div style="width:6px;height:6px;border-radius:99px;background:#8b5cf6;flex-shrink:0;"></div>
      <span style="font-size:13px;color:#475569;font-weight:600;">AI 준법 심의 판단</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
        _time.sleep(3)
        st.rerun()
        return

    if media == "영상":
        st.markdown('<div style="font-size:14.5px;font-weight:800;color:#0f1b2d;margin-bottom:10px;">영상 미리보기 · 실시간 위반 감지</div>', unsafe_allow_html=True)
        timeline = ai.get("timeline") or []
        _video_review(rec["id"], timeline, findings=findings, is_processing=False)
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85])

    # 좌 — AI 심의 결과 + 발견 항목 + AI 수정 제안 + 원본 미리보기
    with left:
        if is_processing:
            st.markdown("""
<style>
@keyframes lb-shimmer2{0%{background-position:-600px 0}100%{background-position:600px 0}}
.lb-sk{background:linear-gradient(90deg,#f1f5f9 25%,#e2e8f0 50%,#f1f5f9 75%);background-size:1200px 100%;
  animation:lb-shimmer2 1.5s infinite linear;border-radius:6px;}
</style>
<div style="background:#fff;border:1px solid #e6eaf1;border-radius:14px;padding:16px 18px;box-shadow:0 1px 2px rgba(16,24,40,.04);">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
    <div style="width:34px;height:34px;border-radius:9px;background:linear-gradient(145deg,#3b82f6,#1d4ed8);display:grid;place-items:center;font-size:12px;font-weight:800;color:#fff;">AI</div>
    <div style="flex:1;">
      <div class="lb-sk" style="height:16px;width:55%;margin-bottom:8px;"></div>
      <div class="lb-sk" style="height:12px;width:75%;"></div>
    </div>
    <div class="lb-sk" style="width:50px;height:50px;border-radius:999px;"></div>
  </div>
  <div style="display:flex;gap:8px;">
    <div class="lb-sk" style="flex:1;height:52px;border-radius:10px;"></div>
    <div class="lb-sk" style="flex:1;height:52px;border-radius:10px;"></div>
    <div class="lb-sk" style="flex:1;height:52px;border-radius:10px;"></div>
  </div>
</div>
<div style="font-size:13px;font-weight:800;color:#0f1b2d;margin:16px 0 8px;">발견 항목</div>
<div style="display:flex;flex-direction:column;gap:10px;">
  <div style="background:#fff;border:1px solid #e6eaf1;border-radius:12px;padding:13px 15px;">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
      <div class="lb-sk" style="width:6px;height:6px;border-radius:999px;flex-shrink:0;"></div>
      <div class="lb-sk" style="width:52px;height:20px;border-radius:999px;"></div>
      <div class="lb-sk" style="width:90px;height:16px;margin-left:4px;"></div>
    </div>
    <div class="lb-sk" style="height:36px;border-radius:8px;margin-bottom:8px;"></div>
    <div class="lb-sk" style="height:13px;width:80%;margin-bottom:6px;"></div>
    <div class="lb-sk" style="height:13px;width:65%;"></div>
  </div>
  <div style="background:#fff;border:1px solid #e6eaf1;border-radius:12px;padding:13px 15px;">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
      <div class="lb-sk" style="width:6px;height:6px;border-radius:999px;flex-shrink:0;"></div>
      <div class="lb-sk" style="width:52px;height:20px;border-radius:999px;"></div>
      <div class="lb-sk" style="width:75px;height:16px;margin-left:4px;"></div>
    </div>
    <div class="lb-sk" style="height:13px;width:70%;margin-bottom:6px;"></div>
    <div class="lb-sk" style="height:13px;width:55%;"></div>
  </div>
</div>
<div style="background:#f0f6ff;border:1px solid #d4e3fb;border-radius:14px;padding:18px 20px;margin-top:14px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">
    <div style="width:24px;height:24px;border-radius:7px;background:linear-gradient(145deg,#3b82f6,#1d4ed8);display:grid;place-items:center;font-size:9px;font-weight:800;color:#fff;">AI</div>
    <div class="lb-sk" style="width:90px;height:14px;"></div>
    <div class="lb-sk" style="width:140px;height:12px;margin-left:4px;"></div>
  </div>
  <div style="background:#fff;border:1px solid #d4e3fb;border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;gap:8px;">
    <div class="lb-sk" style="height:13px;width:90%;"></div>
    <div class="lb-sk" style="height:13px;width:75%;"></div>
    <div class="lb-sk" style="height:13px;width:82%;"></div>
  </div>
</div>
<div style="margin-top:14px;text-align:center;font-size:12px;color:#94a3b8;font-weight:600;padding:12px 0;">
  <span class="lb-pulse" style="display:inline-block;width:7px;height:7px;border-radius:999px;background:#34d399;margin-right:7px;vertical-align:middle;"></span>음성·화면 분석 및 AI 준법 심의를 진행하고 있습니다
</div>
""", unsafe_allow_html=True)
            _time.sleep(3)
            st.rerun()
            return

        # 콘텐츠 미리보기 — AI 결과 카드 위에 표시
        if media == "이미지":
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:10px;">'
                f'<span style="font-size:13.5px;font-weight:800;color:#0f1b2d;">콘텐츠 미리보기</span>'
                f'<span style="margin-left:auto;">{badge(f"위반 {high}", "red")} &nbsp;{badge(f"주의 {mid}", "amber")}</span></div>',
                unsafe_allow_html=True)
            st.image(f"{API}/reviews/{rec['id']}/media", use_container_width=True)
            with st.expander("AI가 추출한 텍스트 보기"):
                st.write(rec["content"])
            st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
        elif media not in ("영상",):
            st.markdown(
                '<div class="lb-anim" style="background:#fff;border:1px solid #e6eaf1;border-radius:14px;'
                'padding:18px 22px;box-shadow:0 1px 2px rgba(16,24,40,.04);margin-bottom:12px;">'
                '<div style="font-size:11px;font-weight:700;color:#94a3b8;letter-spacing:.05em;margin-bottom:10px;">본문 카피 · 위반 문구 하이라이트</div>'
                f'<p style="margin:0;font-size:15px;line-height:1.95;color:#27364e;word-break:keep-all;">{highlight(rec["content"], ai["rule_hits"])}</p>'
                '</div>', unsafe_allow_html=True)

        # AI 1차 심의 결과 카드
        score = ai_score(ai)
        summary_full = html.escape(ai["summary"])
        summary_truncated = html.escape(ai["summary"][:100])
        txt_style = 'font-size:13px;color:#6b7280;line-height:1.7;word-break:keep-all;'
        if len(ai["summary"]) > 100:
            summary_html = (
                f'<details class="lb-summary-toggle" style="cursor:pointer;">'
                f'<summary style="{txt_style}">'
                f'{summary_truncated}… '
                f'<span style="color:#2563eb;font-size:12px;font-weight:600;">더보기</span>'
                f'</summary>'
                f'<p style="margin:0;{txt_style}">{summary_full}</p>'
                f'</details>'
            )
        else:
            summary_html = f'<p style="margin:0;{txt_style}">{summary_full}</p>'
        total = high + mid
        findings_chip = (
            f'<a href="#findings" style="display:inline-flex;align-items:center;gap:5px;margin-top:12px;'
            f'text-decoration:none;font-size:12.5px;font-weight:600;color:#6b7280;">'
            + (f'<span style="color:#ef4444;">위반 {high}건</span>' if high else "")
            + (f'<span style="color:#6b7280;margin:0 2px;">·</span>' if high and mid else "")
            + (f'<span style="color:#f59e0b;">주의 {mid}건</span>' if mid else "")
            + (f'<span style="color:#10b981;">발견 항목 없음</span>' if not total else "")
            + ' <span style="font-size:11px;">↓ 발견 항목 보기</span></a>'
            if total else
            '<span style="display:inline-block;margin-top:12px;font-size:12.5px;font-weight:600;color:#10b981;">발견 항목 없음</span>'
        )
        score_col = "#1ca25b" if score >= 90 else "#e08600" if score >= 70 else "#dc4338"
        st.markdown(
            '<div class="lb-anim" style="background:#fff;border:1px solid #e8ecf3;border-radius:18px;'
            'padding:22px 24px;box-shadow:0 1px 4px rgba(16,24,40,.05);display:flex;align-items:stretch;">'
            # 왼쪽 텍스트 영역
            '<div style="flex:1;min-width:0;padding-right:24px;">'
            '<div style="display:flex;align-items:center;gap:5px;margin-bottom:12px;">'
            '<span style="font-size:11px;color:#2563eb;">✦</span>'
            '<span style="font-size:11.5px;color:#9ca3af;font-weight:500;">AI 자동 심의 · 규정셋 v4.2</span>'
            '</div>'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
            f'<span style="font-size:22px;font-weight:900;color:#111827;letter-spacing:-.03em;">1차 심의 결과</span>'
            f'{badge(vlabel, vtone)}</div>'
            f'{summary_html}'
            f'{findings_chip}'
            '</div>'
            # 세로 구분선
            '<div style="width:1px;background:#f0f2f6;flex-shrink:0;"></div>'
            # 오른쪽 위험도
            f'<div style="flex-shrink:0;display:flex;align-items:center;justify-content:center;gap:12px;padding:0 28px;">'
            f'<div style="display:flex;flex-direction:column;gap:2px;">'
            f'<span style="font-size:11.5px;color:#9ca3af;font-weight:500;">위험도</span>'
            f'<div><span style="font-size:34px;font-weight:900;color:#111827;letter-spacing:-.03em;">{score}</span>'
            f'<span style="font-size:13px;color:#9ca3af;font-weight:500;margin-left:3px;">점</span></div>'
            f'</div>'
            f'{score_arc(score, 64, 7)}'
            f'</div>'
            '</div>',
            unsafe_allow_html=True)

        # finding별 타임스탬프 매핑 (영상만)
        timeline_segs = ai.get("timeline") or [] if media == "영상" else []
        def mmss(t):
            m, s = int(t) // 60, int(t) % 60
            return f"{m:02d}:{s:02d}"
        def finding_timestamps(f):
            """finding의 phrases가 등장하는 타임라인 구간 시작 시각 목록"""
            ts = []
            for seg in timeline_segs:
                if not seg.get("flagged"): continue
                seg_terms = [t.lower() for t in seg.get("terms", [])]
                for phrase in f.get("phrases", []):
                    if phrase.lower() in seg_terms or phrase.lower() in seg.get("text", "").lower():
                        ts.append(seg["start"])
                        break
            return sorted(set(ts))

        # 발견 항목
        st.markdown('<div id="findings" style="font-size:13px;font-weight:800;color:#0f1b2d;margin:16px 0 8px;">발견 항목</div>', unsafe_allow_html=True)
        if not findings:
            st.markdown('<div class="lb-anim" style="background:#f3faf5;border:1px solid #d6efe0;border-radius:12px;padding:12px 14px;color:#15803d;font-weight:650;">'
                        + dot_span("green") + ' &nbsp;모든 심의 항목 이상 없음</div>', unsafe_allow_html=True)
        for i, f in enumerate(findings):
            sv = ("위반", "red") if f["sev"] == "high" else ("주의", "amber")
            timestamps = finding_timestamps(f)
            ts_html = ""
            if timestamps:
                chips = "".join(
                    f'<span style="font-family:ui-monospace,monospace;font-size:10.5px;font-weight:700;'
                    f'color:#2563eb;background:#eff6ff;border-radius:5px;padding:2px 7px;">{mmss(t)}</span>'
                    for t in timestamps)
                ts_html = f'<div style="display:flex;gap:5px;flex-wrap:wrap;margin-top:6px;">{chips}</div>'
            phrases_html = ""
            if f["phrases"]:
                chips_str = "  ".join(
                    f'<span style="color:#b0bacb;">"</span>{html.escape(p)}<span style="color:#b0bacb;">"</span>'
                    for p in f["phrases"])
                phrases_html = (f'<div style="font-size:13.5px;color:#27364e;font-weight:600;background:#f6f8fc;'
                                f'border:1px solid #e8ecf3;border-radius:8px;padding:8px 12px;margin:9px 0;">{chips_str}</div>')
            st.markdown(
                f'<div class="lb-anim lb-finding" style="border:1px solid #e8ecf3;border-radius:12px;'
                f'padding:13px 15px;margin-bottom:10px;background:#fff;animation-delay:{i * 0.12:.2f}s;animation-fill-mode:both;">'
                f'<div style="display:flex;align-items:center;gap:9px;">{dot_span(sv[1], 6)}{badge(sv[0], sv[1])}'
                f'<span style="font-size:13px;font-weight:750;color:#0f1b2d;">{html.escape(f["cat"])}</span>'
                f'<span style="margin-left:auto;">{ts_html}</span></div>'
                f'{phrases_html}'
                f'<div style="font-size:12.8px;color:#3a4a63;line-height:1.6;margin-top:6px;"><b style="color:#8593a8;">문제점</b> · {html.escape(f["issue"])}</div>'
                f'<div style="font-size:12.5px;color:#3a4a63;margin-top:5px;"><b style="color:#8593a8;">근거 규정</b> · '
                f'<span style="font-family:ui-monospace,monospace;">{html.escape(f["rule"])}</span></div></div>',
                unsafe_allow_html=True)

        # AI 수정 제안 — 내부 레이블([음성 자막], [화면 분석], N초:) 제거 후 표시
        if ai.get("alternative_text"):
            import re as _re
            import difflib as _diff
            cleaned = _re.sub(r'\[(음성 자막|화면 분석)\]\s*', '', ai["alternative_text"])
            cleaned = _re.sub(r'\d+초:\s*', '', cleaned)
            cleaned = _re.sub(r'\n{2,}', '\n', cleaned)
            paras = [p.strip() for p in cleaned.split("\n") if p.strip()]

            # 원본과 단어 단위 diff — 추가/변경된 단어 하이라이트
            orig_words = _re.split(r'(\s+)', rec["content"])
            def diff_para(para):
                new_words = _re.split(r'(\s+)', para)
                matcher = _diff.SequenceMatcher(None, orig_words, new_words, autojunk=False)
                result = []
                for tag, _, _, j1, j2 in matcher.get_opcodes():
                    chunk = "".join(new_words[j1:j2])
                    if tag == "equal":
                        result.append(html.escape(chunk))
                    else:
                        result.append(
                            f'<mark style="background:#dbeafe;color:#1d4ed8;border-radius:3px;'
                            f'padding:1px 3px;font-weight:600;">{html.escape(chunk)}</mark>'
                        )
                return "".join(result)

            para_style = 'margin:0 0 8px;font-size:14px;color:#1e293b;line-height:1.75;word-break:keep-all;'
            paras_html = "".join(
                f'<p style="{para_style}">{diff_para(p)}</p>'
                for p in paras[:-1]) + (
                f'<p style="margin:0;font-size:14px;color:#1e293b;line-height:1.75;word-break:keep-all;">{diff_para(paras[-1])}</p>'
                if paras else "")
            st.markdown(
                '<div class="lb-anim" style="background:#f1f7ff;border:1px solid #e3efff;border-radius:18px;'
                'padding:22px 24px;margin-top:4px;box-shadow:0 8px 24px rgba(37,99,235,.04);">'
                '<div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">'
                '<div style="width:30px;height:30px;border-radius:10px;'
                'background:linear-gradient(135deg,#2563eb 0%,#7c3aed 55%,#06b6d4 100%);'
                'display:grid;place-items:center;box-shadow:0 8px 18px rgba(37,99,235,.22);">'
                '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
                '<path d="M12 3l2.4 6.1L21 12l-6.6 2.9L12 21l-2.4-6.1L3 12l6.6-2.9L12 3z" '
                'stroke="white" stroke-width="2.2" stroke-linejoin="round"/>'
                '</svg></div>'
                '<span style="font-size:15px;font-weight:850;color:#1454d4;letter-spacing:-.01em;">AI 수정 제안</span>'
                '<span style="font-size:13px;color:#74a7f8;font-weight:650;">준법 기준에 맞게 수정한 대안 문구입니다</span>'
                '</div>'
                f'<div style="padding:2px 2px 0 40px;">{paras_html}</div>'
                '</div>',
                unsafe_allow_html=True)

        # 이미지·영상 미리보기 (텍스트는 위에서 이미 렌더링)
        if media == "영상":
            st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:10px;">'
                f'<span style="font-size:13.5px;font-weight:800;color:#0f1b2d;">추출 자막</span>'
                f'<span style="margin-left:auto;">{badge(f"위반 {high}", "red")} &nbsp;{badge(f"주의 {mid}", "amber")}</span></div>',
                unsafe_allow_html=True)
            with st.expander("추출 자막 · 위반 문구 하이라이트 보기"):
                st.markdown(
                    f'<p style="margin:0;font-size:14px;line-height:1.9;color:#27364e;word-break:keep-all;">{highlight(rec["content"], ai["rule_hits"])}</p>',
                    unsafe_allow_html=True)

    # 우 — 메타 정보 + 준법관리자 결재
    with right:
        if is_processing:
            st.markdown("""
<div style="background:#fff;border:1px solid #e6eaf1;border-radius:14px;padding:20px;opacity:.55;">
  <div style="font-size:15px;font-weight:800;color:#0f1b2d;margin-bottom:12px;">준법관리자 결재</div>
  <div style="font-size:12px;color:#94a3b8;margin-bottom:14px;">AI 심의가 완료된 후 결재가 가능합니다.</div>
  <div style="display:flex;gap:8px;margin-bottom:16px;">
    <div class="lb-sk" style="flex:1;height:60px;border-radius:10px;"></div>
    <div class="lb-sk" style="flex:1;height:60px;border-radius:10px;"></div>
    <div class="lb-sk" style="flex:1;height:60px;border-radius:10px;"></div>
  </div>
  <div class="lb-sk" style="height:80px;border-radius:10px;margin-bottom:14px;"></div>
  <div class="lb-sk" style="height:44px;border-radius:9px;"></div>
</div>
""", unsafe_allow_html=True)
        else:
            decision_panel(rec, high)
        st.markdown(
            '<div class="lb-anim" style="background:#fff;border:1px solid #e6eaf1;border-radius:12px;padding:4px 18px;margin-top:16px;">'
            + "".join(
                f'<div style="display:flex;padding:11px 0;border-bottom:{"none" if i==2 else "1px solid #f1f3f8"};">'
                f'<div style="width:90px;font-size:12.5px;color:#94a3b8;font-weight:600;">{k}</div>'
                f'<div style="font-size:13px;color:#27364e;font-weight:600;">{html.escape(str(v))}</div></div>'
                for i, (k, v) in enumerate([
                    ("매체", rec["media"] or "-"),
                    ("심의 강도", rec.get("review_mode") or "표준"),
                    ("요청 일시", rec["created_at"].replace("T", " ")),
                    ("적용 규정셋", "금융상품 광고 규정 v4.2"),
                ]))
            + "</div>", unsafe_allow_html=True)


def decision_panel(rec, high):
    if rec["decision_status"] != "대기":
        lbl, tone = DSTATUS[rec["decision_status"]]
        c = TONE[tone]
        st.markdown(
            f'<div class="lb-anim" style="background:{c["bg"]};border:1.5px solid {c["bd"]};border-radius:14px;padding:20px;text-align:center;">'
            f'<div style="font-size:18px;font-weight:850;color:{c["fg"]};">{lbl} 결재 완료</div>'
            f'<div style="font-size:13px;color:{c["fg"]};opacity:.8;margin-top:4px;">{html.escape(rec["reviewer"] or "-")} · {rec["decided_at"] or "-"}</div>'
            + (f'<div style="background:#fff;border:1px solid {c["bd"]};border-radius:9px;padding:10px 12px;margin-top:12px;'
               f'font-size:12.8px;color:#3a4a63;text-align:left;">{html.escape(rec["comment"])}</div>' if rec["comment"] else "")
            + "</div>", unsafe_allow_html=True)
        return

    st.markdown('<div style="font-size:15px;font-weight:800;color:#0f1b2d;margin-bottom:8px;">준법관리자 결재</div>', unsafe_allow_html=True)
    if high:
         st.markdown(f'<div style="padding:10px 13px;background:#fdecea;border:1px solid #f6cfc9;border-radius:10px;'
                     f'font-size:12.5px;color:#a52d27;margin-bottom:12px;"><b>위반 {high}건</b>이 발견되었습니다. 원안 승인 시 사유를 코멘트에 기재하십시오.</div>',
                     unsafe_allow_html=True)


    st.markdown('<div style="font-size:12.5px;font-weight:700;color:#52617a;margin-bottom:7px;">결재 의견을 선택하십시오</div>', unsafe_allow_html=True)
    st.session_state.setdefault("decision", None)
    chosen = st.session_state.get("decision")
    for col, (key, desc, tone) in zip(st.columns(3), DECISIONS):
        sel = chosen == key
        if col.button(f"**{key}**  \n{desc}", key=f"dec-{key}", use_container_width=True,
                      type="primary" if sel else "secondary"):
            st.session_state["decision"] = key
            st.rerun()

    st.markdown('<div style="font-size:12.5px;font-weight:700;color:#52617a;margin:16px 0 7px;">심의 의견 / 수정 지시</div>', unsafe_allow_html=True)
    comment = st.text_area("심의 의견", placeholder="결재 사유·수정 지시·게재 조건 등을 기재하십시오",
                           max_chars=500, label_visibility="collapsed")
    if st.button("결재 확정", type="primary", use_container_width=True):
        if not chosen:
            st.warning("결재 의견(승인·조건부승인·반려)을 먼저 선택하십시오")
        else:
            api_decide(rec["id"], chosen, comment, "김준기")
            st.success(f"{chosen} 결재가 기록되었습니다")
            go("detail", rec["id"])


# --- 진입점 ---

def main():
    st.set_page_config(page_title="LawBee 준법심의 콘솔", page_icon=":material/verified_user:", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("view", "dashboard")
    st.session_state.setdefault("rid", None)
    # URL 쿼리파라미터로 라우팅 (?rid= 상세, ?view= 화면 전환, ?del= 삭제 확인)
    rid_q = st.query_params.get("rid")
    view_q = st.query_params.get("view")
    if rid_q:
        st.session_state["view"] = "detail"
        try:
            st.session_state["rid"] = int(rid_q)
        except ValueError:
            pass
    elif view_q:
        st.session_state["view"] = view_q
        st.session_state["rid"] = None
    sidebar()

    view = st.session_state["view"]
    if view == "new":
        new_review()
    elif view == "detail" and st.session_state["rid"]:
        detail(st.session_state["rid"])
    else:
        dashboard()


main()
