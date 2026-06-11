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
MEDIA_OPTIONS = ["텍스트", "영상", "UI"]


# --- API 클라이언트 ---

def api_list():
    return requests.get(f"{API}/reviews", timeout=60).json()


def api_get(rid):
    return requests.get(f"{API}/reviews/{rid}", timeout=60).json()


def api_create(content, media, title=None):
    return requests.post(f"{API}/reviews", json={"content": content, "media": media, "title": title}, timeout=180).json()


def api_create_image(file_tuple, title=""):
    return requests.post(f"{API}/reviews/image", files={"file": file_tuple}, data={"title": title}, timeout=180).json()


def api_create_video(file_tuple, title=""):
    return requests.post(f"{API}/reviews/video", files={"file": file_tuple}, data={"title": title}, timeout=600).json()


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
    pad, fs, ds = ("2px 8px", "11px", "5px") if sm else ("3px 10px", "12px", "6px")
    d = (f'<span style="width:{ds}px;height:{ds}px;border-radius:999px;background:{c["dot"]};'
         f'margin-right:5px;display:inline-block;"></span>') if dot else ""
    return (f'<span style="display:inline-flex;align-items:center;padding:{pad};border-radius:999px;'
            f'font-size:{fs};font-weight:650;color:{c["fg"]};background:{c["bg"]};border:1px solid {c["bd"]};'
            f'white-space:nowrap;">{d}{html.escape(text)}</span>')


def dot_span(tone, size=7):
    return (f'<span style="display:inline-block;width:{size}px;height:{size}px;border-radius:999px;'
            f'background:{TONE[tone]["dot"]};"></span>')


MEDIA_ICON = {"텍스트": "▭", "영상": "▷", "UI": "▢"}


def media_tag(m):
    icon = MEDIA_ICON.get(m, "▭")
    return (f'<span style="display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:#475569;">'
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
        f'<div style="position:absolute;inset:0;display:grid;place-items:center;font-size:{14 if size>40 else 12}px;'
        f'font-weight:800;color:#0f1b2d;">{value}</div></div>'
    )


CSS = """
<style>
header[data-testid="stHeader"]{display:none;}
#MainMenu, footer{display:none;}
.stApp{background:#f4f6fa;}
.block-container{padding:1.1rem 1.8rem 3rem;max-width:1280px;animation:fadeIn .4s ease;}
[data-testid="stSidebar"]{background:#ffffff;border-right:1px solid #e6eaf1;}
[data-testid="stSidebarHeader"]{display:none !important;}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding-top:1.1rem !important;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.5rem;}
[data-testid="stSidebar"] .stButton>button{background:transparent;border:1px solid transparent;color:#475569;
  justify-content:flex-start !important;text-align:left !important;font-weight:650;border-radius:9px;width:100%;
  padding:9px 13px;transition:all .14s ease;}
[data-testid="stSidebar"] .stButton>button>div{justify-content:flex-start !important;width:100%;}
[data-testid="stSidebar"] .stButton>button [data-testid="stMarkdownContainer"]{width:100%;text-align:left !important;}
[data-testid="stSidebar"] .stButton>button [data-testid="stMarkdownContainer"] p{text-align:left !important;margin:0;width:100%;}
[data-testid="stSidebar"] .stButton>button:hover{background:#eef4ff;color:#1d4ed8;}
[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:linear-gradient(90deg,#1d4ed8,#2563eb) !important;
  color:#fff !important;border:none !important;box-shadow:0 4px 14px rgba(37,99,235,.28);}
[data-testid="stSidebar"] .stButton>button[kind="primary"]:hover{color:#fff !important;}
.stButton>button{border-radius:9px;font-weight:700;transition:all .14s ease;}
.stButton>button[kind="primary"]{background:linear-gradient(180deg,#2f6cf0,#1d4ed8);border:none;color:#fff !important;font-weight:800;}
.stButton>button[kind="primary"]:hover{transform:translateY(-1px);box-shadow:0 6px 16px rgba(37,99,235,.32);}
/* 결재 버튼 — 액션 크게, 설명 줄 작게 */
[class*="st-key-dec-"] button p{font-size:11px !important;line-height:1.5;margin:0;}
[class*="st-key-dec-"] button p strong{font-size:14.5px !important;font-weight:800;}
/* 입력 박스 또렷하게 */
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input{
  border:1px solid #d4dbe6 !important;background:#fff !important;border-radius:10px !important;}
[data-testid="stTextArea"] textarea:focus, [data-testid="stTextInput"] input:focus{
  border-color:#2563eb !important;box-shadow:0 0 0 2px rgba(37,99,235,.12) !important;}
h1,h2,h3{color:#0f1b2d;}
mark{text-decoration:none;}

/* 목록 흰색 카드 */
[class*="st-key-listcard"]{background:#fff !important;border:1px solid #e6eaf1 !important;
  border-radius:14px !important;box-shadow:0 2px 8px rgba(16,24,40,.05);
  padding:8px 26px 18px !important;}
/* 검수 리스트 — 행 링크 + 네이티브 체크박스 */
a.lb-row{color:inherit;transition:background .12s ease;animation:fadeIn .3s ease;border-radius:6px;display:block;}
a.lb-row:hover{background:#f8fafe;}
[class*="st-key-listcard"] [data-testid="stVerticalBlock"]{gap:.2rem;}
[class*="st-key-listcard"] [data-testid="stHorizontalBlock"]{gap:.4rem;}
/* 선택 삭제 버튼 (빨강) */
[class*="st-key-bulk-del-btn"] button{color:#c0322b !important;border-color:#f0c9c4 !important;}
[class*="st-key-bulk-del-btn"] button:hover{background:#fdecea !important;border-color:#f6cfc9 !important;}
/* 삭제 확인 모달의 삭제 버튼 (빨강) */
[class*="st-key-del-confirm"] button{background:linear-gradient(180deg,#e15a52,#c0322b) !important;
  border:none !important;color:#fff !important;font-weight:800;}
a[href="?view=dashboard"]:hover div{color:#fff;}

@keyframes fadeIn{from{opacity:0;transform:translateY(7px);}to{opacity:1;transform:none;}}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.35;}}
.lb-anim{animation:fadeIn .35s ease;}
.lb-pulse{animation:pulse 1.7s infinite;}
/* hover 효과 */
.lb-kpi{transition:all .15s ease;}
.lb-kpi:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(16,24,40,.08);border-color:#d3ddee !important;}
.lb-finding{transition:all .15s ease;}
.lb-finding:hover{transform:translateY(-1px);box-shadow:0 5px 16px rgba(16,24,40,.08);}
</style>
"""

SHIELD = ('<svg width="19" height="19" viewBox="0 0 24 24" fill="none">'
          '<path d="M12 2.5 4 6v6c0 4.4 3.1 7.9 8 9.5 4.9-1.6 8-5.1 8-9.5V6l-8-3.5Z" '
          'fill="rgba(255,255,255,.15)" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/>'
          '<path d="m8.4 12 2.5 2.5L15.8 9.6" stroke="#fff" stroke-width="1.9" '
          'stroke-linecap="round" stroke-linejoin="round"/></svg>')


# --- 네비게이션 ---

def go(view, rid=None):
    st.query_params.clear()
    st.session_state["view"] = view
    st.session_state["rid"] = rid
    st.session_state.pop("decision", None)
    st.rerun()


def sidebar():
    with st.sidebar:
        st.markdown(
            '<a href="?view=dashboard" target="_self" style="text-decoration:none;display:block;cursor:pointer;">'
            '<div style="display:flex;align-items:center;gap:9px;padding:4px 0 14px;">'
            f'<div style="width:34px;height:34px;border-radius:9px;background:linear-gradient(145deg,#3b82f6,#1d4ed8);'
            f'display:grid;place-items:center;box-shadow:0 3px 10px rgba(37,99,235,.3);flex-shrink:0;">{SHIELD}</div>'
            '<div style="line-height:1.1;"><div style="font-weight:800;font-size:20px;color:#0f1b2d;letter-spacing:-.01em;">LawBee</div>'
            '<div style="font-size:9.5px;color:#94a3b8;font-weight:600;letter-spacing:.04em;">준법심의 AGENT</div></div>'
            '</div></a>',
            unsafe_allow_html=True)
        st.markdown(
            '<div style="background:#f6f8fc;border:1px solid #e9edf4;border-radius:10px;padding:11px 12px;'
            'display:flex;align-items:center;gap:10px;margin:2px 0 28px;">'
            '<div style="width:30px;height:30px;border-radius:999px;background:linear-gradient(145deg,#3b82f6,#1d4ed8);'
            'display:grid;place-items:center;font-weight:700;color:#fff;">김</div>'
            '<div><div style="font-size:13px;font-weight:700;color:#27364e;">김준기</div>'
            '<div style="font-size:11px;color:#94a3b8;font-weight:600;">준법관리자 · 결재권자</div></div></div>',
            unsafe_allow_html=True)

        st.markdown('<div style="font-size:10px;font-weight:700;color:#aab4c5;letter-spacing:.09em;padding:6px 4px 10px;">NAVIGATION</div>', unsafe_allow_html=True)
        view = st.session_state.get("view", "dashboard")
        try:
            pending = sum(1 for r in api_list() if r["decision_status"] == "대기")
        except Exception:
            pending = 0
        dash_label = ":material/grid_view: 검수 대시보드"
        if pending:
            dash_label += f" :gray-badge[{pending}]"
        if st.button(dash_label, use_container_width=True, key="nav-dash",
                     type="primary" if view in ("dashboard", "detail", "new") else "secondary"):
            go("dashboard")
        for icon, label in [("menu_book", "심의 규정"), ("history", "결재 이력"), ("bar_chart", "통계·리포트")]:
            if st.button(f":material/{icon}: {label}", use_container_width=True, key=f"nav-{label}"):
                st.toast("준비 중인 메뉴입니다")

        st.markdown(
            '<div style="margin-top:18px;background:#f6f8fc;border:1px solid #e9edf4;border-radius:10px;padding:12px 13px;">'
            '<div style="display:flex;align-items:center;gap:7px;margin-bottom:6px;">'
            '<span class="lb-pulse" style="width:7px;height:7px;border-radius:999px;background:#16934f;box-shadow:0 0 0 3px rgba(22,147,79,.15);"></span>'
            '<span style="font-size:11.5px;font-weight:700;color:#52617a;">AI 심의엔진</span>'
            '<span style="margin-left:auto;font-size:10.5px;color:#16934f;font-weight:700;">정상</span></div>'
            '<div style="font-size:10.5px;color:#94a3b8;">규정셋 v4.2 · 2026.05.30</div></div>'
            '<div style="font-size:10.5px;color:#aab4c5;padding:12px 4px 2px;">드림은행 준법감시부<br>심의 콘솔 v1.0 (MVP)</div>',
            unsafe_allow_html=True)


def topbar(crumb, title, right_html=""):
    st.markdown(
        f'<div class="lb-anim" style="display:flex;align-items:center;padding:4px 2px 16px;border-bottom:1px solid #e6eaf1;margin-bottom:18px;">'
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

    tiles = [("전체 검수", "TOTAL", c["전체"], "건", "#2563eb"),
             ("검토 대기", "PENDING", c["대기"], "결재 필요", "#2563eb"),
             ("승인 완료", "APPROVED", c["승인"], "원안 승인", "#1ca25b"),
             ("결재 완료", "DECIDED", c["승인"] + c["조건부승인"], "승인·조건부", "#7c3aed"),
             ("반려", "REJECTED", c["반려"], "재작업", "#dc4338")]
    for col, (label, en, val, sub, accent) in zip(st.columns(5), tiles):
        col.markdown(
            f'<div class="lb-anim lb-kpi" style="background:#fff;border:1px solid #e6eaf1;border-radius:13px;padding:15px 17px;">'
            f'<div style="display:flex;justify-content:space-between;"><span style="font-size:12.5px;font-weight:700;color:#52617a;">{label}</span>'
            f'<span style="font-size:10px;font-weight:700;color:#aab4c5;">{en}</span></div>'
            f'<div style="display:flex;align-items:flex-end;gap:7px;margin-top:8px;">'
            f'<span style="font-size:26px;font-weight:820;color:{accent};line-height:1;letter-spacing:-.02em;">{val}</span>'
            f'<span style="font-size:12px;font-weight:600;color:#9aa6ba;padding-bottom:4px;">{sub}</span></div></div>',
            unsafe_allow_html=True)

    st.write("")
    flt = st.segmented_control("상태", ["전체", "대기", "승인", "조건부승인", "반려"],
                               default="전체", label_visibility="collapsed")

    # 툴바 — 카드 밖 위에 배치
    tc1, tc2, tc3, tc4 = st.columns([3.6, 1.15, 1.25, 0.5], vertical_alignment="center")
    q = tc1.text_input("검색", placeholder="제목·번호 검색", label_visibility="collapsed")
    media = tc2.selectbox("매체", ["모든 매체"] + MEDIA_OPTIONS, label_visibility="collapsed")
    if tc3.button(":material/add: 새 검수 요청", type="primary", use_container_width=True, key="new-review-btn"):
        go("new")
    rows = [r for r in reviews
            if (flt in (None, "전체") or r["decision_status"] == flt)
            and (media == "모든 매체" or r["media"] == media)
            and (not q or q in title_of(r) or q in f"RV-{r['id']:04d}")]
    selected = [r["id"] for r in rows if st.session_state.get(f"sel-{r['id']}")]
    if tc4.button(f":material/delete: {len(selected) or ''}".strip() or ":material/delete:", use_container_width=True,
                  key="bulk-del-btn", disabled=not selected, help="선택 항목 삭제"):
        st.session_state["pending_bulk"] = set(selected)
        st.rerun()

    with st.container(border=True, key="listcard"):
        st.markdown(
            f'<div style="font-weight:800;font-size:18px;color:#0f1b2d;letter-spacing:-.01em;padding-bottom:8px;padding-top:8px;">검수 목록 &nbsp;'
            f'<span style="font-size:12.5px;font-weight:700;color:#64748b;background:#eef1f6;border:1px solid #dde3ec;border-radius:999px;padding:2px 9px;vertical-align:middle;">{len(rows)}건</span></div>',
            unsafe_allow_html=True)

        grid = "minmax(180px,3.7fr) 0.64fr 0.74fr 0.92fr 0.52fr 0.42fr"
        heads = ["콘텐츠 / 요청", "매체", "AI 심의결과", "발견 항목", "처리 상태", ""]
        hc = st.columns([0.032, 0.968], gap="small", vertical_alignment="center")
        hc[1].markdown(
            f'<div style="display:grid;grid-template-columns:{grid};gap:12px;padding:4px 4px 8px;'
            f'font-size:11.5px;font-weight:700;color:#8593a8;">'
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
            fb = badge("분석 중…", "slate", sm=True) if is_processing else (
                (badge(f"위반 {high}", "red", sm=True) + " " if high else "") + (badge(f"주의 {mid}", "amber", sm=True) if mid else ""))
            if not is_processing and not high and not mid:
                fb = badge("이슈 없음", "green", sm=True)
            trail = ('<span style="font-size:12px;font-weight:700;color:#2563eb;">결재 ›</span>'
                     if r["decision_status"] == "대기"
                     else '<span style="font-size:11.5px;color:#b9c2d0;">완료 ›</span>')
            rc = st.columns([0.032, 0.968], gap="small", vertical_alignment="center")
            rc[0].checkbox("선택", key=f"sel-{r['id']}", label_visibility="collapsed")
            rc[1].markdown(
                f'<a class="lb-row" href="?rid={r["id"]}" target="_self" style="text-decoration:none;display:block;">'
                f'<div style="display:grid;grid-template-columns:{grid};gap:12px;align-items:center;'
                f'padding:8px 4px;border-bottom:1px solid #f1f3f8;">'
                f'<div style="display:flex;align-items:center;gap:12px;min-width:0;">{score_ring(score, 34, 4)}'
                f'<div style="min-width:0;"><div style="font-size:13.5px;font-weight:700;color:#0f1b2d;letter-spacing:-.01em;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{html.escape(title_of(r))}</div>'
                f'<div style="font-size:11px;color:#9aa6ba;font-weight:500;margin-top:2px;">RV-{r["id"]:04d} · {html.escape(r["created_at"][:10])}</div></div></div>'
                f'<div>{media_tag(r["media"])}</div>'
                f'<div>{badge(vlabel, vtone, dot=True, sm=True)}</div>'
                f'<div>{fb}</div>'
                f'<div>{badge(slabel, stone, sm=True)}</div>'
                f'<div style="text-align:right;">{trail}</div></div></a>',
                unsafe_allow_html=True)


# --- 화면: 새 검수 요청 ---

def _ai_guide(msg="AI가 본문을 분석해 위반·주의 항목과 수정 제안을 자동 생성합니다. 결과는 준법관리자가 검토·결재합니다."):
    st.markdown(
        '<div class="lb-anim" style="padding:12px 14px;border-radius:12px;background:#0e1626;color:#9fb0d0;font-size:11.8px;line-height:1.6;">'
        f'<div style="font-weight:700;color:#fff;margin-bottom:5px;">AI 심의 안내</div>{msg}</div>',
        unsafe_allow_html=True)


def new_review():
    topbar("준법심의 콘솔", "새 검수 요청", live_dot("AI 1차 심의 자동 실행"))
    mode = st.radio("입력 방식", ["텍스트", "이미지", "영상"], horizontal=True, label_visibility="collapsed")
    st.markdown('**제목** <span style="color:#dc4338;">*</span>', unsafe_allow_html=True)
    title = st.text_input("제목", label_visibility="collapsed",
                          placeholder="예) 신규 정기적금 「드림플러스」 출시 배너")
    left, right = st.columns([2, 1])

    if mode == "텍스트":
        with left:
            st.markdown('**광고/콘텐츠 본문** <span style="color:#dc4338;">*</span>', unsafe_allow_html=True)
            if st.button("샘플 카피 불러오기"):
                st.session_state["draft"] = ("내 통장이 매달 불어나는 가장 확실한 방법, 원금 손실 걱정 없이 누구나 "
                                             "최고 연 5.0% 수익을 무조건 보장해 드립니다. 업계 1위 드림은행 「드림플러스」 "
                                             "적금으로 지금 바로 시작하세요. 선착순 1만 좌 한정!")
            body = st.text_area("콘텐츠 본문", value=st.session_state.get("draft", ""),
                                height=200, label_visibility="collapsed",
                                placeholder="광고 카피·영상 스크립트·본문 텍스트를 입력하십시오")
        with right:
            media = st.selectbox("매체 (채널)", MEDIA_OPTIONS)
            _ai_guide()
        st.write("")
        if st.button("AI 심의 요청", type="primary", disabled=not (body.strip() and title.strip())):
            with st.spinner("AI 1차 심의 진행 중…"):
                rec = api_create(body, media, title.strip())
            st.session_state.pop("draft", None)
            go("detail", rec["id"])

    elif mode == "이미지":
        with left:
            up = st.file_uploader("이미지 첨부", type=["png", "jpg", "jpeg", "webp"])
            if up:
                st.image(up, use_container_width=True)
        with right:
            _ai_guide("이미지의 문구와 시각 구성을 추출해 심의합니다. 배너·캡처 등을 업로드하십시오.")
        st.write("")
        if st.button("AI 심의 요청", type="primary", disabled=(up is None or not title.strip())):
            with st.spinner("이미지에서 문구 추출 + 심의 중…"):
                rec = api_create_image((up.name, up.getvalue(), up.type), title.strip())
            go("detail", rec["id"])

    else:  # 영상
        with left:
            up = st.file_uploader("영상 첨부", type=["mp4", "mov", "webm", "m4a", "mp3", "wav"])
            if up:
                st.video(up)
        with right:
            _ai_guide("영상의 음성·화면을 분석해 구간별로 심의하고, 위반을 타임라인에 표시합니다.")
        st.write("")
        if st.button("AI 심의 요청", type="primary", disabled=(up is None or not title.strip())):
            with st.spinner("업로드 중…"):
                rec = api_create_video((up.name, up.getvalue(), up.type), title.strip())
            go("detail", rec["id"])


# --- 화면: 검수·결재 상세 ---

_VIDEO_TPL = """
<style>
@keyframes lb-fadein {
  from { opacity:0; transform:translateY(6px); }
  to   { opacity:1; transform:translateY(0); }
}
.lb-seg { animation: lb-fadein .35s ease both; }
</style>
<div style="display:grid;grid-template-columns:1fr 340px;gap:14px;font-family:-apple-system,'Apple SD Gothic Neo',sans-serif;">
  <video id="vid" src="__URL__" controls style="width:100%;border-radius:12px;background:#000;max-height:420px;"></video>
  <div style="display:flex;flex-direction:column;border:1px solid #e6eaf1;border-radius:12px;background:#fff;overflow:hidden;height:420px;">
    <div style="padding:10px 12px;border-bottom:1px solid #eef1f6;font-size:12px;font-weight:700;color:#0f1b2d;display:flex;align-items:center;gap:6px;">
      <span style="width:7px;height:7px;border-radius:9px;background:#34d399;display:inline-block;"></span>실시간 위반 감지
      <span id="feed-hint" style="margin-left:auto;font-size:11px;color:#94a3b8;font-weight:500;">▶ 재생 시 구간별 분석 결과가 표시됩니다</span>
    </div>
    <div id="segs" style="flex:1;overflow:auto;padding:8px 10px;">__FEED_INIT__</div>
  </div>
</div>
<script>
const SEGS = __SEGS__;
const vid = document.getElementById('vid');
const wrap = document.getElementById('segs');
const hint = document.getElementById('feed-hint');
var shown = new Array(SEGS.length).fill(false);

function mmss(t){var m=Math.floor(t/60),s=Math.floor(t%60);return (''+m).padStart(2,'0')+':'+(''+s).padStart(2,'0');}

function makeEl(s, i) {
  var d = document.createElement('div');
  d.id = 'seg'+i;
  d.className = 'lb-seg';
  var baseCss = 'padding:7px 9px;border-radius:7px;margin-bottom:5px;font-size:12.5px;line-height:1.5;border:1px solid transparent;cursor:pointer;transition:background .2s,border-color .2s,color .2s;';
  d.style.cssText = baseCss + (s.flagged ? 'background:#fdecea;border-color:#f6cfc9;color:#7a2a25;' : 'background:#eef4ff;border-color:#c9d8fb;color:#1d4ed8;');
  var terms = s.flagged
    ? s.terms.map(function(t){ return '<span style="background:#fdecea;color:#c0322b;border:1px solid #f6cfc9;border-radius:999px;padding:1px 7px;font-size:11px;font-weight:700;margin-left:4px;">'+t+'</span>'; }).join('')
    : '';
  var k = s.kind==='화면' ? 'background:#f0eafc;color:#6d28d9;' : 'background:#eef2f8;color:#64748b;';
  var kindTag = '<span style="font-size:10px;font-weight:700;border-radius:4px;padding:1px 5px;margin-right:5px;'+k+'">'+s.kind+'</span>';
  d.innerHTML = '<span style="font-family:monospace;color:#2563eb;font-weight:700;">'+mmss(s.start)+'</span> '+kindTag+s.text+terms;
  d.onclick = function(){ vid.currentTime = s.start; vid.play(); };
  d.title = '클릭하면 이 시점으로 이동';
  return d;
}

/* 현재 재생 시각 기준으로 등장해야 할 항목을 추가 */
function syncFeed(t) {
  var appeared = false;
  SEGS.forEach(function(s, i) {
    if (!shown[i] && t >= s.start) {
      var el = makeEl(s, i);
      wrap.appendChild(el);
      el.scrollIntoView({behavior:'smooth', block:'nearest'});
      shown[i] = true;
      appeared = true;
    }
    /* 현재 구간 하이라이트 */
    var el = document.getElementById('seg'+i);
    if (!el) return;
    var active = t >= s.start && t < s.end;
    if (active) {
      el.style.background = s.flagged ? '#fbc6c3' : '#dbeafe';
      el.style.borderColor = s.flagged ? '#e87d77' : '#93c5fd';
    } else {
      el.style.background = s.flagged ? '#fdecea' : '#eef4ff';
      el.style.borderColor = s.flagged ? '#f6cfc9' : '#c9d8fb';
    }
  });
  if (appeared && hint) hint.style.display = 'none';
}

/* seek 시 이미 지난 구간은 즉시 모두 표시 */
vid.addEventListener('seeked', function(){ syncFeed(vid.currentTime); });
vid.addEventListener('timeupdate', function(){ syncFeed(vid.currentTime); });
</script>
"""


_SKELETON_HTML = """
<style>
@keyframes lb-shimmer {
  0%   { background-position: -400px 0; }
  100% { background-position: 400px 0; }
}
.lb-skel {
  background: linear-gradient(90deg, #f1f5f9 25%, #e2e8f0 50%, #f1f5f9 75%);
  background-size: 800px 100%;
  animation: lb-shimmer 1.4s infinite linear;
  border-radius: 6px;
}
</style>
<div style="display:flex;flex-direction:column;gap:8px;padding:10px;">
  <div class="lb-skel" style="height:14px;width:55%;"></div>
  <div class="lb-skel" style="height:14px;width:80%;margin-top:4px;"></div>
  <div class="lb-skel" style="height:14px;width:65%;margin-top:10px;"></div>
  <div class="lb-skel" style="height:14px;width:75%;margin-top:4px;"></div>
  <div class="lb-skel" style="height:14px;width:50%;margin-top:10px;"></div>
  <div class="lb-skel" style="height:14px;width:70%;margin-top:4px;"></div>
  <div style="margin-top:14px;font-size:11.5px;color:#94a3b8;text-align:center;">음성·화면 분석 및 AI 준법 심의를 진행하고 있습니다.</div>
</div>
"""


def _video_review(rid, timeline, is_processing=False):
    """영상 재생 + 재생 시각 동기화 실시간 위반 피드 (클라이언트 사이드)"""
    segs = json.dumps(
        [{"start": s["start"], "end": s["end"], "text": s["text"], "flagged": s["flagged"],
          "terms": s["terms"], "kind": s.get("kind", "음성")}
         for s in timeline], ensure_ascii=False)
    feed_content = _SKELETON_HTML if is_processing else ""
    out = (_VIDEO_TPL
           .replace("__URL__", f"{API}/reviews/{rid}/media")
           .replace("__SEGS__", segs)
           .replace("__FEED_INIT__", feed_content))
    components.html(out, height=440)


def detail(rid):
    import time as _time
    rec = api_get(rid)
    is_processing = rec.get("decision_status") == "처리중"
    ai = rec["ai_result"]
    findings = [] if is_processing else build_findings(ai)
    high, mid = counts_of(findings)
    vlabel, vtone = ("분석 중", "slate") if is_processing else VERDICT[ai["status"]]

    right_badge = (live_dot("AI 분석 중…") if is_processing else badge(vlabel, vtone, dot=True))
    topbar(f"준법심의 · RV-{rec['id']:04d}", title_of(rec), right_badge)

    media = rec["media"]
    if media == "영상":
        st.markdown('<div style="font-size:14.5px;font-weight:800;color:#0f1b2d;margin-bottom:10px;">영상 미리보기 · 실시간 위반 감지</div>', unsafe_allow_html=True)
        timeline = [] if is_processing else (ai.get("timeline") or [])
        _video_review(rec["id"], timeline, is_processing=is_processing)
        if is_processing:
            _time.sleep(3)
            st.rerun()
            return

    left, right = st.columns([1.15, 0.85])

    # 좌 — AI 심의 결과 + 발견 항목 + AI 수정 제안 + 원본 미리보기
    with left:
        # AI 1차 심의 결과 카드
        passed = max(0, 6 - high - mid)
        count_chips = "".join(
            f'<div style="flex:1;padding:9px 12px;border-radius:10px;background:{TONE[t]["bg"]};border:1px solid {TONE[t]["bd"]};">'
            f'<span style="font-size:19px;font-weight:850;color:{TONE[t]["fg"]};">{n}</span>'
            f'<span style="font-size:12px;font-weight:700;color:{TONE[t]["fg"]};opacity:.85;margin-left:6px;">{lab}</span></div>'
            for n, lab, t in [(high, "위반", "red"), (mid, "주의", "amber"), (passed, "통과", "green")])
        st.markdown(
            '<div class="lb-anim" style="background:#fff;border:1px solid #e6eaf1;border-radius:14px;padding:16px 18px;box-shadow:0 1px 2px rgba(16,24,40,.04);">'
            '<div style="display:flex;align-items:center;gap:12px;">'
            '<div style="width:34px;height:34px;border-radius:9px;background:linear-gradient(145deg,#3b82f6,#1d4ed8);display:grid;place-items:center;font-size:12px;font-weight:800;color:#fff;box-shadow:0 3px 10px rgba(37,99,235,.35);">AI</div>'
            f'<div style="flex:1;"><span style="font-size:15.5px;font-weight:800;color:#0f1b2d;">AI 1차 심의 결과</span> &nbsp;{badge(vlabel, vtone)}'
            f'<div style="font-size:12px;color:#8593a8;margin-top:2px;">규정셋 v4.2 · {html.escape(ai["summary"][:80])}{"…" if len(ai["summary"]) > 80 else ""}</div></div>'
            f'{score_ring(ai_score(ai), 50, 5)}</div>'
            f'<div style="display:flex;gap:8px;margin-top:14px;">{count_chips}</div></div>',
            unsafe_allow_html=True)

        # 발견 항목
        st.markdown('<div style="font-size:13px;font-weight:800;color:#0f1b2d;margin:16px 0 8px;">발견 항목</div>', unsafe_allow_html=True)
        if not findings:
            st.markdown('<div class="lb-anim" style="background:#f3faf5;border:1px solid #d6efe0;border-radius:12px;padding:12px 14px;color:#15803d;font-weight:650;">'
                        + dot_span("green") + ' &nbsp;모든 심의 항목 이상 없음</div>', unsafe_allow_html=True)
        for i, f in enumerate(findings):
            sv = ("위반", "red") if f["sev"] == "high" else ("주의", "amber")
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
                f'<span style="font-size:13px;font-weight:750;color:#0f1b2d;">{html.escape(f["cat"])}</span></div>'
                f'{phrases_html}'
                f'<div style="font-size:12.8px;color:#3a4a63;line-height:1.6;margin-top:6px;"><b style="color:#8593a8;">문제점</b> · {html.escape(f["issue"])}</div>'
                f'<div style="font-size:12.5px;color:#3a4a63;margin-top:5px;"><b style="color:#8593a8;">근거 규정</b> · '
                f'<span style="font-family:ui-monospace,monospace;">{html.escape(f["rule"])}</span></div></div>',
                unsafe_allow_html=True)

        # AI 수정 제안 — 내부 레이블([음성 자막], [화면 분석], N초:) 제거 후 표시
        if ai.get("alternative_text"):
            import re as _re
            cleaned = _re.sub(r'\[(음성 자막|화면 분석)\]\s*', '', ai["alternative_text"])
            cleaned = _re.sub(r'\d+초:\s*', '', cleaned)
            paras = [p.strip() for p in cleaned.split("\n") if p.strip()]
            paras_html = "".join(
                f'<p style="margin:0 0 10px;font-size:13.5px;color:#27364e;line-height:1.75;word-break:keep-all;">{html.escape(p)}</p>'
                for p in paras)
            st.markdown(
                '<div class="lb-anim" style="background:#f0f6ff;border:1px solid #d4e3fb;border-radius:14px;padding:18px 20px;margin-top:4px;">'
                '<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">'
                '<div style="width:24px;height:24px;border-radius:7px;background:linear-gradient(145deg,#3b82f6,#1d4ed8);display:grid;place-items:center;font-size:9px;font-weight:800;color:#fff;">AI</div>'
                '<span style="font-size:13.5px;font-weight:800;color:#1d4ed8;">AI 수정 제안</span>'
                '<span style="font-size:11px;color:#93b4f0;margin-left:4px;">준법 기준에 맞게 수정한 대안 문구입니다</span></div>'
                f'<div style="background:#fff;border:1px solid #d4e3fb;border-radius:10px;padding:14px 16px;">{paras_html}</div>'
                '</div>',
                unsafe_allow_html=True)

        # 원본 미리보기
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        head = "추출 자막" if media == "영상" else "콘텐츠 미리보기"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:10px;">'
            f'<span style="font-size:13.5px;font-weight:800;color:#0f1b2d;">{head}</span>'
            f'<span style="margin-left:auto;">{badge(f"위반 {high}", "red")} &nbsp;{badge(f"주의 {mid}", "amber")}</span></div>',
            unsafe_allow_html=True)
        if media == "이미지":
            st.image(f"{API}/reviews/{rec['id']}/media", use_container_width=True)
            with st.expander("AI가 추출한 텍스트 보기"):
                st.write(rec["content"])
        elif media == "영상":
            with st.expander("추출 자막 · 위반 문구 하이라이트 보기"):
                st.markdown(
                    f'<p style="margin:0;font-size:14px;line-height:1.9;color:#27364e;word-break:keep-all;">{highlight(rec["content"], ai["rule_hits"])}</p>',
                    unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="lb-anim" style="background:#fff;border:1px solid #e6eaf1;border-radius:14px;padding:22px 24px;box-shadow:0 1px 2px rgba(16,24,40,.04);">'
                '<div style="font-size:11px;font-weight:700;color:#94a3b8;letter-spacing:.05em;margin-bottom:10px;">본문 카피 · 위반 문구 하이라이트</div>'
                f'<p style="margin:0;font-size:16px;line-height:1.95;color:#27364e;word-break:keep-all;">{highlight(rec["content"], ai["rule_hits"])}</p>'
                '</div>', unsafe_allow_html=True)

    # 우 — 메타 정보 + 준법관리자 결재
    with right:
        decision_panel(rec, high)
        st.markdown(
            '<div class="lb-anim" style="background:#fff;border:1px solid #e6eaf1;border-radius:12px;padding:4px 18px;margin-top:16px;">'
            + "".join(
                f'<div style="display:flex;padding:11px 0;border-bottom:{"none" if i==2 else "1px solid #f1f3f8"};">'
                f'<div style="width:90px;font-size:12.5px;color:#94a3b8;font-weight:600;">{k}</div>'
                f'<div style="font-size:13px;color:#27364e;font-weight:600;">{html.escape(str(v))}</div></div>'
                for i, (k, v) in enumerate([("매체", rec["media"] or "-"), ("요청 일시", rec["created_at"].replace("T", " ")),
                                            ("적용 규정셋", "금융상품 광고 규정 v4.2")]))
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
