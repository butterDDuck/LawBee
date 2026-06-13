"""법령 원문(RTF) 로더

doc/ 폴더의 법제처 다운로드 .doc(실제로는 RTF) 파일을
조(條) 단위 청크로 분할해 LangChain Document 목록으로 변환

파일명 규칙: 법령명(종류)(제N호)(시행일YYYYMMDD).doc
조문 체계가 없는 심사지침류는 로마숫자 섹션(Ⅰ.~) 단위로 분할
"""
import re
from pathlib import Path

from langchain_core.documents import Document
from striprtf.striprtf import rtf_to_text

# 법령명 키워드 → 청크 id 접두 코드 (KB jsonl 의 코드 체계와 맞춤)
LAW_CODES = [
    ("표시·광고의 공정화", "FAD"),
    ("전자상거래", "ECA"),
    ("금융소비자 보호", "FCPA"),
    ("정보통신망", "ICNA"),
    ("추천·보증", "ENDORSE"),
    ("금융상품 등의 표시·광고", "FINAD"),
]

# 법령 종류 → id 구분자 / 법제처 URL 경로
KIND_CODES = {
    "법률": "L",
    "대통령령": "D",
    "금융위원회고시": "R",
    "공정거래위원회예규": "G",
}
ADMIN_RULE_KINDS = {"금융위원회고시", "공정거래위원회예규"}

FILENAME_RE = re.compile(r"^(?P<law>.+?)\((?P<kind>[^()]+)\)\((?P<no>[^()]+)\)\((?P<date>\d{8})\)$")
ARTICLE_RE = re.compile(r"^제(?P<no>\d+)조(?:의(?P<sub>\d+))?(?:\(|\s|$)")
ARTICLE_LABEL_RE = re.compile(r"^제\d+조(?:의\d+)?(?:\([^)]*\))?")
# 심사지침류는 로마숫자 섹션 체계, 첫 섹션(목적)이 "제0조."로 변환되는 경우 포함
SECTION_RE = re.compile(r"^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]\s*\.|제0조\.)\s*\S.*$")
ADDENDA_RE = re.compile(r"^부\s*칙\b|^부\s*칙\s*<")
CLAUSE_RE = re.compile(r"(?=[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])")

MAX_CHUNK = 1800


def _strip_bin(data: bytes) -> bytes:
    """RTF 에 포함된 \\binN 바이너리 블록(이미지 등)을 제거

    바이너리 바이트가 중괄호를 포함해 RTF 그룹 파싱을 깨뜨리므로 선제 제거
    """
    out = bytearray()
    i = 0
    while True:
        j = data.find(b"\\bin", i)
        if j == -1:
            out += data[i:]
            break
        k = j + 4
        num = b""
        while k < len(data) and data[k : k + 1].isdigit():
            num += data[k : k + 1]
            k += 1
        if not num:
            out += data[i:k]
            i = k
            continue
        if data[k : k + 1] == b" ":
            k += 1
        out += data[i:j]
        i = k + int(num)
    return bytes(out)


def _extract_text(path: Path) -> str:
    raw = _strip_bin(path.read_bytes())
    # 한글은 \uN 이스케이프로 들어 있어 ascii 디코딩으로 충분
    return rtf_to_text(raw.decode("ascii", errors="ignore"), errors="ignore")


def _normalize(name: str) -> str:
    """법령명의 가운뎃점 표기(ㆍ/·) 차이를 통일"""
    return name.replace("ㆍ", "·").strip()


def _body_lines(text: str) -> list[str]:
    """본문 라인만 추출 (부칙 이후 제거, 전각공백 정리)"""
    lines = []
    for line in text.splitlines():
        line = line.replace("　", " ").strip()
        if ADDENDA_RE.match(line):
            break
        lines.append(line)
    return lines


def _split_by(lines: list[str], head_re: re.Pattern) -> list[tuple[str, str]]:
    """head_re 에 매칭되는 라인을 경계로 (표제 라인, 본문) 목록 생성"""
    chunks: list[tuple[str, str]] = []
    cur_head, cur_body = None, []
    for line in lines:
        if head_re.match(line):
            if cur_head is not None:
                chunks.append((cur_head, "\n".join(cur_body).strip()))
            cur_head, cur_body = line, []
        elif cur_head is not None and line:
            cur_body.append(line)
    if cur_head is not None:
        chunks.append((cur_head, "\n".join(cur_body).strip()))
    return chunks


def _split_long(body: str) -> list[str]:
    """MAX_CHUNK 초과 본문을 항(①②③) 경계 우선, 그래도 길면 줄 단위로 분할"""
    if len(body) <= MAX_CHUNK:
        return [body]
    parts = []
    for p in CLAUSE_RE.split(body):
        if not p.strip():
            continue
        if len(p) > MAX_CHUNK:
            parts.extend(l for l in p.split("\n") if l.strip())
        else:
            parts.append(p)
    out, buf = [], ""
    for p in parts:
        if buf and len(buf) + len(p) > MAX_CHUNK:
            out.append(buf)
            buf = ""
        buf = p if not buf else buf + "\n" + p
    if buf:
        out.append(buf)
    return out


def _law_code(law: str, kind: str) -> str:
    name = _normalize(law)
    # 금융상품 심사지침이 표시·광고 키워드와 겹치므로 구체적인 것부터 매칭
    for key, code in sorted(LAW_CODES, key=lambda x: -len(x[0])):
        if key in name:
            base = code
            break
    else:
        base = "LAW"
    return f"{base}-{KIND_CODES.get(kind, 'X')}"


def _source_url(law: str, kind: str) -> str:
    path = "행정규칙" if kind in ADMIN_RULE_KINDS else "법령"
    return f"https://www.law.go.kr/{path}/{_normalize(law).replace(' ', '')}"


def _article_no(head: str) -> str:
    m = ARTICLE_RE.match(head)
    no = m.group("no")
    if m.group("sub"):
        no += f"-{m.group('sub')}"
    return no


def load_laws(doc_dir: str) -> list[Document]:
    """doc_dir 의 모든 .doc(RTF) 법령을 조 단위 Document 로 변환"""
    docs: list[Document] = []
    root = Path(doc_dir)
    if not root.is_dir():
        return docs

    for path in sorted(root.glob("*.doc")):
        m = FILENAME_RE.match(path.stem)
        if not m:
            print(f"[law_loader] 파일명 규칙 불일치, 건너뜀: {path.name}")
            continue
        law, kind = _normalize(m.group("law")), m.group("kind")
        code = _law_code(law, kind)
        lines = _body_lines(_extract_text(path))

        chunks = _split_by(lines, ARTICLE_RE)
        if len(chunks) < 3:  # 조문 체계가 아니면 로마숫자 섹션으로 분할
            chunks = _split_by(lines, SECTION_RE)

        seen: dict[str, int] = {}
        for head, body in chunks:
            content = f"{head}\n{body}".strip() if body else head
            if "삭제" in head and not body:
                continue
            am = ARTICLE_RE.match(head)
            # 조문은 첫 줄에 본문이 이어지므로 "제N조(제목)" 표제만 분리
            label = ARTICLE_LABEL_RE.match(head).group() if am else head
            no = _article_no(head) if am else head.split(".")[0].strip()
            chunk_id_base = f"{code}{no}"

            # 같은 조가 현행·시행예정 본문으로 중복 수록된 경우 버전 suffix 부여
            seen[chunk_id_base] = seen.get(chunk_id_base, 0) + 1
            if seen[chunk_id_base] > 1:
                chunk_id_base += f"v{seen[chunk_id_base]}"

            pieces = _split_long(content)
            for i, piece in enumerate(pieces):
                chunk_id = chunk_id_base if len(pieces) == 1 else f"{chunk_id_base}p{i + 1}"
                docs.append(
                    Document(
                        page_content=f"[법령 원문] {law} {label}\n적용 매체: 공통\n규정 내용: {piece}",
                        metadata={
                            "id": chunk_id,
                            "category": "법령 원문",
                            "law": law,
                            "article": label,
                            "media": "공통",
                            "compliance_check": "",
                            "source_law": law,
                            "source_url": _source_url(law, kind),
                        },
                    )
                )
        print(f"[law_loader] {path.name} → {len(chunks)}개 조문 청크")
    return docs
