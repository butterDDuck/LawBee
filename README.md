# LawBee 🐝

사내 마케팅/영업 콘텐츠 **준법심의 자동화 AI** (JB금융그룹 Compliance AI)

마케터가 만든 광고/홍보물을 배포 전 1차로 자동 심의합니다. 규제 조항을 검색·참조해
위반 근거를 제시하고(RAG), 규칙 기반 판단과 LLM 판단을 결합해 대안 문구를 추천합니다.

## 기술 스택
- **언어/프레임워크**: Python · FastAPI · Streamlit
- **RAG/LLM**: LangChain · LangGraph · OpenAI API
- **벡터DB**: FAISS

## 디렉토리 구조
```
LawBee/
├── data/regulations.jsonl   # 규제 조항 54청크 (RAG 지식베이스)
├── app/
│   ├── config.py            # .env 설정 로딩
│   ├── ingest.py            # jsonl → 임베딩 → FAISS 적재
│   ├── retriever.py         # 벡터스토어 로드 + 조항 검색(매체 필터)
│   ├── rules.py             # 결정론적 룰 엔진
│   ├── schema.py            # 심의 입출력 + LLM 구조화 출력 스키마
│   ├── graph.py             # LangGraph 심의 파이프라인
│   └── main.py              # FastAPI 엔드포인트
├── ui/                      # (예정) Streamlit 검수 화면
└── scripts/
    ├── test_search.py       # RAG 검색 동작 확인
    └── test_review.py       # 심의 파이프라인 동작 확인
```

## 시작하기

```bash
# 1. 가상환경 + 의존성
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 환경변수 설정
cp .env.example .env
#   .env 를 열어 OPENAI_API_KEY 입력

# 3. 벡터스토어 생성 (규제 데이터 임베딩)
python -m app.ingest

# 4. 검색 동작 확인
python -m scripts.test_search "원금 100% 보장 확정 수익 펀드"

# 5. 심의 파이프라인 동작 확인
python -m scripts.test_review "원금 100% 보장! 업계 1위 확정 수익 펀드, 누구나 가입 가능"

# 6. API 서버 구동
uvicorn app.main:app --reload
#   POST /review  { "content": "...", "media": "텍스트" }
#   문서: http://127.0.0.1:8000/docs
```

## 개발 로드맵
- [x] 0. 프로젝트 스캐폴딩 + 규제 데이터 확보
- [x] 1. RAG 적재 파이프라인 (jsonl → FAISS)
- [x] 2. 검색 검증 (실 API 키로 동작 확인)
- [x] 3. LangGraph 심의 파이프라인 (룰엔진 → RAG → LLM판단 → 대안문구)
- [x] 4. FastAPI 엔드포인트
- [ ] 5. Streamlit 검수 UI
- [ ] 6. AWS 배포
