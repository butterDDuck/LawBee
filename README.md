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
│   ├── domain/              # 스키마·타입 (의존성 0)
│   │   └── schema.py
│   ├── rag/                 # ingest(적재) · retriever(검색)
│   ├── rules/               # engine(엔진) · lexicon(금칙어 사전)
│   ├── services/            # graph (LangGraph 심의 파이프라인)
│   ├── store/               # sqlite (심의 건 저장소)
│   ├── preprocess/          # image · video 멀티모달 전처리 (인터페이스)
│   └── api/                 # main (FastAPI 엔드포인트)
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
python -m app.rag.ingest

# 4. 검색 동작 확인
python -m scripts.test_search "원금 100% 보장 확정 수익 펀드"

# 5. 심의 파이프라인 동작 확인
python -m scripts.test_review "원금 100% 보장! 업계 1위 확정 수익 펀드, 누구나 가입 가능"

# 6. API 서버 구동
uvicorn app.api.main:app --reload
#   POST /review  { "content": "...", "media": "텍스트" }
#   문서: http://127.0.0.1:8000/docs
```

## Docker 로 실행

로컬 환경 세팅 없이 동일한 환경에서 실행합니다.

```bash
cp .env.example .env          # OPENAI_API_KEY 입력

# 최초 1회 벡터스토어 적재 (임베딩에 키 필요)
docker compose run --rm api python -m app.rag.ingest

# API + UI 서버 구동
docker compose up
#   API  : http://127.0.0.1:8000/docs
#   UI   : http://127.0.0.1:8501
```

> 벡터스토어와 심의 DB 는 호스트의 `vectorstore/`·`var/` 에 영속화됩니다.

## 준법관리자 콘솔 (Streamlit)

API 서버가 떠 있는 상태에서 검수 콘솔을 실행합니다.

```bash
uvicorn app.api.main:app --reload          # 1) API 서버
streamlit run ui/streamlit_app.py          # 2) UI (http://127.0.0.1:8501)
```

- 검수 목록 대시보드에서 심의 건과 결재 상태를 확인
- 새로 검수하기로 콘텐츠를 제출하면 AI 1차 심의 실행
- 검수 상세에서 심의 결과를 검토하고 승인·조건부승인·반려를 기록

## 개발 로드맵
- [x] 0. 프로젝트 스캐폴딩 + 규제 데이터 확보
- [x] 1. RAG 적재 파이프라인 (jsonl → FAISS)
- [x] 2. 검색 검증 (실 API 키로 동작 확인)
- [x] 3. LangGraph 심의 파이프라인 (룰엔진 → RAG → LLM판단 → 대안문구)
- [x] 4. FastAPI 엔드포인트
- [x] 5. Streamlit 검수 UI
- [ ] 6. AWS 배포
