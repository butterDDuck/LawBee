"""심의 건 저장소 (SQLite)

준법관리자 결재 워크플로우를 위해 심의 건을 영속화
저장소 구현은 분리되어 있어 추후 다른 DB 로 교체 가능
"""
import sqlite3
from datetime import datetime

from app.config import settings
from app.services.graph import run_review
from app.domain.schema import DecisionStatus, ReviewRecord, ReviewResult


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """테이블이 없으면 생성"""
    with _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                media TEXT,
                decision_status TEXT NOT NULL DEFAULT '대기',
                ai_result TEXT NOT NULL,
                comment TEXT NOT NULL DEFAULT '',
                reviewer TEXT,
                created_at TEXT NOT NULL,
                decided_at TEXT
            )
            """
        )
        # 기존 DB 마이그레이션 — 컬럼 추가
        for col_ddl in ["ALTER TABLE reviews ADD COLUMN title TEXT",
                         "ALTER TABLE reviews ADD COLUMN review_mode TEXT NOT NULL DEFAULT '표준'"]:
            try:
                c.execute(col_ddl)
            except sqlite3.OperationalError:
                pass


def _to_record(row: sqlite3.Row) -> ReviewRecord:
    return ReviewRecord(
        id=row["id"],
        content=row["content"],
        title=row["title"],
        media=row["media"],
        review_mode=row["review_mode"] if "review_mode" in row.keys() else "표준",
        decision_status=row["decision_status"],
        ai_result=ReviewResult.model_validate_json(row["ai_result"]),
        comment=row["comment"],
        reviewer=row["reviewer"],
        created_at=row["created_at"],
        decided_at=row["decided_at"],
    )


def create_review(content: str, media: str | None = None, title: str | None = None,
                  review_mode: str = "표준") -> ReviewRecord:
    """콘텐츠 제출 → AI 1차 심의 실행 → 대기 상태로 저장"""
    result = run_review(content, media=media, review_mode=review_mode)
    init_db()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO reviews (content, title, media, review_mode, ai_result, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (content, title, media, review_mode, result.model_dump_json(), _now()),
        )
        review_id = cur.lastrowid
    return get_review(review_id)


def create_with_result(content: str, media: str, result: ReviewResult, title: str | None = None,
                       review_mode: str = "표준") -> ReviewRecord:
    """사전 계산된 심의 결과로 대기 상태 저장 (멀티모달 등 전처리가 필요한 경우)"""
    init_db()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO reviews (content, title, media, review_mode, ai_result, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (content, title, media, review_mode, result.model_dump_json(), _now()),
        )
        review_id = cur.lastrowid
    return get_review(review_id)


_PENDING_RESULT = ReviewResult(status="통과", summary="__처리중__", rule_hits=[], violations=[], citations=[])


def create_pending(content: str, media: str, title: str | None = None,
                   review_mode: str = "표준") -> ReviewRecord:
    """즉시 반환용 레코드 생성 — 분석 완료 전 '처리중' 상태로 저장"""
    init_db()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO reviews (content, title, media, review_mode, ai_result, decision_status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (content, title, media, review_mode, _PENDING_RESULT.model_dump_json(), "처리중", _now()),
        )
        review_id = cur.lastrowid
    return get_review(review_id)


def update_ai_result(review_id: int, result: ReviewResult) -> None:
    """분석 완료 후 ai_result 갱신 및 decision_status를 '대기'로 전환"""
    init_db()
    with _conn() as c:
        c.execute(
            "UPDATE reviews SET ai_result = ?, decision_status = '대기' WHERE id = ?",
            (result.model_dump_json(), review_id),
        )


def list_reviews(status: DecisionStatus | None = None) -> list[ReviewRecord]:
    """심의 건 목록 조회 (상태 필터 가능), 최신순"""
    init_db()
    with _conn() as c:
        if status:
            rows = c.execute(
                "SELECT * FROM reviews WHERE decision_status = ? ORDER BY id DESC", (status,)
            ).fetchall()
        else:
            rows = c.execute("SELECT * FROM reviews ORDER BY id DESC").fetchall()
    return [_to_record(r) for r in rows]


def get_review(review_id: int) -> ReviewRecord | None:
    """심의 건 단건 조회"""
    init_db()
    with _conn() as c:
        row = c.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
    return _to_record(row) if row else None


def delete_review(review_id: int) -> bool:
    """심의 건 삭제, 삭제 성공 여부 반환"""
    init_db()
    with _conn() as c:
        cur = c.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
    return cur.rowcount > 0


def decide(
    review_id: int,
    decision: DecisionStatus,
    comment: str = "",
    reviewer: str = "준법관리자",
) -> ReviewRecord | None:
    """준법관리자 최종 결재 기록"""
    init_db()
    with _conn() as c:
        c.execute(
            "UPDATE reviews SET decision_status = ?, comment = ?, reviewer = ?, decided_at = ? WHERE id = ?",
            (decision, comment, reviewer, _now(), review_id),
        )
    return get_review(review_id)
