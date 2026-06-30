"""CRAG on/off 근거 조항 인용 정확도 비교

심의 콘텐츠별 기대 조항(expected)을 두고, CRAG 채점·재작성 루프를
끈 상태(off)와 켠 상태(on)로 run_review 를 실행해 인용 정확도를 비교함.

  python -m scripts.bench_crag
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from app.services.graph import _compiled, run_review, set_crag_enabled


# (콘텐츠, 매체, 기대 근거조항 id) — 기대 id 는 lexicon.CATEGORY_CHUNKS 의 확정 매핑 기준
TEST_CASES = [
    ("원금 100% 보장, 확정 수익 연 8% 펀드에 지금 가입하세요",
     "텍스트", {"FCPA-21-단정"}),
    ("누구나 무조건 받는 연 5.0% 적금, 가입만 하면 끝!",
     "텍스트", {"FCPA-감독규정19"}),
    ("타사 대비 업계 최저 수수료, 어디와 비교해도 가장 저렴합니다",
     "텍스트", {"FAD-3"}),
    ("마감 임박! 오늘 안 하면 평생 후회하는 한정 특가 상품",
     "텍스트", {"FTC-지정고시", "FAD-3"}),
    ("연 4.9% 신용대출, 당일 승인 누구나 가능합니다",
     "텍스트", {"FAG-17-대출필수표기", "FCPA-시행령18"}),
    ("[광고] 신상품 안내드립니다. 자세한 내용은 앱에서 확인하세요",
     "텍스트", {"ICTNA-50"}),
]


def _cited_ids(content: str, media: str) -> tuple[str, set[str], str]:
    """run_review 실행 → (status, 인용 id 집합, 대안 유무)"""
    result = run_review(content, media=media)
    cited = {c.id for c in result.citations}
    return result.status, cited, ("있음" if result.alternative_text else "없음")


def _score(cited: set[str], expected: set[str]) -> tuple[int, float]:
    """recall(기대 조항 1개라도 인용 1/0), precision(인용 중 기대 비율)"""
    recall = 1 if cited & expected else 0
    precision = len(cited & expected) / len(cited) if cited else 0.0
    return recall, precision


def _run(label: str, enabled: bool) -> tuple[float, float]:
    set_crag_enabled(enabled)
    print(f"\n===== CRAG {label} =====")
    header = f"{'콘텐츠':<34} {'기대':<22} {'판정':<6} {'인용':<26} {'R':>2} {'P':>5}"
    print(header)
    print("-" * len(header))

    rec_sum = prec_sum = 0.0
    for content, media, expected in TEST_CASES:
        status, cited, _ = _cited_ids(content, media)
        recall, precision = _score(cited, expected)
        rec_sum += recall
        prec_sum += precision
        exp_s = ",".join(sorted(expected))
        cit_s = ",".join(sorted(cited)) or "-"
        print(f"{content[:32]:<34} {exp_s[:20]:<22} {status:<6} {cit_s[:24]:<26} {recall:>2} {precision:>5.2f}")

    n = len(TEST_CASES)
    print("-" * len(header))
    print(f"{'평균':<34} {'':<22} {'':<6} {'':<26} {rec_sum/n:>4.2f} {prec_sum/n:>5.2f}")
    return rec_sum / n, prec_sum / n


# 룰이 발화하지 않는 해석형 콘텐츠 — 인용이 순수 검색→judge 로 정해져 CRAG 효과가 드러나는 구간
INTERPRETIVE_CASES = [
    "이 펀드는 전문가들이 검증한 전략으로 노후를 든든하게 준비해 드립니다",
    "복잡한 조건 없이, 알아서 척척 굴려 드리는 스마트 자산관리 서비스",
    "지금 시작하면 미래의 당신이 분명 고마워할 현명한 선택입니다",
]


def _telemetry() -> None:
    """CRAG-on 으로 그래프를 직접 호출해 채점·재작성 동작을 계측"""
    set_crag_enabled(True)
    print("\n===== CRAG 메커니즘 텔레메트리 (해석형 콘텐츠) =====")
    header = f"{'콘텐츠':<34} {'판정':<6} {'걸러낸 후보':>10} {'재작성':>6} {'신뢰도':>6} {'인용':<20}"
    print(header)
    print("-" * len(header))
    for content in INTERPRETIVE_CASES:
        final = _compiled().invoke({
            "content": content, "media": "텍스트", "review_mode": "표준",
            "rewrite_count": 0, "alt_retries": 0, "alt_feedback": [], "auto_fix_failed": False,
        })
        j = final["judgment"]
        cited = sorted({cid for v in j.violations for cid in v.citation_ids})
        print(
            f"{content[:32]:<34} {j.status:<6} {final.get('graded_out', 0):>10} "
            f"{final.get('rewrite_count', 0):>6} {final.get('retrieval_confidence', '-'):>6} "
            f"{(','.join(cited) or '-')[:20]:<20}"
        )


def main() -> None:
    off_r, off_p = _run("OFF (기존 검색)", False)
    on_r, on_p = _run("ON (CRAG)", True)

    print("\n===== 종합 비교 (룰 발화 케이스) =====")
    print(f"{'':<10} {'Recall':>8} {'Precision':>10}")
    print(f"{'OFF':<10} {off_r:>8.2f} {off_p:>10.2f}")
    print(f"{'ON':<10} {on_r:>8.2f} {on_p:>10.2f}")
    print(f"{'Δ':<10} {on_r - off_r:>+8.2f} {on_p - off_p:>+10.2f}")
    print("\n* 룰 발화 케이스는 _fix_citations 결정론적 매핑이 인용을 지배하므로 on/off 동일이 정상")

    _telemetry()


if __name__ == "__main__":
    main()
