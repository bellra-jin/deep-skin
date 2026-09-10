"""등급 -> severity 매핑이 실제 라벨 범위를 전부 덮는지 검증한다.

왜 이 테스트인가
----------------
매핑에 없는 등급은 grade_to_severity() 가 None 을 돌려주고, 파서가 그 행을
통째로 버린다("unknown annotation grade skipped"). 예외도 로그도 사용자에게는
보이지 않고, 리포트에서 항목 하나가 조용히 사라진다.

실제로 그렇게 새고 있었다. 문서의 상한이 다섯 항목에서 틀렸고, 정정 전 기준
원본 라벨 137,995건 중 4,511건(3.27%)이 매핑 범위 밖이었다.
가장 심한 glabellus_wrinkle 은 23.94% - 미간 주름이 네 번 중 한 번꼴로
리포트에서 사라지고 있었다.

이 버그는 라벨이 늘어날 때마다 같은 형태로 재발한다. 그래서 값이 아니라
"덮고 있는가"를 검사한다. 새 라벨을 추가하면 OBSERVED_MAX_GRADE 에 실측
상한을 적고, 매핑이 0..max 를 덮지 않으면 여기서 걸린다.
"""

import pytest

from app.services.multivalue_parser import (
    OBSERVED_MAX_GRADE,
    SEVERITY_VOCAB,
    _SEVERITY_BY_ANNOTATION,
    grade_to_severity,
)

ALL_KEYS = sorted(_SEVERITY_BY_ANNOTATION)


# --------------------------------------------------------------------------- #
# 덮는가
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ALL_KEYS)
def test_every_observed_grade_maps_to_a_severity(key):
    """0 부터 실측 최대 등급까지 하나도 None 이 나오면 안 된다."""
    assert key in OBSERVED_MAX_GRADE, (
        f"{key} 의 실측 상한이 OBSERVED_MAX_GRADE 에 없습니다. "
        "라벨을 추가했다면 실측값을 함께 적으세요."
    )
    missing = [g for g in range(OBSERVED_MAX_GRADE[key] + 1)
               if grade_to_severity(key, g) is None]
    assert not missing, (
        f"{key}: 등급 {missing} 이 severity 로 매핑되지 않습니다. "
        "이 등급의 결과는 리포트에서 통째로 사라집니다."
    )


def test_observed_max_covers_every_mapped_key():
    """반대 방향 - 매핑에는 있는데 실측 상한이 없는 키가 없어야 한다."""
    assert set(_SEVERITY_BY_ANNOTATION) == set(OBSERVED_MAX_GRADE)


# --------------------------------------------------------------------------- #
# 어휘를 넓히지 않았는가
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ALL_KEYS)
def test_severity_values_stay_in_vocabulary(key):
    """어휘 밖 값은 소비처의 _SEVERITY_ORDER.get(x, 0) 에서
    가장 낮은 등급으로 조용히 강등된다. 최상위 등급이 최하위로 취급된다."""
    outside = sorted(set(_SEVERITY_BY_ANNOTATION[key].values()) - set(SEVERITY_VOCAB))
    assert not outside, f"{key}: 어휘 밖 severity {outside}"


def test_vocabulary_matches_consumers():
    """소비처가 쓰는 순서 테이블과 어휘가 같아야 한다."""
    from app.services.recommendation_service import _SEVERITY_ORDER as rec_order
    from app.services.report_service import _SEVERITY_ORDER as rep_order

    assert set(SEVERITY_VOCAB) == set(rec_order) == set(rep_order)


# --------------------------------------------------------------------------- #
# 규칙을 지키는가 - 0 은 normal, 최고 등급은 severe
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ALL_KEYS)
def test_zero_is_normal(key):
    assert grade_to_severity(key, 0) == "normal"


@pytest.mark.parametrize("key", ALL_KEYS)
def test_max_grade_is_severe(key):
    top = OBSERVED_MAX_GRADE[key]
    assert grade_to_severity(key, top) == "severe", (
        f"{key}: 최고 등급 {top} 이 severe 가 아닙니다. "
        "척도가 길어지면 severe 도 함께 위로 올라가야 합니다."
    )


@pytest.mark.parametrize("key", ALL_KEYS)
def test_mapping_is_monotonic(key):
    """등급이 오를 때 severity 가 내려가면 안 된다."""
    order = {name: i for i, name in enumerate(SEVERITY_VOCAB)}
    seq = [order[grade_to_severity(key, g)]
           for g in range(OBSERVED_MAX_GRADE[key] + 1)]
    assert seq == sorted(seq), f"{key}: severity 가 단조가 아닙니다 - {seq}"


# --------------------------------------------------------------------------- #
# 범위 밖은 여전히 None 이어야 한다 (조용히 뭉개지 않도록)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ALL_KEYS)
def test_out_of_range_still_returns_none(key):
    assert grade_to_severity(key, OBSERVED_MAX_GRADE[key] + 1) is None
    assert grade_to_severity(key, -1) is None


def test_unknown_key_returns_none():
    assert grade_to_severity("not_a_real_annotation", 0) is None
