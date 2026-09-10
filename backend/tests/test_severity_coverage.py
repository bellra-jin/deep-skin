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

from pathlib import Path

import pytest

from app.services import severity_scale
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


# --------------------------------------------------------------------------- #
# 추론 엔진도 같은 테이블을 쓰는가
# --------------------------------------------------------------------------- #
# 파서만 검사하면 절반이다. 실제로 눈가 주름 상한을 파서에서만 고치고 엔진의
# 자체 테이블을 두는 바람에, 같은 등급이 경로에 따라 다른 severity 로 나갔다.
ENGINE_SRC = Path(__file__).resolve().parents[1] / "scripts" / "inference_engine.py"

# 부위마다 등급 상한이 다르다. 엔진이 라벨 이름을 받아 위임하지 않으면
# 이 셋 중 하나는 반드시 틀린다.
ENGINE_KEYS = ["forehead_pigmentation", "lip_dryness", "l_perocular_wrinkle"]


def test_engine_source_has_no_own_severity_table():
    """엔진이 매핑을 다시 들고 있으면 언젠가 또 갈라진다."""
    src = ENGINE_SRC.read_text(encoding="utf-8")
    assert "severity_map" not in src, (
        "inference_engine 에 자체 severity 테이블이 있습니다. "
        "severity_scale 에 위임하세요.")
    assert "grade_to_severity" in src, (
        "inference_engine 이 severity_scale.grade_to_severity 를 쓰지 않습니다.")


def _engine_module():
    return pytest.importorskip(
        "scripts.inference_engine",
        reason="torch/torchvision 이 없는 환경에서는 건너뜁니다")


def test_engine_delegates_to_the_same_function():
    """값을 복사한 것이 아니라 같은 함수를 부르는지 확인한다."""
    mod = _engine_module()
    assert mod.grade_to_severity is severity_scale.grade_to_severity


@pytest.mark.parametrize("key", ENGINE_KEYS)
def test_engine_covers_observed_range(key):
    """엔진 경로로도 0..실측상한 전 등급이 severity 를 받는가."""
    mod = _engine_module()
    top = OBSERVED_MAX_GRADE[key]
    missing = [g for g in range(top + 1) if mod.grade_to_severity(key, g) is None]
    assert not missing, f"{key}: 엔진 경로에서 등급 {missing} 이 매핑되지 않습니다."
    assert mod.grade_to_severity(key, 0) == "normal"
    assert mod.grade_to_severity(key, top) == "severe"


@pytest.mark.parametrize("key", ENGINE_KEYS)
def test_engine_matches_parser_exactly(key):
    """엔진과 파서가 같은 등급에 같은 severity 를 내야 한다."""
    mod = _engine_module()
    for g in range(OBSERVED_MAX_GRADE[key] + 1):
        assert mod.grade_to_severity(key, g) == grade_to_severity(key, g)


@pytest.mark.parametrize("key,beyond", [("forehead_pigmentation", 6),
                                        ("lip_dryness", 5),
                                        ("l_perocular_wrinkle", 7)])
def test_engine_reports_unmapped_grade_as_none(key, beyond):
    """상한 밖은 None 이어야 한다. "unknown" 같은 값을 지어내면
    소비처에서 최저 등급으로 강등돼 조용히 틀린다."""
    mod = _engine_module()
    assert mod.grade_to_severity(key, beyond) is None


def test_engine_requires_annotation_key():
    """라벨 이름 없이 등급만 받으면 위임이 성립하지 않는다.
    조용히 넘어가지 말고 실패해야 한다."""
    mod = _engine_module()
    eng = mod.DinoInferenceEngine(backbone_ckpt="x", heads_dir="y")
    assert eng.annotation_key is None
    eng.backbone, eng.heads = object(), [object()]
    with pytest.raises(ValueError, match="annotation_key"):
        eng.predict(image=None)


def test_num_classes_for_matches_observed_max():
    for key, top in OBSERVED_MAX_GRADE.items():
        assert severity_scale.num_classes_for(key) == top + 1
    assert severity_scale.num_classes_for("nope") is None
