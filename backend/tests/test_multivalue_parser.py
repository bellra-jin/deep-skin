import copy
import json
from pathlib import Path

from app.services.multivalue_parser import (
    _is_yolo_miss_zero,
    parse_annotations,
    parse_detections,
    parse_equipment,
    parse_multivalue_response,
)

_EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "scripts/face_multivalue_inf/example_response.json"


def _payload():
    with _EXAMPLE_PATH.open() as f:
        return json.load(f)


def _part(payload, facepart):
    return next(part for part in payload["parts"] if part["images"]["facepart"] == facepart)


def _by_key(metrics, key):
    return next(metric for metric in metrics if metric.metric_key == key)


def test_parse_annotations_forehead():
    payload = _payload()

    results = parse_annotations([_part(payload, 1)], session_id=10, user_id=20, image_id=30)

    assert len(results) == 2
    assert {result.metric_name for result in results} == {"pigmentation", "wrinkle"}

    pigmentation = next(result for result in results if result.metric_name == "pigmentation")
    assert pigmentation.raw_part_name == "forehead"
    assert pigmentation.display_part_name == "이마"
    assert pigmentation.grade_value == 2
    # forehead_pigmentation 은 0~3 이 아니라 0~5 다(라벨 전수 집계).
    # 척도가 길어지면서 등급 2 가 moderate 에서 mild 로 내려갔다.
    # 상한을 3 으로 잡고 있을 때는 등급 4·5 가 매핑에 없어 리포트에서 사라지고 있었다.
    assert pigmentation.severity == "mild"
    assert pigmentation.confidence_score is None
    assert pigmentation.predicted_value is None
    assert pigmentation.measured_value is None

    wrinkle = next(result for result in results if result.metric_name == "wrinkle")
    assert wrinkle.grade_value == 3
    assert wrinkle.severity == "moderate"


def test_parse_annotations_skip_null_acne():
    payload = _payload()

    results = parse_annotations([_part(payload, 0)], session_id=10, user_id=20)

    assert results == []


def test_parse_equipment_forehead_metric_group():
    payload = _payload()

    metrics = parse_equipment([_part(payload, 1)], session_id=10, user_id=20)

    moisture = _by_key(metrics, "forehead_moisture")
    assert moisture.metric_group == "moisture"
    assert moisture.metric_name == "moisture"
    assert moisture.value_type == "reg"
    assert moisture.is_dummy is False
    assert moisture.source == "model"

    elasticity_r2 = _by_key(metrics, "forehead_elasticity_R2")
    assert elasticity_r2.metric_group == "elasticity"
    assert elasticity_r2.metric_name == "R2"
    assert elasticity_r2.value_type == "reg"


def test_parse_equipment_eye_wrinkle_keys():
    payload = _payload()

    metrics = parse_equipment([_part(payload, 3)], session_id=10, user_id=20)

    ra = _by_key(metrics, "l_perocular_wrinkle_Ra")
    assert ra.metric_group == "wrinkle"
    assert ra.metric_name == "Ra"
    assert ra.value_type == "reg"

    rz = _by_key(metrics, "l_perocular_wrinkle_Rz=Rtm")
    assert rz.metric_group == "wrinkle"
    assert rz.metric_name == "Rz"
    assert rz.value_type == "reg"


def test_parse_equipment_chin_moisture_dummy():
    payload = _payload()

    metrics = parse_equipment([_part(payload, 8)], session_id=10, user_id=20)

    chin_moisture = _by_key(metrics, "chin_moisture")
    assert chin_moisture.metric_group == "moisture"
    assert chin_moisture.metric_name == "moisture"
    assert chin_moisture.value == 0.0
    assert chin_moisture.is_dummy is True
    assert chin_moisture.dummy_reason == "label_not_trained"
    assert chin_moisture.source == "dummy_fallback"


def test_parse_equipment_pore_count():
    payload = _payload()

    metrics = parse_equipment([_part(payload, 5)], session_id=10, user_id=20)

    pore = _by_key(metrics, "l_cheek_pore")
    assert pore.metric_group == "pore"
    assert pore.metric_name == "pore_count"
    assert pore.value_type == "count"


def test_parse_detections_yolo():
    payload = _payload()

    detections = parse_detections(
        [_part(payload, 3)], payload["detected_parts"], session_id=10, user_id=20
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.raw_part_name == "left_eye"
    assert detection.class_name == "l_eye"
    assert detection.bbox_source == "yolo"
    assert detection.detection_confidence is not None
    left_eye_det = next(d for d in payload["detected_parts"] if d["raw_part_name"] == "left_eye")
    assert detection.bbox_x1 == left_eye_det["bbox_xyxy"][0]


def test_parse_detections_fallback():
    payload = _payload()

    detections = parse_detections(
        [_part(payload, 1)], detected_parts=[], session_id=10, user_id=20
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.raw_part_name == "forehead"
    assert detection.class_name is None
    assert detection.bbox_source == "full_image_fallback"
    assert detection.detection_confidence is None
    assert detection.bbox_x1 == _part(payload, 1)["images"]["bbox"][0]


# ── YOLO miss dummy 처리 테스트 ───────────────────────────────────────────────

def _make_equipment_part(facepart: int, equipment: dict) -> dict:
    """equipment 테스트용 최소 part 딕셔너리."""
    return {
        "images": {
            "facepart": facepart,
            "width": 100,
            "height": 100,
            "device": 0,
            "angle": 0,
            "bbox": [0, 0, 100, 100],
        },
        "info": {},
        "annotations": {},
        "equipment": equipment,
    }


def test_yolo_miss_moisture_zero_is_dummy():
    """left_cheek 미검출 + moisture=0.0 → is_dummy=True, dummy_reason=yolo_miss"""
    parts = [_make_equipment_part(5, {"l_cheek_moisture": 0.0})]
    metrics = parse_equipment(parts, session_id=10, user_id=20, detected_part_names=set())

    moisture = next(m for m in metrics if m.metric_key == "l_cheek_moisture")
    assert moisture.is_dummy is True
    assert moisture.dummy_reason == "yolo_miss_or_model_fallback"
    assert moisture.source == "dummy_fallback"


def test_yolo_miss_elasticity_R2_zero_is_dummy():
    """left_cheek 미검출 + elasticity_R2=0.0 → is_dummy=True"""
    parts = [_make_equipment_part(5, {"l_cheek_elasticity_R2": 0.0})]
    metrics = parse_equipment(parts, session_id=10, user_id=20, detected_part_names=set())

    r2 = next(m for m in metrics if m.metric_key == "l_cheek_elasticity_R2")
    assert r2.is_dummy is True
    assert r2.dummy_reason == "yolo_miss_or_model_fallback"


def test_yolo_detected_moisture_zero_not_dummy():
    """YOLO로 검출된 부위의 0.0은 실제 값 — dummy 처리 안 함"""
    parts = [_make_equipment_part(5, {"l_cheek_moisture": 0.0})]
    metrics = parse_equipment(
        parts, session_id=10, user_id=20, detected_part_names={"left_cheek"}
    )

    moisture = next(m for m in metrics if m.metric_key == "l_cheek_moisture")
    assert moisture.is_dummy is False
    assert moisture.dummy_reason is None
    assert moisture.source == "model"


def test_chin_moisture_label_not_trained_preserved():
    """chin_moisture는 YOLO 검출 여부와 무관하게 label_not_trained dummy 유지"""
    payload = _payload()

    # chin이 YOLO 검출된 상황에서도 chin_moisture는 label_not_trained
    metrics = parse_equipment(
        [_part(payload, 8)], session_id=10, user_id=20, detected_part_names={"chin"}
    )
    chin_moisture = next(m for m in metrics if m.metric_key == "chin_moisture")
    assert chin_moisture.is_dummy is True
    assert chin_moisture.dummy_reason == "label_not_trained"


def test_acne_count_zero_not_dummy():
    """acne_count=0은 full_face — YOLO miss 대상 아님, is_dummy=False"""
    parts = [_make_equipment_part(0, {"acne_count": 0})]
    metrics = parse_equipment(parts, session_id=10, user_id=20, detected_part_names=set())

    acne = next(m for m in metrics if m.metric_key == "acne_count")
    assert acne.is_dummy is False
    assert acne.dummy_reason is None


def test_pigmentation_count_zero_not_dummy():
    """pigmentation_count=0은 full_face — YOLO miss 대상 아님, is_dummy=False"""
    parts = [_make_equipment_part(0, {"pigmentation_count": 0})]
    metrics = parse_equipment(parts, session_id=10, user_id=20, detected_part_names=set())

    pig = next(m for m in metrics if m.metric_key == "pigmentation_count")
    assert pig.is_dummy is False
    assert pig.dummy_reason is None


def test_yolo_miss_nonzero_value_not_dummy():
    """YOLO 미검출이어도 값이 0.0이 아니면 dummy 처리 안 함"""
    parts = [_make_equipment_part(5, {"l_cheek_moisture": 42.5})]
    metrics = parse_equipment(parts, session_id=10, user_id=20, detected_part_names=set())

    moisture = next(m for m in metrics if m.metric_key == "l_cheek_moisture")
    assert moisture.is_dummy is False
    assert moisture.source == "model"


def test_detected_part_names_none_no_yolo_filtering():
    """detected_part_names=None이면 YOLO miss 판별 수행 안 함 (하위 호환)"""
    parts = [_make_equipment_part(5, {"l_cheek_moisture": 0.0})]
    metrics = parse_equipment(parts, session_id=10, user_id=20)  # detected_part_names 미전달

    moisture = next(m for m in metrics if m.metric_key == "l_cheek_moisture")
    assert moisture.is_dummy is False


def test_parse_multivalue_response_passes_detected_to_equipment():
    """parse_multivalue_response가 detected_parts 기반으로 equipment dummy 처리를 수행한다."""
    payload = _payload()
    # left_cheek(facepart=5)를 detected_parts에서 제거
    mutated = copy.deepcopy(payload)
    mutated["detected_parts"] = [
        d for d in mutated["detected_parts"] if d["raw_part_name"] != "left_cheek"
    ]
    # left_cheek의 moisture를 0.0으로 설정
    for part in mutated["parts"]:
        if part["images"]["facepart"] == 5:
            part["equipment"]["l_cheek_moisture"] = 0.0

    parsed = parse_multivalue_response(mutated, session_id=10, user_id=20)
    moisture_metrics = [
        m for m in parsed["metric_values"] if m.metric_key == "l_cheek_moisture"
    ]
    assert len(moisture_metrics) == 1
    assert moisture_metrics[0].is_dummy is True
    assert moisture_metrics[0].dummy_reason == "yolo_miss_or_model_fallback"


# ── _is_yolo_miss_zero 헬퍼 단위 테스트 ──────────────────────────────────────

def test_is_yolo_miss_zero_true():
    assert _is_yolo_miss_zero("left_cheek", 0.0, set(), False) is True


def test_is_yolo_miss_zero_full_face_excluded():
    assert _is_yolo_miss_zero("full_face", 0.0, set(), False) is False


def test_is_yolo_miss_zero_label_dummy_excluded():
    assert _is_yolo_miss_zero("chin", 0.0, set(), True) is False


def test_is_yolo_miss_zero_detected_excluded():
    assert _is_yolo_miss_zero("left_cheek", 0.0, {"left_cheek"}, False) is False


def test_is_yolo_miss_zero_nonzero_excluded():
    assert _is_yolo_miss_zero("left_cheek", 1.5, set(), False) is False


def test_is_yolo_miss_zero_none_skips():
    assert _is_yolo_miss_zero("left_cheek", 0.0, None, False) is False


def test_unknown_key_does_not_crash():
    payload = _payload()
    mutated = copy.deepcopy(payload)
    forehead = _part(mutated, 1)
    forehead["annotations"]["unknown_annotation"] = 3
    forehead["equipment"]["unknown_equipment"] = 1.23

    parsed = parse_multivalue_response(mutated, session_id=10, user_id=20, image_id=30)

    assert parsed["raw_response"].raw_json is mutated
    assert len(parsed["part_results"]) > 0
    assert all(result.metric_name != "unknown_annotation" for result in parsed["part_results"])
    assert all(metric.metric_key != "unknown_equipment" for metric in parsed["metric_values"])
