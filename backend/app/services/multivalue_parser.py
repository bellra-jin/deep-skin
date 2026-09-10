import logging
from collections.abc import Iterable
from typing import Any

from app.models.ai_raw_response import AiRawResponse
from app.models.skin_metric_value import SkinMetricValue
from app.models.skin_part_detection import SkinPartDetection
from app.models.skin_part_result import SkinPartResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "skin_dinov3_multivalue"
_UNKNOWN_MODEL_VERSION = "unknown"

_PART_BY_FACEPART = {
    0: ("full_face", "전체 얼굴"),
    1: ("forehead", "이마"),
    2: ("glabella", "미간"),
    3: ("left_eye", "눈가"),
    4: ("right_eye", "눈가"),
    5: ("left_cheek", "볼"),
    6: ("right_cheek", "볼"),
    7: ("lips", "입술"),
    8: ("chin", "턱"),
}

_ANNOTATION_SPECS = {
    "forehead_pigmentation": ("forehead", "pigmentation", "색소침착", "pigmentation"),
    "forehead_wrinkle": ("forehead", "wrinkle", "주름", "wrinkle"),
    "glabellus_wrinkle": ("glabella", "wrinkle", "주름", "wrinkle"),
    "l_perocular_wrinkle": ("left_eye", "wrinkle", "주름", "wrinkle"),
    "r_perocular_wrinkle": ("right_eye", "wrinkle", "주름", "wrinkle"),
    "l_cheek_pore": ("left_cheek", "pore", "모공", "pore"),
    "l_cheek_pigmentation": ("left_cheek", "pigmentation", "색소침착", "pigmentation"),
    "r_cheek_pore": ("right_cheek", "pore", "모공", "pore"),
    "r_cheek_pigmentation": ("right_cheek", "pigmentation", "색소침착", "pigmentation"),
    "lip_dryness": ("lips", "dryness", "건조", "dryness"),
    "chin_sagging": ("chin", "sagging", "처짐", "sagging"),
}

_ZERO_12_34_5 = {
    0: "normal",
    1: "mild",
    2: "mild",
    3: "moderate",
    4: "moderate",
    5: "severe",
}

# 7등급(0~6) 라벨용. 눈가 주름·미간 주름·이마 주름·턱 처짐이 여기에 해당한다.
# 매핑 범위를 벗어난 등급은 grade_to_severity 가 None 을 돌려주고 해당 부위
# 결과가 통째로 버려진다("unknown annotation grade skipped").
# 기존 테이블의 규칙(0 은 normal, 최고 등급은 severe, 중간을 나눔)을 따른다.
_ZERO_12_345_6 = {
    0: "normal",
    1: "mild",
    2: "mild",
    3: "moderate",
    4: "moderate",
    5: "moderate",
    6: "severe",
}

# 라벨별 실제 등급 범위는 원본 라벨 JSON 112,905건 전수 집계로 확인한 값이다.
# 문서(docs/labeling_codes_guide.md)의 상한이 네 항목에서 틀려 있었고, 그만큼이
# 매핑 범위를 벗어나 리포트에서 사라지고 있었다.
#   glabellus_wrinkle     0~2 로 알고 있었으나 0~6  -> 23.94% (3,003건) 유실
#   forehead_wrinkle      0~4 로 알고 있었으나 0~6  -> 10.78% (1,352건) 유실
#   forehead_pigmentation 0~3 로 알고 있었으나 0~5  ->  1.14% (143건) 유실
#   chin_sagging          0~5 로 알고 있었으나 0~6  ->  0.10% (13건) 유실
# 등급 수가 같은 라벨은 같은 테이블을 재사용한다. 새 어휘를 만들지 않는다.
_SEVERITY_BY_ANNOTATION = {
    "forehead_pigmentation": _ZERO_12_34_5,      # 0~5
    "forehead_wrinkle": _ZERO_12_345_6,          # 0~6
    "glabellus_wrinkle": _ZERO_12_345_6,         # 0~6
    "l_perocular_wrinkle": _ZERO_12_345_6,       # 0~6
    "r_perocular_wrinkle": _ZERO_12_345_6,       # 0~6
    "l_cheek_pore": _ZERO_12_34_5,               # 0~5
    "l_cheek_pigmentation": _ZERO_12_34_5,       # 0~5
    "r_cheek_pore": _ZERO_12_34_5,               # 0~5
    "r_cheek_pigmentation": _ZERO_12_34_5,       # 0~5
    "lip_dryness": {0: "normal", 1: "mild", 2: "mild", 3: "moderate", 4: "severe"},
    "chin_sagging": _ZERO_12_345_6,              # 0~6
}

# 라벨별 실제 최대 등급 (원본 전수 집계). 테스트가 이 값을 기준으로
# 매핑 누락을 잡는다. 라벨이 늘어나면 여기부터 갱신한다.
OBSERVED_MAX_GRADE = {
    "forehead_pigmentation": 5,
    "forehead_wrinkle": 6,
    "glabellus_wrinkle": 6,
    "l_perocular_wrinkle": 6,
    "r_perocular_wrinkle": 6,
    "l_cheek_pore": 5,
    "l_cheek_pigmentation": 5,
    "r_cheek_pore": 5,
    "r_cheek_pigmentation": 5,
    "lip_dryness": 4,
    "chin_sagging": 6,
}

# severity 어휘. 소비처가 _SEVERITY_ORDER.get(x, 0) 패턴이라
# 여기 없는 값을 내보내면 가장 낮은 등급으로 조용히 강등된다.
SEVERITY_VOCAB = ("normal", "mild", "moderate", "severe")

_ELASTICITY_SUFFIXES = {
    *(f"R{i}" for i in range(10)),
    *(f"Q{i}" for i in range(4)),
}
_WRINKLE_SUFFIXES = {
    "Ra",
    "Rmax",
    "Rt",
    "Rz=Rtm",
    "Rp",
    "Rv",
    "Rq",
    "R3z",
}


def parse_multivalue_response(
    payload: dict[str, Any],
    session_id: int,
    user_id: int,
    image_id: int | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("multivalue payload must be a dict")

    parts = payload.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("multivalue payload must include a non-empty parts list")

    detected_parts = payload.get("detected_parts")
    if not isinstance(detected_parts, list):
        logger.warning("detected_parts is missing or not a list; detections will use fallbacks")
        detected_parts = []

    model_name = payload.get("model_name") or _DEFAULT_MODEL_NAME
    model_version = payload.get("model_version") or _UNKNOWN_MODEL_VERSION

    return {
        "raw_response": AiRawResponse(
            session_id=session_id,
            user_id=user_id,
            image_id=image_id,
            model_name=model_name,
            model_version=model_version,
            server_type="multivalue",
            raw_json=payload,
        ),
        "part_results": _parse_annotations(
            parts, session_id, user_id, image_id, model_name, model_version
        ),
        "metric_values": parse_equipment(
            parts, session_id, user_id, image_id,
            detected_part_names={
                d["raw_part_name"]
                for d in detected_parts
                if isinstance(d, dict) and isinstance(d.get("raw_part_name"), str)
            },
        ),
        "detections": parse_detections(parts, detected_parts, session_id, user_id, image_id),
    }


def parse_annotations(
    parts: list[dict[str, Any]],
    session_id: int,
    user_id: int,
    image_id: int | None = None,
) -> list[SkinPartResult]:
    return _parse_annotations(
        parts,
        session_id,
        user_id,
        image_id,
        _DEFAULT_MODEL_NAME,
        _UNKNOWN_MODEL_VERSION,
    )


def parse_equipment(
    parts: list[dict[str, Any]],
    session_id: int,
    user_id: int,
    image_id: int | None = None,
    detected_part_names: set[str] | None = None,
) -> list[SkinMetricValue]:
    """Parse equipment metrics from parts list.

    detected_part_names: YOLO로 검출된 raw_part_name 집합.
      None이면 YOLO miss 판별을 수행하지 않는다 (하위 호환).
      제공되면 미검출 부위의 0.0값을 is_dummy=True로 저장한다.
    """
    results: list[SkinMetricValue] = []
    for part in _iter_part_dicts(parts):
        raw_part_name, display_part_name, facepart = _part_identity(part)
        equipment = part.get("equipment")
        if equipment is None:
            continue
        if not isinstance(equipment, dict):
            logger.warning("equipment is not a dict for part=%s; skipped", raw_part_name)
            continue

        for metric_key, value in equipment.items():
            if value is None:
                continue
            if not _is_number(value):
                logger.warning("equipment value is not numeric: key=%s value=%r", metric_key, value)
                continue

            parsed_metric = _parse_metric_key(metric_key)
            if parsed_metric is None:
                logger.warning("unknown equipment key skipped: %s", metric_key)
                continue
            metric_group, metric_name, value_type = parsed_metric

            is_label_dummy = metric_key == "chin_moisture"
            is_yolo_miss_dummy = _is_yolo_miss_zero(
                raw_part_name, float(value), detected_part_names, is_label_dummy
            )
            is_dummy = is_label_dummy or is_yolo_miss_dummy

            if is_label_dummy:
                dummy_reason: str | None = "label_not_trained"
            elif is_yolo_miss_dummy:
                dummy_reason = "yolo_miss_or_model_fallback"
            else:
                dummy_reason = None

            results.append(
                SkinMetricValue(
                    session_id=session_id,
                    user_id=user_id,
                    image_id=image_id,
                    raw_part_name=raw_part_name,
                    display_part_name=display_part_name,
                    facepart=facepart,
                    metric_group=metric_group,
                    metric_name=metric_name,
                    metric_key=metric_key,
                    value=float(value),
                    value_type=value_type,
                    unit=None,
                    is_dummy=is_dummy,
                    dummy_reason=dummy_reason,
                    source="dummy_fallback" if is_dummy else "model",
                )
            )
    return results


def parse_detections(
    parts: list[dict[str, Any]],
    detected_parts: list[dict[str, Any]] | None,
    session_id: int,
    user_id: int,
    image_id: int | None = None,
) -> list[SkinPartDetection]:
    detections_by_part = _index_detected_parts(detected_parts)
    results: list[SkinPartDetection] = []

    for part in _iter_part_dicts(parts):
        raw_part_name, _display_part_name, facepart = _part_identity(part)
        if facepart == 0:
            continue

        detected = detections_by_part.get(raw_part_name)
        if detected:
            bbox = _valid_bbox(detected.get("bbox_xyxy"))
            if bbox is None:
                logger.warning("invalid detected_parts bbox skipped: part=%s", raw_part_name)
                detected = None

        if detected:
            confidence = detected.get("confidence")
            results.append(
                _make_detection(
                    session_id=session_id,
                    user_id=user_id,
                    image_id=image_id,
                    raw_part_name=raw_part_name,
                    class_name=detected.get("class_name"),
                    facepart=facepart,
                    bbox=bbox,
                    bbox_source="yolo",
                    detection_confidence=float(confidence) if _is_number(confidence) else None,
                )
            )
            continue

        images = part.get("images")
        if not isinstance(images, dict):
            logger.warning("images is missing or invalid for part=%s; detection skipped", raw_part_name)
            continue
        bbox = _valid_bbox(images.get("bbox"))
        if bbox is None:
            logger.warning("invalid images.bbox skipped: part=%s", raw_part_name)
            continue

        results.append(
            _make_detection(
                session_id=session_id,
                user_id=user_id,
                image_id=image_id,
                raw_part_name=raw_part_name,
                class_name=None,
                facepart=facepart,
                bbox=bbox,
                bbox_source="full_image_fallback",
                detection_confidence=None,
            )
        )

    return results


def _parse_annotations(
    parts: list[dict[str, Any]],
    session_id: int,
    user_id: int,
    image_id: int | None,
    model_name: str,
    model_version: str,
) -> list[SkinPartResult]:
    results: list[SkinPartResult] = []
    for part in _iter_part_dicts(parts):
        part_raw_name, display_part_name, _facepart = _part_identity(part)
        annotations = part.get("annotations")
        if annotations is None:
            continue
        if not isinstance(annotations, dict):
            logger.warning("annotations is not a dict for part=%s; skipped", part_raw_name)
            continue

        for key, grade in annotations.items():
            if grade is None:
                continue
            spec = _ANNOTATION_SPECS.get(key)
            if spec is None:
                logger.warning("unknown annotation key skipped: %s", key)
                continue
            raw_part_name, metric_name, metric_display_name, issue_type = spec
            if not _is_number(grade):
                logger.warning("annotation grade is not numeric: key=%s grade=%r", key, grade)
                continue

            grade_value = int(grade)
            severity = grade_to_severity(key, grade_value)
            if severity is None:
                logger.warning("unknown annotation grade skipped: key=%s grade=%s", key, grade)
                continue

            results.append(
                SkinPartResult(
                    session_id=session_id,
                    user_id=user_id,
                    image_id=image_id,
                    raw_part_name=raw_part_name,
                    display_part_name=display_part_name,
                    metric_name=metric_name,
                    metric_display_name=metric_display_name,
                    issue_type=issue_type,
                    grade_value=grade_value,
                    predicted_value=None,
                    measured_value=None,
                    severity=severity,
                    confidence_score=None,
                    model_name=model_name,
                    model_version=model_version,
                )
            )
    return results


def grade_to_severity(annotation_key: str, grade: int) -> str | None:
    return _SEVERITY_BY_ANNOTATION.get(annotation_key, {}).get(grade)


def _parse_metric_key(metric_key: str) -> tuple[str, str, str] | None:
    if metric_key in {"pigmentation_count", "acne_count"}:
        group = metric_key.removesuffix("_count")
        return group, metric_key, "count"

    if metric_key.endswith("_moisture"):
        return "moisture", "moisture", "reg"

    if "_elasticity_" in metric_key:
        suffix = metric_key.rsplit("_elasticity_", 1)[1]
        if suffix in _ELASTICITY_SUFFIXES:
            return "elasticity", suffix, "reg"
        return None

    if metric_key in {"l_cheek_pore", "r_cheek_pore"}:
        return "pore", "pore_count", "count"

    if "_wrinkle_" in metric_key:
        suffix = metric_key.rsplit("_wrinkle_", 1)[1]
        if suffix in _WRINKLE_SUFFIXES:
            metric_name = "Rz" if suffix == "Rz=Rtm" else suffix
            return "wrinkle", metric_name, "reg"
        return None

    return None


def _iter_part_dicts(parts: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(parts, list):
        logger.warning("parts is missing or not a list; skipped")
        return []
    valid_parts = []
    for part in parts:
        if not isinstance(part, dict):
            logger.warning("part item is not a dict; skipped: %r", part)
            continue
        valid_parts.append(part)
    return valid_parts


def _part_identity(part: dict[str, Any]) -> tuple[str, str, int]:
    images = part.get("images")
    if not isinstance(images, dict):
        logger.warning("part.images is missing or invalid; using unknown part identity")
        return "unknown", "알 수 없음", -1

    facepart = images.get("facepart")
    if not isinstance(facepart, int) or facepart not in _PART_BY_FACEPART:
        logger.warning("unknown facepart=%r; using unknown part identity", facepart)
        return "unknown", "알 수 없음", -1

    raw_part_name, display_part_name = _PART_BY_FACEPART[facepart]
    return raw_part_name, display_part_name, facepart


def _index_detected_parts(detected_parts: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(detected_parts, list):
        logger.warning("detected_parts is missing or not a list; skipped")
        return {}

    indexed = {}
    for item in detected_parts:
        if not isinstance(item, dict):
            logger.warning("detected_parts item is not a dict; skipped: %r", item)
            continue
        raw_part_name = item.get("raw_part_name")
        if not isinstance(raw_part_name, str):
            logger.warning("detected_parts item has no raw_part_name; skipped")
            continue
        indexed[raw_part_name] = item
    return indexed


def _valid_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    if not all(_is_number(item) for item in value):
        return None
    return tuple(float(item) for item in value)


def _make_detection(
    session_id: int,
    user_id: int,
    image_id: int | None,
    raw_part_name: str,
    class_name: str | None,
    facepart: int,
    bbox: tuple[float, float, float, float],
    bbox_source: str,
    detection_confidence: float | None,
) -> SkinPartDetection:
    x1, y1, x2, y2 = bbox
    return SkinPartDetection(
        session_id=session_id,
        user_id=user_id,
        image_id=image_id,
        raw_part_name=raw_part_name,
        class_name=class_name,
        facepart=facepart,
        bbox_x1=x1,
        bbox_y1=y1,
        bbox_x2=x2,
        bbox_y2=y2,
        bbox_source=bbox_source,
        detection_confidence=detection_confidence,
    )


def _is_yolo_miss_zero(
    raw_part_name: str,
    value: float,
    detected_part_names: set[str] | None,
    is_label_dummy: bool,
) -> bool:
    """YOLO 미검출 부위의 0.0 fallback 여부를 판단한다.

    조건:
    - detected_part_names가 제공됐을 때만 판별 (None이면 항상 False)
    - full_face는 YOLO 검출 대상이 아니므로 제외
    - label_not_trained dummy는 별도 처리이므로 제외
    - raw_part_name이 검출 목록에 없고 값이 0.0일 때 True
    """
    if is_label_dummy or detected_part_names is None:
        return False
    if raw_part_name == "full_face":
        return False
    return raw_part_name not in detected_part_names and value == 0.0


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
