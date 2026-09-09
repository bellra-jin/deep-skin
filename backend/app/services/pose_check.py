"""업로드된 사진이 정면인지 판정한다.

왜 필요한가
-----------
고개가 돌아간 사진에서는 가려진 쪽 부위가 검출되지 않고, 그 부위는
얼굴 전체 이미지로 추론된다(bbox_source="full_image_fallback").
값은 나오지만 추정값이다. 사용자가 다시 찍을 기회를 주는 것이
조용히 추정값을 내보내는 것보다 낫다.

판정 방법
---------
YOLO 검출 결과 자체가 각도 판정기다. 미간이 두 볼 중점에서 좌우로
얼마나 벗어났는지를 볼 간격으로 정규화한 값(dx)을 쓴다.
추가 모델이 필요 없고, AI 서버 응답의 detected_parts 만으로 계산된다.

임계값 근거
-----------
AI-Hub 검증셋 270장(각도 0~8 각 30장)으로 실측 보정했다.
정면(각도 0·1·2) 대 측면(3~8) 이분 기준:

  |dx| <= 0.05   정확도 94.8%   오탐 0건   미탐 14건   <- 채택
  |dx| <= 0.06   정확도 95.6%   오탐 5건   미탐  7건

0.06 이 정확도는 높지만 오탐이 생긴다. 두 실패의 비용이 다르다.
  오탐 = 측면 사진을 정면으로 통과시켜 추정값을 실측값처럼 내보낸다
  미탐 = 정면 사진에 재촬영을 한 번 더 권한다
전자가 나쁘므로 오탐 0 인 0.05 를 쓴다.

측정했지만 쓰지 않은 것
-----------------------
  눈가 쌍 존재 여부  단독 정확도 62.2%. L15/R15 에서 두 눈가가 모두
                     검출되므로 강한 회전만 잡는다. dx 와 AND 로 묶어도
                     추가 이득이 없었다.
  세로 항 dy         "고개를 숙이면 세로로 밀린다"는 가설로 추가했으나,
                     아래를 본 사진(각도 2)이 정면보다 오히려 0 에 가까웠다.
                     오탐 0 을 유지하면서 각도 2 를 올리는 조합이 없어 기각.
                     (notebooks/jh/frontality_calibration.py 참고)

남은 약점
---------
아래를 본 사진(각도 2)은 66.7% 만 정면으로 잡힌다. 그래서 안내 문구를
"정면으로 찍어주세요"가 아니라 "카메라를 눈높이에 맞춰"로 쓴다.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

FRONTALITY_THRESHOLD = 0.05

# 판정에 필요한 부위. 하나라도 없으면 점수를 낼 수 없다.
_REQUIRED_PARTS = ("glabella", "left_cheek", "right_cheek")

STATUS_FRONTAL = "frontal"
STATUS_TURNED = "turned"
STATUS_FACE_NOT_DETECTED = "face_not_detected"

MESSAGES = {
    STATUS_FRONTAL: None,
    STATUS_TURNED: (
        "고개가 돌아간 것 같습니다. 카메라를 눈높이에 맞춰 정면을 봐주세요. "
        "이대로 진행하면 가려진 부위는 얼굴 전체 이미지로 추정한 값이 표시됩니다."
    ),
    STATUS_FACE_NOT_DETECTED: (
        "얼굴을 충분히 인식하지 못했습니다. 밝은 곳에서 얼굴 전체가 나오도록 "
        "카메라를 눈높이에 맞춰 다시 찍어주세요."
    ),
}


def _center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def frontality_offset(detected_parts: Any) -> Optional[float]:
    """미간이 두 볼 중점에서 좌우로 벗어난 정도. 계산 불가면 None.

    볼 간격으로 나누므로 얼굴 크기와 촬영 거리에 무관하다.
    """
    if not isinstance(detected_parts, list):
        return None

    by_part: dict[str, dict] = {}
    for d in detected_parts:
        if not isinstance(d, dict):
            continue
        name = d.get("raw_part_name")
        bbox = d.get("bbox_xyxy")
        if name and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            by_part[name] = d

    if not all(k in by_part for k in _REQUIRED_PARTS):
        return None

    try:
        gx, _ = _center([float(v) for v in by_part["glabella"]["bbox_xyxy"]])
        lx, _ = _center([float(v) for v in by_part["left_cheek"]["bbox_xyxy"]])
        rx, _ = _center([float(v) for v in by_part["right_cheek"]["bbox_xyxy"]])
    except (TypeError, ValueError):
        logger.warning("pose_check: bbox_xyxy 값을 숫자로 읽을 수 없습니다")
        return None

    span = abs(rx - lx)
    if span < 1e-6:
        return None
    return (gx - (lx + rx) / 2.0) / span


def judge_pose(detected_parts: Any) -> dict:
    """('frontal' | 'turned' | 'face_not_detected') 판정 결과를 dict 로 반환.

    점수를 낼 수 없는 경우를 정면으로 취급하지 않는다. 그렇게 하면
    가장 나쁜 입력(얼굴이 거의 안 보이는 사진)이 무사통과한다.
    """
    score = frontality_offset(detected_parts)
    if score is None:
        status = STATUS_FACE_NOT_DETECTED
    elif abs(score) <= FRONTALITY_THRESHOLD:
        status = STATUS_FRONTAL
    else:
        status = STATUS_TURNED
    return {
        "status": status,
        "score": None if score is None else round(score, 5),
        "threshold": FRONTALITY_THRESHOLD,
        "message": MESSAGES[status],
    }
