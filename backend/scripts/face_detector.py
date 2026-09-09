# -*- coding: utf-8 -*-
"""
Face part detector using a YOLO model.

Given an image (path, PIL.Image, or numpy array), returns per-part bounding
boxes in xyxy format. Class names follow the trained YOLO model and are
normalized to the names used by the inference server (e.g. ``l_eye`` →
``left_eye``).
"""

import os
from typing import Any, Dict, List, Optional, Union

from PIL import Image

# YOLO class index → raw class name as trained
CLASS_NAMES = [
    "forehead", "glabella", "l_eye", "r_eye",
    "l_cheek", "r_cheek", "lips", "chin",
]

# Map raw YOLO class names → server-side raw_part_name
PART_NAME_MAP = {
    "forehead": "forehead",
    "glabella": "glabella",
    "l_eye": "left_eye",
    "r_eye": "right_eye",
    "l_cheek": "left_cheek",
    "r_cheek": "right_cheek",
    "lips": "lips",
    "chin": "chin",
}

ImageInput = Union[str, "os.PathLike[str]", Image.Image, Any]

# 좌우 쌍을 이루는 부위. 검출기가 좌우를 자주 혼동하므로 x좌표로 재할당한다.
#   좌우 규약은 관찰자(이미지) 기준 - 이미지 왼쪽이 l_*, 오른쪽이 r_*.
#   라벨 108,070건 전수 검증 결과이며 세 촬영 기기 모두 동일하다.
#   (근거: 정면 bbox 중심 정규화 x 중앙값 fp3 0.220 / fp5 0.307, 미러링 없음)
LR_PAIRS = [
    ("left_eye", "right_eye"),
    ("left_cheek", "right_cheek"),
]

_LR_NAMES = {name for pair in LR_PAIRS for name in pair}


def _cx(det: Dict[str, Any]) -> float:
    x1, _, x2, _ = det["bbox_xyxy"]
    return (x1 + x2) / 2.0


def _reassign_lr(dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """좌우 쌍 부위의 예측 클래스를 x좌표로 다시 정하는 방어적 안전망.

    detect_best_per_part 는 부위명당 한 박스만 남기므로, 두 박스가 같은 클래스로
    분류되면 반대쪽이 통째로 사라지고 전체 이미지 fallback 으로 넘어간다.
    AI-Hub 검증셋 기준으로 이런 좌우 혼동은 관측되지 않았지만
    (혼동행렬 l_eye <-> r_eye 오분류 0건, mAP50 0.994),
    도메인이 다른 실사용 입력에서는 보장되지 않으므로 가드로 남긴다.

    좌우 규약은 관찰자(이미지) 기준 - 이미지 왼쪽이 l_*, 오른쪽이 r_*.
    라벨 108,070건 전수 검증 결과이며 세 촬영 기기 모두 동일하다.

    쌍에 속한 박스가 정확히 2개일 때만 재할당한다.
    1개면 어느 쪽인지 알 수 없으므로 원래 예측을 존중하고, 3개 이상이면
    신뢰도 상위 2개만 남긴다.
    """
    out = [d for d in dets if d["raw_part_name"] not in _LR_NAMES]

    for left_name, right_name in LR_PAIRS:
        pair = [d for d in dets if d["raw_part_name"] in (left_name, right_name)]
        if len(pair) >= 2:
            pair = sorted(pair, key=lambda d: d["confidence"], reverse=True)[:2]
            pair = sorted(pair, key=_cx)          # x 오름차순
            pair[0] = {**pair[0], "raw_part_name": left_name, "lr_source": "x_position"}
            pair[1] = {**pair[1], "raw_part_name": right_name, "lr_source": "x_position"}
        else:
            pair = [{**d, "lr_source": "detector"} for d in pair]
        out.extend(pair)
    return out


class FaceDetector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load(self) -> bool:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            print(f"[face-detector] ultralytics not installed: {e}")
            return False

        if not os.path.exists(self.model_path):
            print(f"[face-detector] model file not found: {self.model_path}")
            return False

        try:
            self.model = YOLO(self.model_path)
            return True
        except Exception as e:
            print(f"[face-detector] failed to load YOLO model: {e}")
            return False

    def detect(
        self,
        image: ImageInput,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 1280,
    ) -> List[Dict[str, Any]]:
        if self.model is None:
            return []

        r = self.model.predict(
            source=image, imgsz=imgsz, conf=conf, iou=iou, verbose=False
        )[0]

        boxes = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        detections: List[Dict[str, Any]] = []
        for box, cid, cf in zip(boxes, cls_ids, confs):
            cid = int(cid)
            raw = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)
            detections.append({
                "class_id": cid,
                "class_name": raw,
                "raw_part_name": PART_NAME_MAP.get(raw, raw),
                "confidence": float(cf),
                "bbox_xyxy": [float(x) for x in box],
            })
        return detections

    def detect_best_per_part(
        self,
        image: ImageInput,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 1280,
    ) -> Dict[str, Dict[str, Any]]:
        """Return one detection per part — the highest-confidence box keyed by
        the server-side ``raw_part_name``.

        좌우 쌍 부위는 _reassign_lr 로 x좌표 기준 재할당을 거친다. 각 검출 dict 에는
        좌우가 검출기 판단인지(``lr_source="detector"``) 후처리 결과인지
        (``"x_position"``) 남는다. DB 저장은 하지 않고 반환 dict 에만 남긴다.
        """
        best: Dict[str, Dict[str, Any]] = {}
        dets = _reassign_lr(self.detect(image, conf=conf, iou=iou, imgsz=imgsz))
        for d in dets:
            name = d["raw_part_name"]
            if name not in best or d["confidence"] > best[name]["confidence"]:
                best[name] = d
        return best
