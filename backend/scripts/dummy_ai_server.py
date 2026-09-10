"""
AI inference server — Real inference for Perocular (Part 3) using DinoInferenceEngine.
Other parts remain mock for now.

실행:
    cd backend
    python -m uvicorn scripts.dummy_ai_server:app --reload --port 9000
"""

import os
import io
import json
from typing import Optional
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from contextlib import asynccontextmanager

# Import Modular Engine
try:
    from scripts.inference_engine import DinoInferenceEngine
    from scripts.face_detector import FaceDetector
except ImportError:
    # If run directly as 'python scripts/dummy_ai_server.py'
    from inference_engine import DinoInferenceEngine
    from face_detector import FaceDetector

# ===================================================================
# Config & Paths
# ===================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKBONE_CKPT = os.path.join(SCRIPT_DIR, "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth")
BASE_DIR = os.path.dirname(SCRIPT_DIR)
HEADS_DIR = os.path.join(BASE_DIR, "ckpt_kfold_vits_part3")
# 이 서버가 실제로 추론하는 부위는 facepart 3(왼쪽 눈가) 하나다.
# 엔진이 등급을 severity 로 옮기려면 어떤 라벨인지 알아야 한다 -
# 부위마다 등급 상한이 다르기 때문이다(눈가 주름 0~6 / 이마 색소 0~5 / 입술 0~4).
PART3_ANNOTATION_KEY = "l_perocular_wrinkle"
YOLO_CKPT = os.path.join(BASE_DIR, "model", "yolo_facecrop_best.pt")

# Initialize Engines
engine = DinoInferenceEngine(
    backbone_ckpt=BACKBONE_CKPT,
    heads_dir=HEADS_DIR,
    annotation_key=PART3_ANNOTATION_KEY,
)
face_detector = FaceDetector(model_path=YOLO_CKPT)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    success = engine.load_models()
    if not success:
        print("Warning: Failed to load models in DinoInferenceEngine.")
    if not face_detector.load():
        print("Warning: Failed to load YOLO face detector.")
    yield

app = FastAPI(title="Modular AI Inference Server", version="0.2.0", lifespan=lifespan)

_MOCK_PARTS_TEMPLATE = [
    {
        "raw_part_name": "left_cheek",
        "display_part_name": "볼",
        "metric_name": "pore",
        "metric_display_name": "모공",
        "issue_type": "pore",
        "grade_value": 2,
        "predicted_value": 0.62,
        "severity": "moderate",
        "confidence_score": 0.82,
    },
    {
        "raw_part_name": "right_cheek",
        "display_part_name": "볼",
        "metric_name": "pore",
        "metric_display_name": "모공",
        "issue_type": "pore",
        "grade_value": 1,
        "predicted_value": 0.35,
        "severity": "mild",
        "confidence_score": 0.79,
    },
    {
        "raw_part_name": "forehead",
        "display_part_name": "이마",
        "metric_name": "wrinkle",
        "metric_display_name": "주름",
        "issue_type": "wrinkle",
        "grade_value": 0,
        "predicted_value": 0.18,
        "severity": "normal",
        "confidence_score": 0.91,
    },
    {
        "raw_part_name": "glabella",
        "display_part_name": "미간",
        "metric_name": "wrinkle",
        "metric_display_name": "주름",
        "issue_type": "wrinkle",
        "grade_value": 2,
        "predicted_value": 0.65,
        "severity": "moderate",
        "confidence_score": 0.84,
    },
    {
        "raw_part_name": "left_eye",
        "display_part_name": "눈가",
        "metric_name": "wrinkle",
        "metric_display_name": "주름",
        "issue_type": "wrinkle",
        "grade_value": 3,
        "predicted_value": 0.87,
        "severity": "severe",
        "confidence_score": 0.88,
    },
    {
        "raw_part_name": "lips",
        "display_part_name": "입술",
        "metric_name": "dryness",
        "metric_display_name": "건조",
        "issue_type": "dryness",
        "grade_value": 1,
        "predicted_value": 0.38,
        "severity": "mild",
        "confidence_score": 0.77,
    },
    {
        "raw_part_name": "chin",
        "display_part_name": "턱",
        "metric_name": "sagging",
        "metric_display_name": "처짐",
        "issue_type": "sagging",
        "grade_value": 2,
        "predicted_value": 0.67,
        "severity": "moderate",
        "confidence_score": 0.74,
    },
]

@app.post("/inference/skin")
async def inference_skin(
    file: UploadFile = File(..., description="분석할 얼굴 이미지 파일"),
    session_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    image_id: Optional[str] = Form(None),
    bbox_left_eye: Optional[str] = Form(None, description="JSON string [x1, y1, x2, y2]"),
):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    print(f"[ai-server] 수신 | filename={file.filename} size={len(content)}bytes")

    # 1) YOLO face-part detection → per-part bboxes
    part_bboxes: dict = {}
    if face_detector.model is not None:
        try:
            part_bboxes = face_detector.detect_best_per_part(image)
        except Exception as e:
            print(f"Error during face-part detection: {e}")

    # 2) bbox for left_eye: caller override wins, else YOLO result
    bbox = None
    if bbox_left_eye:
        try:
            bbox = json.loads(bbox_left_eye)
        except Exception:
            print(f"Warning: Failed to parse bbox_left_eye: {bbox_left_eye}")
    if bbox is None and "left_eye" in part_bboxes:
        bbox = part_bboxes["left_eye"]["bbox_xyxy"]

    # 3) Actual inference for left_eye
    eye_result = None
    if len(engine.heads) > 0:
        try:
            eye_result = engine.predict(image, bbox)
        except Exception as e:
            print(f"Error during eye inference: {e}")

    # 4) Construct response — attach detected bbox to each part
    parts = []
    for p in _MOCK_PARTS_TEMPLATE:
        new_p = p.copy()
        det = part_bboxes.get(p["raw_part_name"])
        if det is not None:
            new_p["bbox_xyxy"] = det["bbox_xyxy"]
            new_p["detection_confidence"] = det["confidence"]
        if p["raw_part_name"] == "left_eye" and eye_result:
            new_p.update(eye_result)
        parts.append(new_p)

    return {
        "model_name": "skin_dinov3_ensemble_model",
        "model_version": "0.2.0",
        "parts": parts,
        "detected_parts": [
            {
                "raw_part_name": name,
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "bbox_xyxy": d["bbox_xyxy"],
            }
            for name, d in part_bboxes.items()
        ],
    }

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "service": "ai-inference",
        "device": str(engine.device),
        "models_loaded": len(engine.heads) > 0,
        "face_detector_loaded": face_detector.model is not None,
    }
