from typing import Optional

from pydantic import BaseModel, ConfigDict


class PartResult(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "raw_part_name": "left_cheek",
                "display_part_name": "볼",
                "metric_name": "pore",
                "metric_display_name": "모공",
                "issue_type": "pore",
                "grade_value": 2,
                "severity": "moderate",
                "confidence_score": 0.82,
            }
        }
    )

    raw_part_name: str
    display_part_name: str
    metric_name: str
    metric_display_name: str
    issue_type: str
    grade_value: int
    predicted_value: Optional[float] = None
    measured_value: Optional[float] = None
    severity: str
    confidence_score: Optional[float] = None


class PoseCheck(BaseModel):
    """업로드 사진의 정면 여부. 차단이 아니라 경고용이다.

    status  frontal | turned | face_not_detected
    score   미간이 두 볼 중점에서 좌우로 벗어난 정도(볼 간격으로 정규화).
            판정 불가 시 None - 이 경우를 정면으로 취급하지 않는다.
    """

    status: str
    score: Optional[float] = None
    threshold: Optional[float] = None
    message: Optional[str] = None


class InferenceResult(BaseModel):
    model_name: str
    model_version: str
    parts: list[PartResult]


class ImageUploadResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_id": 1,
                "session_id": 1,
                "original_filename": "face.jpg",
                "stored_filename": "550e8400-e29b-41d4-a716-446655440000.jpg",
                "file_path": "uploads/1/1/550e8400-e29b-41d4-a716-446655440000.jpg",
                "width": 1920,
                "height": 1080,
                "upload_status": "processed",
                "session_status": "completed",
                "inference_result": {
                    "model_name": "mock_skin_model",
                    "model_version": "0.0.1",
                    "parts": [
                        {
                            "raw_part_name": "left_cheek",
                            "display_part_name": "볼",
                            "metric_name": "pore",
                            "metric_display_name": "모공",
                            "issue_type": "pore",
                            "grade_value": 2,
                            "severity": "moderate",
                            "confidence_score": 0.82,
                        },
                        {
                            "raw_part_name": "left_eye",
                            "display_part_name": "눈가",
                            "metric_name": "wrinkle",
                            "metric_display_name": "주름",
                            "issue_type": "wrinkle",
                            "grade_value": 3,
                            "severity": "severe",
                            "confidence_score": 0.88,
                        },
                    ],
                },
            }
        }
    )

    image_id: int
    session_id: int
    original_filename: str
    stored_filename: str
    file_path: str
    width: Optional[int]
    height: Optional[int]
    upload_status: str
    session_status: str
    inference_result: InferenceResult
    # 정면 여부 경고. flat 모드처럼 검출 결과가 없으면 None 이다.
    pose_check: Optional[PoseCheck] = None
