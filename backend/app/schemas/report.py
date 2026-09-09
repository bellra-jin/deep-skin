from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.schemas.recommendation import ExcludedIngredientItem, IngredientItem


class IssueItem(BaseModel):
    metric_name: str
    metric_display_name: str
    issue_type: str
    severity: str
    grade_value: Optional[int]
    predicted_value: Optional[float] = None
    measured_value: Optional[float] = None
    reason: Optional[str]


class MainIssue(BaseModel):
    issue_type: str
    severity: str


class OverallSummary(BaseModel):
    status: str
    main_message: str
    main_issues: list[MainIssue]


class RecommendationSummary(BaseModel):
    categories: list[str]
    ingredients: list[IngredientItem]
    excluded_ingredients: list[ExcludedIngredientItem]
    exclusion_reason: Optional[str]
    care_tips: list[str]


class PartReport(BaseModel):
    display_part_name: str
    summary: str
    issues: list[IssueItem]
    recommendation: Optional[RecommendationSummary]
    # 이 부위의 bbox 출처. skin_part_detections.bbox_source 를 부위 단위로 집계한 값.
    #   "yolo"                 검출된 bbox 로 추론 - 신뢰 가능
    #   "full_image_fallback"  미검출이라 얼굴 전체 이미지로 추론 - 추정값
    #   "mixed"                좌우 중 한쪽만 검출됨 (눈가/볼은 좌우 두 부위를 묶는다)
    #   None                   검출 기록이 없음 (dev JSON 경로 등)
    bbox_source: Optional[str] = None


class ReportResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": 1,
                "status": "completed",
                "overall_summary": {
                    "status": "집중 관리 필요",
                    "main_message": "눈가 부위의 주름, 볼 부위의 모공, 턱 부위의 처짐 관리가 필요합니다.",
                    "main_issues": [
                        {"issue_type": "wrinkle", "severity": "severe"},
                        {"issue_type": "pore", "severity": "moderate"},
                        {"issue_type": "sagging", "severity": "moderate"},
                    ],
                },
                "part_reports": [
                    {
                        "display_part_name": "눈가",
                        "summary": "눈가 부위에서 주름 관리가 필요합니다.",
                        "issues": [
                            {
                                "metric_name": "wrinkle",
                                "metric_display_name": "주름",
                                "issue_type": "wrinkle",
                                "severity": "severe",
                                "grade_value": 3,
                                "reason": None,
                            }
                        ],
                        "recommendation": {
                            "categories": ["아이크림", "주름 개선 세럼"],
                            "ingredients": [
                                {"key": "retinol", "name": "레티놀"},
                                {"key": "peptide", "name": "펩타이드"},
                            ],
                            "excluded_ingredients": [],
                            "exclusion_reason": None,
                            "care_tips": ["눈가는 가볍게 두드리며 흡수시키는 것이 좋습니다."],
                        },
                    },
                    {
                        "display_part_name": "볼",
                        "summary": "볼 부위에서 모공 관리가 필요합니다.",
                        "issues": [
                            {
                                "metric_name": "pore",
                                "metric_display_name": "모공",
                                "issue_type": "pore",
                                "severity": "moderate",
                                "grade_value": 2,
                                "reason": None,
                            }
                        ],
                        "recommendation": {
                            "categories": ["모공 케어 토너", "피지 조절 세럼"],
                            "ingredients": [{"key": "niacinamide", "name": "나이아신아마이드"}],
                            "excluded_ingredients": [],
                            "exclusion_reason": None,
                            "care_tips": ["피지 조절과 모공 케어 중심의 제품을 사용하는 것이 좋습니다."],
                        },
                    },
                ],
            }
        }
    )

    session_id: int
    status: str
    overall_summary: OverallSummary
    part_reports: list[PartReport]
