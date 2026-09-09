from collections import defaultdict
from typing import Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.core.exceptions import (
    completed_report_not_found,
    session_access_denied,
    session_not_found,
)
from app.models.analysis_session import AnalysisSession
from app.models.part_recommendation import PartRecommendation
from app.models.skin_part_detection import SkinPartDetection
from app.models.skin_part_result import SkinPartResult
from app.schemas.recommendation import ExcludedIngredientItem, IngredientItem
from app.schemas.report import (
    IssueItem,
    MainIssue,
    OverallSummary,
    PartReport,
    RecommendationSummary,
    ReportResponse,
)

_SEVERITY_ORDER = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}


def get_latest_report(db: Session, user_id: int) -> ReportResponse:
    session = (
        db.query(AnalysisSession)
        .filter(
            AnalysisSession.user_id == user_id,
            AnalysisSession.status == "completed",
        )
        .order_by(
            desc(AnalysisSession.analyzed_at),
            desc(AnalysisSession.updated_at),
            desc(AnalysisSession.created_at),
        )
        .first()
    )
    if session is None:
        raise completed_report_not_found()

    return get_report(db, session.id, user_id)


def get_report(db: Session, session_id: int, user_id: int) -> ReportResponse:
    session = db.get(AnalysisSession, session_id)
    if session is None:
        raise session_not_found()
    if session.user_id != user_id:
        raise session_access_denied()

    if session.status != "completed":
        return ReportResponse(
            session_id=session_id,
            status=session.status,
            overall_summary=OverallSummary(
                status="분석 중",
                main_message="아직 분석이 완료되지 않았습니다.",
                main_issues=[],
            ),
            part_reports=[],
        )

    results: list[SkinPartResult] = (
        db.query(SkinPartResult)
        .filter(SkinPartResult.session_id == session_id)
        .all()
    )

    recommendations: list[PartRecommendation] = (
        db.query(PartRecommendation)
        .filter(PartRecommendation.session_id == session_id)
        .all()
    )

    rec_map: dict[tuple[str, str], PartRecommendation] = {
        (r.display_part_name, r.issue_type): r for r in recommendations
    }

    # bbox 출처. 미검출 부위는 얼굴 전체 이미지로 추론되므로 추정값임을 알려야 한다.
    detections: list[SkinPartDetection] = (
        db.query(SkinPartDetection)
        .filter(SkinPartDetection.session_id == session_id)
        .all()
    )
    source_by_raw_part: dict[str, str] = {d.raw_part_name: d.bbox_source for d in detections}

    part_map: dict[str, list[SkinPartResult]] = defaultdict(list)
    for r in results:
        part_map[r.display_part_name].append(r)

    part_reports: list[PartReport] = []
    non_normal_results: list[SkinPartResult] = []

    for display_part_name, part_results in part_map.items():
        issues = [
            IssueItem(
                metric_name=r.metric_name,
                metric_display_name=r.metric_display_name,
                issue_type=r.issue_type,
                severity=r.severity,
                grade_value=r.grade_value,
                predicted_value=r.predicted_value,
                measured_value=r.measured_value,
                reason=r.reason_text,
            )
            for r in part_results
        ]
        issues.sort(key=lambda x: _SEVERITY_ORDER.get(x.severity, 0), reverse=True)

        for r in part_results:
            if r.severity != "normal":
                non_normal_results.append(r)

        rec_summary = _build_recommendation(display_part_name, issues, rec_map)
        summary = _build_part_summary(display_part_name, issues)

        part_reports.append(PartReport(
            display_part_name=display_part_name,
            summary=summary,
            issues=issues,
            recommendation=rec_summary,
            bbox_source=_aggregate_bbox_source(part_results, source_by_raw_part),
        ))

    part_reports.sort(
        key=lambda p: max((_SEVERITY_ORDER.get(i.severity, 0) for i in p.issues), default=0),
        reverse=True,
    )

    overall = _build_overall_summary(non_normal_results)

    return ReportResponse(
        session_id=session_id,
        status=session.status,
        overall_summary=overall,
        part_reports=part_reports,
    )


def _aggregate_bbox_source(
    part_results: list[SkinPartResult],
    source_by_raw_part: dict[str, str],
) -> str | None:
    """부위 단위 bbox 출처를 집계한다.

    display_part_name 하나가 raw_part_name 여럿을 묶는다
    ("눈가" = left_eye + right_eye, "볼" = left_cheek + right_cheek).
    고개를 돌린 사진에서는 한쪽만 미검출될 수 있으므로 이진값으로 뭉개지 않고
    "mixed" 를 따로 둔다.
    """
    sources = {
        source_by_raw_part[r.raw_part_name]
        for r in part_results
        if r.raw_part_name in source_by_raw_part
    }
    if not sources:
        return None
    if len(sources) == 1:
        return next(iter(sources))
    return "mixed"


def _build_recommendation(
    display_part_name: str,
    issues: list[IssueItem],
    rec_map: dict[tuple[str, str], PartRecommendation],
) -> Optional[RecommendationSummary]:
    merged_categories: list[str] = []
    merged_ingredients: dict[str, IngredientItem] = {}
    merged_excluded: dict[str, ExcludedIngredientItem] = {}
    merged_tips: list[str] = []
    exclusion_reason: Optional[str] = None

    for issue in issues:
        rec = rec_map.get((display_part_name, issue.issue_type))
        if rec is None:
            continue
        for cat in (rec.recommend_categories or []):
            if cat not in merged_categories:
                merged_categories.append(cat)
        for ing in (rec.recommend_ingredients or []):
            k = ing["key"] if isinstance(ing, dict) else ing.key
            n = ing["name"] if isinstance(ing, dict) else ing.name
            merged_ingredients[k] = IngredientItem(key=k, name=n)
        for ing in (rec.excluded_ingredients or []):
            k = ing["key"] if isinstance(ing, dict) else ing.key
            n = ing["name"] if isinstance(ing, dict) else ing.name
            rt = ing.get("reason_type") if isinstance(ing, dict) else getattr(ing, "reason_type", None)
            # allergy가 이미 있으면 reason_type을 내려받지 않음
            if k not in merged_excluded or merged_excluded[k].reason_type != "allergy":
                merged_excluded[k] = ExcludedIngredientItem(key=k, name=n, reason_type=rt)
        for tip in (rec.care_tips or []):
            if tip not in merged_tips:
                merged_tips.append(tip)
        if rec.exclusion_reason and not exclusion_reason:
            exclusion_reason = rec.exclusion_reason

    if not merged_categories and not merged_ingredients:
        return None

    # exclusion_reason: allergy 우선, 없으면 sensitive
    has_allergy = any(e.reason_type == "allergy" for e in merged_excluded.values())
    has_sensitive = any(e.reason_type == "sensitive" for e in merged_excluded.values())
    if has_allergy and has_sensitive:
        exclusion_reason = "사용자가 피해야 할 성분으로 등록된 성분과 민감 피부 주의 성분이 제외되었습니다."
    elif has_allergy:
        exclusion_reason = "사용자가 피해야 할 성분으로 등록된 성분이 제외되었습니다."
    elif has_sensitive:
        exclusion_reason = "민감 피부 주의 성분이 제외되었습니다."

    return RecommendationSummary(
        categories=merged_categories,
        ingredients=list(merged_ingredients.values()),
        excluded_ingredients=list(merged_excluded.values()),
        exclusion_reason=exclusion_reason,
        care_tips=merged_tips,
    )


def _build_part_summary(display_part_name: str, issues: list[IssueItem]) -> str:
    non_normal = [i for i in issues if i.severity != "normal"]
    if not non_normal:
        return f"{display_part_name} 부위는 전반적으로 양호한 상태입니다."
    worst = max(_SEVERITY_ORDER.get(i.severity, 0) for i in non_normal)
    top = [i for i in non_normal if _SEVERITY_ORDER.get(i.severity, 0) == worst]
    names = [i.metric_display_name for i in top]
    if len(names) == 1:
        return f"{display_part_name} 부위에서 {names[0]} 관리가 필요합니다."
    return f"{display_part_name} 부위에서 {', '.join(names[:-1])}과 {names[-1]} 관리가 필요합니다."


def _build_overall_summary(non_normal: list[SkinPartResult]) -> OverallSummary:
    if not non_normal:
        return OverallSummary(
            status="양호",
            main_message="전반적으로 피부 상태가 양호합니다.",
            main_issues=[],
        )

    severities = {r.severity for r in non_normal}
    if "severe" in severities:
        status = "집중 관리 필요"
    elif "moderate" in severities:
        status = "주의"
    else:
        status = "주의"

    worst_per_type: dict[str, str] = {}
    for r in non_normal:
        cur = worst_per_type.get(r.issue_type, "normal")
        if _SEVERITY_ORDER.get(r.severity, 0) > _SEVERITY_ORDER.get(cur, 0):
            worst_per_type[r.issue_type] = r.severity

    main_issues = sorted(
        [MainIssue(issue_type=it, severity=sv) for it, sv in worst_per_type.items()],
        key=lambda x: _SEVERITY_ORDER.get(x.severity, 0),
        reverse=True,
    )
    main_message = _build_main_message(non_normal)

    return OverallSummary(status=status, main_message=main_message, main_issues=main_issues)


def _build_main_message(non_normal: list[SkinPartResult]) -> str:
    seen: set[tuple[str, str]] = set()
    mentions: list[tuple[str, str]] = []
    for r in sorted(non_normal, key=lambda x: _SEVERITY_ORDER.get(x.severity, 0), reverse=True):
        key = (r.display_part_name, r.metric_display_name)
        if key not in seen:
            seen.add(key)
            mentions.append(key)
        if len(mentions) >= 3:
            break

    if not mentions:
        return "피부 상태를 주의 깊게 관리하세요."

    part_to_metrics: dict[str, list[str]] = defaultdict(list)
    for part, metric in mentions:
        part_to_metrics[part].append(metric)

    parts_text = []
    for part, metrics in part_to_metrics.items():
        parts_text.append(f"{part} 부위의 {', '.join(metrics)}")

    return f"{', '.join(parts_text)} 관리가 필요합니다."
