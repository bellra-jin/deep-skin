import base64
import math
import os

import streamlit as st

from components.common import severity_badge_html, ingredient_chip_html, category_chip_html
from styles.theme import SEVERITY_LABEL, SEVERITY_SCORE, MAIN_CONCERN_OPTIONS


# ── 아이콘 헬퍼 ──────────────────────────────────────────────────────────────

def _b64_src(rel: str) -> str:
    frontend_root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(frontend_root, "assets", "icons", rel)
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def _img(rel: str, size: int = 24) -> str:
    src = _b64_src(rel)
    if not src:
        return ""
    return (
        f'<img src="{src}" width="{size}" height="{size}" '
        f'style="object-fit:contain;vertical-align:middle;">'
    )


# ── 얼굴 부위 → 아이콘 매핑 ──────────────────────────────────────────────────

_PART_ICON: dict[str, str] = {
    "볼":   "skincare/cheek_face.png",
    "눈가": "skincare/eye_area_face.png",
    "입술": "skincare/lips_face.png",
    "턱":   "skincare/chin_face.png",
    "이마": "skincare/forehead_face.png",
    "미간": "Glabella.png",
}
_PART_EMOJI: dict[str, str] = {
    "볼": "🫧", "눈가": "👁️", "입술": "👄",
    "턱": "🫦", "이마": "😐", "미간": "◌", "전체": "🧖",
}


def _part_icon(part_name: str, size: int = 64) -> str:
    for key, icon_rel in _PART_ICON.items():
        if key in part_name:
            src = _b64_src(icon_rel)
            if src:
                return (
                    f'<img src="{src}" width="{size}" height="{size}" '
                    f'style="object-fit:contain;border-radius:12px;">'
                )
    emoji = next((v for k, v in _PART_EMOJI.items() if k in part_name), "🌀")
    return f'<div style="font-size:{size * 0.55:.0f}px;line-height:1;">{emoji}</div>'


# ── 점수 계산 ─────────────────────────────────────────────────────────────────

def compute_report_score(part_reports: list) -> int:
    scores = []
    for part in part_reports:
        for issue in (part.get("issues") or []):
            sev = issue.get("severity", "normal")
            scores.append(SEVERITY_SCORE.get(sev, 70))
    return round(sum(scores) / len(scores)) if scores else 85


# ── 동적 점수 게이지 (SVG) ────────────────────────────────────────────────────

def render_score_gauge(score: int):
    """CSS/SVG 기반 동적 원형 점수 게이지."""
    r     = 42
    circ  = 2 * math.pi * r
    filled = circ * score / 100

    if score >= 75:
        stroke, label_color = "#10B981", "#065F46"
    elif score >= 50:
        stroke, label_color = "#F59E0B", "#92400E"
    elif score >= 30:
        stroke, label_color = "#EF4444", "#991B1B"
    else:
        stroke, label_color = "#7F1D1D", "#7F1D1D"

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:center;padding:8px 0;">
        <svg width="120" height="120" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="{r}" fill="none"
                    stroke="#E4EAF5" stroke-width="10"/>
            <circle cx="50" cy="50" r="{r}" fill="none"
                    stroke="{stroke}" stroke-width="10"
                    stroke-dasharray="{filled:.1f} {circ:.1f}"
                    stroke-linecap="round"
                    transform="rotate(-90 50 50)"/>
            <text x="50" y="46" text-anchor="middle"
                  font-size="22" font-weight="800" fill="#0F2447">{score}</text>
            <text x="50" y="62" text-anchor="middle"
                  font-size="11" fill="#6B7894">/100</text>
        </svg>
    </div>
    """, unsafe_allow_html=True)


# ── 동적 별점 ────────────────────────────────────────────────────────────────

def render_star_rating(severity: str):
    """severity 기준 동적 별점 렌더링 (정적 이미지 미사용)."""
    mapping = {"normal": 1, "mild": 2, "moderate": 3, "severe": 4}
    colors  = {"normal": "#10B981", "mild": "#F59E0B",
               "moderate": "#F97316", "severe": "#EF4444"}
    filled  = mapping.get(severity, 1)
    color   = colors.get(severity, "#A0AABB")
    stars   = "".join(
        f'<span style="color:{color};font-size:16px;">★</span>' if i < filled
        else '<span style="color:#E4EAF5;font-size:16px;">★</span>'
        for i in range(4)
    )
    st.markdown(f'<div style="margin:4px 0;">{stars}</div>', unsafe_allow_html=True)


def stars_html(severity: str) -> str:
    """HTML 문자열 반환 버전 (inline 사용용)."""
    mapping = {"normal": 1, "mild": 2, "moderate": 3, "severe": 4}
    colors  = {"normal": "#10B981", "mild": "#F59E0B",
               "moderate": "#F97316", "severe": "#EF4444"}
    filled  = mapping.get(severity, 1)
    color   = colors.get(severity, "#A0AABB")
    return "".join(
        f'<span style="color:{color};font-size:15px;">★</span>' if i < filled
        else '<span style="color:#E4EAF5;font-size:15px;">★</span>'
        for i in range(4)
    )


# ── 동적 Progress Dots ────────────────────────────────────────────────────────

def render_progress_dots(active_count: int, total: int = 5):
    """active_count 개수만큼 파란 점, 나머지는 회색 (정적 이미지 미사용)."""
    dots = "".join(
        '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        f'background:{"#4B7BFF" if i < active_count else "#E4EAF5"};'
        'margin:0 3px;"></span>'
        for i in range(total)
    )
    st.markdown(f'<div style="text-align:center;margin:6px 0;">{dots}</div>', unsafe_allow_html=True)


# ── 전체 요약 카드 ────────────────────────────────────────────────────────────

def render_overall_summary(overall: dict, score: int, analyzed_at: str = ""):
    status      = overall.get("status", "")
    message     = overall.get("main_message", "")
    main_issues = overall.get("main_issues") or []

    badge = severity_badge_html(_overall_severity(status))

    issue_chips = ""
    if main_issues:
        chips = " ".join(
            f'<span class="ds-chip ds-chip-concern">'
            f'{MAIN_CONCERN_OPTIONS.get(i.get("issue_type",""), i.get("issue_type",""))} '
            f'{severity_badge_html(i.get("severity","normal"))}</span>'
            for i in main_issues
        )
        issue_chips = f'<div style="margin-top:14px;">{chips}</div>'

    date_icon = _img("report/date_analyzed.png", 14)
    date_line = (
        f'<div style="font-size:12px;color:#A0AABB;margin-bottom:10px;">'
        f'{date_icon} 분석 일시: {analyzed_at}</div>'
        if analyzed_at else ""
    )

    col_text, col_score = st.columns([3, 1])
    with col_text:
        st.markdown(f"""
        <div class="ds-score-card">
            {date_line}
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap;">
                <span style="font-size:20px;font-weight:800;color:#0F2447;">{status}</span>
                {badge}
            </div>
            <p style="font-size:14px;color:#374151;margin:0;line-height:1.7;">{message}</p>
            {issue_chips}
        </div>
        """, unsafe_allow_html=True)
    with col_score:
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:center;height:100%;">',
            unsafe_allow_html=True,
        )
        render_score_gauge(score)
        st.markdown('</div>', unsafe_allow_html=True)


def _overall_severity(status: str) -> str:
    if "집중" in status:
        return "severe"
    if "관리 필요" in status and "약한" not in status:
        return "moderate"
    if "약한" in status:
        return "mild"
    return "normal"


# ── 부위별 카드 헤더 (컴팩트 그리드용) ───────────────────────────────────────

# ── 추정값 표시 ──────────────────────────────────────────────────────────────

_ESTIMATION_LABEL = {
    "full_image_fallback": ("추정값", "이 부위가 사진에서 검출되지 않아 얼굴 전체 이미지로 추정한 값입니다."),
    "mixed": ("일부 추정", "좌우 중 한쪽이 검출되지 않아 해당 쪽은 얼굴 전체 이미지로 추정했습니다."),
}


def estimation_badge_html(bbox_source: str = None) -> str:
    """bbox 출처가 검출(yolo)이 아니면 추정값 배지를 만든다.

    값을 숨기지 않고 추정임을 밝히는 쪽이 맞다. 숨기면 사용자는
    "왜 항목이 없지?" 가 되고, 그냥 두면 추정값을 측정값으로 오해한다.
    """
    label = _ESTIMATION_LABEL.get(bbox_source or "")
    if not label:
        return ""
    text, tip = label
    return (
        f'<span title="{tip}" style="display:inline-block;padding:1px 7px;'
        f'border:1px solid #C9A227;border-radius:9px;background:#FFF8E1;'
        f'color:#8A6D0B;font-size:10px;font-weight:600;">{text}</span>'
    )


def render_part_card_header(part: dict):
    part_name = part.get("display_part_name", "")
    issues    = part.get("issues") or []

    order = {"severe": 4, "moderate": 3, "mild": 2, "normal": 1}
    worst = max(
        (i.get("severity", "normal") for i in issues),
        key=lambda s: order.get(s, 0),
        default="normal",
    )

    concern_chips = " ".join(
        f'<span class="ds-chip ds-chip-concern" style="font-size:11px;">'
        f'{MAIN_CONCERN_OPTIONS.get(i.get("issue_type",""), i.get("metric_display_name",""))}'
        f'</span>'
        for i in issues[:2]
    )

    icon_html = _part_icon(part_name, 60)
    stars     = stars_html(worst)
    badge     = severity_badge_html(worst)
    est_badge = estimation_badge_html(part.get("bbox_source"))

    st.markdown(f"""
    <div class="ds-part-card">
        <div style="margin-bottom:8px;">{icon_html}</div>
        <div style="font-size:14px;font-weight:700;color:#0F2447;margin-bottom:6px;">{part_name}</div>
        <div style="margin:4px 0;">{stars}</div>
        <div style="margin-top:6px;">{badge}</div>
        <div style="margin-top:4px;">{est_badge}</div>
        <div style="margin-top:8px;line-height:1.8;">{concern_chips}</div>
    </div>
    """, unsafe_allow_html=True)


# ── 부위별 상세 (expander) ────────────────────────────────────────────────────

def render_part_detail(part: dict):
    part_name = part.get("display_part_name", "")
    summary   = part.get("summary", "")
    issues    = part.get("issues") or []
    rec       = part.get("recommendation") or {}

    categories  = rec.get("categories")   or []
    ingredients = rec.get("ingredients")  or []
    excluded    = rec.get("excluded_ingredients") or []
    care_tips   = rec.get("care_tips")    or []

    with st.expander(f"**{part_name}** 상세 분석", expanded=False):
        est = _ESTIMATION_LABEL.get(part.get("bbox_source") or "")
        if est:
            st.markdown(
                f'<div style="padding:8px 12px;margin:0 0 12px;border-radius:8px;'
                f'background:#FFF8E1;border:1px solid #F0DFA0;color:#8A6D0B;'
                f'font-size:12px;line-height:1.6;">'
                f'<b>{est[0]}</b> - {est[1]}<br>'
                f'정면을 바라보고 다시 촬영하면 더 정확한 결과를 받을 수 있습니다.</div>',
                unsafe_allow_html=True,
            )
        if summary:
            st.markdown(
                f'<p style="color:#6B7894;font-size:14px;margin:0 0 14px;">{summary}</p>',
                unsafe_allow_html=True,
            )

        # 이슈 목록
        for issue in issues:
            sev   = issue.get("severity", "normal")
            dname = issue.get("metric_display_name", "")
            grade = issue.get("grade_value")
            grade_txt = f"  (등급 {grade})" if grade is not None else ""
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                f'<span style="font-size:14px;font-weight:600;color:#0F2447;">{dname}</span>'
                f'{severity_badge_html(sev)}'
                f'<span style="font-size:12px;color:#A0AABB;">{grade_txt}</span></div>',
                unsafe_allow_html=True,
            )

        if not any([categories, ingredients, excluded, care_tips]):
            return

        # 추천 4열 카드
        col_cat, col_ingr, col_excl, col_tips = st.columns(4)

        with col_cat:
            cat_icon = _img("report/category.png", 20)
            st.markdown(f'<div class="ds-rec-label">{cat_icon} 추천 카테고리</div>', unsafe_allow_html=True)
            st.markdown(
                " ".join(category_chip_html(c) for c in categories)
                if categories else '<span style="color:#A0AABB;font-size:13px;">—</span>',
                unsafe_allow_html=True,
            )

        with col_ingr:
            ingr_icon = _img("report/ingredient.png", 20)
            st.markdown(f'<div class="ds-rec-label">{ingr_icon} 추천 성분</div>', unsafe_allow_html=True)
            st.markdown(
                " ".join(ingredient_chip_html(i.get("name", "")) for i in ingredients)
                if ingredients else '<span style="color:#A0AABB;font-size:13px;">—</span>',
                unsafe_allow_html=True,
            )

        with col_excl:
            excl_icon = _img("report/excluded.png", 20)
            st.markdown(f'<div class="ds-rec-label">{excl_icon} 제외 성분</div>', unsafe_allow_html=True)
            st.markdown(
                " ".join(
                    ingredient_chip_html(e.get("name", ""), excluded=True,
                                        reason_type=e.get("reason_type"))
                    for e in excluded
                ) if excluded else '<span style="color:#A0AABB;font-size:13px;">없음</span>',
                unsafe_allow_html=True,
            )

        with col_tips:
            tip_icon = _img("report/care_tip.png", 20)
            st.markdown(f'<div class="ds-rec-label">{tip_icon} 관리 팁</div>', unsafe_allow_html=True)
            if care_tips:
                tips_html = "".join(
                    f'<li style="font-size:12px;color:#374151;margin-bottom:3px;">{t}</li>'
                    for t in care_tips
                )
                st.markdown(
                    f'<ul style="margin:0;padding-left:16px;">{tips_html}</ul>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<span style="color:#A0AABB;font-size:13px;">—</span>', unsafe_allow_html=True)


# ── 추천 섹션 카드 (하단 4 카드) ──────────────────────────────────────────────

def render_recommendation_cards(part_reports: list):
    """모든 부위의 추천을 합산해 4개의 대형 카드로 표시한다."""
    if not part_reports:
        return

    all_cats:  list[str] = []
    all_ingr:  list[dict] = []
    all_excl:  list[dict] = []
    all_tips:  list[str] = []

    seen_cats: set  = set()
    seen_ingr: set  = set()
    seen_excl: set  = set()

    for part in part_reports:
        rec = part.get("recommendation") or {}
        for c in (rec.get("categories") or []):
            if c not in seen_cats:
                all_cats.append(c); seen_cats.add(c)
        for i in (rec.get("ingredients") or []):
            n = i.get("name", "")
            if n and n not in seen_ingr:
                all_ingr.append(i); seen_ingr.add(n)
        for e in (rec.get("excluded_ingredients") or []):
            n = e.get("name", "")
            if n and n not in seen_excl:
                all_excl.append(e); seen_excl.add(n)
        for t in (rec.get("care_tips") or []):
            if t and t not in all_tips:
                all_tips.append(t)

    st.markdown(
        '<div style="font-size:17px;font-weight:700;color:#0F2447;margin:24px 0 16px;">'
        '💊 통합 추천 결과</div>',
        unsafe_allow_html=True,
    )

    col_cat, col_ingr, col_excl, col_tips = st.columns(4, gap="medium")

    with col_cat:
        cat_src = _b64_src("report/category.png")
        cat_img = (
            f'<img src="{cat_src}" width="28" style="object-fit:contain;margin-bottom:8px;">'
            if cat_src else "🛍️"
        )
        chips = " ".join(category_chip_html(c) for c in all_cats) if all_cats else "—"
        st.markdown(f"""
        <div class="ds-rec-card">
            <div style="text-align:center;margin-bottom:12px;">{cat_img}</div>
            <div class="ds-card-title" style="font-size:14px;">추천 카테고리</div>
            <div style="line-height:2;">{chips}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_ingr:
        ingr_src = _b64_src("report/ingredient.png")
        ingr_img = (
            f'<img src="{ingr_src}" width="28" style="object-fit:contain;margin-bottom:8px;">'
            if ingr_src else "💧"
        )
        chips = " ".join(ingredient_chip_html(i.get("name","")) for i in all_ingr) if all_ingr else "—"
        st.markdown(f"""
        <div class="ds-rec-card">
            <div style="text-align:center;margin-bottom:12px;">{ingr_img}</div>
            <div class="ds-card-title" style="font-size:14px;">추천 성분</div>
            <div style="line-height:2;">{chips}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_excl:
        excl_src = _b64_src("report/excluded.png")
        excl_img = (
            f'<img src="{excl_src}" width="28" style="object-fit:contain;margin-bottom:8px;">'
            if excl_src else "🚫"
        )
        chips = " ".join(
            ingredient_chip_html(e.get("name",""), excluded=True, reason_type=e.get("reason_type"))
            for e in all_excl
        ) if all_excl else "없음"
        st.markdown(f"""
        <div class="ds-rec-card">
            <div style="text-align:center;margin-bottom:12px;">{excl_img}</div>
            <div class="ds-card-title" style="font-size:14px;">제외 성분</div>
            <div style="line-height:2;">{chips}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_tips:
        tip_src = _b64_src("report/care_tip.png")
        tip_img = (
            f'<img src="{tip_src}" width="28" style="object-fit:contain;margin-bottom:8px;">'
            if tip_src else "💡"
        )
        tips_html = "".join(
            f'<li style="font-size:13px;color:#374151;margin-bottom:5px;">{t}</li>'
            for t in all_tips[:6]
        ) if all_tips else "<li>—</li>"
        st.markdown(f"""
        <div class="ds-rec-card">
            <div style="text-align:center;margin-bottom:12px;">{tip_img}</div>
            <div class="ds-card-title" style="font-size:14px;">관리 팁</div>
            <ul style="margin:0;padding-left:16px;">{tips_html}</ul>
        </div>
        """, unsafe_allow_html=True)
