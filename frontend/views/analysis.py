import base64
import os
import threading

import streamlit as st

from services import analysis_api, api_client
from components.common import show_error, handle_401
from components import upload_panel
from styles.css_loader import load_css


_STATUS_MESSAGES = [
    "이미지를 확인하고 있어요.",
    "얼굴 영역을 감지하고 있어요.",
    "피부 상태를 분석하고 있어요.",
    "피부 고민 부위를 확인하고 있어요.",
    "맞춤 성분과 제품 타입을 추천하고 있어요.",
    "리포트를 생성하고 있어요.",
]


def show():
    _ensure_state()
    _inject_analysis_css()

    state = st.session_state.get("analysis_flow_state", "idle")

    with st.container(key="ds_analysis_page"):
        _render_header()

        if state == "analyzing":
            _hide_idle_analysis_ui()
            _render_steps("analyzing")
            _start_api_if_needed()
            _analyzing_fragment()
            return

        if state == "pose_warning":
            _hide_idle_analysis_ui()
            _render_steps("complete")
            _render_pose_warning()
            return

        if state == "complete":
            _hide_idle_analysis_ui()
            _render_steps("complete")
            _render_complete()
            return

        if state == "error":
            _hide_idle_analysis_ui()
            _render_steps("error")
            _render_error()
            return

        _render_steps("idle")
        image_data = upload_panel.render()
        st.markdown('<div class="ds-analysis-guide-wrap"></div>', unsafe_allow_html=True)
        upload_panel.render_guide_cards()
        _render_start_button(image_data)


def _ensure_state():
    st.session_state.setdefault("analysis_flow_state", "idle")
    st.session_state.setdefault("analysis_image_data", None)
    st.session_state.setdefault("analysis_error_message", "")
    st.session_state.setdefault("analysis_pose_check", None)


def _start_api_if_needed():
    """API 호출을 백그라운드 스레드로 한 번만 시작한다."""
    if "_api_result" in st.session_state:
        return

    st.session_state["_analysis_step"] = 0
    token = st.session_state.get("access_token")
    image_data = st.session_state.get("analysis_image_data")
    api_result: dict = {"ok": None, "error": None, "status_code": None,
                        "session_id": None, "pose_check": None}

    def _run():
        s = analysis_api.create_session(token, "피부 분석")
        if api_client.is_error(s):
            api_result["ok"] = False
            api_result["error"] = api_client.get_error_message(s)
            api_result["status_code"] = s.get("_status")
            return
        sid = s.get("id")
        api_result["session_id"] = sid
        u = analysis_api.upload_image(token, sid, image_data["bytes"], image_data["name"])
        if api_client.is_error(u):
            api_result["ok"] = False
            api_result["error"] = api_client.get_error_message(u)
            api_result["status_code"] = u.get("_status")
            return
        api_result["pose_check"] = u.get("pose_check")
        api_result["ok"] = True

    threading.Thread(target=_run, daemon=True).start()
    st.session_state["_api_result"] = api_result


@st.fragment(run_every=1.2)
def _analyzing_fragment():
    """페이지 플래시 없이 1.2초마다 독립 업데이트되는 분석 진행 UI."""
    step = st.session_state.get("_analysis_step", 0)
    api_result = st.session_state.get("_api_result")

    # API 완료 감지 → 전체 페이지 rerun으로 상태 전환
    if api_result is not None and api_result.get("ok") is not None:
        ok = api_result["ok"]
        if api_result.get("session_id"):
            st.session_state["current_session_id"] = api_result["session_id"]
        del st.session_state["_api_result"]
        st.session_state.pop("_analysis_step", None)
        if ok:
            st.session_state["analysis_image_data"] = None
            # 정면이 아니면 리포트로 바로 넘기지 않고 한 번 알린다.
            # 차단이 아니라 경고다 - 사용자가 "이대로 진행"을 고를 수 있다.
            pose = api_result.get("pose_check") or {}
            st.session_state["analysis_pose_check"] = pose
            if pose.get("status") in ("turned", "face_not_detected"):
                st.session_state["analysis_flow_state"] = "pose_warning"
            else:
                st.session_state["analysis_flow_state"] = "complete"
        else:
            if api_result.get("status_code") == 401:
                handle_401()
                return
            st.session_state["analysis_error_message"] = (
                api_result.get("error") or "이미지를 다시 확인한 후 재시도해주세요."
            )
            st.session_state["analysis_flow_state"] = "error"
        st.rerun()
        return

    # 진행 단계 HTML 조합
    steps_done = min(step, len(_STATUS_MESSAGES))
    rows = []
    for i, msg in enumerate(_STATUS_MESSAGES):
        if i < steps_done:
            cls, mark = "ds-ai-progress-done", "✓"
        elif i == steps_done:
            cls, mark = "ds-ai-progress-active", str(i + 1)
        else:
            cls, mark = "ds-ai-progress-pending", str(i + 1)
        rows.append(
            f'<div class="ds-ai-progress-row {cls}">'
            f'<span class="ds-ai-progress-mark">{mark}</span>'
            f'{msg}</div>'
        )

    st.markdown(
        f"""
        <div class="ds-analysis-state-card">
            <div class="ds-state-badge ds-state-badge-running">✨ AI 분석 중</div>
            <div class="ds-ai-orb">
                <div class="ds-ai-orb-ring"></div>
                <div class="ds-ai-orb-core">✦</div>
            </div>
            <div class="ds-state-title">AI가 피부를 분석하고 있어요</div>
            <div class="ds-state-desc">잠시만 기다려주세요.</div>
            <div class="ds-ai-current-box">
                <div class="ds-progress-list">{"".join(rows)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.session_state["_analysis_step"] = min(step + 1, len(_STATUS_MESSAGES))


def _hide_idle_analysis_ui():
    """Hide stale upload-screen DOM while Streamlit is rendering a state screen."""
    st.markdown(
        """
        <style>
        .st-key-ds_camera_card,
        .st-key-ds_upload_card,
        [data-testid="stHorizontalBlock"]:has(.st-key-ds_camera_card),
        [data-testid="stHorizontalBlock"]:has(.st-key-ds_upload_card),
        [data-testid="stHorizontalBlock"]:has(.ds-analysis-guide-card),
        [data-testid="stElementContainer"]:has(.ds-analysis-guide-wrap),
        [data-testid="stElementContainer"]:has(.ds-analysis-guide-card),
        [data-testid="stElementContainer"]:has(.ds-analysis-privacy),
        .ds-preview-frame,
        .ds-analysis-guide-wrap,
        .ds-analysis-guide-card,
        .st-key-ds_analysis_action,
        .ds-analysis-privacy {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            pointer-events: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header():
    st.markdown(
        """
        <div class="ds-analysis-header">
            <div>
                <div class="ds-analysis-title">피부 분석 시작</div>
                <div class="ds-analysis-subtitle">
                    사진을 업로드하고 AI가 당신의 피부 상태를 정밀하게 분석해드려요.
                </div>
            </div>
            <div class="ds-security-pill">🛡 보안 인증 완료</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _steps_html(state: str) -> str:
    if state == "idle":
        statuses = ["active", "pending", "pending"]
        labels = ["사진 업로드", "AI 분석", "리포트 확인"]
    elif state == "analyzing":
        statuses = ["done", "active", "pending"]
        labels = ["사진 업로드", "AI 분석 중", "리포트 확인"]
    elif state == "complete":
        statuses = ["done", "done", "active"]
        labels = ["사진 업로드", "AI 分석 완료", "리포트 확인"]
    else:
        statuses = ["done", "active", "pending"]
        labels = ["사진 업로드", "AI 분석", "리포트 확인"]

    items = []
    for idx, (status, label) in enumerate(zip(statuses, labels), start=1):
        if idx > 1:
            line_cls = "ds-step-connector ds-step-connector-done" if statuses[idx - 2] == "done" else "ds-step-connector"
            items.append(f'<div class="{line_cls}"></div>')
        inner = "✓" if status == "done" else str(idx)
        items.append(
            f"""
            <div class="ds-step-item ds-step-{status}">
                <div class="ds-step-dot">{inner}</div>
                <div class="ds-step-text">{label}</div>
            </div>
            """
        )
    return '<div class="ds-analysis-steps">' + "".join(items) + "</div>"


def _render_steps(state: str):
    st.markdown(_steps_html(state), unsafe_allow_html=True)


def _render_start_button(image_data: dict | None):
    with st.container(key="ds_analysis_action"):
        if st.button("✦  분석 시작", key="analysis_start", width="stretch", disabled=False):
            if not image_data:
                show_error("분석할 이미지를 먼저 업로드해주세요.")
            else:
                st.session_state["analysis_image_data"] = image_data
                st.session_state["analysis_flow_state"] = "analyzing"
                st.session_state["analysis_error_message"] = ""
                st.rerun()
    st.markdown(
        """
        <div class="ds-analysis-privacy">
            🛡 업로드된 이미지는 분석 후 즉시 안전하게 삭제되며, 외부에 저장되지 않습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_pose_warning():
    """정면이 아닌 사진에 경고를 띄운다. 업로드를 막지는 않는다.

    판정이 틀렸을 때 사용자가 아무것도 못 하게 되면 안 되므로
    "이대로 진행"을 항상 남겨 둔다. 그대로 진행하면 리포트의
    "추정값" 배지가 두 번째 방어선이 된다.
    """
    pose = st.session_state.get("analysis_pose_check") or {}
    message = pose.get("message") or "사진을 다시 확인해주세요."
    score = pose.get("score")
    detail = "" if score is None else f"좌우 대칭 편차 {abs(score):.3f} (기준 {pose.get('threshold', 0.05)})"

    st.markdown(
        f"""
        <div style="padding:16px 18px;margin:8px 0 18px;border-radius:12px;
                    background:#FFF8E1;border:1px solid #F0DFA0;color:#8A6D0B;">
            <div style="font-size:15px;font-weight:700;margin-bottom:6px;">
                분석은 끝났지만, 사진을 한 번 더 확인해주세요
            </div>
            <div style="font-size:13px;line-height:1.7;">{message}</div>
            <div style="font-size:11px;color:#A08A3C;margin-top:8px;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_retake, col_proceed = st.columns([1, 1], gap="medium")
    with col_retake:
        if st.button("다시 찍기", use_container_width=True, key="ds_pose_retake"):
            st.session_state["analysis_pose_check"] = None
            st.session_state["analysis_image_data"] = None
            st.session_state["analysis_flow_state"] = "idle"
            st.rerun()
    with col_proceed:
        if st.button("이대로 진행", type="primary", use_container_width=True,
                     key="ds_pose_proceed"):
            st.session_state["analysis_flow_state"] = "complete"
            st.rerun()


def _render_complete():
    with st.container(key="ds_complete_card"):
        st.markdown(
            """
            <div class="ds-state-badge ds-state-badge-complete">✓ 분석 완료</div>
            <div class="ds-complete-orb">✓</div>
            <div class="ds-state-title">분석이 완료되었습니다</div>
            <div class="ds-state-desc">
                맞춤 피부 리포트가 준비되었어요.<br>
                아래 버튼을 눌러 분석 결과를 확인해주세요.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("리포트 확인하기 →", key="go_report_after_analysis", width="stretch"):
            st.session_state["current_page"] = "report"
            st.session_state["analysis_flow_state"] = "idle"
            st.rerun()


def _render_error():
    msg = st.session_state.get("analysis_error_message") or "이미지를 다시 확인한 후 재시도해주세요."
    st.markdown(
        f"""
        <div class="ds-analysis-state-card ds-analysis-error-card">
            <div class="ds-state-badge ds-state-badge-error">분석 실패</div>
            <div class="ds-state-title">분석 중 문제가 발생했습니다</div>
            <div class="ds-state-desc">{msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        if st.button("다시 분석하기", key="retry_analysis", width="stretch"):
            st.session_state["analysis_flow_state"] = "analyzing"
            st.rerun()
    with c2:
        if st.button("이미지 다시 선택", key="back_to_upload", width="stretch"):
            st.session_state["analysis_flow_state"] = "idle"
            st.session_state["analysis_image_data"] = None
            st.rerun()


def _b64_src(rel: str) -> str:
    path = os.path.join("assets", "icons", rel)
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def _inject_analysis_css():
    load_css("analysis.css")
