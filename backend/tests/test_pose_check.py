"""정면 여부 판정(pose_check) 단위 테스트.

확인 항목
---------
  1. 미간이 두 볼 중점에 있으면 frontal
  2. 임계값 0.05 경계 - 안쪽은 frontal, 바깥은 turned
  3. 좌우 어느 쪽으로 밀려도 대칭적으로 판정된다
  4. 얼굴 크기·촬영 거리에 무관하다 (볼 간격으로 정규화)
  5. 필요한 부위가 하나라도 없으면 face_not_detected
     - 이 경우를 frontal 로 취급하면 최악의 입력이 무사통과한다
  6. detected_parts 가 None/빈 리스트/깨진 값이어도 죽지 않는다
  7. 상태별로 안내 문구가 붙는다 (frontal 은 없음)
"""

from app.services.pose_check import (
    FRONTALITY_THRESHOLD,
    STATUS_FACE_NOT_DETECTED,
    STATUS_FRONTAL,
    STATUS_TURNED,
    frontality_offset,
    judge_pose,
)


def _det(name, x1, x2, y1=100.0, y2=300.0):
    return {"raw_part_name": name, "confidence": 0.9, "bbox_xyxy": [x1, y1, x2, y2]}


def _face(glabella_cx, l_cx=250.0, r_cx=750.0, half=50.0):
    """볼 중점 500, 간격 500 인 얼굴. 미간 중심만 옮겨 가며 쓴다."""
    return [
        _det("glabella", glabella_cx - half, glabella_cx + half),
        _det("left_cheek", l_cx - half, l_cx + half),
        _det("right_cheek", r_cx - half, r_cx + half),
        _det("forehead", 400.0, 600.0),
    ]


# --------------------------------------------------------------------------- #
def test_centered_glabella_is_frontal():
    out = judge_pose(_face(500.0))
    assert out["status"] == STATUS_FRONTAL
    assert out["score"] == 0.0
    assert out["message"] is None


def test_threshold_boundary():
    # 볼 간격 500 이므로 편차 25px = dx 0.05 (경계 안쪽, 포함)
    assert judge_pose(_face(525.0))["status"] == STATUS_FRONTAL
    # 편차 30px = dx 0.06 (경계 바깥)
    assert judge_pose(_face(530.0))["status"] == STATUS_TURNED


def test_symmetric_in_both_directions():
    left = judge_pose(_face(400.0))
    right = judge_pose(_face(600.0))
    assert left["status"] == right["status"] == STATUS_TURNED
    assert left["score"] == -right["score"]


def test_scale_invariant():
    """얼굴이 2배로 크게 찍혀도 같은 판정이어야 한다."""
    small = frontality_offset(_face(560.0, l_cx=250.0, r_cx=750.0, half=50.0))
    big = frontality_offset(_face(1120.0, l_cx=500.0, r_cx=1500.0, half=100.0))
    assert small is not None and big is not None
    assert abs(small - big) < 1e-9


# --------------------------------------------------------------------------- #
# 판정 불가를 정면으로 취급하지 않는다
# --------------------------------------------------------------------------- #
def test_missing_cheek_is_not_frontal():
    dets = [d for d in _face(500.0) if d["raw_part_name"] != "right_cheek"]
    out = judge_pose(dets)
    assert out["status"] == STATUS_FACE_NOT_DETECTED
    assert out["score"] is None


def test_missing_glabella_is_not_frontal():
    dets = [d for d in _face(500.0) if d["raw_part_name"] != "glabella"]
    assert judge_pose(dets)["status"] == STATUS_FACE_NOT_DETECTED


def test_degenerate_inputs_do_not_crash():
    for bad in (None, [], "not a list", [{}], [{"raw_part_name": "glabella"}],
                [_det("glabella", 0, 0), _det("left_cheek", 0, 0),
                 _det("right_cheek", 0, 0)]):
        out = judge_pose(bad)
        assert out["status"] == STATUS_FACE_NOT_DETECTED
        assert out["score"] is None


def test_non_numeric_bbox_is_handled():
    dets = _face(500.0)
    dets[0]["bbox_xyxy"] = ["x", "y", "z", "w"]
    assert judge_pose(dets)["status"] == STATUS_FACE_NOT_DETECTED


# --------------------------------------------------------------------------- #
def test_messages_present_only_when_action_needed():
    assert judge_pose(_face(500.0))["message"] is None
    for dets in (_face(700.0), []):
        out = judge_pose(dets)
        assert out["message"]
        # 아래를 본 사진이 약점이라 위아래를 포함한 문구여야 한다
        assert "눈높이" in out["message"]


def test_threshold_is_the_calibrated_value():
    """270장 실측 보정에서 오탐 0 이던 값. 바꾸려면 재보정이 필요하다."""
    assert FRONTALITY_THRESHOLD == 0.05
