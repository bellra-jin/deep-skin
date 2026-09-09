"""face_detector 의 좌우 재할당(_reassign_lr) 단위 테스트.

YOLO 가중치가 저장소에 없으므로 합성 검출 결과로 로직만 검증한다.

배경
----
좌우 눈가는 내부 특징이 거의 없는 서로 대칭인 피부 패치라 검출기가 좌우를 혼동한다.
혼동하면 detect_best_per_part 가 부위명당 한 박스만 남기므로 한쪽 키가 통째로
사라지고 전체 이미지 fallback 으로 넘어간다. 좌우 규약은 관찰자(이미지) 기준으로
확정돼 있으므로(라벨 108,070건 전수 검증), 좌우는 x좌표로 다시 정한다.

확인 항목
---------
  1. 두 박스가 모두 left_eye 로 예측됨      -> x 순서대로 left/right 로 갈라진다
  2. 두 박스가 올바르게 left/right 로 예측됨 -> 결과가 바뀌지 않는다
  3. 두 박스가 좌우 뒤바뀌어 예측됨          -> x 기준으로 교정된다
  4. 박스가 1개뿐                          -> 원래 예측 유지, lr_source="detector"
  5. 박스가 3개                            -> 신뢰도 상위 2개만 남고 x 순서로 할당
  6. 눈가와 볼이 섞여 있음                  -> 서로 간섭하지 않는다
  7. 쌍이 아닌 부위(forehead, chin 등)      -> 그대로 통과한다
  8. detect_best_per_part 통합             -> 한쪽 눈 키가 사라지지 않는다
"""

from scripts.face_detector import FaceDetector, _cx, _reassign_lr


def _det(part, x1, x2, conf=0.9):
    """합성 검출 하나. y 는 판정에 쓰이지 않으므로 고정값."""
    return {
        "class_id": 0,
        "class_name": part,
        "raw_part_name": part,
        "confidence": conf,
        "bbox_xyxy": [float(x1), 100.0, float(x2), 300.0],
    }


def _by_name(dets):
    return {d["raw_part_name"]: d for d in dets}


# --------------------------------------------------------------------------- #
# 1. 두 박스가 모두 같은 클래스로 예측된 경우 (핵심 케이스)
# --------------------------------------------------------------------------- #
def test_both_predicted_as_left_eye_are_split_by_x():
    dets = [
        _det("left_eye", 100, 200, conf=0.91),   # 이미지 왼쪽
        _det("left_eye", 800, 900, conf=0.88),   # 이미지 오른쪽
    ]
    out = _by_name(_reassign_lr(dets))

    assert set(out) == {"left_eye", "right_eye"}
    assert _cx(out["left_eye"]) == 150.0
    assert _cx(out["right_eye"]) == 850.0
    assert out["left_eye"]["lr_source"] == "x_position"
    assert out["right_eye"]["lr_source"] == "x_position"


# --------------------------------------------------------------------------- #
# 2. 이미 올바른 경우 - 결과가 바뀌면 안 된다
# --------------------------------------------------------------------------- #
def test_correct_prediction_is_preserved():
    dets = [
        _det("left_eye", 100, 200, conf=0.91),
        _det("right_eye", 800, 900, conf=0.88),
    ]
    out = _by_name(_reassign_lr(dets))

    assert _cx(out["left_eye"]) == 150.0
    assert _cx(out["right_eye"]) == 850.0
    # 신뢰도도 원래 박스의 것을 그대로 들고 있어야 한다
    assert out["left_eye"]["confidence"] == 0.91
    assert out["right_eye"]["confidence"] == 0.88


# --------------------------------------------------------------------------- #
# 3. 좌우가 뒤바뀐 경우 - 교정돼야 한다
# --------------------------------------------------------------------------- #
def test_swapped_prediction_is_corrected():
    dets = [
        _det("right_eye", 100, 200, conf=0.91),  # 왼쪽에 있는데 right 로 예측
        _det("left_eye", 800, 900, conf=0.88),   # 오른쪽에 있는데 left 로 예측
    ]
    out = _by_name(_reassign_lr(dets))

    assert _cx(out["left_eye"]) == 150.0
    assert _cx(out["right_eye"]) == 850.0
    # 원래 right 로 예측됐던 박스(conf 0.91)가 left 로 넘어와야 한다
    assert out["left_eye"]["confidence"] == 0.91
    assert out["right_eye"]["confidence"] == 0.88


# --------------------------------------------------------------------------- #
# 4. 한쪽만 검출된 경우 - 원래 예측을 존중
# --------------------------------------------------------------------------- #
def test_single_box_keeps_detector_prediction():
    dets = [_det("right_eye", 100, 200, conf=0.7)]
    out = _reassign_lr(dets)

    assert len(out) == 1
    # x 가 왼쪽이어도 재할당하지 않는다 - 짝이 없으면 어느 쪽인지 알 수 없다
    assert out[0]["raw_part_name"] == "right_eye"
    assert out[0]["lr_source"] == "detector"


# --------------------------------------------------------------------------- #
# 5. 3개 이상 - 신뢰도 상위 2개만
# --------------------------------------------------------------------------- #
def test_three_boxes_keep_top_two_by_confidence():
    dets = [
        _det("left_eye", 800, 900, conf=0.88),   # 유지 (오른쪽)
        _det("left_eye", 100, 200, conf=0.91),   # 유지 (왼쪽)
        _det("right_eye", 450, 550, conf=0.30),  # 탈락
    ]
    out = _by_name(_reassign_lr(dets))

    assert set(out) == {"left_eye", "right_eye"}
    assert _cx(out["left_eye"]) == 150.0
    assert _cx(out["right_eye"]) == 850.0
    assert 0.30 not in {d["confidence"] for d in out.values()}


# --------------------------------------------------------------------------- #
# 6. 눈가와 볼이 섞여 있어도 서로 간섭하지 않는다
# --------------------------------------------------------------------------- #
def test_eye_and_cheek_pairs_do_not_interfere():
    dets = [
        _det("left_eye", 100, 200, conf=0.91),
        _det("left_eye", 800, 900, conf=0.88),    # 둘 다 left_eye 로 오예측
        _det("right_cheek", 150, 350, conf=0.80),
        _det("right_cheek", 700, 900, conf=0.75),  # 둘 다 right_cheek 로 오예측
    ]
    out = _by_name(_reassign_lr(dets))

    assert set(out) == {"left_eye", "right_eye", "left_cheek", "right_cheek"}
    assert _cx(out["left_eye"]) == 150.0
    assert _cx(out["right_eye"]) == 850.0
    assert _cx(out["left_cheek"]) == 250.0
    assert _cx(out["right_cheek"]) == 800.0


# --------------------------------------------------------------------------- #
# 7. 쌍이 아닌 부위는 그대로 통과
# --------------------------------------------------------------------------- #
def test_non_paired_parts_pass_through_untouched():
    dets = [
        _det("forehead", 300, 700, conf=0.95),
        _det("chin", 350, 650, conf=0.85),
        _det("lips", 400, 600, conf=0.80),
        _det("glabella", 420, 580, conf=0.77),
    ]
    out = _reassign_lr(dets)

    assert len(out) == 4
    for d in out:
        assert "lr_source" not in d
    assert {d["raw_part_name"] for d in out} == {
        "forehead", "chin", "lips", "glabella",
    }


def test_input_dicts_are_not_mutated():
    dets = [
        _det("left_eye", 100, 200, conf=0.91),
        _det("left_eye", 800, 900, conf=0.88),
    ]
    _reassign_lr(dets)

    # 원본은 그대로여야 한다 (_reassign_lr 은 복사본을 만든다)
    assert [d["raw_part_name"] for d in dets] == ["left_eye", "left_eye"]
    assert all("lr_source" not in d for d in dets)


# --------------------------------------------------------------------------- #
# 8. detect_best_per_part 통합 - 한쪽 눈 키가 사라지지 않는다
# --------------------------------------------------------------------------- #
class _StubDetector(FaceDetector):
    """detect() 만 합성 결과로 바꾼 FaceDetector. YOLO 를 로드하지 않는다."""

    def __init__(self, dets):
        super().__init__(model_path="unused")
        self._dets = dets

    def detect(self, image, conf=0.25, iou=0.5, imgsz=1280):
        return list(self._dets)


def test_detect_best_per_part_recovers_missing_eye():
    # 재할당이 없으면 left_eye 하나만 남고 right_eye 는 통째로 사라진다.
    dets = [
        _det("left_eye", 100, 200, conf=0.91),
        _det("left_eye", 800, 900, conf=0.88),
        _det("forehead", 300, 700, conf=0.95),
    ]
    best = _StubDetector(dets).detect_best_per_part(image=None)

    assert set(best) == {"left_eye", "right_eye", "forehead"}
    assert _cx(best["left_eye"]) == 150.0
    assert _cx(best["right_eye"]) == 850.0


def test_detect_best_per_part_keeps_highest_confidence_for_non_paired():
    dets = [
        _det("forehead", 300, 700, conf=0.60),
        _det("forehead", 310, 710, conf=0.95),
    ]
    best = _StubDetector(dets).detect_best_per_part(image=None)

    assert set(best) == {"forehead"}
    assert best["forehead"]["confidence"] == 0.95
