"""정면 여부 판정기를 라벨 각도로 실측 보정한다.

착상
----
YOLO 검출 결과 자체가 각도 판정기다. 두 눈가가 다 잡히면 정면, 하나만 잡히면 측면이다.
데이터셋에 정답 각도 코드가 붙어 있으므로 판정 규칙을 감으로 정하지 않고 실측으로
보정할 수 있다.

두 지표
-------
  (a) 눈가 쌍 존재 여부  - 이진 신호. 단순하지만 경계에서 불안정할 수 있다.
  (b) 좌우 대칭 점수     - 연속 신호. 미간이 두 볼 중점에서 얼마나 벗어났는가.

왜 이게 성립하는가
------------------
검출기는 좌우를 혼동하지 않는다(혼동행렬 l_eye <-> r_eye 오분류 0건, mAP50 0.994).
한쪽 눈가가 없는 것은 검출 실패가 아니라, 그 각도에서 가려진 쪽에 데이터셋이
애초에 라벨을 붙이지 않았고 검출기가 그대로 학습했기 때문이다.
즉 "눈가 하나만 검출됨"은 신뢰할 수 있는 측면 신호다.

산출물
------
results/frontality_calibration.csv

사용법
------
    uv run --with ultralytics python notebooks/jh/frontality_calibration.py
    uv run python notebooks/jh/frontality_calibration.py --analyze-only
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from angle_check import DEVICE_NAME, guess_image_path, split_roots  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLO_CKPT = PROJECT_ROOT / "backend" / "model" / "yolo_facecrop_best.pt"

ALL_PARTS = [
    "forehead", "glabella", "left_eye", "right_eye",
    "left_cheek", "right_cheek", "lips", "chin",
]

# 정답 각도의 이분. 0 정면 / 1 위 / 2 아래 는 정면으로 본다.
FRONT_ANGLES = {0, 1, 2}

CSV_COLUMNS = [
    "image_path", "device", "angle", "is_front",
    "n_eyes_detected", "frontality_score",
    "detected_parts", "missing_parts",
]


def frontality_score(dets_by_part: Dict[str, Dict[str, Any]]) -> Optional[float]:
    """0에 가까울수록 정면. 필요한 부위가 없으면 None.

    미간이 두 볼 중심의 중점에 있으면 정면이고,
    고개가 돌수록 한쪽으로 밀린다. 볼 간격으로 나눠 얼굴 크기에 무관하게 만든다.
    """
    need = ("glabella", "left_cheek", "right_cheek")
    if not all(k in dets_by_part for k in need):
        return None
    cx = lambda k: (dets_by_part[k]["bbox_xyxy"][0] + dets_by_part[k]["bbox_xyxy"][2]) / 2
    lc, rc = cx("left_cheek"), cx("right_cheek")
    span = abs(rc - lc)
    if span < 1e-6:
        return None
    return (cx("glabella") - (lc + rc) / 2) / span


# --------------------------------------------------------------------------- #
# 수집
# --------------------------------------------------------------------------- #
def pick_samples(audit_csv: Path, per_angle: int, seed: int) -> list[dict]:
    """각도별로 고르게, 기기를 섞어서 뽑는다."""
    rows = list(csv.DictReader(audit_csv.open(encoding="utf-8-sig")))
    # facepart 0 (전체 얼굴) 레코드가 이미지 1장에 정확히 하나씩 대응한다.
    pool: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["split"] != "val" or r["facepart"] != "0":
            continue
        pool[(int(r["angle"]), int(r["device"]))].append(r)

    rng = random.Random(seed)
    picks: list[dict] = []
    for angle in range(9):
        devs = [d for d in (0, 1, 2) if pool.get((angle, d))]
        if not devs:
            continue
        # 각도당 per_angle 장을 사용 가능한 기기에 균등 배분
        share = max(1, per_angle // len(devs))
        for d in devs:
            lst = pool[(angle, d)][:]
            rng.shuffle(lst)
            picks += lst[:share]
    return picks


def collect(picks: list[dict], data_root: Path, out_csv: Path) -> None:
    from scripts.face_detector import FaceDetector  # noqa: E402

    det = FaceDetector(model_path=str(YOLO_CKPT))
    if not det.load():
        raise SystemExit(f"YOLO 로드 실패: {YOLO_CKPT}")

    rows: list[dict] = []
    t0 = last = time.time()
    for i, r in enumerate(picks, 1):
        label_root, image_root = split_roots(data_root, r["split"])
        p = guess_image_path(Path(r["json_path"]), label_root, image_root, r["filename"])
        if p is None:
            continue
        best = det.detect_best_per_part(str(p))
        n_eyes = sum(1 for k in ("left_eye", "right_eye") if k in best)
        score = frontality_score(best)
        detected = [k for k in ALL_PARTS if k in best]
        missing = [k for k in ALL_PARTS if k not in best]
        angle = int(r["angle"])
        rows.append({
            "image_path": str(p),
            "device": int(r["device"]),
            "angle": angle,
            "is_front": int(angle in FRONT_ANGLES),
            "n_eyes_detected": n_eyes,
            "frontality_score": "" if score is None else round(score, 5),
            "detected_parts": ",".join(detected),
            "missing_parts": ",".join(missing),
        })
        if time.time() - last > 20:
            el = time.time() - t0
            rate = i / el
            print(f"  {i}/{len(picks)}  ({i/len(picks)*100:.1f}%)  {rate:.2f} img/s  "
                  f"남은 시간 약 {(len(picks)-i)/max(rate,1e-9)/60:.1f}분", flush=True)
            last = time.time()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        wri = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        wri.writeheader()
        wri.writerows(rows)
    print(f"  완료 - {len(rows)}장, {(time.time()-t0)/60:.1f}분")
    print(f"  저장: {out_csv}")


# --------------------------------------------------------------------------- #
# 분석
# --------------------------------------------------------------------------- #
def confusion(rows, predict) -> tuple[int, int, int, int]:
    """(정면->정면, 정면->측면, 측면->정면, 측면->측면)"""
    tp = fn = fp = tn = 0
    for r in rows:
        actual = r["is_front"] == "1"
        pred = predict(r)
        if actual and pred:
            tp += 1
        elif actual and not pred:
            fn += 1
        elif not actual and pred:
            fp += 1
        else:
            tn += 1
    return tp, fn, fp, tn


def show(name, rows, predict) -> float:
    tp, fn, fp, tn = confusion(rows, predict)
    n = tp + fn + fp + tn
    acc = (tp + tn) / n if n else 0.0
    print(f"  -- {name} --")
    print(f"    {'':<14}{'정면으로 판정':>13}{'측면으로 판정':>13}")
    print(f"    {'실제 정면':<14}{tp:>13}{fn:>13}")
    print(f"    {'실제 측면':<14}{fp:>13}{tn:>13}")
    print(f"    정확도 {acc*100:.1f}%   오탐(측면을 정면으로) {fp}건   미탐(정면을 측면으로) {fn}건")
    return acc


def analyze(csv_path: Path) -> None:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    print(f"\n표본 {len(rows)}장 "
          f"(정면 {sum(r['is_front']=='1' for r in rows)} / "
          f"측면 {sum(r['is_front']=='0' for r in rows)})")

    print("\n== 각도별 검출 현황 ==")
    print(f"  {'angle':>5} {'n':>4} {'눈가2개':>8} {'눈가1개':>8} {'눈가0개':>8} "
          f"{'대칭점수 중앙':>13} {'점수없음':>8}")
    for ang in range(9):
        sub = [r for r in rows if int(r["angle"]) == ang]
        if not sub:
            continue
        sc = [abs(float(r["frontality_score"])) for r in sub if r["frontality_score"]]
        med = f"{statistics.median(sc):.4f}" if sc else "-"
        print(f"  {ang:>5} {len(sub):>4} "
              f"{sum(r['n_eyes_detected']=='2' for r in sub):>8} "
              f"{sum(r['n_eyes_detected']=='1' for r in sub):>8} "
              f"{sum(r['n_eyes_detected']=='0' for r in sub):>8} "
              f"{med:>13} {len(sub)-len(sc):>8}")

    print("\n== 지표 (a) 눈가 쌍 존재 여부 ==")
    acc_a = show("두 눈가가 모두 검출되면 정면", rows,
                 lambda r: r["n_eyes_detected"] == "2")

    print("\n== 지표 (b) 좌우 대칭 점수 - 임계값별 정확도 ==")
    print("    점수를 못 내면(부위 부족) 측면으로 본다.")
    print(f"    {'임계값':>8} {'정확도':>8} {'오탐':>6} {'미탐':>6}")
    best_thr, best_acc = None, -1.0
    for thr in [round(0.01 * i, 2) for i in range(1, 26)]:
        def pred(r, t=thr):
            s = r["frontality_score"]
            return bool(s) and abs(float(s)) <= t
        tp, fn, fp, tn = confusion(rows, pred)
        acc = (tp + tn) / len(rows)
        print(f"    {thr:>8.2f} {acc*100:>7.1f}% {fp:>6} {fn:>6}")
        if acc > best_acc:
            best_thr, best_acc = thr, acc
    print(f"\n    권장 임계값: |score| <= {best_thr}  (정확도 {best_acc*100:.1f}%)")

    print("\n== 두 지표 AND (둘 다 정면이라고 해야 정면) ==")
    def pred_and(r):
        s = r["frontality_score"]
        return r["n_eyes_detected"] == "2" and bool(s) and abs(float(s)) <= best_thr
    acc_and = show(f"눈가 2개 AND |score| <= {best_thr}", rows, pred_and)

    print("\n== 위/아래 각도(1·2)가 정면 쪽에 붙는가 ==")
    for ang in (0, 1, 2):
        sub = [r for r in rows if int(r["angle"]) == ang]
        if not sub:
            continue
        a = sum(r["n_eyes_detected"] == "2" for r in sub) / len(sub)
        b = sum(bool(r["frontality_score"]) and abs(float(r["frontality_score"])) <= best_thr
                for r in sub) / len(sub)
        c = sum(pred_and(r) for r in sub) / len(sub)
        print(f"  angle {ang}: (a) {a*100:5.1f}%   (b) {b*100:5.1f}%   AND {c*100:5.1f}%  (n={len(sub)})")

    print(f"\n요약: (a) {acc_a*100:.1f}%   (b) {best_acc*100:.1f}% @ {best_thr}   "
          f"AND {acc_and*100:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description="정면 판정기 실측 보정")
    ap.add_argument("--data-root", type=Path,
                    default=PROJECT_ROOT / "data" / "raw" / "aihub_skin")
    ap.add_argument("--audit-csv", type=Path,
                    default=PROJECT_ROOT / "results" / "facepart_audit.csv")
    ap.add_argument("--out-csv", type=Path,
                    default=PROJECT_ROOT / "results" / "frontality_calibration.csv")
    ap.add_argument("--per-angle", type=int, default=30,
                    help="각도당 표본 수 (전체가 300장을 넘지 않게 한다)")
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--analyze-only", action="store_true",
                    help="YOLO 를 다시 돌리지 않고 기존 CSV 만 분석한다")
    args = ap.parse_args()

    if not args.analyze_only:
        picks = pick_samples(args.audit_csv, args.per_angle, args.seed)
        print(f"표본 {len(picks)}장 선정 (각도당 최대 {args.per_angle})")
        by_dev = defaultdict(int)
        for r in picks:
            by_dev[int(r["device"])] += 1
        print("  기기 분포: " + "  ".join(
            f"{DEVICE_NAME.get(d, d)} {n}" for d, n in sorted(by_dev.items())))
        collect(picks, args.data_root, args.out_csv)

    analyze(args.out_csv)


if __name__ == "__main__":
    main()
