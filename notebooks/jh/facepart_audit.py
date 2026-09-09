"""눈가(facepart 3·4) 좌우 정의와 각도 필터를 라벨 JSON 통계로 진단한다.

왜 이 스크립트인가
------------------
"한쪽 눈이 검출되지 않는다"는 관찰을 숫자로 확정하기 위한 것이다.
볼(facepart 5·6)에서 각도 필터가 직관과 정반대였던 전례가 있으므로,
설계를 고치기 전에 라벨 자체를 먼저 검증한다.

검증하는 가설
-------------
  H1  "왼쪽/오른쪽"이 피사체 기준인가 관찰자 기준인가
      -> 정면(angle 0)에서 facepart 3·4 bbox 중심 x 를 비교
  H2  기기에 따라 좌우가 반전되는가 (폰 셀피 미러링)
      -> 같은 통계를 device 0/1/2 로 분리
  H3  눈가 각도 필터가 볼과 같은 구조인가
      -> facepart 3·4 의 각도별 bbox 기하 통계

bbox 형식
---------
docs/labeling_codes_guide.md 는 [x, y, w, h] 라고 적고 있으나 이는 오류다.
실측 결과 [x1, y1, x2, y2] 이고(그렇게 읽지 않으면 71%가 이미지 밖으로 나간다),
crop.py 와 angle_check.py 도 [x1, y1, x2, y2] 로 읽는다. 여기서도 그렇게 읽는다.

산출물
------
results/facepart_audit.csv          레코드별 정규화 기하 통계
results/eye_bbox_check_*.jpg        정면 이미지에 facepart 3·4 bbox 를 겹쳐 그린 대조본

사용법
------
    uv run python notebooks/jh/facepart_audit.py
    uv run python notebooks/jh/facepart_audit.py --split val --no-images
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from angle_check import (  # noqa: E402
    ANGLE_NAME,
    DEVICE_NAME,
    FACEPART_NAME,
    guess_image_path,
    load_font,
    split_roots,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# facepart -> annotations 키. 하나라도 non-null 이면 라벨이 있는 것으로 본다.
ANNOTATION_KEYS = {
    0: ["acne"],
    1: ["forehead_pigmentation", "forehead_wrinkle"],
    2: ["glabellus_wrinkle"],
    3: ["l_perocular_wrinkle"],
    4: ["r_perocular_wrinkle"],
    5: ["l_cheek_pigmentation", "l_cheek_pore"],
    6: ["r_cheek_pigmentation", "r_cheek_pore"],
    7: ["lip_dryness"],
    8: ["chin_sagging"],
}

CSV_COLUMNS = [
    "split", "id", "device", "angle", "facepart",
    "img_w", "img_h", "x1", "y1", "x2", "y2",
    "cx_norm", "cy_norm", "ratio", "area_frac", "has_label",
    "json_path", "filename",
]


# --------------------------------------------------------------------------- #
# 수집
# --------------------------------------------------------------------------- #
def collect(label_root: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    seen = bad = 0

    for jp in label_root.rglob("*.json"):
        seen += 1
        if seen % 20000 == 0:
            print(f"    {seen:,}건 훑음 (유효 {len(rows):,})", flush=True)
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            bad += 1
            continue

        images = data.get("images") or {}
        info = data.get("info") or {}
        ann = data.get("annotations") or {}

        bbox = images.get("bbox")
        W, H = images.get("width"), images.get("height")
        fp = images.get("facepart")
        if not (isinstance(bbox, list) and len(bbox) == 4 and W and H):
            bad += 1
            continue

        x1, y1, x2, y2 = (int(v) for v in bbox)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            bad += 1
            continue

        keys = ANNOTATION_KEYS.get(fp, [])
        has_label = any(ann.get(k) is not None for k in keys)

        rows.append({
            "split": split,
            "id": str(info.get("id", "")),
            "device": images.get("device"),
            "angle": images.get("angle"),
            "facepart": fp,
            "img_w": W, "img_h": H,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx_norm": round((x1 + w / 2) / W, 5),
            "cy_norm": round((y1 + h / 2) / H, 5),
            "ratio": round(w / h, 4),
            "area_frac": round((w * h) / (W * H), 6),
            "has_label": int(has_label),
            "json_path": str(jp),
            "filename": info.get("filename", ""),
        })

    print(f"    완료: {seen:,}건 중 유효 {len(rows):,}, 제외 {bad:,}")
    return rows


def med(values) -> float | str:
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 4) if vals else "-"


def sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r[k] == v]
    return out


# --------------------------------------------------------------------------- #
# H1 - 좌우 규약
# --------------------------------------------------------------------------- #
def report_h1(rows) -> dict:
    print()
    print("=" * 78)
    print("H1  좌우 규약 - 정면(angle 0) bbox 중심 x")
    print("=" * 78)
    print("  판정 기준: facepart 3(왼쪽 눈가)의 cx > 0.5 이면 피사체 기준,")
    print("             cx < 0.5 이면 관찰자 기준(이미지 왼쪽에 찍힘).")
    print()
    print(f"  {'facepart':>8} {'이름':<10} {'n':>7} {'cx 중앙값':>10} {'위치':>8}")

    result = {}
    for fp in (3, 4, 5, 6):
        sub = sel(rows, facepart=fp, angle=0)
        m = med([r["cx_norm"] for r in sub])
        side = "-" if m == "-" else ("이미지 오른쪽" if m > 0.5 else "이미지 왼쪽")
        result[fp] = m
        print(f"  {fp:>8} {FACEPART_NAME.get(fp, ''):<10} {len(sub):>7} {str(m):>10} {side:>8}")

    print()
    eye_conv = "피사체 기준" if (result[3] != "-" and result[3] > 0.5) else "관찰자 기준"
    cheek_conv = "피사체 기준" if (result[5] != "-" and result[5] > 0.5) else "관찰자 기준"
    print(f"  눈가(3/4) 규약: {eye_conv}")
    print(f"  볼  (5/6) 규약: {cheek_conv}")
    print(f"  두 부위가 같은 규약인가: {'예' if eye_conv == cheek_conv else '아니오 [!] 심각'}")
    return result


# --------------------------------------------------------------------------- #
# H2 - 기기별 반전
# --------------------------------------------------------------------------- #
def report_h2(rows) -> dict:
    print()
    print("=" * 78)
    print("H2  기기별 좌우 반전 - 정면(angle 0), 기기별 bbox 중심 x")
    print("=" * 78)
    print("  판정 기준: 같은 facepart 에서 기기에 따라 0.5 기준 부호가 뒤집히면 미러링.")
    print()
    print(f"  {'facepart':>8} {'device':>8} {'이름':<7} {'n':>7} {'cx 중앙값':>10} {'위치':>8}")

    result: dict = defaultdict(dict)
    for fp in (3, 4, 5, 6):
        for dev in (0, 1, 2):
            sub = sel(rows, facepart=fp, angle=0, device=dev)
            m = med([r["cx_norm"] for r in sub])
            side = "-" if m == "-" else ("오른쪽" if m > 0.5 else "왼쪽")
            result[fp][dev] = m
            print(f"  {fp:>8} {dev:>8} {DEVICE_NAME.get(dev, ''):<7} {len(sub):>7} "
                  f"{str(m):>10} {side:>8}")
        print()

    flipped = []
    for fp, per_dev in result.items():
        sides = {d: (m > 0.5) for d, m in per_dev.items() if m != "-"}
        if len(set(sides.values())) > 1:
            flipped.append((fp, sides))
    if flipped:
        print("  [!] 기기별 반전 발견:")
        for fp, sides in flipped:
            txt = "  ".join(f"{DEVICE_NAME.get(d,'')}={'오른쪽' if s else '왼쪽'}"
                            for d, s in sorted(sides.items()))
            print(f"      facepart {fp}: {txt}")
    else:
        print("  반전 없음 - 모든 기기에서 같은 쪽을 가리킨다.")
    return result


# --------------------------------------------------------------------------- #
# H3 - 각도 필터
# --------------------------------------------------------------------------- #
def report_h3(rows) -> dict:
    print()
    print("=" * 78)
    print("H3  각도별 bbox 기하 - 비율(w/h)이 클수록 그 부위가 카메라 정면")
    print("=" * 78)
    print("  볼(5·6)은 확정된 정답이 있으므로 대조군이다.")
    print("  crop.py ALLOWED_ANGLES = {5: {0,1,2,5,6,8}, 6: {0,1,2,3,4,7}}")
    print()

    result: dict = defaultdict(dict)
    for fp in (5, 6, 3, 4):
        print(f"  -- facepart {fp} ({FACEPART_NAME.get(fp, '')}) --")
        print(f"  {'angle':>5} {'이름':<7} {'n':>7} {'ratio':>7} {'area':>9} "
              f"{'cx':>7} {'중앙 W':>7}")
        base = sel(rows, facepart=fp)
        for ang in range(9):
            sub = [r for r in base if r["angle"] == ang]
            if not sub:
                continue
            r_med = med([r["ratio"] for r in sub])
            a_med = med([r["area_frac"] for r in sub])
            c_med = med([r["cx_norm"] for r in sub])
            w_med = int(statistics.median([r["x2"] - r["x1"] for r in sub]))
            result[fp][ang] = {"n": len(sub), "ratio": r_med,
                               "area": a_med, "cx": c_med, "w": w_med}
            print(f"  {ang:>5} {ANGLE_NAME.get(ang, ''):<7} {len(sub):>7} "
                  f"{str(r_med):>7} {str(a_med):>9} {str(c_med):>7} {w_med:>7}")
        print()
    return result


# --------------------------------------------------------------------------- #
# 라벨 커버리지
# --------------------------------------------------------------------------- #
def report_coverage(rows) -> None:
    print()
    print("=" * 78)
    print("라벨 커버리지 - facepart 별 레코드 수와 실제 라벨 보유 비율")
    print("=" * 78)
    print(f"  {'fp':>3} {'이름':<11} {'레코드':>9} {'라벨 있음':>10} {'비율':>7}")
    for fp in range(9):
        sub = sel(rows, facepart=fp)
        if not sub:
            continue
        got = sum(r["has_label"] for r in sub)
        print(f"  {fp:>3} {FACEPART_NAME.get(fp, ''):<11} {len(sub):>9,} "
              f"{got:>10,} {got/len(sub)*100:>6.1f}%")

    print()
    n3, n4 = len(sel(rows, facepart=3)), len(sel(rows, facepart=4))
    l3 = sum(r["has_label"] for r in sel(rows, facepart=3))
    l4 = sum(r["has_label"] for r in sel(rows, facepart=4))
    print(f"  좌우 눈가 표본: fp3={n3:,} (라벨 {l3:,}) / fp4={n4:,} (라벨 {l4:,})")
    if max(n3, n4):
        gap = abs(n3 - n4) / max(n3, n4) * 100
        print(f"  레코드 수 차이 {gap:.2f}% -> {'불균형 [!]' if gap > 5 else '균형'}")


# --------------------------------------------------------------------------- #
# 육안 확인
# --------------------------------------------------------------------------- #
def visual_check(rows, data_root: Path, out_dir: Path, per_device: int = 1) -> None:
    print()
    print("=" * 78)
    print("육안 확인 - 정면 이미지에 facepart 3(파랑)·4(주황) bbox 겹쳐 그리기")
    print("=" * 78)

    # 같은 원본 사진을 공유하는 fp3/fp4 레코드를 (split, id, device, angle) 로 묶는다.
    pairs: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        if r["facepart"] in (3, 4) and r["angle"] == 0:
            pairs[(r["split"], r["id"], r["device"], r["angle"])][r["facepart"]] = r

    out_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(40)
    made_per_dev: dict = defaultdict(int)

    for key in sorted(pairs, key=lambda k: (k[2] if k[2] is not None else 9, k[1])):
        split, pid, dev, _ = key
        both = pairs[key]
        if 3 not in both or 4 not in both:
            continue
        if made_per_dev[dev] >= per_device:
            continue

        rec = both[3]
        label_root, image_root = split_roots(data_root, split)
        img_path = guess_image_path(Path(rec["json_path"]), label_root,
                                    image_root, rec["filename"])
        if img_path is None:
            continue
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        draw = ImageDraw.Draw(im)
        for fp, color in ((3, "#1e6fff"), (4, "#ff7a1e")):
            r = both[fp]
            draw.rectangle([r["x1"], r["y1"], r["x2"], r["y2"]], outline=color, width=10)
            draw.text((r["x1"], max(0, r["y1"] - 48)),
                      f"fp{fp} {FACEPART_NAME.get(fp,'')} cx={r['cx_norm']:.3f}",
                      fill=color, font=font)
        draw.text((20, 20),
                  f"{DEVICE_NAME.get(dev, dev)}  id={pid}  angle=0  ({split})",
                  fill="#000000", font=font)

        im.thumbnail((1100, 1100))
        out = out_dir / f"eye_bbox_check_{DEVICE_NAME.get(dev, dev)}_{pid}.jpg"
        im.save(out, quality=90)
        made_per_dev[dev] += 1
        print(f"  저장: {out.name}  fp3 cx={both[3]['cx_norm']:.3f}  "
              f"fp4 cx={both[4]['cx_norm']:.3f}")

    if not made_per_dev:
        print("  [!] 그릴 표본을 찾지 못했습니다 (원본 이미지 압축 해제 확인)")


def main() -> None:
    ap = argparse.ArgumentParser(description="눈가 좌우 정의·각도 필터 진단")
    ap.add_argument("--data-root", type=Path,
                    default=PROJECT_ROOT / "data" / "raw" / "aihub_skin")
    ap.add_argument("--split", choices=["train", "val", "all"], default="all")
    ap.add_argument("--out-csv", type=Path,
                    default=PROJECT_ROOT / "results" / "facepart_audit.csv")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results")
    ap.add_argument("--no-images", action="store_true", help="육안 확인 생략")
    args = ap.parse_args()

    splits = ["train", "val"] if args.split == "all" else [args.split]

    rows: list[dict] = []
    print("[1/3] 라벨 수집")
    for sp in splits:
        label_root, _ = split_roots(args.data_root, sp)
        print(f"  {sp}: {label_root}")
        if not label_root.exists():
            raise SystemExit(f"라벨 폴더가 없습니다: {label_root}")
        rows += collect(label_root, sp)

    print()
    print("[2/3] CSV 저장")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        wri = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        wri.writeheader()
        for r in rows:
            wri.writerow({k: r[k] for k in CSV_COLUMNS})
    print(f"  저장: {args.out_csv}  ({len(rows):,}행)")

    print()
    print("[3/3] 진단")
    report_h1(rows)
    report_h2(rows)
    report_h3(rows)
    report_coverage(rows)
    if not args.no_images:
        visual_check(rows, args.data_root, args.out_dir)


if __name__ == "__main__":
    main()
