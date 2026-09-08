"""부위 x 각도별 crop 품질 점검 시트 생성기.

목적
----
AI-Hub 라벨의 bbox 로 부위를 잘랐을 때, 각도(angle 0~8) 와 촬영기기(device 0~2)
별로 crop 이 어떻게 달라지는지 한 장의 대조 시트로 확인한다.

현재 crop.py 의 ALLOWED_ANGLES 는 볼 기준 angle {0,1,2,3} / {0,1,2,5} 만 허용해
전체의 46% 만 쓰고 있고, 스마트패드/스마트폰 전용 각도인 7·8 은 통째로 빠진다.
필터를 넓힐지 판단하려면 7·8 crop 이 실제로 해당 부위를 담고 있는지 눈으로 봐야 한다.

산출물
------
results/angle_check/facepart{N}_grid.jpg   각도별 crop 대조 시트
results/angle_check/bbox_stats.csv         부위 x 각도 x 기기별 bbox 통계

사용법
------
    python notebooks/jh/angle_check.py                    # 볼(5,6), 각도별 5장씩
    python notebooks/jh/angle_check.py --faceparts 1 2 8  # 이마/미간/턱
    python notebooks/jh/angle_check.py --samples 8 --split train
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACEPART_NAME = {
    0: "full_face", 1: "forehead", 2: "glabella", 3: "l_eye", 4: "r_eye",
    5: "l_cheek", 6: "r_cheek", 7: "lips", 8: "chin",
}
DEVICE_NAME = {0: "DSLR", 1: "PAD", 2: "PHONE"}
ANGLE_NAME = {
    0: "front", 1: "up", 2: "down", 3: "L15", 4: "L30",
    5: "R15", 6: "R30", 7: "side-a", 8: "side-b",
}

# crop.py 와 동일한 기본 여백
DEFAULT_MARGIN = (0.06, 0.08)

CELL = 190          # 시트 셀 한 변(px)
CAPTION_H = 34      # 셀 아래 캡션 높이
PAD = 8


# --------------------------------------------------------------------------- #
# 경로 해석
# --------------------------------------------------------------------------- #
def split_roots(data_root: Path, split: str) -> tuple[Path, Path]:
    """(label_root, image_root) 반환."""
    if split == "train":
        return (
            data_root / "Training" / "02.라벨링데이터" / "TL",
            data_root / "Training" / "01.원천데이터" / "TS",
        )
    return (
        data_root / "Validation" / "02.라벨링데이터" / "VL",
        data_root / "Validation" / "01.원천데이터" / "VS",
    )


def guess_image_path(json_path: Path, label_root: Path, image_root: Path,
                     filename: str) -> Path | None:
    """라벨 경로 구조를 그대로 미러링해 원본 이미지 경로를 추정한다.

    TL/1. 디지털카메라/0002/0002_01_F_00.json
      -> TS/1. 디지털카메라/0002/0002_01_F.jpg
    """
    rel = json_path.parent.relative_to(label_root)
    direct = image_root / rel / filename
    if direct.exists():
        return direct

    # 확장자만 다른 경우
    stem = Path(filename).stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        cand = image_root / rel / f"{stem}{ext}"
        if cand.exists():
            return cand

    # 사람 폴더 안에서 탐색 (마지막 수단)
    folder = image_root / rel
    if folder.is_dir():
        for cand in folder.iterdir():
            if cand.stem == stem:
                return cand
    return None


# --------------------------------------------------------------------------- #
# 수집
# --------------------------------------------------------------------------- #
def collect(label_root: Path, faceparts: set[int], per_bucket_cap: int,
            seed: int) -> tuple[dict, dict]:
    """(buckets, stats) 반환.

    buckets[(facepart, angle)] = [record, ...]   시트용 표본
    stats[(facepart, angle, device)] = [(w, h), ...]  bbox 통계용 전량
    """
    rng = random.Random(seed)
    buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    stats: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    seen = 0

    for jp in label_root.rglob("*.json"):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        seen += 1

        images = data.get("images") or {}
        info = data.get("info") or {}
        fp = images.get("facepart")
        if fp not in faceparts:
            continue

        bbox = images.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue

        angle = images.get("angle")
        device = images.get("device")
        stats[(fp, angle, device)].append((w, h))

        rec = {
            "json_path": jp,
            "filename": info.get("filename", ""),
            "person": str(info.get("id", "")),
            "facepart": fp,
            "angle": angle,
            "device": device,
            "bbox": (x1, y1, x2, y2),
            "img_w": images.get("width"),
            "img_h": images.get("height"),
        }
        # 저수지 표본추출 — 앞쪽 사람에게 쏠리지 않게
        key = (fp, angle)
        pool = buckets[key]
        if len(pool) < per_bucket_cap:
            pool.append(rec)
        else:
            j = rng.randint(0, len(pool))
            if j < per_bucket_cap:
                pool[j] = rec

    print(f"  라벨 {seen:,}건 훑음")
    return buckets, stats


# --------------------------------------------------------------------------- #
# crop
# --------------------------------------------------------------------------- #
def expand_bbox(bbox, img_w, img_h, margin=DEFAULT_MARGIN):
    x1, y1, x2, y2 = bbox
    mx = int(round((x2 - x1) * margin[0]))
    my = int(round((y2 - y1) * margin[1]))
    return (
        max(0, x1 - mx), max(0, y1 - my),
        min(img_w, x2 + mx), min(img_h, y2 + my),
    )


def load_font(size: int):
    for path in (r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_grid(facepart: int, buckets, label_root: Path, image_root: Path,
               samples: int, out_path: Path) -> int:
    angles = sorted({a for (fp, a) in buckets if fp == facepart},
                    key=lambda x: (x is None, x))
    if not angles:
        print(f"  facepart {facepart}: 표본 없음")
        return 0

    font = load_font(13)
    font_hdr = load_font(16)

    header_w = 118
    grid_w = header_w + samples * (CELL + PAD) + PAD
    grid_h = PAD + len(angles) * (CELL + CAPTION_H + PAD) + 40

    sheet = Image.new("RGB", (grid_w, grid_h), "#f2f3f5")
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD, 10),
              f"facepart {facepart} ({FACEPART_NAME.get(facepart, '?')})"
              f"  —  각 행 = angle, 셀 = bbox+여백 crop 을 정사각형으로 resize한 모습",
              fill="#1b2a44", font=font_hdr)

    made = 0
    y = 40
    for angle in angles:
        recs = list(buckets[(facepart, angle)])
        random.shuffle(recs)

        devs = {r["device"] for r in recs}
        dev_txt = "/".join(DEVICE_NAME.get(d, str(d)) for d in sorted(devs))
        draw.rectangle([PAD, y, header_w - 4, y + CELL + CAPTION_H], fill="#e2e6ee")
        draw.text((PAD + 8, y + 12), f"angle {angle}", fill="#0f2350", font=font_hdr)
        draw.text((PAD + 8, y + 36), ANGLE_NAME.get(angle, ""), fill="#42557c", font=font)
        draw.text((PAD + 8, y + 56), dev_txt, fill="#7b8aa6", font=font)
        draw.text((PAD + 8, y + 76), f"n={len(recs)}", fill="#7b8aa6", font=font)

        x = header_w
        placed = 0
        for rec in recs:
            if placed >= samples:
                break
            img_path = guess_image_path(rec["json_path"], label_root, image_root,
                                        rec["filename"])
            if img_path is None:
                continue
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    box = expand_bbox(rec["bbox"], im.width, im.height)
                    crop = im.crop(box)
            except Exception:
                continue

            cw, ch = crop.size
            ratio = cw / ch if ch else 0
            cell = crop.resize((CELL, CELL), Image.LANCZOS)
            sheet.paste(cell, (x, y))
            draw.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline="#c7d5ea")

            draw.rectangle([x, y + CELL, x + CELL, y + CELL + CAPTION_H], fill="#ffffff")
            draw.text((x + 6, y + CELL + 4),
                      f"{cw}x{ch}  ratio {ratio:.2f}", fill="#0f2350", font=font)
            draw.text((x + 6, y + CELL + 18),
                      f"{DEVICE_NAME.get(rec['device'], rec['device'])} · id {rec['person']}",
                      fill="#7b8aa6", font=font)

            x += CELL + PAD
            placed += 1
            made += 1

        if placed == 0:
            draw.text((header_w + 6, y + CELL // 2),
                      "원본 이미지를 찾지 못했습니다 (TS/VS 압축 해제 확인)",
                      fill="#b91c1c", font=font)
        y += CELL + CAPTION_H + PAD

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
    print(f"  시트 저장: {out_path}  (crop {made}장)")
    return made


# --------------------------------------------------------------------------- #
# 통계
# --------------------------------------------------------------------------- #
def write_stats(stats, out_csv: Path, faceparts):
    rows = []
    for (fp, angle, dev), wh in sorted(stats.items(),
                                       key=lambda kv: (kv[0][0], kv[0][1] if kv[0][1] is not None else -1,
                                                       kv[0][2] if kv[0][2] is not None else -1)):
        ws = [w for w, _ in wh]
        hs = [h for _, h in wh]
        ratios = [w / h for w, h in wh if h]
        areas = [w * h for w, h in wh]
        rows.append({
            "facepart": fp,
            "facepart_name": FACEPART_NAME.get(fp, ""),
            "angle": angle,
            "angle_name": ANGLE_NAME.get(angle, ""),
            "device": dev,
            "device_name": DEVICE_NAME.get(dev, ""),
            "n": len(wh),
            "median_w": int(statistics.median(ws)),
            "median_h": int(statistics.median(hs)),
            "median_ratio": round(statistics.median(ratios), 3) if ratios else "",
            "median_area": int(statistics.median(areas)),
            "min_w": min(ws),
            "min_h": min(hs),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  통계 저장: {out_csv}")

    for fp in sorted(faceparts):
        sub = [r for r in rows if r["facepart"] == fp]
        if not sub:
            continue
        print(f"\n  ── facepart {fp} ({FACEPART_NAME.get(fp,'')}) bbox 통계 ──")
        print(f"  {'angle':>5} {'device':>7} {'n':>6} {'중앙 W x H':>14} {'비율':>7} {'최소 W':>7}")
        for r in sub:
            print(f"  {str(r['angle']):>5} {r['device_name']:>7} {r['n']:>6} "
                  f"{r['median_w']:>6} x {r['median_h']:<5} {str(r['median_ratio']):>7} {r['min_w']:>7}")

    # 현재 필터가 버리는 양
    ALLOWED = {5: {0, 1, 2, 3}, 6: {0, 1, 2, 5}}
    print("\n  ── 현재 crop.py ALLOWED_ANGLES 기준 사용률 ──")
    for fp in sorted(faceparts):
        if fp not in ALLOWED:
            continue
        tot = sum(r["n"] for r in rows if r["facepart"] == fp)
        use = sum(r["n"] for r in rows if r["facepart"] == fp and r["angle"] in ALLOWED[fp])
        if tot:
            print(f"  facepart {fp}: 전체 {tot:,} → 사용 {use:,} "
                  f"({use / tot * 100:.1f}%) | 버려짐 {tot - use:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description="부위 x 각도별 crop 품질 점검 시트")
    ap.add_argument("--data-root", type=Path,
                    default=PROJECT_ROOT / "data" / "raw" / "aihub_skin")
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "results" / "angle_check")
    ap.add_argument("--split", choices=["train", "val"], default="val",
                    help="기본은 val — 가볍고 빠릅니다.")
    ap.add_argument("--faceparts", type=int, nargs="+", default=[5, 6])
    ap.add_argument("--samples", type=int, default=5, help="각도당 표시할 crop 수")
    ap.add_argument("--pool", type=int, default=40, help="각도당 후보 표본 수")
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    label_root, image_root = split_roots(args.data_root, args.split)
    print(f"\nsplit={args.split}")
    print(f"  label: {label_root}")
    print(f"  image: {image_root}")
    if not label_root.exists():
        raise SystemExit(f"라벨 폴더가 없습니다: {label_root}")
    if not image_root.exists():
        raise SystemExit(f"원본 이미지 폴더가 없습니다: {image_root}")

    print("\n[1/3] 라벨 수집")
    buckets, stats = collect(label_root, set(args.faceparts), args.pool, args.seed)

    print("\n[2/3] bbox 통계")
    write_stats(stats, args.out_dir / f"bbox_stats_{args.split}.csv", set(args.faceparts))

    print("\n[3/3] 대조 시트 생성")
    random.seed(args.seed)
    for fp in args.faceparts:
        build_grid(fp, buckets, label_root, image_root, args.samples,
                   args.out_dir / f"facepart{fp}_{args.split}_grid.jpg")

    print(f"\n완료. 결과 폴더를 열어 확인하세요:\n  {args.out_dir}\n")
    print("확인 포인트")
    print("  1. angle 7·8 (패드/폰) crop 이 해당 부위를 제대로 담고 있는가")
    print("  2. 정사각형 resize 후 찌그러짐이 심한 각도가 있는가 (ratio 열 참고)")
    print("  3. 해상도가 지나치게 작은 각도가 있는가 (최소 W 열 참고)")


if __name__ == "__main__":
    main()
