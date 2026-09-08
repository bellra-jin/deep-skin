"""Build cheek crops and metadata CSV files from the AI-Hub skin dataset."""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Optional

from PIL import Image


logger = logging.getLogger(__name__)

FACEPART_SIDE = {5: "left", 6: "right"}
FACEPART_DIR = {5: "l_cheek", 6: "r_cheek"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 각도 코드는 "얼굴이 도는 방향" 기준이다.
#   R15/R30 (5,6) -> 얼굴이 오른쪽으로 회전 -> 왼볼이 카메라 정면으로 온다
#   L15/L30 (3,4) -> 얼굴이 왼쪽으로 회전   -> 오른볼이 카메라 정면으로 온다
#   7/8 은 스마트패드·스마트폰 전용 측면. 7=오른볼용, 8=왼볼용.
#
# 초기 구현은 이를 반대로 해석해 각 부위에서 가장 선명한 각도를 버리고
# 가장 눌린 각도를 학습에 넣고 있었다. bbox 통계와 crop 육안 검증으로 확인 후 수정.
# (검증 스크립트: notebooks/jh/angle_check.py)
ALLOWED_ANGLES = {
    5: {0, 1, 2, 5, 6, 8},  # 왼볼:   정면·위·아래 + R15·R30 + 패드/폰 측면(8)
    6: {0, 1, 2, 3, 4, 7},  # 오른볼: 정면·위·아래 + L15·L30 + 패드/폰 측면(7)
}

DEFAULT_MARGIN = (0.06, 0.08)
# 남긴 각도는 모두 가로 폭이 충분하다(비율 1퍼센타일 0.49 이상).
# 각도별 여백 보정이 필요 없어 비워 둔다.
ANGLE_MARGIN: dict[int, tuple[float, float]] = {}

# 각도별 규칙 대신 전 각도 공통 가드.
# Training 15,444건 실측: min_width=150 -> 0.48% 제거, min_ratio=0.45 -> 0.06% 제거.
# 주로 스마트폰 저해상도 촬영분의 꼬리를 잘라낸다.
DEFAULT_QUALITY_RULE = {"min_width": 150, "min_ratio": 0.45, "min_area": 60000}
ANGLE_QUALITY_RULES: dict[int, dict] = {}

CSV_COLUMNS = [
    "image_path",
    "original_image_path",
    "original_filename",
    "json_path",
    "id",
    "gender",
    "age",
    "date",
    "skin_type",
    "sensitive",
    "device",
    "image_width",
    "image_height",
    "angle",
    "facepart",
    "side",
    "bbox",
    "pore_label",
    "pigmentation_label",
    "split",
    "info_json",
    "images_json",
    "annotations_json",
    "equipment_json",
    "raw_json",
]

ISSUE_LOG_COLUMNS = ["json_path", "reason", "detail"]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Build cheek crop images and metadata CSV files from AI-Hub JSON."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "data" / "raw" / "aihub_skin",
        help="Root directory for the raw AI-Hub dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "cropped",
        help="Directory where cropped cheek images will be stored.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=project_root / "data" / "processed",
        help="Directory where train/val metadata CSV files will be stored.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=project_root / "results" / "cheek",
        help="Directory for crop samples and issue logs.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional cap for train samples. Leave unset for full dataset.",
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=None,
        help="Optional cap for val samples. Leave unset for full dataset.",
    )
    parser.add_argument(
        "--save-samples",
        type=int,
        default=10,
        help="How many crop samples to copy into the results directory per split.",
    )
    return parser.parse_args()


def _load_json(json_path: Path) -> Optional[dict]:
    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("JSON parse failed: %s (%s)", json_path, exc)
        return None


def _build_image_index(image_root: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    duplicate_count = 0

    for path in image_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.name in image_index:
            duplicate_count += 1
            continue
        image_index[path.name] = path

    logger.info(
        "indexed source images=%d under %s (duplicates ignored=%d)",
        len(image_index),
        image_root,
        duplicate_count,
    )
    return image_index


def _find_image(filename: str, image_index: dict[str, Path]) -> Optional[Path]:
    return image_index.get(filename)


def _extract_labels(annotations: dict, facepart: int) -> Optional[tuple[int, int]]:
    if facepart == 5:
        pore = annotations.get("l_cheek_pore")
        pigmentation = annotations.get("l_cheek_pigmentation")
    else:
        pore = annotations.get("r_cheek_pore")
        pigmentation = annotations.get("r_cheek_pigmentation")

    if pore is None or pigmentation is None:
        return None
    return int(pore), int(pigmentation)


def _to_json_text(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _expand_bbox(
    bbox: list[int],
    img_w: int,
    img_h: int,
    angle: Optional[int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    margin_x_ratio, margin_y_ratio = ANGLE_MARGIN.get(angle, DEFAULT_MARGIN)
    margin_x = int(round(width * margin_x_ratio))
    margin_y = int(round(height * margin_y_ratio))
    return (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(img_w, x2 + margin_x),
        min(img_h, y2 + margin_y),
    )


def _passes_quality_filter(angle: Optional[int], width: int, height: int) -> bool:
    rules = ANGLE_QUALITY_RULES.get(angle, DEFAULT_QUALITY_RULE)
    if rules is None:
        return True

    ratio = width / height
    area = width * height
    return (
        width >= rules["min_width"]
        and ratio >= rules["min_ratio"]
        and area >= rules["min_area"]
    )


def _issue(json_path: Path, reason: str, detail: str = "") -> dict[str, str]:
    return {
        "json_path": str(json_path),
        "reason": reason,
        "detail": detail,
    }


def process_json_file(
    json_path: Path,
    image_index: dict[str, Path],
    output_dir: Path,
    split: str,
    min_crop_px: int = 10,
) -> tuple[str, Optional[dict], dict[str, str]]:
    data = _load_json(json_path)
    if data is None:
        return "error", None, _issue(json_path, "json_parse_failed")

    info = data.get("info", {})
    images = data.get("images", {})
    annotations = data.get("annotations", {})
    equipment = data.get("equipment", {})

    facepart = images.get("facepart")
    if facepart not in (5, 6):
        return "skip", None, _issue(json_path, "non_cheek_facepart", str(facepart))

    angle = images.get("angle")
    if angle not in ALLOWED_ANGLES[facepart]:
        return "skip", None, _issue(json_path, "angle_filtered", str(angle))

    bbox = images.get("bbox")
    if not bbox or len(bbox) != 4:
        return "error", None, _issue(json_path, "invalid_bbox", str(bbox))

    filename = info.get("filename")
    if not filename:
        return "error", None, _issue(json_path, "missing_filename")

    labels = _extract_labels(annotations, facepart)
    if labels is None:
        return "error", None, _issue(json_path, "missing_labels")
    pore_label, pigmentation_label = labels

    img_path = _find_image(filename, image_index)
    if img_path is None:
        return "error", None, _issue(json_path, "missing_source_image", filename)

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as exc:
        return "error", None, _issue(json_path, "image_load_failed", str(exc))

    img_w, img_h = img.size
    x1, y1, x2, y2 = _expand_bbox(bbox, img_w, img_h, angle)
    bbox_clipped = x1 == 0 or y1 == 0 or x2 == img_w or y2 == img_h

    if x2 <= x1 or y2 <= y1:
        return "error", None, _issue(json_path, "collapsed_bbox_after_clipping")

    crop_img = img.crop((x1, y1, x2, y2))
    crop_w, crop_h = crop_img.size
    if not _passes_quality_filter(angle, crop_w, crop_h):
        return "skip", None, _issue(
            json_path,
            "quality_filtered",
            f"angle={angle}, width={crop_w}, height={crop_h}",
        )
    if crop_w < min_crop_px or crop_h < min_crop_px:
        return "error", None, _issue(
            json_path,
            "crop_too_small",
            f"width={crop_w}, height={crop_h}",
        )

    out_subdir = output_dir / split / FACEPART_DIR[facepart]
    out_subdir.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem
    user_id = str(info.get("id", "unk"))
    out_name = f"{user_id}_{stem}_{facepart:02d}.jpg"
    out_path = out_subdir / out_name
    crop_img.save(out_path, quality=95)

    detail = "bbox_clipped" if bbox_clipped else ""
    row = {
        "image_path": str(out_path),
        "original_image_path": str(img_path),
        "original_filename": filename,
        "json_path": str(json_path),
        "id": user_id,
        "gender": info.get("gender", ""),
        "age": info.get("age", ""),
        "date": info.get("date", ""),
        "skin_type": info.get("skin_type", ""),
        "sensitive": info.get("sensitive", ""),
        "device": images.get("device", ""),
        "image_width": images.get("width", ""),
        "image_height": images.get("height", ""),
        "angle": images.get("angle", ""),
        "facepart": facepart,
        "side": FACEPART_SIDE[facepart],
        "bbox": str([x1, y1, x2, y2]),
        "pore_label": pore_label,
        "pigmentation_label": pigmentation_label,
        "split": split,
        "info_json": _to_json_text(info),
        "images_json": _to_json_text(images),
        "annotations_json": _to_json_text(annotations),
        "equipment_json": _to_json_text(equipment),
        "raw_json": _to_json_text(data),
    }
    return "success", row, _issue(json_path, "success", detail)


def build_cheek_dataset(
    data_root: Path,
    output_dir: Path,
    split: str,
    max_samples: Optional[int] = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    label_subdir = "TL" if split == "train" else "VL"
    image_subdir = "TS" if split == "train" else "VS"
    data_subdir = "Training" if split == "train" else "Validation"

    label_root = data_root / data_subdir / "02.라벨링데이터" / label_subdir
    image_root = data_root / data_subdir / "01.원천데이터" / image_subdir

    json_files = sorted(label_root.rglob("*.json"))
    logger.info("[%s] json files=%d", split, len(json_files))

    image_index = _build_image_index(image_root)

    rows: list[dict] = []
    skip_records: list[dict] = []
    error_records: list[dict] = []

    for json_path in json_files:
        status, row, issue = process_json_file(json_path, image_index, output_dir, split)
        if status == "success":
            rows.append(row or {})
            if max_samples is not None and len(rows) >= max_samples:
                break
        elif status == "skip":
            skip_records.append(issue)
        else:
            error_records.append(issue)

    logger.info(
        "[%s] done success=%d skip=%d error=%d",
        split,
        len(rows),
        len(skip_records),
        len(error_records),
    )
    return rows, skip_records, error_records


def save_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("saved csv: %s (%d rows)", csv_path, len(rows))


def save_issue_log(records: list[dict], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ISSUE_LOG_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "json_path": record.get("json_path", ""),
                    "reason": record.get("reason", ""),
                    "detail": record.get("detail", ""),
                }
            )
    logger.info("saved issue log: %s (%d rows)", log_path, len(records))


def save_error_log(errors: list[str] | list[dict], log_path: Path) -> None:
    if errors and isinstance(errors[0], str):
        records = [_issue(Path(path), "legacy_error") for path in errors]  # type: ignore[index]
    else:
        records = errors  # type: ignore[assignment]
    save_issue_log(records, log_path)


def save_skip_log(skips: list[dict], log_path: Path) -> None:
    save_issue_log(skips, log_path)


def save_crop_samples(rows: list[dict], samples_dir: Path, n: int = 10) -> None:
    import shutil

    samples_dir.mkdir(parents=True, exist_ok=True)
    for row in rows[:n]:
        src = Path(row["image_path"])
        if src.exists():
            shutil.copy2(src, samples_dir / src.name)
    logger.info("saved crop samples: %s (%d files)", samples_dir, min(n, len(rows)))


def run_dataset_build(
    data_root: Path,
    output_dir: Path,
    processed_dir: Path,
    results_dir: Path,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    save_samples: int = 10,
) -> dict:
    if not data_root.exists():
        raise FileNotFoundError(f"Missing raw dataset root: {data_root}")

    summary: dict[str, dict | str] = {}
    all_skips: list[dict] = []
    all_errors: list[dict] = []

    for split, max_samples in (("train", max_train), ("val", max_val)):
        rows, skip_records, error_records = build_cheek_dataset(
            data_root=data_root,
            output_dir=output_dir,
            split=split,
            max_samples=max_samples,
        )
        csv_path = processed_dir / f"cheek_{split}_metadata.csv"
        save_csv(rows, csv_path)

        if save_samples > 0 and rows:
            save_crop_samples(rows, results_dir / "crop_samples" / split, n=save_samples)

        summary[split] = {
            "rows": len(rows),
            "skips": len(skip_records),
            "errors": len(error_records),
            "csv_path": str(csv_path),
        }
        all_skips.extend(skip_records)
        all_errors.extend(error_records)

    if all_skips:
        skip_log_path = results_dir / "crop_skip_log.csv"
        save_skip_log(all_skips, skip_log_path)
        summary["skip_log_path"] = str(skip_log_path)

    if all_errors:
        error_log_path = results_dir / "crop_error_log.csv"
        save_error_log(all_errors, error_log_path)
        summary["error_log_path"] = str(error_log_path)

    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    summary = run_dataset_build(
        data_root=args.data_root,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        max_train=args.max_train,
        max_val=args.max_val,
        save_samples=args.save_samples,
    )
    logger.info("dataset build summary: %s", summary)


if __name__ == "__main__":
    main()
