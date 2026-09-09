"""README 의 실험 수치 표를 실험 로그에서 다시 만들어 낸다.

왜 필요한가
-----------
README 에 적힌 수치가 `results/head_experiments.csv` 에서 나온 것인지,
아니면 어딘가에서 손으로 옮겨 적힌 것인지 구분할 방법이 없었다.
이 스크립트는 표를 로그에서 직접 생성하므로, 출력이 README 와 다르면
둘 중 하나가 틀린 것이다.

로그에 없는 표는 "재현 불가"로 보고한다 - 침묵하지 않는다.

사용법
------
    uv run python notebooks/jh/report_tables.py            # 전부 출력
    uv run python notebooks/jh/report_tables.py --table 6.8-3
    uv run python notebooks/jh/report_tables.py --check    # 재현 가능 여부만
"""

from __future__ import annotations

import argparse
import csv
import statistics as stat
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "results" / "head_experiments.csv"


# --------------------------------------------------------------------------- #
# 표 정의 - (라벨, 필터) 목록.  필터는 CSV 컬럼에 대한 완전일치 조건이다.
# --------------------------------------------------------------------------- #
def f(**kw) -> dict:
    return {k: str(v) for k, v in kw.items()}


TABLES: dict[str, dict] = {
    "6.8-1": {
        "title": "6.8-(1) 불균형 보정 3종 (모공 6등급 · ResNet-50)",
        "cols": ["macro_f1", "accuracy"],
        "rows": [
            ("기준선 (CE)", f(tag="기준선", arch="resnet50", target="pore", grades=6, loss="ce", sampler="none", class_weight="False")),
            ("Focal Loss", f(tag="focal", arch="resnet50", target="pore", grades=6, loss="focal")),
            ("WeightedRandomSampler", f(tag="가중샘플러", arch="resnet50", target="pore", grades=6, sampler="weighted")),
            ("클래스 가중치", f(tag="클래스가중치", arch="resnet50", target="pore", grades=6, class_weight="True")),
        ],
    },
    "6.8-2": {
        "title": "6.8-(2) 등급 병합 6 → 3 (ResNet-50)",
        "cols": ["macro_f1"],
        "rows": [
            ("색소침착 6등급", f(tag="기준선", arch="resnet50", target="pigmentation", grades=6)),
            ("색소침착 3등급", f(tag="3등급", arch="resnet50", target="pigmentation", grades=3)),
            ("모공 6등급", f(tag="기준선", arch="resnet50", target="pore", grades=6)),
            ("모공 3등급", f(tag="3등급", arch="resnet50", target="pore", grades=3)),
        ],
    },
    "6.8-3": {
        "title": "6.8-(3) 백본 비교 (좌우반전 증강 양쪽 모두 끔)",
        "cols": ["macro_f1", "accuracy", "qwk"],
        "rows": [
            ("색소침착 3등급 · ResNet-50", f(tag="R50/증강없음", target="pigmentation", grades=3)),
            ("색소침착 3등급 · DINOv3", f(tag="DINOv3", target="pigmentation", grades=3)),
            ("모공 3등급 · ResNet-50", f(tag="R50/증강없음", target="pore", grades=3)),
            ("모공 3등급 · DINOv3", f(tag="DINOv3", target="pore", grades=3)),
            ("모공 3등급 · DINOv3+CORAL", f(tag="DINOv3/CORAL", target="pore", grades=3)),
            ("색소침착 6등급 · ResNet-50", f(tag="R50/증강없음", target="pigmentation", grades=6)),
            ("색소침착 6등급 · DINOv3", f(tag="DINOv3", target="pigmentation", grades=6)),
            ("색소침착 6등급 · DINOv3+CORAL", f(tag="DINOv3/CORAL", target="pigmentation", grades=6)),
        ],
    },
    "6.8-4": {
        "title": "6.8-(4) 도메인 갭 - 디카 학습 → 폰 검증 (3등급)",
        "cols": ["macro_f1"],
        "rows": [
            ("모공 · ResNet-50", f(tag="도메인갭", arch="resnet50", target="pore", eval="device")),
            ("모공 · DINOv3", f(tag="DINOv3/도메인갭", target="pore", eval="device")),
            ("색소침착 · ResNet-50", f(tag="도메인갭", arch="resnet50", target="pigmentation", eval="device")),
            ("색소침착 · DINOv3", f(tag="DINOv3/도메인갭", target="pigmentation", eval="device")),
        ],
    },
    "6.8-5": {
        "title": "6.8-(5) 기기 열화 증강 (3등급 · DINOv3 · 5시드)",
        "cols": ["macro_f1"],
        "rows": [
            ("모공 · 디카 학습 → 폰 검증 · 기준선",
             f(tag="기기증강/기준선", target="pore", eval="device")),
            ("모공 · 디카 학습 → 폰 검증 · 증강",
             f(tag="기기증강/증강", target="pore", eval="device")),
            ("색소침착 · 디카 학습 → 폰 검증 · 기준선",
             f(tag="기기증강/기준선", target="pigmentation", eval="device")),
            ("색소침착 · 디카 학습 → 폰 검증 · 증강",
             f(tag="기기증강/증강", target="pigmentation", eval="device")),
            ("모공 · 전체검증 · 기준선",
             f(tag="기기증강/기준선", target="pore", eval="standard")),
            ("모공 · 전체검증 · 증강",
             f(tag="기기증강/증강", target="pore", eval="standard")),
        ],
    },
    "6.9-1": {
        "title": "6.9 눈가 - 볼 결론의 전이 여부",
        "cols": ["macro_f1", "qwk"],
        "rows": [
            ("7등급 CE", f(tag="5.1/7등급/ce")),
            ("7등급 CORAL", f(tag="5.1/7등급/coral")),
            ("3등급 CE (boundary)", f(tag="5.1/3등급-boundary")),
            ("3등급 CE (proportional)", f(tag="5.1/3등급/ce")),
            ("3등급 CORAL (proportional)", f(tag="5.1/3등급/coral")),
            ("디카 학습 → 폰 검증", f(tag="도메인갭/눈가/전체")),
        ],
    },
    "6.9-2": {
        "title": "6.9 종횡비 왜곡 - ResizeAndPad vs 강제 정사각형",
        "cols": ["macro_f1"],
        "rows": [
            ("볼 모공 · 전체검증 · pad", f(tag="pore/standard/pad")),
            ("볼 모공 · 전체검증 · squash", f(tag="pore/standard/squash")),
            ("볼 모공 · 폰검증 · pad", f(tag="pore/device/pad")),
            ("볼 모공 · 폰검증 · squash", f(tag="pore/device/squash")),
            ("볼 색소침착 · 폰검증 · pad", f(tag="pigmentation/device/pad")),
            ("볼 색소침착 · 폰검증 · squash", f(tag="pigmentation/device/squash")),
            ("눈가 · 전체검증 · pad", f(tag="5.1/3등급/ce")),
            ("눈가 · 전체검증 · squash", f(tag="5.2/squash/전체")),
            ("눈가 · 폰검증 · pad", f(tag="5.1/도메인갭")),
            ("눈가 · 폰검증 · squash", f(tag="5.2/squash/폰")),
        ],
    },
    "6.9-3": {
        "title": "6.9 크롭 폭 임계값 (검증셋 고정, train 만 필터)",
        "cols": ["n_train", "macro_f1"],
        "rows": [
            ("하한 없음 (90)", f(tag="5.3/w90")),
            ("120", f(tag="5.3fix/w120")),
            ("150", f(tag="5.3fix/w150")),
            ("180", f(tag="5.3fix/w180")),
        ],
    },
    "6.10-1": {
        "title": "6.10 flip TTA (3등급 · DINOv3 · 전체검증)",
        "cols": ["macro_f1", "qwk"],
        "rows": [
            ("색소침착 · TTA 없음", f(tag="pigmentation/noTTA")),
            ("색소침착 · TTA 있음", f(tag="pigmentation/TTA")),
            ("모공 · TTA 없음", f(tag="pore/noTTA")),
            ("모공 · TTA 있음", f(tag="pore/TTA")),
        ],
    },
    "6.10-2": {
        "title": "6.10 도메인 갭의 각도 교란 배제 (디카 학습 → 폰 검증)",
        "cols": ["n_val", "macro_f1"],
        "rows": [
            ("볼 · 전체", f(tag="도메인갭/볼/전체")),
            ("볼 · 각도 0만", f(tag="도메인갭/볼/각도0만")),
            ("볼 · 각도 7·8만", f(tag="도메인갭/볼/각도78만")),
            ("눈가 · 전체", f(tag="도메인갭/눈가/전체")),
            ("눈가 · 각도 0만", f(tag="도메인갭/눈가/각도0만")),
            ("눈가 · 각도 7·8만", f(tag="도메인갭/눈가/각도78만")),
        ],
    },
}

# 로그에서 재현할 수 없는 표.  침묵하지 않고 이유와 함께 보고한다.
NOT_REPRODUCIBLE = {
    "6.9-0": "눈가 크롭 건수·오류 수 - 학습 로그가 아니라 crop.py 산출물이다.",
    "6.10-3": "라벨 문서 오류 3건 - facepart_audit.py 의 전수 집계 결과다.",
    "7.4-1": "YOLO 혼동행렬 - 팀의 학습 산출물(images/yolo_confusion_matrix.png)이다.",
    "7.6-1": "정면 판정 보정 - frontality_calibration.py 의 산출물이다.",
}


# --------------------------------------------------------------------------- #
def load(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise SystemExit(f"실험 로그가 없습니다: {csv_path}")
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def select(rows: list[dict], flt: dict) -> list[dict]:
    return [r for r in rows if all(r.get(k, "") == v for k, v in flt.items())]


def fmt(values: list[str], col: str) -> str:
    nums = []
    for v in values:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    if not nums:
        return "-"
    if col in ("n_train", "n_val", "epochs_run", "best_epoch"):
        return f"{int(round(stat.mean(nums))):,}"
    if len(nums) == 1:
        return f"{nums[0]:.4f}"
    return f"{stat.mean(nums):.4f}±{stat.stdev(nums):.4f}"


def render(name: str, spec: dict, rows: list[dict]) -> tuple[str, list[str]]:
    header = ["조건", *spec["cols"], "n"]
    out = [f"#### {spec['title']}", "",
           "| " + " | ".join(header) + " |",
           "|" + "---|" * len(header)]
    missing = []
    for label, flt in spec["rows"]:
        sel = select(rows, flt)
        if not sel:
            missing.append(label)
            out.append(f"| {label} | " + " | ".join(["**없음**"] * len(spec["cols"])) + " | 0 |")
            continue
        cells = [fmt([r[c] for r in sel], c) for c in spec["cols"]]
        out.append(f"| {label} | " + " | ".join(cells) + f" | {len(sel)} |")
    return "\n".join(out), missing


def main() -> None:
    ap = argparse.ArgumentParser(description="README 실험 표를 로그에서 재생성")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--table", default=None, help="특정 표만 (예: 6.8-3)")
    ap.add_argument("--check", action="store_true", help="재현 가능 여부만 요약")
    args = ap.parse_args()

    rows = load(args.csv)
    print(f"실험 로그 {len(rows):,}건  ({args.csv.relative_to(PROJECT_ROOT)})\n")

    targets = {args.table: TABLES[args.table]} if args.table else TABLES
    if args.table and args.table not in TABLES:
        raise SystemExit(f"모르는 표: {args.table}\n사용 가능: {', '.join(TABLES)}")

    problems: dict[str, list[str]] = {}
    for name, spec in targets.items():
        body, missing = render(name, spec, rows)
        if missing:
            problems[name] = missing
        if not args.check:
            print(body, "\n")

    print("=" * 66)
    if problems:
        print("로그에서 찾지 못한 행:")
        for name, labels in problems.items():
            print(f"  {name}: {', '.join(labels)}")
    else:
        print(f"표 {len(targets)}개, 모든 행을 로그에서 재현했습니다.")

    if not args.table:
        print("\n로그로 재현할 수 없는 표 (학습 실험이 아님):")
        for name, why in NOT_REPRODUCIBLE.items():
            print(f"  {name}  {why}")


if __name__ == "__main__":
    main()
