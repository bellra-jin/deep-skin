"""장비 측정값(equipment)의 가용성과, 전문가 등급과의 관계를 잰다.

왜 이걸 먼저 재는가
-------------------
지금까지 학습한 것은 전부 전문가 진단 등급(annotations)이다. 같은 라벨 JSON 에는
장비 측정 수치(equipment)가 함께 들어 있는데 한 번도 쓰지 않았다.

볼 모공은 같은 이름으로 등급(0~5)과 개수(629.0)가 둘 다 있다.
등급이 개수를 구간으로 나눈 것이라면 둘은 강하게 단조 관계여야 한다.
그렇지 않다면 전문가 등급과 장비 측정이 다른 것을 보고 있다는 뜻이고,
그건 "라벨 경계가 모호하다"는 문제 정의에 직접적인 근거가 된다.

회귀 헤드를 붙이기 전에 이 관계를 먼저 재야 한다. 관계가 없으면
회귀 성능보다 그 사실 자체가 중요한 결과다.

산출물
------
results/equipment_audit.csv        키별 결측률·기기별 편차·분포
results/label_vs_equipment.png     등급별 박스플롯 (통계만 믿지 않는다)

사용법
------
    uv run --with matplotlib python notebooks/jh/equipment_audit.py
    uv run python notebooks/jh/equipment_audit.py --no-plot
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(10 ** 9)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVICE_NAME = {0: "디카", 1: "패드", 2: "폰"}

# (CSV 접두어, 등급 컬럼) 목록. 등급 컬럼은 crop.py 가 만든 {키}_label 이다.
SPLITS = [
    ("cheek", ["pore", "pigmentation"]),
    ("eye", ["wrinkle"]),
]

# 등급과 측정값이 같은 대상을 가리키는 조합.
# facepart -> (등급 컬럼, equipment 키 접미사 목록)
# 눈가 키는 문서의 wrinkle_{l,r}_eye_* 가 아니라 {l,r}_perocular_wrinkle_* 이다.
GRADE_VS_EQUIP = {
    5: ("pore", ["l_cheek_pore"]),
    6: ("pore", ["r_cheek_pore"]),
    3: ("wrinkle", [f"l_perocular_wrinkle_{s}"
                    for s in ("Ra", "Rq", "Rmax", "Rt", "Rp", "Rv", "R3z", "Rz=Rtm")]),
    4: ("wrinkle", [f"r_perocular_wrinkle_{s}"
                    for s in ("Ra", "Rq", "Rmax", "Rt", "Rp", "Rv", "R3z", "Rz=Rtm")]),
}


def load(prefix: str, split: str) -> list[dict]:
    p = PROJECT_ROOT / "data" / "processed" / f"{prefix}_{split}_metadata.csv"
    if not p.exists():
        print(f"  [!] 없음: {p}")
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_equipment(row: dict) -> dict:
    raw = row.get("equipment_json") or ""
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {}


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f          # NaN 제외


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """순위 상관. 선형을 가정하지 않는다. 동점은 평균 순위로 처리한다."""
    if len(xs) < 3:
        return None

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return None if den == 0 else num / den


def overlap_ratio(a: list[float], b: list[float]) -> float | None:
    """인접 등급 분포의 겹침. 두 사분위 구간(IQR)이 겹치는 비율."""
    if len(a) < 4 or len(b) < 4:
        return None
    qa = (st.quantiles(a, n=4)[0], st.quantiles(a, n=4)[2])
    qb = (st.quantiles(b, n=4)[0], st.quantiles(b, n=4)[2])
    lo, hi = max(qa[0], qb[0]), min(qa[1], qb[1])
    inter = max(0.0, hi - lo)
    union = max(qa[1], qb[1]) - min(qa[0], qb[0])
    return None if union <= 0 else inter / union


# --------------------------------------------------------------------------- #
# A. 가용성
# --------------------------------------------------------------------------- #
def audit(rows: list[dict], prefix: str, split: str, out: list[dict]) -> None:
    if not rows:
        return
    n = len(rows)
    has_col = "equipment_json" in rows[0]
    nonempty = sum(1 for r in rows if parse_equipment(r))
    print(f"\n  -- {prefix}/{split}  n={n:,}  equipment_json 컬럼={has_col}  "
          f"비어있지 않은 레코드 {nonempty:,} ({nonempty / n * 100:.1f}%)")

    # 키별 값 수집 (facepart 별로 키가 다르므로 부위를 함께 센다)
    vals: dict[str, list[float]] = defaultdict(list)
    by_dev: dict[tuple[str, int], int] = defaultdict(int)
    dev_total: dict[int, int] = defaultdict(int)
    key_fp: dict[str, set] = defaultdict(set)
    for r in rows:
        dev = int(r["device"]) if r.get("device") not in ("", None) else -1
        dev_total[dev] += 1
        eq = parse_equipment(r)
        for k, v in eq.items():
            f = _num(v)
            key_fp[k].add(int(r["facepart"]))
            if f is None:
                continue
            vals[k].append(f)
            by_dev[(k, dev)] += 1

    print(f"     키 {len(vals)}종")
    print(f"     {'key':<32}{'n':>8}{'채움률':>8}{'min':>11}{'중앙':>11}{'max':>12}  기기별 채움률")
    for k in sorted(vals):
        xs = vals[k]
        # 이 키가 등장하는 부위의 레코드 수를 분모로 쓴다
        denom = sum(1 for r in rows if int(r["facepart"]) in key_fp[k])
        devtxt = "  ".join(
            f"{DEVICE_NAME.get(d, d)} "
            f"{by_dev[(k, d)] / max(sum(1 for r in rows if int(r['device']) == d and int(r['facepart']) in key_fp[k]), 1) * 100:.0f}%"
            for d in (0, 1, 2) if dev_total.get(d)
        )
        print(f"     {k:<32}{len(xs):>8,}{len(xs) / max(denom, 1) * 100:>7.1f}%"
              f"{min(xs):>11.3f}{st.median(xs):>11.3f}{max(xs):>12.3f}  {devtxt}")
        out.append({
            "prefix": prefix, "split": split, "key": k,
            "faceparts": ",".join(str(x) for x in sorted(key_fp[k])),
            "n": len(xs), "fill_rate": round(len(xs) / max(denom, 1), 4),
            "min": round(min(xs), 4), "p25": round(st.quantiles(xs, n=4)[0], 4),
            "median": round(st.median(xs), 4), "p75": round(st.quantiles(xs, n=4)[2], 4),
            "max": round(max(xs), 4),
            "fill_dslr": round(by_dev[(k, 0)] / max(sum(1 for r in rows if int(r["device"]) == 0 and int(r["facepart"]) in key_fp[k]), 1), 4),
            "fill_pad": round(by_dev[(k, 1)] / max(sum(1 for r in rows if int(r["device"]) == 1 and int(r["facepart"]) in key_fp[k]), 1), 4),
            "fill_phone": round(by_dev[(k, 2)] / max(sum(1 for r in rows if int(r["device"]) == 2 and int(r["facepart"]) in key_fp[k]), 1), 4),
        })


# --------------------------------------------------------------------------- #
# B. 등급 vs 측정값
# --------------------------------------------------------------------------- #
def compare(rows: list[dict], prefix: str) -> list[dict]:
    """등급과 측정값이 같은 것을 보고 있는지 잰다."""
    results = []
    print(f"\n{'=' * 78}")
    print(f"B. 등급 vs 장비 측정값  ({prefix})")
    print(f"{'=' * 78}")

    pairs: dict[str, tuple[list[float], list[float]]] = {}
    for r in rows:
        fp = int(r["facepart"])
        spec = GRADE_VS_EQUIP.get(fp)
        if not spec:
            continue
        grade_col, keys = spec
        g = _num(r.get(f"{grade_col}_label"))
        if g is None or g < 0:
            continue
        eq = parse_equipment(r)
        for k in keys:
            m = _num(eq.get(k))
            if m is None:
                continue
            # 좌우를 합쳐서 본다 - 같은 지표이고 부위만 다르다.
            # 접두사 두 글자만 떼어낸다. replace 로 지우면 키 안쪽의
            # "perocula[r_]wrinkle" 까지 지워져 이름이 뭉개진다.
            base = k[2:] if k[:2] in ("l_", "r_") else k
            pairs.setdefault(base, ([], []))
            pairs[base][0].append(g)
            pairs[base][1].append(m)

    for base, (gs, ms) in sorted(pairs.items()):
        rho = spearman(gs, ms)
        print(f"\n  -- {base}  n={len(gs):,}  Spearman rho = "
              f"{'-' if rho is None else f'{rho:+.4f}'}")
        by_grade: dict[int, list[float]] = defaultdict(list)
        for g, m in zip(gs, ms):
            by_grade[int(g)].append(m)
        print(f"     {'등급':>4}{'n':>8}{'p25':>12}{'중앙':>12}{'p75':>12}{'인접겹침':>10}")
        keys_sorted = sorted(by_grade)
        for i, g in enumerate(keys_sorted):
            xs = by_grade[g]
            q = st.quantiles(xs, n=4) if len(xs) >= 4 else [float("nan")] * 3
            ov = ""
            if i + 1 < len(keys_sorted):
                o = overlap_ratio(xs, by_grade[keys_sorted[i + 1]])
                ov = "-" if o is None else f"{o * 100:.0f}%"
            print(f"     {g:>4}{len(xs):>8,}{q[0]:>12.2f}{st.median(xs):>12.2f}"
                  f"{q[2]:>12.2f}{ov:>10}")
        results.append({"metric": base, "n": len(gs), "spearman": rho,
                        "by_grade": {g: by_grade[g] for g in keys_sorted}})
    return results


def plot(results: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [!] matplotlib 이 없어 그림을 건너뜁니다 "
              "(uv run --with matplotlib ... 로 실행하세요)")
        return
    show = [r for r in results if r["by_grade"]]
    if not show:
        return
    cols = min(4, len(show))
    rows_n = (len(show) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.4 * rows_n), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, r in enumerate(show):
        ax = axes[i // cols][i % cols]
        ax.axis("on")
        grades = sorted(r["by_grade"])
        ax.boxplot([r["by_grade"][g] for g in grades],
                   tick_labels=[str(g) for g in grades], showfliers=False)
        rho = r["spearman"]
        ax.set_title(f"{r['metric']}\nrho={'-' if rho is None else f'{rho:+.3f}'}  n={r['n']:,}",
                     fontsize=9)
        ax.set_xlabel("expert grade", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("expert grade vs equipment measurement", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"\n  그림 저장: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="장비 측정값 가용성과 등급 대비 관계")
    ap.add_argument("--out-csv", type=Path,
                    default=PROJECT_ROOT / "results" / "equipment_audit.csv")
    ap.add_argument("--out-png", type=Path,
                    default=PROJECT_ROOT / "results" / "label_vs_equipment.png")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("A. 장비 측정값 가용성")
    print("=" * 78)
    audit_rows: list[dict] = []
    all_rows: dict[str, list[dict]] = {}
    for prefix, _ in SPLITS:
        merged = []
        for split in ("train", "val"):
            rows = load(prefix, split)
            audit(rows, prefix, split, audit_rows)
            merged += rows
        all_rows[prefix] = merged

    if audit_rows:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            w.writeheader()
            w.writerows(audit_rows)
        print(f"\n  저장: {args.out_csv}  ({len(audit_rows)}행)")

    results = []
    for prefix, _ in SPLITS:
        results += compare(all_rows[prefix], prefix)

    if not args.no_plot:
        plot(results, args.out_png)

    print(f"\n{'=' * 78}")
    print("B 요약 - Spearman rho")
    for r in sorted(results, key=lambda x: -(abs(x["spearman"]) if x["spearman"] else 0)):
        rho = r["spearman"]
        verdict = ("-" if rho is None else
                   "등급은 측정값의 이산화에 가깝다" if abs(rho) >= 0.7 else
                   "관련은 있으나 등급이 다른 요소도 반영" if abs(rho) >= 0.4 else
                   "둘은 다른 것을 재고 있다")
        print(f"  {r['metric']:<28} {('-' if rho is None else f'{rho:+.4f}'):>9}  "
              f"n={r['n']:>7,}  {verdict}")


if __name__ == "__main__":
    main()
