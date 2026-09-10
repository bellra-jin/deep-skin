"""모델이 자기가 틀릴 것 같을 때 그걸 아는가.

왜 이걸 재는가
--------------
지금까지 잰 것은 등급을 맞히는 능력(macro-F1)이다. 모델이 예측의 88%를 한 등급에
몰아넣는다는 것은 알고 있었지만, 몰면서 어떤 확률을 내는지는 재지 않았다.

  등급 2로 몰면서 확률 0.95 -> 틀리면서 확신한다. 나쁘다
  등급 2로 몰면서 확률 0.40 -> 모른다고 말하고 있다. 서비스가 쓸 수 있다

후자라면 지금 리포트는 그 정보를 버리고 있는 것이다. bbox_source 를 화면에 노출한
것과 같은 문제다 - 추정값을 실측값처럼 내보내지 않는다. 지금은 부위 검출 실패만
표시하고 등급 예측이 불확실한 경우는 표시가 없다.

성능 개선이 목적이 아니다. temperature scaling 은 argmax 를 바꾸지 않으므로
macro-F1 과 accuracy 는 정의상 그대로다. 바뀌면 구현이 틀린 것이고,
이 스크립트가 그걸 단언한다.

ECE 의 함정
-----------
기본 구현은 확률을 등폭(equal-width) 구간으로 나눈다. 확률이 좁은 범위에 몰려
있으면 대부분의 구간이 비고 한 구간이 전부를 먹는데, 그 상태의 ECE 는 작게
나오지만 아무것도 재지 않는다. 등폭과 등질량(equal-mass)을 둘 다 내고
구간별 표본 수를 함께 적는다.

산출물
------
results/calibration_experiments.csv
results/confidence_hist.png
results/reliability_{조건}.png

사용법
------
    uv run --with matplotlib python notebooks/jh/calibration.py --probs-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 사람 단위 분할 비율. val 을 calib / report 로 나눈다.
# 같은 사람이 양쪽에 들어가면 안 된다 (README 5.2 규칙).
CALIB_FRACTION = 0.5


# --------------------------------------------------------------------------- #
# 지표
# --------------------------------------------------------------------------- #
def ece_equal_width(conf: np.ndarray, correct: np.ndarray, bins: int = 15):
    """등폭 구간 ECE. 구간별 (표본 수, 평균 확률, 정확도) 도 함께 돌려준다."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, bins - 1)
    return _ece_from_bins(conf, correct, idx, bins)


def ece_equal_mass(conf: np.ndarray, correct: np.ndarray, bins: int = 15):
    """등질량 구간 ECE. 구간마다 표본 수가 같아지도록 분위수로 나눈다.

    확률이 좁게 몰려 있어도 빈 구간이 생기지 않는다.
    """
    order = np.argsort(conf, kind="mergesort")
    idx = np.empty(len(conf), dtype=np.int64)
    splits = np.array_split(order, bins)
    for b, part in enumerate(splits):
        idx[part] = b
    return _ece_from_bins(conf, correct, idx, bins)


def _ece_from_bins(conf, correct, idx, bins):
    n = len(conf)
    ece = 0.0
    rows = []
    for b in range(bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            rows.append({"bin": b, "n": 0, "conf": float("nan"), "acc": float("nan")})
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        ece += cnt / n * abs(a - c)
        rows.append({"bin": b, "n": cnt, "conf": c, "acc": a})
    return ece, rows


def brier_multiclass(probs: np.ndarray, y: np.ndarray) -> float:
    """다중분류 Brier score. 원-핫과의 제곱거리 평균."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.log(p).mean())


# --------------------------------------------------------------------------- #
# temperature scaling
# --------------------------------------------------------------------------- #
def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """확률에 온도를 적용한다.

    저장된 것이 로짓이 아니라 softmax 확률이므로 log 를 취해 로짓 대용으로 쓴다.
    softmax(log(p)/T) 는 softmax(logit/T) 와 같다 - softmax 는 상수 평행이동에
    불변이고 log(p) 는 원래 로짓과 상수만큼만 다르기 때문이다.
    """
    logit = np.log(np.clip(probs, 1e-12, 1.0))
    z = logit / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, y: np.ndarray,
                    lo: float = 0.05, hi: float = 10.0, iters: int = 60) -> float:
    """calib 조각에서 NLL 을 최소화하는 T. 삼분 탐색."""
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if nll(apply_temperature(probs, m1), y) < nll(apply_temperature(probs, m2), y):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


def person_split(person: np.ndarray, frac: float, seed: int):
    """사람 단위로 calib / report 를 나눈다. 같은 사람이 양쪽에 들어가지 않는다."""
    people = sorted({str(p) for p in person})
    rng = np.random.RandomState(seed)
    rng.shuffle(people)
    cut = max(1, int(len(people) * frac))
    calib_people = set(people[:cut])
    is_calib = np.array([str(p) in calib_people for p in person])
    return is_calib, len(calib_people), len(people) - cut


# --------------------------------------------------------------------------- #
def load_runs(probs_dir: Path) -> dict:
    runs = defaultdict(list)
    for f in sorted(probs_dir.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        meta = [str(x) for x in z["meta"]]
        group, target, grades, loss, ev, seed, tag = meta
        if loss != "ce":
            continue          # CORAL 출력은 확률분포가 아니라 여기서 다루지 않는다
        runs[(group, target, int(grades))].append({
            "probs": z["probs"].astype(np.float64),
            "y": z["y"].astype(np.int64),
            "pred": z["pred"].astype(np.int64),
            "person": z["person"],
            "seed": int(seed),
        })
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description="예측 신뢰도와 calibration")
    ap.add_argument("--probs-dir", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path,
                    default=PROJECT_ROOT / "results" / "calibration_experiments.csv")
    ap.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "results")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--split-seed", type=int, default=20260910)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    runs = load_runs(args.probs_dir)
    if not runs:
        raise SystemExit(f"확률 파일이 없습니다: {args.probs_dir}")

    out_rows = []
    hist_data = {}

    for key in sorted(runs):
        group, target, grades = key
        name = f"{group}_{target}_{grades}"
        rs = sorted(runs[key], key=lambda r: r["seed"])
        print(f"\n{'=' * 78}")
        print(f"{name}  ({len(rs)}시드)")
        print(f"{'=' * 78}")

        # ---------- A. 예측 분포와 확률 ----------
        r0 = rs[0]
        n = len(r0["y"])
        print(f"\n  A. 등급별 실제 vs 예측 비율 (seed {r0['seed']}, n={n:,})")
        print(f"     {'등급':>4}{'실제':>10}{'예측':>10}{'평균확률':>10}")
        for g in range(grades):
            act = float((r0["y"] == g).mean())
            prd = float((r0["pred"] == g).mean())
            sel = r0["pred"] == g
            mp = float(r0["probs"][sel, g].mean()) if sel.any() else float("nan")
            print(f"     {g:>4}{act * 100:>9.1f}%{prd * 100:>9.1f}%{mp:>10.3f}")
        top = int(np.bincount(r0["pred"], minlength=grades).argmax())
        share = float((r0["pred"] == top).mean())
        print(f"     -> 최다 예측 등급 {top} 에 {share * 100:.1f}% 쏠림")

        # 맞힘 / 틀림 확률 분포
        conf_all, corr_all = [], []
        for r in rs:
            c = r["probs"].max(axis=1)
            conf_all.append(c)
            corr_all.append((r["pred"] == r["y"]).astype(np.float64))
        conf = np.concatenate(conf_all)
        corr = np.concatenate(corr_all)
        hist_data[name] = (conf, corr)
        ok, ng = conf[corr == 1], conf[corr == 0]
        print(f"\n  A-4. 맞힌 예측 확률 {ok.mean():.3f} (중앙 {np.median(ok):.3f}, n={len(ok):,})")
        print(f"       틀린 예측 확률 {ng.mean():.3f} (중앙 {np.median(ng):.3f}, n={len(ng):,})")
        print(f"       차이 {ok.mean() - ng.mean():+.3f}   "
              f"분리도(AUC) {_auc(ng, ok):.3f}")

        # ---------- B. calibration ----------
        ew, em, br = [], [], []
        for r in rs:
            c = r["probs"].max(axis=1)
            k = (r["pred"] == r["y"]).astype(np.float64)
            ew.append(ece_equal_width(c, k, args.bins)[0])
            em.append(ece_equal_mass(c, k, args.bins)[0])
            br.append(brier_multiclass(r["probs"], r["y"]))
        print(f"\n  B. ECE 등폭 {st.mean(ew):.4f}±{st.stdev(ew):.4f}  "
              f"등질량 {st.mean(em):.4f}±{st.stdev(em):.4f}  "
              f"Brier {st.mean(br):.4f}±{st.stdev(br):.4f}")

        _, ew_rows = ece_equal_width(conf_all[0], corr_all[0], args.bins)
        empty = sum(1 for x in ew_rows if x["n"] == 0)
        biggest = max(x["n"] for x in ew_rows)
        print(f"     등폭 구간 {args.bins}개 중 빈 구간 {empty}개, "
              f"최대 구간이 {biggest / len(conf_all[0]) * 100:.1f}% 를 차지")

        # ---------- C. temperature scaling ----------
        Ts, pre, post, pre_b, post_b = [], [], [], [], []
        argmax_changed = 0
        rep_missing = set()
        for r in rs:
            is_cal, n_cal_p, n_rep_p = person_split(r["person"], CALIB_FRACTION,
                                                    args.split_seed + r["seed"])
            T = fit_temperature(r["probs"][is_cal], r["y"][is_cal])
            Ts.append(T)
            rp, ry = r["probs"][~is_cal], r["y"][~is_cal]
            scaled = apply_temperature(rp, T)
            argmax_changed += int((scaled.argmax(1) != rp.argmax(1)).sum())
            for g in range(grades):
                if (ry == g).sum() == 0:
                    rep_missing.add(g)
            pre.append(ece_equal_mass(rp.max(1), (rp.argmax(1) == ry).astype(float),
                                      args.bins)[0])
            post.append(ece_equal_mass(scaled.max(1), (scaled.argmax(1) == ry).astype(float),
                                       args.bins)[0])
            pre_b.append(brier_multiclass(rp, ry))
            post_b.append(brier_multiclass(scaled, ry))
        print(f"\n  C. T = {st.mean(Ts):.4f}±{st.stdev(Ts):.4f}  "
              f"({'과확신' if st.mean(Ts) > 1 else '과소확신'})")
        print(f"     사람 단위 분할: calib {n_cal_p}명 / report {n_rep_p}명")
        print(f"     report 조각 ECE(등질량) {st.mean(pre):.4f}±{st.stdev(pre):.4f} "
              f"-> {st.mean(post):.4f}±{st.stdev(post):.4f}  "
              f"차이 {st.mean(post) - st.mean(pre):+.4f}")
        print(f"     report 조각 Brier {st.mean(pre_b):.4f} -> {st.mean(post_b):.4f}")
        print(f"     argmax 바뀐 표본 {argmax_changed}건 "
              f"{'(정상 - 온도는 순서를 바꾸지 않는다)' if argmax_changed == 0 else '[!] 구현 오류'}")
        if rep_missing:
            print(f"     [!] report 조각에 표본이 0건인 등급: {sorted(rep_missing)}")

        out_rows.append({
            "condition": name, "group": group, "target": target, "grades": grades,
            "seeds": len(rs), "n_val": n,
            "top_pred_grade": top, "top_pred_share": round(share, 4),
            "conf_correct": round(float(ok.mean()), 4),
            "conf_wrong": round(float(ng.mean()), 4),
            "conf_auc": round(_auc(ng, ok), 4),
            "ece_equal_width": round(st.mean(ew), 4),
            "ece_equal_width_sd": round(st.stdev(ew), 4),
            "ece_equal_mass": round(st.mean(em), 4),
            "ece_equal_mass_sd": round(st.stdev(em), 4),
            "brier": round(st.mean(br), 4), "brier_sd": round(st.stdev(br), 4),
            "temperature": round(st.mean(Ts), 4),
            "temperature_sd": round(st.stdev(Ts), 4),
            "ece_report_before": round(st.mean(pre), 4),
            "ece_report_after": round(st.mean(post), 4),
            "brier_report_before": round(st.mean(pre_b), 4),
            "brier_report_after": round(st.mean(post_b), 4),
            "argmax_changed": argmax_changed,
            "report_missing_grades": ",".join(str(g) for g in sorted(rep_missing)),
        })

        # ---------- D. 임계값 ----------
        print(f"\n  D. 신뢰도 임계값 (5시드 평균)")
        print(f"     {'t':>6}{'coverage':>10}{'남은 acc':>10}{'버린 acc':>10}"
              f"{'등급별 coverage':>22}")
        for t in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8):
            cov, acc_in, acc_out, per_g = [], [], [], defaultdict(list)
            for r in rs:
                c = r["probs"].max(1)
                k = (r["pred"] == r["y"])
                keep = c >= t
                cov.append(float(keep.mean()))
                if keep.any():
                    acc_in.append(float(k[keep].mean()))
                if (~keep).any():
                    acc_out.append(float(k[~keep].mean()))
                for g in range(grades):
                    m = r["y"] == g
                    if m.any():
                        per_g[g].append(float(keep[m].mean()))
            pg = " ".join(f"{g}:{st.mean(v) * 100:.0f}%" for g, v in sorted(per_g.items()))
            print(f"     {t:>6.1f}{st.mean(cov) * 100:>9.1f}%"
                  f"{(st.mean(acc_in) * 100 if acc_in else float('nan')):>9.1f}%"
                  f"{(st.mean(acc_out) * 100 if acc_out else float('nan')):>9.1f}%"
                  f"   {pg}")

    # ---------- 저장 ----------
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n저장: {args.out_csv}  ({len(out_rows)}행)")

    if not args.no_plot:
        _plot_hist(hist_data, args.fig_dir / "confidence_hist.png")
        for key in sorted(runs):
            name = f"{key[0]}_{key[1]}_{key[2]}"
            _plot_reliability(runs[key], name, args.bins,
                              args.fig_dir / f"reliability_{name}.png")


def _auc(neg: np.ndarray, pos: np.ndarray) -> float:
    """두 분포의 분리도. 0.5 면 전혀 못 가른다."""
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[len(neg):].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def _plot_hist(hist_data: dict, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [!] matplotlib 없음 - 그림 생략")
        return
    cols = len(hist_data)
    fig, axes = plt.subplots(1, cols, figsize=(4.2 * cols, 3.4), squeeze=False)
    for i, (name, (conf, corr)) in enumerate(sorted(hist_data.items())):
        ax = axes[0][i]
        bins = np.linspace(conf.min(), 1.0, 30)
        ax.hist(conf[corr == 1], bins=bins, alpha=0.6, label="correct", density=True)
        ax.hist(conf[corr == 0], bins=bins, alpha=0.6, label="wrong", density=True)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("max probability", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    fig.suptitle("confidence: correct vs wrong", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"  그림 저장: {out}")


def _plot_reliability(rs: list, name: str, bins: int, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    conf = np.concatenate([r["probs"].max(1) for r in rs])
    corr = np.concatenate([(r["pred"] == r["y"]).astype(float) for r in rs])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, (fn, label) in zip(axes, ((ece_equal_width, "equal-width"),
                                      (ece_equal_mass, "equal-mass"))):
        e, rows = fn(conf, corr, bins)
        xs = [r["conf"] for r in rows if r["n"] > 0]
        ys = [r["acc"] for r in rows if r["n"] > 0]
        ns = [r["n"] for r in rows if r["n"] > 0]
        ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
        ax.scatter(xs, ys, s=[max(8, min(160, v / 20)) for v in ns], alpha=0.75)
        ax.set_title(f"{label}  ECE={e:.4f}  bins={len(xs)}/{bins}", fontsize=9)
        ax.set_xlabel("confidence", fontsize=8)
        ax.set_ylabel("accuracy", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(name, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"  그림 저장: {out}")


if __name__ == "__main__":
    main()
