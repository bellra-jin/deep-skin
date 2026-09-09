"""캐싱된 특징 위에서 얕은 헤드만 학습하는 실험 러너.

특징이 이미 뽑혀 있으므로 에폭당 수 초로 끝난다. 조건을 바꿔가며
반복 실행하고, 모든 결과를 results/head_experiments.csv 에 누적한다.

비교할 수 있는 축
----------------
  --target   pore | pigmentation          어떤 지표가 더 학습 가능한가
  --grades   6 | 3                        등급 병합이 얼마나 이득인가
  --loss     ce | focal | coral           서수(ordinal) 손실의 효과
  --sampler  none | weighted              불균형 보정의 효과
  --eval     standard | device            학습/서비스 기기 도메인 갭

사용법
------
    uv run python notebooks/jh/train_head.py --target pigmentation
    uv run python notebooks/jh/train_head.py --target pore --grades 3 --loss coral
    uv run python notebooks/jh/train_head.py --target pore --eval device
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crop import FACEPARTS  # noqa: E402

# 지표별 등급 수는 crop.py 의 부위 테이블이 정본이다.
# 미간(0~2)/이마(0~3) 확장 시 그쪽만 고치면 된다.
TARGET_NUM_CLASSES: dict[str, int] = {}
for _part in FACEPARTS.values():
    for _k, _n in _part["num_classes"].items():
        TARGET_NUM_CLASSES[_k] = max(TARGET_NUM_CLASSES.get(_k, 0), _n)

# 3등급 병합 룩업. 등급 수가 부위마다 다르므로 전체 등급 수로 키를 잡는다.
#   볼   0~5 (6등급): 0-1 양호 / 2 보통 / 3-5 주의  - README 6.3 의 확정값
#   눈가 0~6 (7등급): 경계를 어디에 두느냐에 따라 분포가 크게 달라져 두 가지를 둔다.
#     boundary      0-1 / 2   / 3-6  -> 40.6 / 14.6 / 44.8 %  (기본값)
#     proportional  0-1 / 2-3 / 4-6  -> 40.6 / 29.2 / 30.2 %
#   기본값은 boundary 다. proportional 이 macro-F1 은 0.12 높게 나오지만,
#   그 차이의 상당 부분은 macro-F1 이 균형 잡힌 클래스를 유리하게 평가하는 성질에서
#   온다. 병합 경계를 지표로 고르면 순환 논리가 되므로 척도 위치를 볼과 맞춘다.
MERGE_3_LUTS = {
    (6, "proportional"): [0, 0, 1, 2, 2, 2],
    (6, "boundary"):     [0, 0, 1, 2, 2, 2],
    (7, "proportional"): [0, 0, 1, 1, 2, 2, 2],
    (7, "boundary"):     [0, 0, 1, 2, 2, 2, 2],
}


def merge3_lut(full_grades: int, scheme: str) -> np.ndarray:
    key = (full_grades, scheme)
    if key not in MERGE_3_LUTS:
        raise SystemExit(f"{full_grades}등급용 3등급 병합 규칙이 없습니다 (scheme={scheme})")
    return np.array(MERGE_3_LUTS[key], dtype=np.int64)
DEVICE_NAME = {0: "디카", 1: "패드", 2: "폰"}

# 변형 특징 파일의 접미사. 자동 선택이 이들을 기본 특징과 헷갈리지 않게 한다.
#   _devaug  기기 열화 사이드카 (원본과 짝을 이룸)
#   _squash  종횡비 무시 리사이즈로 뽑은 별도 특징 세트
FEAT_VARIANTS = ("_devaug", "_squash")


def _matches_variant(stem: str, want: str) -> bool:
    """want 가 빈 문자열이면 변형이 아닌 기본 특징만 참."""
    if want:
        return stem.endswith(want)
    return not any(stem.endswith(v) for v in FEAT_VARIANTS)


# --------------------------------------------------------------------------- #
# 데이터
# --------------------------------------------------------------------------- #
def load_npz(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"특징 파일이 없습니다: {path}\n먼저 extract_features.py 를 실행하세요.")
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def prepare(pack: dict, target: str, grades: int, device_filter: int | None,
            min_width: int = 0, lut: np.ndarray | None = None,
            angles: list[int] | None = None):
    y = pack[f"labels_{target}"].astype(np.int64)
    keep = y >= 0
    if device_filter is not None:
        keep &= pack["device"] == device_filter
    # 크롭 폭 하한. train 과 val 에 똑같이 건다 - 검증만 깨끗하게 두면
    # 실제 서비스보다 낙관적인 수치가 나온다.
    if min_width and "crop_width" in pack:
        keep &= pack["crop_width"] >= min_width
    # 각도 필터. 기기 도메인 갭이 "기기 차이"인지 "각도 구성 차이"인지 분해할 때 쓴다.
    # AI-Hub 는 각도와 기기가 독립이 아니다 - 디카는 0~6, 패드/폰은 0·7·8 뿐이다.
    if angles:
        keep &= np.isin(pack["angle"], list(angles))
    idx = np.nonzero(keep)[0]

    y = y[idx]
    if grades == 3:
        # 라벨이 예상 범위를 벗어나도 죽지 않도록 룩업 테이블로 병합한다.
        y = lut[np.clip(y, 0, len(lut) - 1)]

    out = {
        "feat": pack["feat"][idx],
        "y": y,
        "device": pack["device"][idx],
        "person": pack["person"][idx],
    }
    out["feat_flip"] = pack["feat_flip"][idx] if "feat_flip" in pack else None
    out["feat_devaug"] = pack["feat_devaug"][idx] if "feat_devaug" in pack else None
    return out


class FeatDataset(Dataset):
    """캐싱 환경의 증강 - 좌우반전본과 기기 열화본을 확률적으로 바꿔 쓴다.

    한 샘플에 대해 한 번에 하나만 고른다(원본 / 반전 / 열화).
    섞어서 적용할 수는 없다 - 특징이 이미 뽑혀 있기 때문이다.
    """

    def __init__(self, feat, feat_flip, feat_devaug, y,
                 augment: bool, aug_prob: float = 0.5):
        self.feat, self.y = feat, y
        self.feat_flip = feat_flip if augment else None
        self.feat_devaug = feat_devaug
        self.aug_prob = aug_prob

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        x = self.feat[i]
        if self.feat_devaug is not None and np.random.rand() < self.aug_prob:
            x = self.feat_devaug[i]
        elif self.feat_flip is not None and np.random.rand() < 0.5:
            x = self.feat_flip[i]
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.y[i])


# --------------------------------------------------------------------------- #
# 헤드 · 손실
# --------------------------------------------------------------------------- #
class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=0, dropout=0.3):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class CoralHead(nn.Module):
    """CORAL - 가중치를 공유하고 절편만 분리해 등급 순서를 구조로 보장한다."""

    def __init__(self, in_dim, num_classes, hidden=0, dropout=0.3):
        super().__init__()
        self.k = num_classes
        if hidden:
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True), nn.Dropout(dropout),
            )
            feat_dim = hidden
        else:
            self.trunk = nn.Dropout(dropout)
            feat_dim = in_dim
        self.fc = nn.Linear(feat_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        return self.fc(self.trunk(x)) + self.bias      # (N, K-1)


def coral_loss(logits, y, weight=None):
    k1 = logits.shape[1]
    levels = (y[:, None] > torch.arange(k1, device=y.device)[None, :]).float()
    term = F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels)
    per_sample = -term.sum(dim=1)
    if weight is not None:
        per_sample = per_sample * weight[y]
    return per_sample.mean()


def focal_loss(logits, y, gamma=2.0, weight=None):
    logp = F.log_softmax(logits, dim=1)
    logpt = logp.gather(1, y[:, None]).squeeze(1)
    loss = -((1 - logpt.exp()) ** gamma) * logpt
    if weight is not None:
        loss = loss * weight[y]
    return loss.mean()


def coral_predict(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


# --------------------------------------------------------------------------- #
# 평가
# --------------------------------------------------------------------------- #
def evaluate(model, feat, y, loss_kind, feat_flip=None, batch=1024):
    """feat_flip 이 주어지면 좌우반전 출력을 평균한다 (추론 엔진과 동일한 TTA).

    로짓이 아니라 확률을 평균한다. 추론 엔진이
    probs = (softmax(f(x)) + softmax(f(flip(x)))) / 2 를 쓰므로 그대로 맞춘다.
    CORAL 은 softmax 가 아니라 sigmoid 출력을 평균한 뒤 임계 처리한다.
    """
    model.eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(y), batch):
            xb = torch.from_numpy(np.ascontiguousarray(feat[i:i + batch]))
            out = model(xb)
            xf = None
            if feat_flip is not None:
                xf = torch.from_numpy(np.ascontiguousarray(feat_flip[i:i + batch]))
            if loss_kind == "coral":
                q = torch.sigmoid(out)
                if xf is not None:
                    q = (q + torch.sigmoid(model(xf))) / 2
                p = (q > 0.5).sum(dim=1)
            else:
                q = torch.softmax(out, dim=-1)
                if xf is not None:
                    q = (q + torch.softmax(model(xf), dim=-1)) / 2
                p = q.argmax(1)
            preds.append(p.numpy())
    p = np.concatenate(preds)
    return {
        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "accuracy": accuracy_score(y, p),
        "qwk": cohen_kappa_score(y, p, weights="quadratic"),
    }, p


def main() -> None:
    ap = argparse.ArgumentParser(description="캐싱 특징 기반 헤드 학습 실험")
    ap.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "data" / "features")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--group", default="cheek",
                    help="특징 파일 접두어 (cheek | eye | ...)")
    ap.add_argument("--target", default="pore",
                    help="pore | pigmentation | wrinkle 등. 특징 파일의 labels_* 키")
    ap.add_argument("--grades", type=int, default=0,
                    help="0이면 부위 테이블의 등급 수를 그대로 쓴다. 3이면 3등급 병합.")
    ap.add_argument("--merge3", choices=["boundary", "proportional"],
                    default="boundary",
                    help="3등급 병합 경계. 볼(6등급)은 두 값이 동일하다.")
    ap.add_argument("--min-width", type=int, default=0,
                    help="크롭 폭 하한. 0이면 필터 없음. "
                         "npz 에 crop_width 가 없으면 무시하고 경고한다.")
    ap.add_argument("--eval-angles", type=int, nargs="+", default=None,
                    help="검증 세트를 각도로 한 번 더 거른다 (도메인 갭 분해용). "
                         "학습 세트에는 적용하지 않는다.")
    ap.add_argument("--min-width-split", choices=["both", "train"], default="both",
                    help="both: train/val 모두 거른다 (서비스 조건에 맞는 수치). "
                         "train: train 만 거른다 (검증셋을 고정해 임계값끼리 비교할 때).")
    ap.add_argument("--loss", choices=["ce", "focal", "coral"], default="ce")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--sampler", choices=["none", "weighted"], default="none")
    ap.add_argument("--class-weight", action="store_true", help="손실에 클래스 가중치 적용")
    ap.add_argument("--eval", choices=["standard", "device"], default="standard",
                    help="device: 디카(0)로 학습, 폰(2)으로 검증")
    ap.add_argument("--hidden", type=int, default=512, help="0이면 선형 헤드")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--no-flip-aug", action="store_true")
    ap.add_argument("--device-aug", action="store_true",
                    help="기기 열화 특징(_devaug 사이드카)을 학습에 섞는다")
    ap.add_argument("--aug-prob", type=float, default=0.5,
                    help="열화본을 쓸 확률 (0.3 / 0.5 / 0.7 비교 권장)")
    ap.add_argument("--squash", action="store_true",
                    help="종횡비 무시 리사이즈로 뽑은 _squash 특징 세트를 쓴다")
    ap.add_argument("--tta-flip", action="store_true",
                    help="평가 시 좌우반전 특징의 확률을 평균한다 (추론 엔진과 동일한 TTA)")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--results-csv", type=Path,
                    default=PROJECT_ROOT / "results" / "head_experiments.csv")
    ap.add_argument("--tag", default="", help="실험 로그에 남길 메모")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    variant = "_squash" if args.squash else ""
    # 볼(cheek)은 접두어 없이 기존 파일명을 쓴다 (기존 npz 재현성 유지).
    gp = "" if args.group == "cheek" else f"{args.group}_"
    suffix = f"{args.arch}_{args.image_size}{variant}.npz"
    if not (args.features_dir / f"{gp}train_{suffix}").exists():
        # --image-size 를 안 줘도 해당 arch 의 특징 파일을 알아서 찾는다.
        # 변형 특징(_devaug 사이드카, _squash)이 기본 특징과 섞이지 않게 거른다.
        cand = sorted(c for c in args.features_dir.glob(f"{gp}train_{args.arch}_*.npz")
                      if _matches_variant(c.stem, variant))
        if len(cand) == 1:
            suffix = cand[0].name[len(f"{gp}train_"):]
            print(f"  특징 파일 자동 선택: {cand[0].name}")
        elif len(cand) > 1:
            raise SystemExit("특징 파일이 여러 개입니다. --image-size 로 지정하세요:\n  "
                             + "\n  ".join(c.name for c in cand))
    tr_pack = load_npz(args.features_dir / f"{gp}train_{suffix}")
    va_pack = load_npz(args.features_dir / f"{gp}val_{suffix}")

    label_key = f"labels_{args.target}"
    if label_key not in tr_pack:
        have = sorted(k[len("labels_"):] for k in tr_pack if k.startswith("labels_"))
        raise SystemExit(f"특징 파일에 {label_key} 가 없습니다. 사용 가능한 target: {have}")
    if args.min_width and "crop_width" not in tr_pack:
        print("  [!] 특징 파일에 crop_width 가 없어 --min-width 를 무시합니다")

    # 열화 특징은 학습 세트에만 붙인다. 검증은 실제 기기 분포 그대로여야 한다.
    if args.device_aug:
        aug_name = f"{gp}train_{suffix}".replace(".npz", "_devaug.npz")
        aug_path = args.features_dir / aug_name
        if not aug_path.exists():
            raise SystemExit(
                f"열화 특징 파일이 없습니다: {aug_path}\n"
                "먼저 extract_features.py 를 --device-aug 로 실행하세요.")
        aug_pack = load_npz(aug_path)
        if len(aug_pack["feat"]) != len(tr_pack["feat"]):
            raise SystemExit("열화 특징의 행 수가 원본과 다릅니다. 같은 CSV로 다시 추출하세요.")
        tr_pack["feat_devaug"] = aug_pack["feat"]
        print(f"  열화 특징 사이드카: {aug_path.name}  (aug_prob={args.aug_prob})")

    full_grades = TARGET_NUM_CLASSES.get(args.target, 6)
    if args.grades == 0:
        args.grades = full_grades
    elif args.grades not in (3, full_grades):
        raise SystemExit(f"--grades 는 3 또는 {full_grades} 여야 합니다 "
                         f"(target={args.target}).")

    train_dev = 0 if args.eval == "device" else None
    val_dev = 2 if args.eval == "device" else None
    lut = merge3_lut(full_grades, args.merge3) if args.grades == 3 else None
    va_min_width = args.min_width if args.min_width_split == "both" else 0
    tr = prepare(tr_pack, args.target, args.grades, train_dev, args.min_width, lut)
    va = prepare(va_pack, args.target, args.grades, val_dev, va_min_width, lut,
                 angles=args.eval_angles)

    if args.tta_flip and va["feat_flip"] is None:
        raise SystemExit(
            f"검증 특징에 feat_flip 이 없습니다: val_{suffix}. "
            "extract_features.py 를 --split val --flip 으로 다시 실행하세요.")

    num_classes = args.grades
    counts = np.bincount(tr["y"], minlength=num_classes)
    print(f"\n{'='*66}")
    print(f"target={args.target}  grades={args.grades}  loss={args.loss}  "
          f"sampler={args.sampler}  eval={args.eval}")
    print(f"  train {len(tr['y']):,}건" + (f" (디카만)" if train_dev is not None else ""))
    print(f"  val   {len(va['y']):,}건" + (f" (폰만)" if val_dev is not None else "")
          + (f" 각도 {args.eval_angles}" if args.eval_angles else ""))
    print(f"  train 클래스 분포: " +
          " ".join(f"{c}:{n}({n/len(tr['y'])*100:.1f}%)" for c, n in enumerate(counts)))
    va_counts = np.bincount(va["y"], minlength=num_classes)
    print(f"  val   클래스 분포: " +
          " ".join(f"{c}:{n}" for c, n in enumerate(va_counts)))
    empty = [c for c, n in enumerate(va_counts) if n == 0]
    if empty:
        print(f"  [!] 검증셋에 없는 클래스: {empty} - macro-F1 이 왜곡됩니다")

    # 특징 표준화 (train 기준)
    mu, sd = tr["feat"].mean(0), tr["feat"].std(0) + 1e-6
    for pack in (tr, va):
        pack["feat"] = ((pack["feat"] - mu) / sd).astype(np.float32)
        if pack["feat_flip"] is not None:
            pack["feat_flip"] = ((pack["feat_flip"] - mu) / sd).astype(np.float32)
        if pack.get("feat_devaug") is not None:
            pack["feat_devaug"] = ((pack["feat_devaug"] - mu) / sd).astype(np.float32)

    ds = FeatDataset(tr["feat"], tr["feat_flip"], tr.get("feat_devaug"), tr["y"],
                     augment=not args.no_flip_aug, aug_prob=args.aug_prob)
    if args.sampler == "weighted":
        w = (1.0 / np.maximum(counts, 1))[tr["y"]]
        sampler = WeightedRandomSampler(torch.DoubleTensor(w), len(w), replacement=True)
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, drop_last=True)
    else:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if len(loader) == 0:
        raise SystemExit(
            f"학습 배치를 만들 수 없습니다 (표본 {len(tr['y'])}건 < 배치 {args.batch_size}). "
            "--batch-size 를 줄이세요.")

    in_dim = tr["feat"].shape[1]
    if args.loss == "coral":
        model = CoralHead(in_dim, num_classes, args.hidden, args.dropout)
    else:
        model = MLPHead(in_dim, num_classes, args.hidden, args.dropout)

    cw = None
    if args.class_weight:
        cw = torch.tensor(len(tr["y"]) / (num_classes * np.maximum(counts, 1)),
                          dtype=torch.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)

    best = {"macro_f1": -1.0}
    best_epoch, bad, t0 = 0, 0, time.time()
    print(f"\n{'epoch':>5} {'loss':>8} {'macroF1':>8} {'acc':>7} {'QWK':>7}")
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            yb = yb.long()
            out = model(xb)
            if args.loss == "coral":
                loss = coral_loss(out, yb, cw)
            elif args.loss == "focal":
                loss = focal_loss(out, yb, args.focal_gamma, cw)
            else:
                loss = F.cross_entropy(out, yb, weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(yb)

        m, _ = evaluate(model, va["feat"], va["y"], args.loss,
                        feat_flip=va["feat_flip"] if args.tta_flip else None)
        sched.step(m["macro_f1"])
        mark = ""
        if m["macro_f1"] > best["macro_f1"]:
            best, best_epoch, bad = m, ep, 0
            mark = "  *"
        else:
            bad += 1
        if ep <= 3 or ep % 5 == 0 or mark:
            print(f"{ep:>5} {tot/len(ds):>8.4f} {m['macro_f1']:>8.4f} "
                  f"{m['accuracy']:>7.4f} {m['qwk']:>7.4f}{mark}")
        if args.patience and bad >= args.patience:
            print(f"  조기 종료 (macro-F1 {args.patience}에폭 개선 없음)")
            break

    elapsed = time.time() - t0
    print(f"\n최고 성능 - epoch {best_epoch}")
    print(f"  macro-F1 {best['macro_f1']:.4f} | accuracy {best['accuracy']:.4f} | QWK {best['qwk']:.4f}")
    print(f"  소요 {elapsed:.1f}초 ({elapsed/max(ep,1):.2f}초/에폭)")

    # 기기별 검증 성능 - 도메인 갭 확인
    per_dev = {}
    if args.eval == "standard":
        for d in (0, 1, 2):
            sel = va["device"] == d
            if sel.sum() < 10:
                continue
            m, _ = evaluate(model, va["feat"][sel], va["y"][sel], args.loss,
                            feat_flip=(va["feat_flip"][sel]
                                       if args.tta_flip and va["feat_flip"] is not None
                                       else None))
            per_dev[DEVICE_NAME[d]] = round(m["macro_f1"], 4)
        if per_dev:
            print("  기기별 macro-F1: " + "  ".join(f"{k} {v}" for k, v in per_dev.items()))

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "target": args.target, "grades": args.grades, "loss": args.loss,
        "sampler": args.sampler, "class_weight": args.class_weight,
        "eval": args.eval, "arch": args.arch, "hidden": args.hidden,
        "lr": args.lr, "seed": args.seed, "epochs_run": ep, "best_epoch": best_epoch,
        "n_train": len(tr["y"]), "n_val": len(va["y"]),
        "macro_f1": round(best["macro_f1"], 4),
        "accuracy": round(best["accuracy"], 4),
        "qwk": round(best["qwk"], 4),
        "per_device_f1": json.dumps(per_dev, ensure_ascii=False),
        "seconds": round(elapsed, 1),
        "tag": args.tag,
    }
    new = not args.results_csv.exists()
    with args.results_csv.open("a", newline="", encoding="utf-8-sig") as f:
        wri = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            wri.writeheader()
        wri.writerow(row)
    print(f"  실험 로그 누적: {args.results_csv}")


if __name__ == "__main__":
    main()
