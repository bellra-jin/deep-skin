"""크롭 이미지에서 백본 특징을 1회 추출해 캐싱한다.

왜 캐싱인가
-----------
CUDA 미지원 환경(내장 GPU)에서 ResNet-50 전체 미세조정은 에폭당 1시간을 넘는다.
백본을 얼리고 특징을 한 번만 뽑아 두면, 이후 헤드 학습은 에폭당 수 초로 끝난다.
손실함수 · 등급 병합 · 샘플러 · 평가 프로토콜을 수십 번 바꿔 비교할 수 있다.

전처리는 학습과 동일하게 맞춘다(ResizeAndPad → ToTensor → Normalize).
증강은 캐싱과 상충하므로, 좌우반전 특징을 함께 저장해 헤드 학습 때 선택하도록 한다.

산출물
------
{out}/{split}_features.npz
    feat       (N, D) float32   원본 특징
    feat_flip  (N, D) float32   좌우반전 특징 (--flip 일 때만)
    labels_pore, labels_pigmentation  (N,) int16   (-1 = 결측)
    person, device, angle, facepart, image_path    (N,) 메타

사용법
------
    uv run python notebooks/jh/extract_features.py --split train --flip
    uv run python notebooks/jh/extract_features.py --split val
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PAD_FILL = (124, 116, 104)  # dataset.py 의 ResizeAndPad 와 동일


class ResizeAndPad:
    """비율을 유지한 채 정사각형 캔버스에 패딩. dataset.py 와 동일한 규칙."""

    def __init__(self, image_size: int, fill=PAD_FILL):
        self.image_size = image_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.image_size / max(w, h)
        rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((rw, rh), Image.Resampling.BILINEAR)
        pw, ph = self.image_size - rw, self.image_size - rh
        return ImageOps.expand(
            img,
            border=(pw // 2, ph // 2, pw - pw // 2, ph - ph // 2),
            fill=self.fill,
        )


class CropDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, flip: bool,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.paths = df["image_path"].tolist()
        self.flip = flip
        self.tf = transforms.Compose([
            ResizeAndPad(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.tf(img)
        if self.flip:
            return x, self.tf(ImageOps.mirror(img))
        return x, torch.zeros(1)


def arch_slug(arch: str) -> str:
    """arch 이름을 파일명에 쓸 수 있는 형태로."""
    return (arch.replace("timm:", "").replace("hf_hub:", "")
                .replace("/", "-").replace(":", "-").replace(".", "-"))


# 자주 쓰는 백본의 짧은 별칭.  파라미터 수를 ResNet-50(25.6M)과 맞춰 두어
# "사전학습 방식만 다른 비교"가 되게 한다.
ARCH_ALIAS = {
    "dinov3_s":  "timm:vit_small_patch16_dinov3.lvd1689m",   # 21.6M, 384d, 256px
    "dinov3_b":  "timm:vit_base_patch16_dinov3.lvd1689m",    # 86M,  768d, 256px
    "dinov2_s":  "timm:vit_small_patch14_dinov2.lvd142m",
    "dinov2_b":  "timm:vit_base_patch14_dinov2.lvd142m",
    "siglip2_b": "timm:vit_base_patch16_siglip_224.v2_webli",
}


def build_backbone(arch: str, weights_path: Path | None, image_size: int):
    """분류 head 를 제거한 특징 추출기, 출력 차원, 전처리 규격을 반환.

    반환: (net, dim, mean, std, image_size)
    """
    arch = ARCH_ALIAS.get(arch, arch)

    if arch.startswith("timm:") or arch.startswith("hf_hub:"):
        try:
            import timm
        except ImportError:
            raise SystemExit("timm 이 필요합니다.  pyproject 에 timm 을 추가하고 uv sync 하세요.")
        name = arch.split(":", 1)[1]
        # "org/repo" 형태면 HF 저장소를 직접 지정하고, 아니면 timm 레지스트리 이름으로 본다.
        ref = f"hf_hub:{name}" if "/" in name else name
        net = timm.create_model(ref, pretrained=True, num_classes=0)
        cfg = timm.data.resolve_model_data_config(net)
        dim = net.num_features
        size = image_size or cfg["input_size"][-1]
        mean, std = list(cfg["mean"]), list(cfg["std"])
        print(f"  timm 백본: {name}  ({sum(p.numel() for p in net.parameters())/1e6:.1f}M "
              f"파라미터, {dim}차원, {size}px)")
    elif arch == "resnet50":
        net = models.resnet50(weights=None)
        dim = net.fc.in_features
        if weights_path and weights_path.exists():
            net.load_state_dict(torch.load(weights_path, map_location="cpu"))
            print(f"  로컬 가중치 사용: {weights_path}")
        else:
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            print("  torchvision ImageNet 가중치 사용 (최초 1회 다운로드)")
        net.fc = nn.Identity()
        size, mean, std = image_size or 224, IMAGENET_MEAN, IMAGENET_STD
    elif arch == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        dim = 512
        net.fc = nn.Identity()
        size, mean, std = image_size or 224, IMAGENET_MEAN, IMAGENET_STD
    else:
        raise SystemExit(f"지원하지 않는 arch: {arch}")

    net.eval()
    return net, dim, mean, std, size


@torch.inference_mode()
def extract(net, loader, dim: int, n: int, flip: bool, threads: int):
    torch.set_num_threads(threads)
    feat = np.zeros((n, dim), dtype=np.float32)
    feat_flip = np.zeros((n, dim), dtype=np.float32) if flip else None

    done, t0, last = 0, time.time(), time.time()
    for x, xf in loader:
        b = x.shape[0]
        feat[done:done + b] = net(x).numpy()
        if flip:
            feat_flip[done:done + b] = net(xf).numpy()
        done += b
        if time.time() - last > 20:
            el = time.time() - t0
            rate = done / el
            print(f"  {done:,}/{n:,}  ({done/n*100:.1f}%)  "
                  f"{rate:.1f} img/s  남은 시간 약 {(n-done)/max(rate,1e-9)/60:.1f}분",
                  flush=True)
            last = time.time()
    print(f"  완료 - {done:,}장, {(time.time()-t0)/60:.1f}분")
    return feat, feat_flip


def main() -> None:
    ap = argparse.ArgumentParser(description="크롭 특징 추출 및 캐싱")
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--csv", type=Path, default=None,
                    help="기본: data/processed/cheek_{split}_metadata.csv")
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "data" / "features")
    ap.add_argument("--arch", default="resnet50",
                    help="resnet50 | resnet18 | dinov3_s | dinov3_b | dinov2_s | "
                         "dinov2_b | siglip2_b | timm:<모델명>")
    ap.add_argument("--weights", type=Path,
                    default=PROJECT_ROOT / "checkpoints" / "pretrained" / "resnet50-0676ba61.pth",
                    help="로컬 사전학습 가중치. 없으면 torchvision 에서 받는다.")
    ap.add_argument("--image-size", type=int, default=0,
                    help="0이면 백본의 기본 입력 크기를 따른다 (resnet=224)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU 스레드 수. 0이면 자동.")
    ap.add_argument("--flip", action="store_true",
                    help="좌우반전 특징도 함께 저장 (train 권장)")
    ap.add_argument("--limit", type=int, default=0, help="디버그용 상한")
    args = ap.parse_args()

    csv_path = args.csv or (PROJECT_ROOT / "data" / "processed"
                            / f"cheek_{args.split}_metadata.csv")
    if not csv_path.exists():
        raise SystemExit(f"메타데이터 CSV 가 없습니다: {csv_path}")

    cols = ["image_path", "id", "device", "angle", "facepart",
            "pore_label", "pigmentation_label"]
    df = pd.read_csv(csv_path, usecols=cols)
    if args.limit:
        df = df.head(args.limit)
    missing = [p for p in df["image_path"].head(5) if not Path(p).exists()]
    if missing:
        raise SystemExit(f"크롭 이미지를 찾을 수 없습니다: {missing[0]}")

    threads = args.threads or torch.get_num_threads()
    print(f"\nsplit={args.split}  n={len(df):,}  arch={args.arch}  "
          f"flip={args.flip}  threads={threads}")

    net, dim, mean, std, size = build_backbone(args.arch, args.weights, args.image_size)

    ds = CropDataset(df, size, args.flip, mean, std)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)

    print(f"\n특징 추출 시작 (차원 {dim}, 입력 {size}px)")
    feat, feat_flip = extract(net, loader, dim, len(df), args.flip, threads)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.split}_{arch_slug(args.arch)}_{size}.npz"

    payload = dict(
        feat=feat,
        labels_pore=df["pore_label"].fillna(-1).astype(np.int16).to_numpy(),
        labels_pigmentation=df["pigmentation_label"].fillna(-1).astype(np.int16).to_numpy(),
        person=df["id"].astype(str).to_numpy(),
        device=df["device"].fillna(-1).astype(np.int16).to_numpy(),
        angle=df["angle"].fillna(-1).astype(np.int16).to_numpy(),
        facepart=df["facepart"].fillna(-1).astype(np.int16).to_numpy(),
        image_path=df["image_path"].astype(str).to_numpy(),
    )
    if feat_flip is not None:
        payload["feat_flip"] = feat_flip

    np.savez_compressed(out, **payload)
    mb = out.stat().st_size / 1e6
    print(f"\n저장: {out}  ({mb:.0f} MB)")
    print(f"  feat shape = {feat.shape}")
    if feat_flip is not None:
        print(f"  feat_flip shape = {feat_flip.shape}")


if __name__ == "__main__":
    main()
