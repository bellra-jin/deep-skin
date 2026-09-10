# -*- coding: utf-8 -*-
"""
Inference Engine for DINOv3 Ensemble Models.
Encapsulates backbone/head loading and prediction logic.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Optional, List, Dict, Any

# severity 매핑은 backend/app/services/severity_scale.py 가 정본이다.
# 이 엔진은 자체 테이블을 두지 않는다 - 두면 파서 쪽과 갈라진다.
# 파서를 직접 import 하지 않는 이유는 그쪽이 app.models 를 통해 SQLAlchemy 를
# 끌고 오기 때문이다. AI 서버가 매핑 하나 때문에 DB 스택을 로드할 이유가 없다.
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)
from app.services.severity_scale import (  # noqa: E402
    grade_to_severity,
    num_classes_for,
)

# ===================================================================
# DINOv3 Model Definitions
# ===================================================================
class LinearHead(nn.Module):
    def __init__(self, feature_dim, num_classes, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.dropout(self.norm(x)))

class DinoInferenceEngine:
    def __init__(
        self, 
        backbone_ckpt: str, 
        heads_dir: str, 
        device: str = None,
        image_size: int = 448,
        annotation_key: Optional[str] = None,
    ):
        # 이 엔진 인스턴스가 어떤 라벨을 예측하는지. severity 위임에 필요하다.
        # 부위마다 등급 상한이 다르므로(이마 색소 0~5 / 눈가 주름 0~6 / 입술 0~4)
        # 라벨 이름 없이는 등급을 severity 로 옮길 수 없다.
        self.annotation_key = annotation_key
        self.backbone_ckpt = backbone_ckpt
        self.heads_dir = heads_dir
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = image_size
        
        self.backbone = None
        self.heads = []
        self.num_classes = 0
        self.feature_dim = 0
        
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        

    def _setup_dinov3_path(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_path = os.path.join(script_dir, "dinov3")
        if os.path.isdir(repo_path) and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        
        try:
            from dinov3.models.vision_transformer import DinoVisionTransformer
            return DinoVisionTransformer
        except ImportError:
            print(f"Error: DINOv3 repository not found at {repo_path}.")
            return None

    def load_models(self):
        DinoVisionTransformer = self._setup_dinov3_path()
        if DinoVisionTransformer is None:
            return False

        print(f"Loading backbone from {self.backbone_ckpt}...")
        # ViT-S Params default
        self.backbone = DinoVisionTransformer(
            img_size=self.image_size, patch_size=16, in_chans=3,
            pos_embed_rope_base=100.0,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2.0,
            pos_embed_rope_dtype="fp32",
            embed_dim=384, depth=12, num_heads=6, ffn_ratio=6,
            qkv_bias=True, drop_path_rate=0.0, layerscale_init=1e-5,
            norm_layer="layernormbf16", ffn_layer="swiglu",
            ffn_bias=True, proj_bias=True,
            n_storage_tokens=4, mask_k_bias=True,
        )
        
        if os.path.exists(self.backbone_ckpt):
            sd = torch.load(self.backbone_ckpt, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(sd, strict=False)
        else:
            print(f"Warning: Backbone checkpoint not found: {self.backbone_ckpt}")
            return False
        
        self.backbone = self.backbone.to(self.device).eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Load Ensemble Heads
        head_paths = [os.path.join(self.heads_dir, f"head_fold{i+1}.pth") for i in range(5)]
        self.heads = []
        for hp in head_paths:
            if not os.path.exists(hp):
                continue
            sd = torch.load(hp, map_location=self.device, weights_only=True)
            if self.num_classes == 0:
                self.num_classes = sd["num_classes"]
                self.feature_dim = sd["feature_dim"]
            
            model = LinearHead(self.feature_dim, self.num_classes).to(self.device)
            model.load_state_dict(sd["head_state"])
            model.eval()
            self.heads.append(model)
        
        print(f"Loaded {len(self.heads)} models for ensemble. Device: {self.device}")
        return len(self.heads) > 0

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        bbox: Optional[List[int]] = None,
        use_tta: bool = True,
        annotation_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.backbone or not self.heads:
            raise RuntimeError("Models are not loaded.")

        key = annotation_key or self.annotation_key
        if not key:
            raise ValueError(
                "annotation_key 가 필요합니다. 부위마다 등급 상한이 달라 "
                "라벨 이름 없이는 severity 를 정할 수 없습니다 "
                "(예: 'l_perocular_wrinkle'). 생성자나 predict 인자로 주세요.")

        if bbox:
            # bbox: [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            # Simple expansion logic
            mx, my = int(w * 0.15), int(h * 0.15)
            W, H = image.size
            crop_coords = (max(0, x1-mx), max(0, y1-my), min(W, x2+mx), min(H, y2+my))
            image = image.crop(crop_coords)

        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        # Autocast for efficiency if CUDA
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16) if self.device.type == 'cuda' else torch.no_grad():
            feat = self.backbone(input_tensor)
        
        feat = feat.float()

        all_probs = []
        for model in self.heads:
            probs = torch.softmax(model(feat), dim=-1)
            if use_tta:
                feat_flip = self.backbone(self.transform(image.transpose(Image.FLIP_LEFT_RIGHT).convert("RGB")).unsqueeze(0).to(self.device)).float()
                probs = (probs + torch.softmax(model(feat_flip), dim=-1)) / 2
            all_probs.append(probs)
        
        ensemble_probs = torch.stack(all_probs).mean(dim=0)
        probs_np = ensemble_probs[0].cpu().numpy()
        pred_class = int(np.argmax(probs_np))
        
        severity = grade_to_severity(key, pred_class)
        if severity is None:
            # 조용히 "unknown" 을 내보내면 소비처에서 최저 등급으로 강등된다.
            # 매핑이 못 덮는 등급이 나왔다는 것 자체가 드러나야 한다.
            print(f"[inference-engine] severity 매핑 없음: "
                  f"key={key} grade={pred_class} "
                  f"(등급 수 {self.num_classes}, 매핑 상한 {num_classes_for(key)})")
        return {
            "grade_value": pred_class,
            "severity": severity,
            "confidence_score": round(float(probs_np[pred_class]), 4),
        }
