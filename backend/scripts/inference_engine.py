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
        image_size: int = 448
    ):
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
        
        # 부위별 등급 상한이 다르다 (docs/labeling_codes_guide.md 참고).
        #   미간 0~2 / 이마 색소 0~3 / 이마 주름·입술 0~4
        #   눈가 주름·볼 모공·볼 색소·턱 처짐 0~5
        # severity 어휘는 계약(backend/docs/ai_inference_contract.md)상
        # normal / mild / moderate / severe 네 가지로 고정이므로 그 안에서 매핑한다.
        # 값 자체는 multivalue_parser._ZERO_12_34_5 (dev JSON 경로가 쓰는 정본)와 동일하다.
        #
        # 기존 map 은 두 가지로 새고 있었다.
        #   등급 5 -> .get(pred, "unknown") 이 걸려 "unknown"
        #   등급 4 -> 어휘 밖 값 "very_severe"
        # 소비처(recommendation_service/report_service/report_cards)가 모두
        # _SEVERITY_ORDER.get(severity, 0) 패턴이라, 어휘 밖 값은 예외 없이
        # 가장 낮은 등급으로 조용히 강등된다. 즉 최상위 등급이 정렬에서 최하위로
        # 취급되고 있었다.
        self.severity_map = {
            0: "normal", 1: "mild", 2: "mild",
            3: "moderate", 4: "moderate", 5: "severe",
        }

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
    def predict(self, image: Image.Image, bbox: Optional[List[int]] = None, use_tta: bool = True) -> Dict[str, Any]:
        if not self.backbone or not self.heads:
            raise RuntimeError("Models are not loaded.")

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
        
        return {
            "grade_value": pred_class,
            "severity": self.severity_map.get(pred_class, "unknown"),
            "confidence_score": round(float(probs_np[pred_class]), 4),
        }
