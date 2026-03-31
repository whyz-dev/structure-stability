from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from augmentations import build_default_transforms


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint_path: Path
    arch: str
    backbone_name: str
    img_size: int
    saved_logloss: float | None


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(q_tokens)
        kv = self.norm_kv(kv_tokens)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        return q_tokens + attn_out


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)


class DevDataset(Dataset):
    def __init__(self, data_dir: Path, img_size: int):
        self.df = pd.read_csv(data_dir / "dev.csv", encoding="utf-8-sig").reset_index(drop=True)
        self.root_dir = data_dir / "dev"
        _, self.transform = build_default_transforms(img_size)
        self.label_map = {"stable": 0, "unstable": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sample_id = str(row["id"])
        views = []
        for name in ("front", "top"):
            image = Image.open(self.root_dir / sample_id / f"{name}.png").convert("RGB")
            views.append(self.transform(image))
        label = self.label_map[str(row["label"])]
        return sample_id, views, label


class MultiViewBidir(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        attn_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        pos_grid: int = 7,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 512,
        classifier_mid_dim: int = 128,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="")
        feature_dim = self.backbone.num_features
        self.feature_dim = feature_dim
        self.proj = nn.Conv2d(feature_dim, attn_dim, kernel_size=1, bias=False)
        self.pos_embed = nn.Parameter(torch.randn(1, attn_dim, pos_grid, pos_grid) * 0.02)
        self.cross_12 = nn.ModuleList([CrossAttentionBlock(attn_dim, num_heads, dropout) for _ in range(num_layers)])
        self.cross_21 = nn.ModuleList([CrossAttentionBlock(attn_dim, num_heads, dropout) for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim * 2, classifier_hidden_dim),
            nn.BatchNorm1d(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, classifier_mid_dim),
            nn.ReLU(),
            nn.Linear(classifier_mid_dim, 1),
        )

    def _to_tokens(self, feat_map: torch.Tensor) -> torch.Tensor:
        if feat_map.ndim == 4 and feat_map.shape[-1] == self.feature_dim:
            feat_map = feat_map.permute(0, 3, 1, 2)
        feat_map = self.proj(feat_map)
        pos = F.interpolate(self.pos_embed, size=feat_map.shape[-2:], mode="bilinear", align_corners=False)
        feat_map = feat_map + pos
        return feat_map.flatten(2).transpose(1, 2)

    def extract_features(self, views: list[torch.Tensor]) -> torch.Tensor:
        f1 = self.backbone.forward_features(views[0])
        f2 = self.backbone.forward_features(views[1])
        t1 = self._to_tokens(f1)
        t2 = self._to_tokens(f2)
        for blk12, blk21 in zip(self.cross_12, self.cross_21):
            t1_prev, t2_prev = t1, t2
            t1 = blk12(t1_prev, t2_prev)
            t2 = blk21(t2_prev, t1_prev)
        return torch.cat([t1.mean(dim=1), t2.mean(dim=1)], dim=1)

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        feat = self.extract_features(views)
        return self.classifier(feat).view(-1)


class FlexibleCAF(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        attn_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 512,
        classifier_mid_dim: int = 128,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="")
        self.feature_dim = self.backbone.num_features
        self.cnn_proj = nn.Conv2d(self.feature_dim, attn_dim, kernel_size=1, bias=False)
        self.token_proj = nn.Linear(self.feature_dim, attn_dim, bias=False)
        self.cross_12 = nn.ModuleList([CrossAttentionBlock(attn_dim, num_heads, dropout) for _ in range(num_layers)])
        self.cross_21 = nn.ModuleList([CrossAttentionBlock(attn_dim, num_heads, dropout) for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim * 2, classifier_hidden_dim),
            nn.BatchNorm1d(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, classifier_mid_dim),
            nn.ReLU(),
            nn.Linear(classifier_mid_dim, 1),
        )

    def _to_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim == 4:
            if feat.shape[1] == self.feature_dim:
                feat = self.cnn_proj(feat)
                return feat.flatten(2).transpose(1, 2)
            if feat.shape[-1] == self.feature_dim:
                feat = feat.permute(0, 3, 1, 2)
                feat = self.cnn_proj(feat)
                return feat.flatten(2).transpose(1, 2)
        if feat.ndim == 3:
            if feat.shape[-1] == self.feature_dim:
                return self.token_proj(feat)
            if feat.shape[1] == self.feature_dim:
                return self.token_proj(feat.transpose(1, 2))
        raise ValueError(f"Unsupported feature shape: {tuple(feat.shape)}")

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        f1 = self.backbone.forward_features(views[0])
        f2 = self.backbone.forward_features(views[1])
        t1 = self._to_tokens(f1)
        t2 = self._to_tokens(f2)
        for blk12, blk21 in zip(self.cross_12, self.cross_21):
            t1_prev, t2_prev = t1, t2
            t1 = blk12(t1_prev, t2_prev)
            t2 = blk21(t2_prev, t1_prev)
        feat = torch.cat([t1.mean(dim=1), t2.mean(dim=1)], dim=1)
        return self.classifier(feat).view(-1)


class DANNModel(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="")
        feature_dim = self.backbone.num_features
        self.proj = nn.Conv2d(feature_dim, 256, kernel_size=1, bias=False)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, 7, 7) * 0.02)
        self.cross_12 = nn.ModuleList([CrossAttentionBlock(256, 8, 0.1) for _ in range(2)])
        self.cross_21 = nn.ModuleList([CrossAttentionBlock(256, 8, 0.1) for _ in range(2)])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def _to_tokens(self, feat_map: torch.Tensor) -> torch.Tensor:
        feat_map = self.proj(feat_map)
        pos = F.interpolate(self.pos_embed, size=feat_map.shape[-2:], mode="bilinear", align_corners=False)
        feat_map = feat_map + pos
        return feat_map.flatten(2).transpose(1, 2)

    def extract_features(self, views: list[torch.Tensor]) -> torch.Tensor:
        f1 = self.backbone.forward_features(views[0])
        f2 = self.backbone.forward_features(views[1])
        t1 = self._to_tokens(f1)
        t2 = self._to_tokens(f2)
        for blk12, blk21 in zip(self.cross_12, self.cross_21):
            t1_prev, t2_prev = t1, t2
            t1 = blk12(t1_prev, t2_prev)
            t2 = blk21(t2_prev, t1_prev)
        return torch.cat([t1.mean(dim=1), t2.mean(dim=1)], dim=1)

    def forward_class_logits(self, views: list[torch.Tensor]) -> torch.Tensor:
        feat = self.extract_features(views)
        return self.classifier(feat).view(-1)


class TeacherRegularizedModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        physics_dim: int,
        image_dim: int,
        attn_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        pos_grid: int = 7,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 512,
        classifier_mid_dim: int = 128,
        classifier_dropout: float = 0.3,
        domain_hidden_dim: int = 256,
        domain_dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="")
        feature_dim = self.backbone.num_features
        self.feature_dim = feature_dim
        self.proj = nn.Conv2d(feature_dim, attn_dim, kernel_size=1, bias=False)
        self.pos_embed = nn.Parameter(torch.randn(1, attn_dim, pos_grid, pos_grid) * 0.02)
        self.cross_12 = nn.ModuleList([CrossAttentionBlock(attn_dim, num_heads, dropout) for _ in range(num_layers)])
        self.cross_21 = nn.ModuleList([CrossAttentionBlock(attn_dim, num_heads, dropout) for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim * 2, classifier_hidden_dim),
            nn.BatchNorm1d(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, classifier_mid_dim),
            nn.ReLU(),
            nn.Linear(classifier_mid_dim, 1),
        )
        self.physics_head = nn.Sequential(
            nn.Linear(attn_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, physics_dim),
        )
        self.image_head = nn.Sequential(
            nn.Linear(attn_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, image_dim),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(attn_dim * 2, domain_hidden_dim),
            nn.ReLU(),
            nn.Dropout(domain_dropout),
            nn.Linear(domain_hidden_dim, 1),
        )

    def _to_tokens(self, feat_map: torch.Tensor) -> torch.Tensor:
        if feat_map.ndim == 4 and feat_map.shape[-1] == self.feature_dim:
            feat_map = feat_map.permute(0, 3, 1, 2)
        feat_map = self.proj(feat_map)
        pos = F.interpolate(self.pos_embed, size=feat_map.shape[-2:], mode="bilinear", align_corners=False)
        feat_map = feat_map + pos
        return feat_map.flatten(2).transpose(1, 2)

    def extract_features(self, views: list[torch.Tensor]) -> torch.Tensor:
        f1 = self.backbone.forward_features(views[0])
        f2 = self.backbone.forward_features(views[1])
        t1 = self._to_tokens(f1)
        t2 = self._to_tokens(f2)
        for blk12, blk21 in zip(self.cross_12, self.cross_21):
            t1_prev, t2_prev = t1, t2
            t1 = blk12(t1_prev, t2_prev)
            t2 = blk21(t2_prev, t1_prev)
        return torch.cat([t1.mean(dim=1), t2.mean(dim=1)], dim=1)

    def forward(self, views: list[torch.Tensor], lambda_=0.0) -> dict[str, torch.Tensor]:
        feat = self.extract_features(views)
        return {
            "class_logit": self.classifier(feat).view(-1),
            "physics_pred": self.physics_head(feat),
            "image_pred": self.image_head(feat),
            "domain_logit": self.domain_head(grad_reverse(feat, lambda_)).view(-1),
        }


class FlexibleTeacherRegularizedModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        physics_dim: int,
        image_dim: int,
        attn_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 512,
        classifier_mid_dim: int = 128,
        classifier_dropout: float = 0.3,
        domain_hidden_dim: int = 256,
        domain_dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="")
        self.feature_dim = self.backbone.num_features
        self.cnn_proj = nn.Conv2d(self.feature_dim, attn_dim, kernel_size=1, bias=False)
        self.token_proj = nn.Linear(self.feature_dim, attn_dim, bias=False)
        self.cross_12 = nn.ModuleList([CrossAttentionBlock(attn_dim, num_heads, dropout) for _ in range(num_layers)])
        self.cross_21 = nn.ModuleList([CrossAttentionBlock(attn_dim, num_heads, dropout) for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim * 2, classifier_hidden_dim),
            nn.BatchNorm1d(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, classifier_mid_dim),
            nn.ReLU(),
            nn.Linear(classifier_mid_dim, 1),
        )
        self.physics_head = nn.Sequential(
            nn.Linear(attn_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, physics_dim),
        )
        self.image_head = nn.Sequential(
            nn.Linear(attn_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, image_dim),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(attn_dim * 2, domain_hidden_dim),
            nn.ReLU(),
            nn.Dropout(domain_dropout),
            nn.Linear(domain_hidden_dim, 1),
        )

    def _to_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim == 4:
            if feat.shape[1] == self.feature_dim:
                feat = self.cnn_proj(feat)
                return feat.flatten(2).transpose(1, 2)
            if feat.shape[-1] == self.feature_dim:
                feat = feat.permute(0, 3, 1, 2)
                feat = self.cnn_proj(feat)
                return feat.flatten(2).transpose(1, 2)
        if feat.ndim == 3:
            if feat.shape[-1] == self.feature_dim:
                return self.token_proj(feat)
            if feat.shape[1] == self.feature_dim:
                return self.token_proj(feat.transpose(1, 2))
        raise ValueError(f"Unsupported feature shape: {tuple(feat.shape)}")

    def extract_features(self, views: list[torch.Tensor]) -> torch.Tensor:
        f1 = self.backbone.forward_features(views[0])
        f2 = self.backbone.forward_features(views[1])
        t1 = self._to_tokens(f1)
        t2 = self._to_tokens(f2)
        for blk12, blk21 in zip(self.cross_12, self.cross_21):
            t1_prev, t2_prev = t1, t2
            t1 = blk12(t1_prev, t2_prev)
            t2 = blk21(t2_prev, t1_prev)
        return torch.cat([t1.mean(dim=1), t2.mean(dim=1)], dim=1)

    def forward(self, views: list[torch.Tensor], lambda_=0.0) -> dict[str, torch.Tensor]:
        feat = self.extract_features(views)
        return {
            "class_logit": self.classifier(feat).view(-1),
            "physics_pred": self.physics_head(feat),
            "image_pred": self.image_head(feat),
            "domain_logit": self.domain_head(grad_reverse(feat, lambda_)).view(-1),
        }


BACKBONE_PATTERNS = [
    ("swin_tiny_patch4_window7_224", "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"),
    ("swin_tiny_patch4_window7_224_ms_in22k_ft_in1k", "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"),
    ("swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"),
    ("convnext_small_fb_in22k_ft_in1k", "convnext_small.fb_in22k_ft_in1k"),
    ("convnext_small.fb_in22k_ft_in1k", "convnext_small.fb_in22k_ft_in1k"),
    ("convnext_tiny_fb_in22k_ft_in1k", "convnext_tiny.fb_in22k_ft_in1k"),
    ("convnext_tiny.fb_in22k_ft_in1k", "convnext_tiny.fb_in22k_ft_in1k"),
    ("deit3_small_patch16_224_fb_in22k_ft_in1k", "deit3_small_patch16_224.fb_in22k_ft_in1k"),
    ("deit3_small_patch16_224.fb_in22k_ft_in1k", "deit3_small_patch16_224.fb_in22k_ft_in1k"),
    ("vit_small_patch16_224_augreg_in21k_ft_in1k", "vit_small_patch16_224.augreg_in21k_ft_in1k"),
    ("vit_base_patch16_224_augreg_in21k_ft_in1k", "vit_base_patch16_224.augreg_in21k_ft_in1k"),
    ("efficientnetv2_rw_s", "efficientnetv2_rw_s"),
]


def infer_backbone_name(checkpoint_path: Path, cfg: dict | None) -> str:
    if isinstance(cfg, dict) and "BACKBONE_NAME" in cfg:
        raw = str(cfg["BACKBONE_NAME"])
        for pattern, backbone in BACKBONE_PATTERNS:
            if raw == pattern:
                return backbone
        return raw
    text = checkpoint_path.as_posix()
    for pattern, backbone in BACKBONE_PATTERNS:
        if pattern in text:
            return backbone
    if any(prefix in text for prefix in ["/baseline_", "/dann_", "/teacher_regularization_", "/mv_caf_efficientnet_"]):
        return "efficientnetv2_rw_s"
    raise ValueError(f"Unable to infer backbone from {checkpoint_path}")


def infer_model_spec(checkpoint_path: Path) -> ModelSpec | None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["ema_state_dict"] if "ema_state_dict" in ckpt else ckpt["model_state_dict"]
    cfg = ckpt.get("cfg")
    keys = set(state)

    if "front_backbone.conv_stem.weight" in keys:
        return None

    if "domain_classifier.0.weight" in keys:
        arch = "dann"
    elif "physics_head.3.weight" in keys and "cnn_proj.weight" in keys:
        arch = "teacher_flexible"
    elif "physics_head.3.weight" in keys:
        arch = "teacher"
    elif "cnn_proj.weight" in keys and "token_proj.weight" in keys:
        arch = "flexible_caf"
    elif "pos_embed" in keys:
        arch = "bidir"
    else:
        return None

    img_size = int(cfg.get("IMG_SIZE", 320 if arch in {"bidir", "dann", "teacher"} else 224)) if isinstance(cfg, dict) else (320 if arch in {"bidir", "dann", "teacher"} else 224)
    return ModelSpec(
        name=checkpoint_path.stem,
        checkpoint_path=checkpoint_path,
        arch=arch,
        backbone_name=infer_backbone_name(checkpoint_path, cfg),
        img_size=img_size,
        saved_logloss=ckpt.get("dev_logloss"),
    )


def build_model(spec: ModelSpec):
    ckpt = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg") if isinstance(ckpt, dict) else None

    def cfg_get(name: str, default):
        return cfg.get(name, default) if isinstance(cfg, dict) else default

    if spec.arch == "bidir":
        return MultiViewBidir(
            spec.backbone_name,
            attn_dim=cfg_get("ATTN_DIM", 256),
            num_heads=cfg_get("NUM_HEADS", 8),
            num_layers=cfg_get("NUM_LAYERS", 2),
            pos_grid=cfg_get("POS_GRID", 7),
            dropout=cfg_get("DROPOUT", 0.1),
            classifier_hidden_dim=cfg_get("CLASSIFIER_HIDDEN_DIM", 512),
            classifier_mid_dim=cfg_get("CLASSIFIER_MID_DIM", 128),
            classifier_dropout=cfg_get("CLASSIFIER_DROPOUT", 0.3),
        )
    if spec.arch == "flexible_caf":
        return FlexibleCAF(
            spec.backbone_name,
            attn_dim=cfg_get("ATTN_DIM", 256),
            num_heads=cfg_get("NUM_HEADS", 8),
            num_layers=cfg_get("NUM_LAYERS", 2),
            dropout=cfg_get("DROPOUT", 0.1),
            classifier_hidden_dim=cfg_get("CLASSIFIER_HIDDEN_DIM", 512),
            classifier_mid_dim=cfg_get("CLASSIFIER_MID_DIM", 128),
            classifier_dropout=cfg_get("CLASSIFIER_DROPOUT", 0.3),
        )
    if spec.arch == "dann":
        return DANNModel(spec.backbone_name)
    if spec.arch == "teacher":
        state = ckpt["ema_state_dict"] if "ema_state_dict" in ckpt else ckpt["model_state_dict"]
        physics_dim = state["physics_head.3.weight"].shape[0]
        image_dim = state["image_head.3.weight"].shape[0]
        return TeacherRegularizedModel(
            spec.backbone_name,
            physics_dim,
            image_dim,
            attn_dim=cfg_get("ATTN_DIM", 256),
            num_heads=cfg_get("NUM_HEADS", 8),
            num_layers=cfg_get("NUM_LAYERS", 2),
            pos_grid=cfg_get("POS_GRID", 7),
            dropout=cfg_get("DROPOUT", 0.1),
            classifier_hidden_dim=cfg_get("CLASSIFIER_HIDDEN_DIM", 512),
            classifier_mid_dim=cfg_get("CLASSIFIER_MID_DIM", 128),
            classifier_dropout=cfg_get("CLASSIFIER_DROPOUT", 0.3),
            domain_hidden_dim=cfg_get("DOMAIN_HIDDEN_DIM", 256),
            domain_dropout=cfg_get("DOMAIN_DROPOUT", 0.2),
        )
    if spec.arch == "teacher_flexible":
        state = ckpt["ema_state_dict"] if "ema_state_dict" in ckpt else ckpt["model_state_dict"]
        physics_dim = state["physics_head.3.weight"].shape[0]
        image_dim = state["image_head.3.weight"].shape[0]
        return FlexibleTeacherRegularizedModel(
            spec.backbone_name,
            physics_dim,
            image_dim,
            attn_dim=cfg_get("ATTN_DIM", 256),
            num_heads=cfg_get("NUM_HEADS", 8),
            num_layers=cfg_get("NUM_LAYERS", 2),
            dropout=cfg_get("DROPOUT", 0.1),
            classifier_hidden_dim=cfg_get("CLASSIFIER_HIDDEN_DIM", 512),
            classifier_mid_dim=cfg_get("CLASSIFIER_MID_DIM", 128),
            classifier_dropout=cfg_get("CLASSIFIER_DROPOUT", 0.3),
            domain_hidden_dim=cfg_get("DOMAIN_HIDDEN_DIM", 256),
            domain_dropout=cfg_get("DOMAIN_DROPOUT", 0.2),
        )
    raise ValueError(spec.arch)


def load_state(model: nn.Module, checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["ema_state_dict"] if "ema_state_dict" in ckpt else ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys for {checkpoint_path}: {missing}")
    if unexpected:
        print(f"load_state warning for {checkpoint_path.name}: ignoring unexpected keys {unexpected}")
    return ckpt


def predict_probs(model: nn.Module, spec: ModelSpec, loader: DataLoader, device: torch.device):
    model.eval()
    ids, probs, labels = [], [], []
    with torch.no_grad():
        for batch_ids, views, batch_labels in tqdm(loader, desc=spec.name, dynamic_ncols=True, ascii=True):
            views = [v.to(device) for v in views]
            if spec.arch == "dann":
                logits = model.forward_class_logits(views)
            elif spec.arch in {"teacher", "teacher_flexible"}:
                logits = model(views, lambda_=0.0)["class_logit"]
            else:
                logits = model(views)
            prob = torch.sigmoid(logits).cpu().numpy()
            ids.extend(batch_ids)
            probs.extend(prob.tolist())
            labels.extend(batch_labels.numpy().tolist())
    out = pd.DataFrame({"id": ids, "label_int": labels, "prob": probs})
    loss = log_loss(out["label_int"], np.clip(out["prob"], 1e-15, 1 - 1e-15), labels=[0, 1])
    return out, float(loss)


def pair_search(prob_map: dict[str, np.ndarray], labels: np.ndarray, step: float = 0.05) -> pd.DataFrame:
    names = list(prob_map)
    rows = []
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pa, pb = prob_map[a], prob_map[b]
            best_loss = math.inf
            best_alpha = None
            for alpha in grid:
                blend = alpha * pa + (1.0 - alpha) * pb
                loss = log_loss(labels, np.clip(blend, 1e-15, 1 - 1e-15), labels=[0, 1])
                if loss < best_loss:
                    best_loss = float(loss)
                    best_alpha = float(alpha)
            best_single = min(
                log_loss(labels, np.clip(pa, 1e-15, 1 - 1e-15), labels=[0, 1]),
                log_loss(labels, np.clip(pb, 1e-15, 1 - 1e-15), labels=[0, 1]),
            )
            rows.append({
                "model_a": a,
                "model_b": b,
                "alpha_for_a": best_alpha,
                "alpha_for_b": 1.0 - best_alpha,
                "blend_logloss": best_loss,
                "best_single_logloss": best_single,
                "improvement": best_single - best_loss,
            })
    return pd.DataFrame(rows).sort_values(["blend_logloss", "improvement"], ascending=[True, False]).reset_index(drop=True)


def triple_search(prob_map: dict[str, np.ndarray], labels: np.ndarray, step: float = 0.1) -> pd.DataFrame:
    names = list(prob_map)
    rows = []
    steps = int(round(1.0 / step))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for k in range(j + 1, len(names)):
                a, b, c = names[i], names[j], names[k]
                pa, pb, pc = prob_map[a], prob_map[b], prob_map[c]
                best_loss = math.inf
                best_weights = None
                for ia in range(steps + 1):
                    wa = ia * step
                    for ib in range(steps + 1 - ia):
                        wb = ib * step
                        wc = 1.0 - wa - wb
                        blend = wa * pa + wb * pb + wc * pc
                        loss = log_loss(labels, np.clip(blend, 1e-15, 1 - 1e-15), labels=[0, 1])
                        if loss < best_loss:
                            best_loss = float(loss)
                            best_weights = (wa, wb, wc)
                best_single = min(
                    log_loss(labels, np.clip(pa, 1e-15, 1 - 1e-15), labels=[0, 1]),
                    log_loss(labels, np.clip(pb, 1e-15, 1 - 1e-15), labels=[0, 1]),
                    log_loss(labels, np.clip(pc, 1e-15, 1 - 1e-15), labels=[0, 1]),
                )
                rows.append({
                    "model_a": a,
                    "model_b": b,
                    "model_c": c,
                    "w_a": best_weights[0],
                    "w_b": best_weights[1],
                    "w_c": best_weights[2],
                    "blend_logloss": best_loss,
                    "best_single_logloss": best_single,
                    "improvement": best_single - best_loss,
                })
    return pd.DataFrame(rows).sort_values(["blend_logloss", "improvement"], ascending=[True, False]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "dev_prob_ensemble")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    specs = []
    skipped = []
    for checkpoint_path in sorted((ROOT / "outputs" / "weights").rglob("*.pt")):
        spec = infer_model_spec(checkpoint_path)
        if spec is None:
            skipped.append(str(checkpoint_path))
            continue
        specs.append(spec)

    print(f"loaded_specs={len(specs)} skipped={len(skipped)}")
    if skipped:
        print("skipped_checkpoints:")
        for path in skipped:
            print(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    label_df = pd.read_csv(args.data_dir / "dev.csv", encoding="utf-8-sig")
    labels = label_df["label"].map({"stable": 0, "unstable": 1}).to_numpy(dtype=np.int64)

    prob_map: dict[str, np.ndarray] = {}
    individual_rows = []

    for spec in specs:
        print(f"\n### {spec.name}")
        dataset = DevDataset(args.data_dir, spec.img_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        model = build_model(spec).to(device)
        load_state(model, spec.checkpoint_path)
        df, loss = predict_probs(model, spec, loader, device)
        df.to_csv(args.output_dir / f"{spec.name}_dev_probs.csv", index=False)
        prob_map[spec.name] = df["prob"].to_numpy(dtype=np.float64)
        individual_rows.append({
            "model": spec.name,
            "arch": spec.arch,
            "backbone": spec.backbone_name,
            "saved_logloss": spec.saved_logloss,
            "dev_logloss": loss,
            "prob_path": str(args.output_dir / f"{spec.name}_dev_probs.csv"),
        })
        print({"saved_logloss": spec.saved_logloss, "dev_logloss": loss})

    individual_df = pd.DataFrame(individual_rows).sort_values("dev_logloss").reset_index(drop=True)
    pair_df = pair_search(prob_map, labels, step=0.05)
    triple_df = triple_search(prob_map, labels, step=0.1)

    individual_df.to_csv(args.output_dir / "individual_results.csv", index=False)
    pair_df.to_csv(args.output_dir / "pair_results.csv", index=False)
    triple_df.to_csv(args.output_dir / "triple_results.csv", index=False)

    print("\n=== Top Individuals ===")
    print(individual_df.head(10).to_string(index=False))
    print("\n=== Top Improved Pairs ===")
    print(pair_df[pair_df["improvement"] > 0].head(10).to_string(index=False))
    print("\n=== Top Improved Triples ===")
    print(triple_df[triple_df["improvement"] > 0].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
