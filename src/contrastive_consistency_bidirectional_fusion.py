from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ContrastiveConsistencyBidirectionalFusionConfig:
    backbone_name: str = "efficientnetv2_rw_s"
    pretrained: bool = True
    attn_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2
    pos_grid: int = 7
    dropout: float = 0.1
    classifier_hidden_dim: int = 512
    classifier_mid_dim: int = 128
    classifier_dropout: float = 0.3
    projection_hidden_dim: int = 256
    projection_dim: int = 128
    aux_hidden_dim: int = 256
    out_dim: int = 1


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


class ContrastiveConsistencyBidirectionalFusion(nn.Module):
    """Bidirectional cross-attention fusion with contrastive and consistency training heads."""

    def __init__(self, config: ContrastiveConsistencyBidirectionalFusionConfig | None = None):
        super().__init__()
        self.config = config or ContrastiveConsistencyBidirectionalFusionConfig()

        self.backbone = timm.create_model(
            self.config.backbone_name,
            pretrained=self.config.pretrained,
            num_classes=0,
            global_pool="",
        )
        self.feature_dim = self.backbone.num_features

        self.proj = nn.Conv2d(self.feature_dim, self.config.attn_dim, kernel_size=1, bias=False)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.config.attn_dim, self.config.pos_grid, self.config.pos_grid) * 0.02
        )

        self.cross_12 = nn.ModuleList(
            [
                CrossAttentionBlock(self.config.attn_dim, self.config.num_heads, self.config.dropout)
                for _ in range(self.config.num_layers)
            ]
        )
        self.cross_21 = nn.ModuleList(
            [
                CrossAttentionBlock(self.config.attn_dim, self.config.num_heads, self.config.dropout)
                for _ in range(self.config.num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.config.attn_dim * 2, self.config.classifier_hidden_dim),
            nn.BatchNorm1d(self.config.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(self.config.classifier_hidden_dim, self.config.classifier_mid_dim),
            nn.ReLU(),
            nn.Linear(self.config.classifier_mid_dim, self.config.out_dim),
        )

        self.aux_head_top = nn.Sequential(
            nn.Linear(self.feature_dim, self.config.aux_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(self.config.aux_hidden_dim, self.config.out_dim),
        )
        self.aux_head_front = nn.Sequential(
            nn.Linear(self.feature_dim, self.config.aux_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(self.config.aux_hidden_dim, self.config.out_dim),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.config.projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.projection_hidden_dim, self.config.projection_dim),
        )

    def _global_pool_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            return features.mean(dim=(2, 3))
        if features.ndim == 3:
            return features.mean(dim=1)
        raise ValueError(f"Unsupported feature shape: {tuple(features.shape)}")

    def _to_tokens(self, feat_map: torch.Tensor) -> torch.Tensor:
        if feat_map.ndim != 4:
            raise ValueError("Bidirectional cross-attention expects CNN feature maps with shape [B, C, H, W].")
        feat_map = self.proj(feat_map)
        pos = F.interpolate(
            self.pos_embed,
            size=feat_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        feat_map = feat_map + pos
        return feat_map.flatten(2).transpose(1, 2)

    def forward(self, views: list[torch.Tensor], return_aux: bool = False):
        front_feat_map = self.backbone.forward_features(views[0])
        top_feat_map = self.backbone.forward_features(views[1])

        feat_front = self._global_pool_features(front_feat_map)
        feat_top = self._global_pool_features(top_feat_map)

        proj_front = F.normalize(self.projection_head(feat_front), dim=1)
        proj_top = F.normalize(self.projection_head(feat_top), dim=1)

        aux_front_logits = self.aux_head_front(feat_front)
        aux_top_logits = self.aux_head_top(feat_top)

        t_front = self._to_tokens(front_feat_map)
        t_top = self._to_tokens(top_feat_map)

        for blk_front, blk_top in zip(self.cross_12, self.cross_21):
            t_front_prev, t_top_prev = t_front, t_top
            t_front = blk_front(t_front_prev, t_top_prev)
            t_top = blk_top(t_top_prev, t_front_prev)

        fused_feat = torch.cat([t_front.mean(dim=1), t_top.mean(dim=1)], dim=1)
        logits = self.classifier(fused_feat)

        if not return_aux:
            return logits

        return {
            "logits": logits,
            "aux_top_logits": aux_top_logits,
            "aux_front_logits": aux_front_logits,
            "proj_top": proj_top,
            "proj_front": proj_front,
            "feat_top": feat_top,
            "feat_front": feat_front,
        }


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"features must be [B, D], got {tuple(features.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    labels = labels.view(-1)
    if labels.shape[0] != features.shape[0]:
        raise ValueError("labels and features batch size must match")

    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    device = features.device
    batch_size = features.shape[0]
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~self_mask)

    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positive_count = positive_mask.sum(dim=1)
    mean_log_prob_pos = (log_prob * positive_mask).sum(dim=1) / positive_count.clamp_min(1)
    valid = positive_count > 0
    if not torch.any(valid):
        return features.new_zeros(())

    loss = -mean_log_prob_pos[valid].mean()
    return loss


def consistency_loss_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    probs_a = torch.sigmoid(logits_a)
    probs_b = torch.sigmoid(logits_b)
    return F.mse_loss(probs_a, probs_b)
