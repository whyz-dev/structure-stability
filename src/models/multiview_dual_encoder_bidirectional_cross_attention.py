import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MultiViewDualEncoderBidirectionalCrossAttentionConfig
from .multiview_bidirectional_cross_attention import CrossAttentionBlock


class MultiViewDualEncoderBidirectionalCrossAttention(nn.Module):
    """Dual-encoder variant of the baseline v2.1 bidirectional cross-attention model."""

    def __init__(self, config: MultiViewDualEncoderBidirectionalCrossAttentionConfig | None = None):
        super().__init__()
        self.config = config or MultiViewDualEncoderBidirectionalCrossAttentionConfig()

        top_backbone_name = self.config.top_backbone_name or self.config.front_backbone_name

        self.front_backbone = timm.create_model(
            self.config.front_backbone_name,
            pretrained=self.config.pretrained,
            num_classes=0,
            global_pool="",
        )
        self.top_backbone = timm.create_model(
            top_backbone_name,
            pretrained=self.config.pretrained,
            num_classes=0,
            global_pool="",
        )

        self.front_proj = nn.Conv2d(
            self.front_backbone.num_features, self.config.attn_dim, kernel_size=1, bias=False
        )
        self.top_proj = nn.Conv2d(
            self.top_backbone.num_features, self.config.attn_dim, kernel_size=1, bias=False
        )
        self.front_pos_embed = nn.Parameter(
            torch.randn(1, self.config.attn_dim, self.config.pos_grid, self.config.pos_grid) * 0.02
        )
        self.top_pos_embed = nn.Parameter(
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

    @staticmethod
    def _to_tokens(feat_map: torch.Tensor, proj: nn.Module, pos_embed: torch.Tensor) -> torch.Tensor:
        feat_map = proj(feat_map)
        pos = F.interpolate(
            pos_embed,
            size=feat_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        feat_map = feat_map + pos
        return feat_map.flatten(2).transpose(1, 2)

    def forward(self, views):
        front_feat = self.front_backbone.forward_features(views[0])
        top_feat = self.top_backbone.forward_features(views[1])

        t_front = self._to_tokens(front_feat, self.front_proj, self.front_pos_embed)
        t_top = self._to_tokens(top_feat, self.top_proj, self.top_pos_embed)

        for blk_front, blk_top in zip(self.cross_12, self.cross_21):
            t_front_prev, t_top_prev = t_front, t_top
            t_front = blk_front(t_front_prev, t_top_prev)
            t_top = blk_top(t_top_prev, t_front_prev)

        fused_feat = torch.cat([t_front.mean(dim=1), t_top.mean(dim=1)], dim=1)
        return self.classifier(fused_feat)
