import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MultiViewBidirectionalCrossAttentionConfig


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim) # 각 샘플 / 각 토큰마다 독립적으로 정규화
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(q_tokens)
        kv = self.norm_kv(kv_tokens)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        return q_tokens + attn_out


class MultiViewBidirectionalCrossAttention(nn.Module):
    """Cross-attention baseline from baseline2.ipynb."""

    def __init__(self, config: MultiViewBidirectionalCrossAttentionConfig | None = None):
        super().__init__()
        self.config = config or MultiViewBidirectionalCrossAttentionConfig()

        self.backbone = timm.create_model(
            self.config.backbone_name,
            pretrained=self.config.pretrained,
            num_classes=0,
            global_pool="",
        )
        feature_dim = self.backbone.num_features

        self.proj = nn.Conv2d(feature_dim, self.config.attn_dim, kernel_size=1, bias=False)
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

    def _to_tokens(self, feat_map: torch.Tensor) -> torch.Tensor:
        feat_map = self.proj(feat_map)
        pos = F.interpolate(
            self.pos_embed,
            size=feat_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        feat_map = feat_map + pos
        return feat_map.flatten(2).transpose(1, 2)

    def forward(self, views):
        f1 = self.backbone.forward_features(views[0])
        f2 = self.backbone.forward_features(views[1])

        t1 = self._to_tokens(f1)
        t2 = self._to_tokens(f2)

        for blk12, blk21 in zip(self.cross_12, self.cross_21):
            t1_prev, t2_prev = t1, t2
            t1 = blk12(t1_prev, t2_prev)
            t2 = blk21(t2_prev, t1_prev)

        feat = torch.cat([t1.mean(dim=1), t2.mean(dim=1)], dim=1)
        return self.classifier(feat)
