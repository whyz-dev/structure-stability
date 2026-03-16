import torch.nn as nn
import timm

from .config import MultiViewCrossAttentionConfig


class MultiViewCrossAttention(nn.Module):
    """Cross-attention model for paired front/top views."""

    def __init__(self, config: MultiViewCrossAttentionConfig | None = None):
        super().__init__()
        self.config = config or MultiViewCrossAttentionConfig()

        self.backbone = timm.create_model(
            self.config.backbone_name,
            pretrained=self.config.pretrained,
            num_classes=0,
            global_pool="",
        )
        backbone_dim = self.backbone.num_features

        self.proj = nn.Conv2d(backbone_dim, self.config.attn_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(self.config.attn_dim)
        self.attn = nn.MultiheadAttention(self.config.attn_dim, self.config.num_heads, batch_first=True)
        self.fc = nn.Linear(self.config.attn_dim, self.config.out_dim)

    def forward(self, views):
        img1, img2 = views

        f1 = self.backbone.forward_features(img1)
        f2 = self.backbone.forward_features(img2)

        if f1.ndim != 4 or f2.ndim != 4:
            raise ValueError("Expected 4D feature maps from backbone.forward_features.")

        f1 = self.proj(f1)
        f2 = self.proj(f2)

        t1 = self.norm(f1.flatten(2).transpose(1, 2))
        t2 = self.norm(f2.flatten(2).transpose(1, 2))

        attn_out, _ = self.attn(t1, t2, t2, need_weights=False)
        feat = attn_out.mean(dim=1)
        return self.fc(feat)
