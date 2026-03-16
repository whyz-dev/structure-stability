import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MultiViewFeatureFusionConfig


class MultiViewFeatureFusion(nn.Module):
    """Baseline multi-view model from baseline.ipynb."""

    def __init__(self, config: MultiViewFeatureFusionConfig | None = None):
        super().__init__()
        self.config = config or MultiViewFeatureFusionConfig()
        self.backbone = timm.create_model(
            self.config.backbone_name,
            pretrained=self.config.pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 4, self.config.hidden_dim),
            nn.BatchNorm1d(self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.mid_dim),
            nn.ReLU(),
            nn.Linear(self.config.mid_dim, self.config.out_dim),
        )

    @staticmethod
    def _to_vector(x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x

    def forward(self, views):
        f1 = self._to_vector(self.backbone(views[0]))
        f2 = self._to_vector(self.backbone(views[1]))

        combined = torch.cat([f1, f2, torch.abs(f1 - f2), f1 * f2], dim=1)
        return self.classifier(combined)
