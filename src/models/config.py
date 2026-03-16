from dataclasses import dataclass


@dataclass(frozen=True)
class MultiViewCrossAttentionConfig:
    backbone_name: str = "efficientnetv2_rw_s"
    pretrained: bool = True
    attn_dim: int = 256
    num_heads: int = 8
    out_dim: int = 1


@dataclass(frozen=True)
class EMAConfig:
    decay: float = 0.999


@dataclass(frozen=True)
class MultiViewFeatureFusionConfig:
    backbone_name: str = "efficientnetv2_rw_s"
    pretrained: bool = True
    hidden_dim: int = 512
    mid_dim: int = 128
    dropout: float = 0.3
    out_dim: int = 1


@dataclass(frozen=True)
class MultiViewBidirectionalCrossAttentionConfig:
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
    out_dim: int = 1
