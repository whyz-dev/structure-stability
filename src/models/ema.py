import copy

import torch
import torch.nn as nn

from .config import EMAConfig


class ModelEMA:
    """Exponential Moving Average for model parameters/buffers."""

    def __init__(self, model: nn.Module, config: EMAConfig | None = None):
        self.config = config or EMAConfig()
        self.ema_model = copy.deepcopy(model).to(next(model.parameters()).device)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @property
    def decay(self) -> float:
        return self.config.decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        model_state = model.state_dict()
        ema_state = self.ema_model.state_dict()

        for k, v_ema in ema_state.items():
            v_model = model_state[k].detach()
            if torch.is_floating_point(v_ema):
                v_ema.mul_(self.decay).add_(v_model, alpha=1.0 - self.decay)
            else:
                v_ema.copy_(v_model)
