from typing import Optional

import torch
from torch import nn


def get_mlp_hidden_features(channels: int, grid_size: int = 5, spline_order: int = 3) -> int:
    """
    Match MLP parameter count to the equivalent KAN bottleneck block,
    keeping the comparison as fair as possible.

    For KANBottleneckBlock:
        params = C^2 * (grid_size + spline_order + 1) + C
    For MLP block:
        params = (C * hidden) + hidden + (hidden * C) + C
               = hidden * (2 * C + 1) + C

    Solving for hidden gives:
        hidden = (C^2 * (grid_size + spline_order + 1)) / (2 * C + 1)
    """
    kan_params = channels * channels * (grid_size + spline_order + 1) + channels
    hidden = (kan_params - channels) / (2 * channels + 1)
    return max(1, int(round(hidden)))


class MLPBottleneckBlock(nn.Module):
    """
    Applies a lightweight channel-wise MLP at each spatial location.

    Input/output: [B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        hidden_features: Optional[int] = None,
        residual_scale: float = 0.1,
    ):
        super().__init__()

        if hidden_features is None:
            hidden_features = get_mlp_hidden_features(channels)

        self.channels = channels
        self.hidden_features = hidden_features
        self.norm = nn.LayerNorm(channels)

        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, channels),
        )

        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        flat = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        flat = self.norm(flat)

        mlp_out = self.mlp(flat)
        mlp_out = mlp_out.reshape(b, h, w, c).permute(0, 3, 1, 2)

        return x + self.residual_scale * mlp_out
