import math

import torch
import torch.nn.functional as F
from torch import nn


class KANLinear(nn.Module):
    """
    Fixed-grid spline-based KAN linear layer.

    Input:  [N, in_features]
    Output: [N, out_features]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-3.0, 3.0),
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size

        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        )
        grid = grid.expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.bias = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.01)

        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes B-spline basis values.
        x: [N, in_features]
        returns: [N, in_features, grid_size + spline_order]
        """
        assert x.dim() == 2
        assert x.size(1) == self.in_features

        x = x.unsqueeze(-1)
        grid = self.grid

        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            left = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)] + 1e-8)
                * bases[:, :, :-1]
            )

            right = (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:-k] + 1e-8)
                * bases[:, :, 1:]
            )

            bases = left + right

        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = F.linear(F.silu(x), self.base_weight, self.bias)

        spline_basis = self.b_splines(x)
        spline_output = F.linear(
            spline_basis.reshape(x.size(0), -1),
            self.spline_weight.reshape(self.out_features, -1),
        )

        return base_output + spline_output


class KANBottleneckBlock(nn.Module):
    """
    Applies KAN to channel vectors at every spatial location.

    Input/output: [B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        grid_size: int = 5,
        spline_order: int = 3,
        residual_scale: float = 0.1,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(channels)

        self.kan = KANLinear(
            in_features=channels,
            out_features=channels,
            grid_size=grid_size,
            spline_order=spline_order,
        )

        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        flat = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        flat = self.norm(flat)

        kan_out = self.kan(flat)
        kan_out = kan_out.reshape(b, h, w, c).permute(0, 3, 1, 2)

        return x + self.residual_scale * kan_out
