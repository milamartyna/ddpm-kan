import math

import torch
from torch import nn


def get_num_groups(num_channels: int, max_groups: int = 8) -> int:
    for groups in reversed(range(1, max_groups + 1)):
        if num_channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2

        scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=device) * -scale
        )

        embeddings = timesteps.float()[:, None] * frequencies[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)

        return embeddings


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.GroupNorm(get_num_groups(in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels),
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(get_num_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)

        time_projection = self.time_mlp(time_embedding)
        h = h + time_projection[:, :, None, None]

        h = self.block2(h)

        return h + self.residual_conv(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels,
                              kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        time_embedding_dim: int = 256,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        self.down1 = ResBlock(c1, c1, time_embedding_dim)
        self.downsample1 = Downsample(c1)

        self.down2 = ResBlock(c1, c2, time_embedding_dim)
        self.downsample2 = Downsample(c2)

        self.down3 = ResBlock(c2, c3, time_embedding_dim)

        self.mid1 = ResBlock(c3, c3, time_embedding_dim)
        self.mid2 = ResBlock(c3, c3, time_embedding_dim)

        self.up1 = ResBlock(c3 + c3, c2, time_embedding_dim)
        self.upsample1 = Upsample(c2)

        self.up2 = ResBlock(c2 + c2, c1, time_embedding_dim)
        self.upsample2 = Upsample(c1)

        self.up3 = ResBlock(c1 + c1, c1, time_embedding_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(get_num_groups(c1), c1),
            nn.SiLU(),
            nn.Conv2d(c1, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_mlp(timesteps)

        x = self.init_conv(x)

        skip1 = self.down1(x, time_embedding)
        x = self.downsample1(skip1)

        skip2 = self.down2(x, time_embedding)
        x = self.downsample2(skip2)

        skip3 = self.down3(x, time_embedding)

        x = self.mid1(skip3, time_embedding)
        x = self.mid2(x, time_embedding)

        x = torch.cat([x, skip3], dim=1)
        x = self.up1(x, time_embedding)
        x = self.upsample1(x)

        x = torch.cat([x, skip2], dim=1)
        x = self.up2(x, time_embedding)
        x = self.upsample2(x)

        x = torch.cat([x, skip1], dim=1)
        x = self.up3(x, time_embedding)

        return self.out(x)
