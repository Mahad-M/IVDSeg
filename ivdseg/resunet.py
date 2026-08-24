"""A 12-channel residual U-Net used for the B1 semantic baseline."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


def _group_count(channels: int) -> int:
    """Choose a stable GroupNorm grouping for small medical-image batches."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    raise ValueError(f"no valid GroupNorm grouping for {channels} channels")


class ResidualBlock(nn.Module):
    """Pre-activation-free two-convolution residual block with GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1 or stride < 1:
            raise ValueError("residual block channels and stride must be positive")
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
        )

    def forward(self, tensor: Tensor) -> Tensor:
        identity = self.shortcut(tensor)
        tensor = self.activation(self.norm1(self.conv1(tensor)))
        tensor = self.norm2(self.conv2(tensor))
        return self.activation(tensor + identity)


def _residual_stage(in_channels: int, out_channels: int, *, blocks: int, stride: int) -> nn.Sequential:
    if blocks < 1:
        raise ValueError("a residual stage needs at least one block")
    layers: list[nn.Module] = [ResidualBlock(in_channels, out_channels, stride=stride)]
    layers.extend(ResidualBlock(out_channels, out_channels) for _ in range(blocks - 1))
    return nn.Sequential(*layers)


class UpFuse(nn.Module):
    """Upsample one decoder map, concatenate its skip, and refine residually."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.refine = nn.Sequential(
            ResidualBlock(out_channels + skip_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, tensor: Tensor, skip: Tensor) -> Tensor:
        tensor = self.up(tensor)
        if tensor.shape[-2:] != skip.shape[-2:]:
            tensor = functional.interpolate(tensor, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat((tensor, skip), dim=1))


class ResUNet34(nn.Module):
    """A ResNet-34-style residual encoder with a U-Net decoder.

    The encoder has residual stage depths ``(3, 4, 6, 3)``, matching the
    canonical ResNet-34 layout, but starts from the project's 12-channel MRI
    tensor. GroupNorm avoids batch-statistic instability at the fixed micro
    batch size of two. This baseline is intentionally initialized from scratch.
    """

    stage_depths: tuple[int, int, int, int] = (3, 4, 6, 3)

    def __init__(self, *, in_channels: int = 12, base_channels: int = 32, out_channels: int = 1) -> None:
        super().__init__()
        if in_channels < 1 or base_channels < 8 or out_channels < 1:
            raise ValueError("ResUNet channel counts must be positive and base_channels at least 8")
        encoder_channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channels = encoder_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, encoder_channels[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(encoder_channels[0]), encoder_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.encoder1 = _residual_stage(
            encoder_channels[0], encoder_channels[1], blocks=self.stage_depths[0], stride=2
        )
        self.encoder2 = _residual_stage(
            encoder_channels[1], encoder_channels[2], blocks=self.stage_depths[1], stride=2
        )
        self.encoder3 = _residual_stage(
            encoder_channels[2], encoder_channels[3], blocks=self.stage_depths[2], stride=2
        )
        self.encoder4 = _residual_stage(
            encoder_channels[3], encoder_channels[4], blocks=self.stage_depths[3], stride=2
        )
        self.decoder3 = UpFuse(encoder_channels[4], encoder_channels[3], encoder_channels[3])
        self.decoder2 = UpFuse(encoder_channels[3], encoder_channels[2], encoder_channels[2])
        self.decoder1 = UpFuse(encoder_channels[2], encoder_channels[1], encoder_channels[1])
        self.decoder0 = UpFuse(encoder_channels[1], encoder_channels[0], encoder_channels[0])
        self.head = nn.Conv2d(encoder_channels[0], out_channels, kernel_size=1)

    def forward(self, tensor: Tensor) -> Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != self.in_channels:
            raise ValueError(
                f"ResUNet34 expects [B, {self.in_channels}, H, W], got {tuple(tensor.shape)}"
            )
        height, width = tensor.shape[-2:]
        if height % 16 or width % 16:
            raise ValueError("ResUNet34 input height and width must be divisible by 16")
        skip0 = self.stem(tensor)
        skip1 = self.encoder1(skip0)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        encoded = self.encoder4(skip3)
        tensor = self.decoder3(encoded, skip3)
        tensor = self.decoder2(tensor, skip2)
        tensor = self.decoder1(tensor, skip1)
        tensor = self.decoder0(tensor, skip0)
        return self.head(tensor)


def residual_encoder_depths(model: ResUNet34) -> Sequence[int]:
    """Expose the fixed B1 residual-backbone depths for configuration records."""
    return model.stage_depths
