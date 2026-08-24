"""Pretrained Swin V2 encoders with a direct-12-channel 2D U-Net decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import Swin_V2_S_Weights, Swin_V2_T_Weights, swin_v2_s, swin_v2_t


def _group_count(channels: int) -> int:
    """Use the largest GroupNorm group count up to eight that divides channels."""
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    raise ValueError(f"cannot choose a GroupNorm group count for {channels} channels")


class ConvNormGELU(nn.Sequential):
    """A decoder convolution with batch-size-stable normalization."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class SwinDecoderBlock(nn.Module):
    """Upsample one Swin feature scale and fuse its same-resolution skip."""

    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(
            ConvNormGELU(output_channels + skip_channels, output_channels),
            ConvNormGELU(output_channels, output_channels),
        )

    def forward(self, features: Tensor, skip: Tensor) -> Tensor:
        features = self.up(features)
        if features.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                "Swin decoder skip shape does not match the upsampled feature; "
                f"got feature={tuple(features.shape)}, skip={tuple(skip.shape)}"
            )
        return self.fuse(torch.cat((features, skip), dim=1))


def adapt_pretrained_patch_projection(projection: nn.Conv2d, *, in_channels: int) -> nn.Conv2d:
    """Expand Swin's RGB patch projection to direct modality-first 2.5D input.

    Repeating the RGB kernel across modality groups and dividing by the number
    of groups preserves its response exactly when every three-channel group is
    equal, while keeping the pretrained feature scale stable for fine tuning.
    """
    if projection.in_channels != 3:
        raise ValueError("Swin ImageNet patch projection must have exactly three input channels")
    if in_channels < 3 or in_channels % 3:
        raise ValueError("direct pretrained stem adaptation requires a positive multiple of three channels")
    group_count = in_channels // 3
    adapted = nn.Conv2d(
        in_channels,
        projection.out_channels,
        kernel_size=projection.kernel_size,
        stride=projection.stride,
        padding=projection.padding,
        dilation=projection.dilation,
        groups=projection.groups,
        bias=projection.bias is not None,
        padding_mode=projection.padding_mode,
        device=projection.weight.device,
        dtype=projection.weight.dtype,
    )
    with torch.no_grad():
        adapted.weight.copy_(projection.weight.repeat(1, group_count, 1, 1) / group_count)
        if projection.bias is not None:
            adapted.bias.copy_(projection.bias)
    return adapted


class _SwinV2UNet(nn.Module):
    """Shared direct-12-channel decoder around one torchvision Swin V2 encoder."""

    encoder_channels = (96, 192, 384, 768)

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        pretrained: bool,
        backbone_builder: object,
        pretrained_weights: Swin_V2_T_Weights | Swin_V2_S_Weights,
        architecture_name: str,
    ) -> None:
        super().__init__()
        if in_channels != 12:
            raise ValueError("Swin U-Net is locked to the project 12-channel modality-first 2.5D tensor")
        if out_channels != 1:
            raise ValueError("Swin U-Net is a binary semantic segmenter with one output channel")
        if not callable(backbone_builder):
            raise TypeError("Swin backbone builder must be callable")
        weights = pretrained_weights if pretrained else None
        backbone = backbone_builder(weights=weights)
        patch_projection = backbone.features[0][0]
        if not isinstance(patch_projection, nn.Conv2d):
            raise RuntimeError("unexpected torchvision Swin V2 patch-projection layout")
        backbone.features[0][0] = adapt_pretrained_patch_projection(
            patch_projection,
            in_channels=in_channels,
        )
        self.encoder = backbone.features
        self.encoder_norm = backbone.norm
        self.decoder_3 = SwinDecoderBlock(768, 384, 384)
        self.decoder_2 = SwinDecoderBlock(384, 192, 192)
        self.decoder_1 = SwinDecoderBlock(192, 96, 96)
        self.full_resolution = nn.Sequential(
            nn.ConvTranspose2d(96, 64, kernel_size=2, stride=2),
            ConvNormGELU(64, 64),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            ConvNormGELU(64, 64),
        )
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pretrained = pretrained
        self.architecture_name = architecture_name

    @staticmethod
    def _to_nchw(features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError(f"Swin features must be NHWC tensors, got {tuple(features.shape)}")
        return features.permute(0, 3, 1, 2).contiguous()

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != self.in_channels:
            raise ValueError(
                f"Swin U-Net inputs must have shape [B, {self.in_channels}, H, W], got {tuple(images.shape)}"
            )
        if images.shape[-2] % 32 or images.shape[-1] % 32:
            raise ValueError("Swin V2 U-Net input dimensions must be divisible by 32")
        stage_1 = self.encoder[1](self.encoder[0](images))
        stage_2 = self.encoder[3](self.encoder[2](stage_1))
        stage_3 = self.encoder[5](self.encoder[4](stage_2))
        bottleneck = self.encoder_norm(self.encoder[7](self.encoder[6](stage_3)))
        decoded = self.decoder_3(self._to_nchw(bottleneck), self._to_nchw(stage_3))
        decoded = self.decoder_2(decoded, self._to_nchw(stage_2))
        decoded = self.decoder_1(decoded, self._to_nchw(stage_1))
        return self.head(self.full_resolution(decoded))


class SwinV2TinyUNet(_SwinV2UNet):
    """A direct-12-channel U-Net with ImageNet-pretrained Swin V2 Tiny encoder."""

    pretrained_weights_name = "Swin_V2_T_Weights.IMAGENET1K_V1"
    pretrained_weight_url = "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth"

    def __init__(self, *, in_channels: int = 12, out_channels: int = 1, pretrained: bool = True) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            pretrained=pretrained,
            backbone_builder=swin_v2_t,
            pretrained_weights=Swin_V2_T_Weights.IMAGENET1K_V1,
            architecture_name="Swin V2 Tiny",
        )


class SwinV2SmallUNet(_SwinV2UNet):
    """A direct-12-channel U-Net with ImageNet-pretrained Swin V2 Small encoder."""

    pretrained_weights_name = "Swin_V2_S_Weights.IMAGENET1K_V1"
    pretrained_weight_url = "https://download.pytorch.org/models/swin_v2_s-637d8ceb.pth"

    def __init__(self, *, in_channels: int = 12, out_channels: int = 1, pretrained: bool = True) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            pretrained=pretrained,
            backbone_builder=swin_v2_s,
            pretrained_weights=Swin_V2_S_Weights.IMAGENET1K_V1,
            architecture_name="Swin V2 Small",
        )
