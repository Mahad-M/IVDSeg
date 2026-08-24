"""Synchronized tensor augmentations for 2.5D IVD instance segmentation.

The input tensor is modality-first: three adjacent slices for each of the four
modalities.  A geometric operation therefore acts once on all 12 channels and
on every instance mask.  Intensity perturbations are independent across
modalities but shared by a modality's three slice channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from ivdseg.manifest import MODALITY_ORDER


CHANNELS_PER_MODALITY = 3
INPUT_CHANNELS = len(MODALITY_ORDER) * CHANNELS_PER_MODALITY


@dataclass(frozen=True)
class AugmentationConfig:
    """Conservative in-plane and modality-specific augmentation settings.

    No horizontal or vertical flip is offered deliberately.  A flip along
    either in-plane anatomical axis would change anatomy laterality or
    superior/inferior orientation, which is not part of the experiment design.
    """

    geometric_probability: float = 0.8
    max_rotation_degrees: float = 7.0
    max_translate_fraction: float = 0.05
    minimum_scale: float = 0.95
    maximum_scale: float = 1.05
    intensity_probability: float = 0.8
    minimum_intensity_gain: float = 0.90
    maximum_intensity_gain: float = 1.10
    maximum_intensity_bias: float = 0.10

    def __post_init__(self) -> None:
        probabilities = (self.geometric_probability, self.intensity_probability)
        if any(not 0.0 <= probability <= 1.0 for probability in probabilities):
            raise ValueError("augmentation probabilities must be in [0, 1]")
        if self.max_rotation_degrees < 0.0 or self.max_translate_fraction < 0.0:
            raise ValueError("rotation and translation magnitudes must be non-negative")
        if self.minimum_scale <= 0.0 or self.maximum_scale < self.minimum_scale:
            raise ValueError("scale range must be positive and ordered")
        if self.minimum_intensity_gain < 0.0 or self.maximum_intensity_gain < self.minimum_intensity_gain:
            raise ValueError("intensity gain range must be non-negative and ordered")
        if self.maximum_intensity_bias < 0.0:
            raise ValueError("maximum intensity bias must be non-negative")

    @property
    def uses_anatomical_axis_flips(self) -> bool:
        """Expose the fixed no-flip policy for configuration and tests."""
        return False


def _uniform(low: float, high: float, *, device: torch.device) -> float:
    """Draw one reproducible scalar from the active PyTorch RNG stream."""
    return float(torch.empty((), device=device).uniform_(low, high).item())


def boxes_from_masks(masks: Tensor) -> tuple[Tensor, Tensor]:
    """Return absolute exclusive-XYXY boxes and a nonempty-instance indicator.

    Args:
        masks: Boolean ``[N, H, W]`` instance masks.

    Returns:
        A ``([N_kept, 4], [N])`` tuple.  The indicator indexes the original
        instance fields and lets callers discard masks cropped fully away by an
        affine transformation.
    """
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape [N, H, W], got {tuple(masks.shape)}")
    masks = masks.bool()
    count = masks.shape[0]
    if count == 0:
        return torch.empty((0, 4), dtype=torch.float32, device=masks.device), torch.zeros(
            (0,), dtype=torch.bool, device=masks.device
        )

    rows = masks.any(dim=2)
    columns = masks.any(dim=1)
    keep = rows.any(dim=1) & columns.any(dim=1)
    row_indices = torch.arange(masks.shape[1], device=masks.device)
    column_indices = torch.arange(masks.shape[2], device=masks.device)
    ymin = torch.where(rows, row_indices.unsqueeze(0), masks.shape[1]).min(dim=1).values
    ymax = torch.where(rows, row_indices.unsqueeze(0), -1).max(dim=1).values + 1
    xmin = torch.where(columns, column_indices.unsqueeze(0), masks.shape[2]).min(dim=1).values
    xmax = torch.where(columns, column_indices.unsqueeze(0), -1).max(dim=1).values + 1
    boxes = torch.stack((xmin, ymin, xmax, ymax), dim=1).to(torch.float32)
    return boxes[keep], keep


def normalize_boxes_xyxy(boxes: Tensor, *, height: int, width: int) -> Tensor:
    """Convert absolute exclusive-XYXY boxes to RF-DETR normalized CXCYWH."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must have shape [N, 4], got {tuple(boxes.shape)}")
    if height <= 0 or width <= 0:
        raise ValueError("image height and width must be positive")
    x0, y0, x1, y1 = boxes.unbind(dim=1)
    return torch.stack(
        ((x0 + x1) / (2.0 * width), (y0 + y1) / (2.0 * height), (x1 - x0) / width, (y1 - y0) / height),
        dim=1,
    )


def target_from_masks(
    masks: Tensor,
    *,
    labels: Tensor | None = None,
    template: dict[str, Any] | None = None,
    original_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build a model-ready RF-DETR target from aligned boolean masks.

    The output boxes use RF-DETR's normalized ``cx, cy, width, height`` format;
    masks remain at image resolution for the segmentation criterion.
    """
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape [N, H, W], got {tuple(masks.shape)}")
    masks = masks.bool()
    height, width = (int(masks.shape[1]), int(masks.shape[2]))
    source_labels = (
        torch.zeros((masks.shape[0],), dtype=torch.int64, device=masks.device)
        if labels is None
        else labels.to(device=masks.device, dtype=torch.int64)
    )
    if source_labels.shape != (masks.shape[0],):
        raise ValueError("labels must contain one class ID per mask")
    boxes_xyxy, keep = boxes_from_masks(masks)
    kept_masks = masks[keep]
    kept_labels = source_labels[keep]
    area = kept_masks.flatten(1).sum(dim=1).to(torch.float32)
    target = dict(template or {})
    target.update(
        {
            "boxes": normalize_boxes_xyxy(boxes_xyxy, height=height, width=width),
            "labels": kept_labels,
            "masks": kept_masks,
            "area": area,
            "iscrowd": torch.zeros((kept_masks.shape[0],), dtype=torch.int64, device=masks.device),
            "size": torch.tensor((height, width), dtype=torch.int64, device=masks.device),
            "orig_size": torch.tensor(
                original_size or (height, width), dtype=torch.int64, device=masks.device
            ),
        }
    )
    return target


def resize_tensor_and_target(
    image: Tensor, target: dict[str, Any], *, resolution: int
) -> tuple[Tensor, dict[str, Any]]:
    """Resize a CHW tensor and masks, then regenerate exact normalized targets."""
    if image.ndim != 3:
        raise ValueError(f"image must have shape [C, H, W], got {tuple(image.shape)}")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    original_size = tuple(int(value) for value in target["orig_size"].tolist())
    image = torch.nn.functional.interpolate(
        image.unsqueeze(0), size=(resolution, resolution), mode="bilinear", align_corners=False
    ).squeeze(0)
    masks = torch.nn.functional.interpolate(
        target["masks"].unsqueeze(1).to(torch.float32), size=(resolution, resolution), mode="nearest"
    ).squeeze(1).bool()
    return image, target_from_masks(
        masks,
        labels=target["labels"],
        template={"image_id": target["image_id"]},
        original_size=original_size,
    )


class Synchronized2p5DAugmentation:
    """Apply legal geometry once and modality-wise intensity variation to one sample."""

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()

    def __call__(self, image: Tensor, target: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
        if image.ndim != 3 or image.shape[0] != INPUT_CHANNELS:
            raise ValueError(
                f"2.5D input must have shape [{INPUT_CHANNELS}, H, W], got {tuple(image.shape)}"
            )
        if "masks" not in target or "labels" not in target:
            raise ValueError("target must contain masks and labels for synchronized augmentation")
        image = image.clone()
        masks = target["masks"].to(device=image.device, dtype=torch.bool)
        labels = target["labels"].to(device=image.device, dtype=torch.int64)
        original_size = tuple(int(value) for value in target["orig_size"].tolist())

        if torch.rand((), device=image.device) < self.config.geometric_probability:
            height, width = image.shape[-2:]
            angle = _uniform(-self.config.max_rotation_degrees, self.config.max_rotation_degrees, device=image.device)
            translate = [
                int(round(_uniform(-self.config.max_translate_fraction, self.config.max_translate_fraction, device=image.device) * width)),
                int(round(_uniform(-self.config.max_translate_fraction, self.config.max_translate_fraction, device=image.device) * height)),
            ]
            scale = _uniform(self.config.minimum_scale, self.config.maximum_scale, device=image.device)
            image = transform_functional.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=(0.0, 0.0),
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            masks = transform_functional.affine(
                masks.to(torch.float32),
                angle=angle,
                translate=translate,
                scale=scale,
                shear=(0.0, 0.0),
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            ).bool()

        if torch.rand((), device=image.device) < self.config.intensity_probability:
            for modality_index in range(len(MODALITY_ORDER)):
                start = modality_index * CHANNELS_PER_MODALITY
                end = start + CHANNELS_PER_MODALITY
                gain = _uniform(
                    self.config.minimum_intensity_gain,
                    self.config.maximum_intensity_gain,
                    device=image.device,
                )
                bias = _uniform(
                    -self.config.maximum_intensity_bias,
                    self.config.maximum_intensity_bias,
                    device=image.device,
                )
                block = image[start:end]
                # The fitted data transform uses zero as background.  Preserve that
                # convention even after adding a modality-specific intensity bias.
                image[start:end] = torch.where(block != 0.0, block * gain + bias, block)

        return image, target_from_masks(
            masks,
            labels=labels,
            template={"image_id": target["image_id"]},
            original_size=original_size,
        )
