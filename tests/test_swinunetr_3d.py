from pathlib import Path

import numpy as np
import pytest
import torch

from ivdseg.swinunetr_3d import (
    B5_FALLBACK_ROI_SIZE,
    B5_PRIMARY_ROI_SIZE,
    B5DevelopmentConfig,
    B5PatchDataset,
    B5Volume,
    _adapt_ssl_patch_projection,
    b5_threshold_grid,
)


def _config(**overrides: object) -> B5DevelopmentConfig:
    values: dict[str, object] = {
        "manifest_path": Path("manifest.json"),
        "dataset_root": Path("data"),
        "normalization_profile": Path("profile.json"),
        "run_dir": Path("run"),
        "pretrained_weights_path": Path("ssl.pth"),
        "seed": 17,
    }
    values.update(overrides)
    return B5DevelopmentConfig(**values)  # type: ignore[arg-type]


def test_b5_configuration_locks_only_the_predeclared_roi_sizes() -> None:
    assert _config().roi_size == B5_PRIMARY_ROI_SIZE
    assert _config(roi_size=B5_FALLBACK_ROI_SIZE).roi_size == B5_FALLBACK_ROI_SIZE
    with pytest.raises(ValueError, match="ROI"):
        _config(roi_size=(32, 128, 128))
    with pytest.raises(ValueError, match="num_workers"):
        _config(num_workers=1)


def test_b5_patch_dataset_is_repeatable_and_keeps_native_channel_order() -> None:
    image = np.stack([np.full((4, 8, 8), channel + 1, dtype=np.float32) for channel in range(4)])
    target = np.zeros((4, 8, 8), dtype=bool)
    target[2, 4, 4] = True
    dataset = B5PatchDataset(
        [B5Volume(subject_id="01", image=image, target=target)],
        roi_size=(4, 8, 8),
        patches_per_epoch=2,
        foreground_probability=1.0,
        seed=17,
    )
    first = dataset[0]
    second = dataset[0]
    assert first["image"].shape == (4, 4, 8, 8)
    assert first["target"].shape == (4, 8, 8)
    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["target"], second["target"])
    assert first["target"].any()
    dataset.set_epoch(1)
    assert not torch.equal(first["image"], dataset[0]["image"])


def test_b5_threshold_grid_breaks_ties_toward_lower_cutoff() -> None:
    probabilities = {
        "02": np.array([[[0.8, 0.2]]], dtype=np.float32),
        "12": np.array([[[0.8, 0.2]]], dtype=np.float32),
    }
    targets = {
        "02": np.array([[[True, False]]]),
        "12": np.array([[[True, False]]]),
    }
    selection = b5_threshold_grid(probabilities, targets, thresholds=(0.40, 0.50, 0.90))
    assert selection.selected.probability_threshold == 0.40
    assert selection.selected.macro_dice == 1.0


def test_ssl_stem_adaptation_preserves_equal_modality_response() -> None:
    source = torch.arange(48 * 2 * 2 * 2, dtype=torch.float32).reshape(48, 1, 2, 2, 2)
    adapted = _adapt_ssl_patch_projection(source, torch.Size((48, 4, 2, 2, 2)))
    assert adapted.shape == (48, 4, 2, 2, 2)
    assert torch.equal(adapted.sum(dim=1), source[:, 0])
    with pytest.raises(ValueError, match="unexpected SSL"):
        _adapt_ssl_patch_projection(torch.ones((48, 3, 2, 2, 2)), torch.Size((48, 4, 2, 2, 2)))
