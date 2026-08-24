from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from ivdseg.manifest import MODALITY_ORDER, build_manifest
from ivdseg.normalization import (
    ModalityNormalizationStats,
    fit_normalization_profile,
    normalize_volume,
    select_training_records,
)


def write_record_images(dataset_root: Path, subject_id: str, value: float) -> dict:
    modalities = {}
    for index, modality in enumerate(MODALITY_ORDER):
        relative_path = Path(f"train/{subject_id}/{subject_id}_{modality}.nii")
        path = dataset_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(np.array([[[0.0, value + index]]], dtype=np.float32), np.eye(4)),
            path,
        )
        modalities[modality] = str(relative_path)
    return {
        "subject_id": subject_id,
        "partition": "train",
        "modalities": modalities,
        "label_alignment": {"reference_modality": "fat"},
    }


def test_fit_uses_only_nonzero_training_voxels_per_modality(tmp_path: Path) -> None:
    first = write_record_images(tmp_path, "01", 2.0)
    second = write_record_images(tmp_path, "02", 6.0)

    profile = fit_normalization_profile([first, second], tmp_path)

    assert profile.fitted_subject_ids == ("01", "02")
    assert profile.modality_stats["fat"] == ModalityNormalizationStats(
        mean=4.0, std=2.0, nonzero_voxel_count=2
    )
    assert profile.modality_stats["water"] == ModalityNormalizationStats(
        mean=7.0, std=2.0, nonzero_voxel_count=2
    )


def test_normalization_preserves_zero_background_and_clips_nonzero_values() -> None:
    normalized = normalize_volume(
        np.array([0.0, 2.0, 4.0, 20.0], dtype=np.float32),
        ModalityNormalizationStats(mean=2.0, std=2.0, nonzero_voxel_count=3),
    )

    np.testing.assert_allclose(normalized, [0.0, 0.0, 1.0, 5.0])
    assert normalized.dtype == np.float32


def test_normalization_selection_rejects_test_subjects() -> None:
    manifest = build_manifest(Path("IVDM3Seg"))

    with pytest.raises(ValueError, match="only use training subjects"):
        select_training_records(manifest, ["03"])
