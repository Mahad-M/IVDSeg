from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from ivdseg.manifest import build_manifest
from ivdseg.spatial import (
    GeometryMismatchError,
    grids_match,
    load_canonical_subject,
    require_matching_grid,
    resample_image_to_reference,
    resample_label_to_reference,
)


DATASET_ROOT = Path("IVDM3Seg")


def record_for(subject_id: str) -> dict:
    manifest = build_manifest(DATASET_ROOT)
    return next(record for record in manifest["subjects"] if record["subject_id"] == subject_id)


def test_aligned_subject_is_ras_and_keeps_its_label_grid() -> None:
    subject = load_canonical_subject(record_for("01"), DATASET_ROOT)

    assert subject.label_was_resampled is False
    assert tuple(subject.modalities) == ("fat", "inn", "opp", "water")
    assert nib.aff2axcodes(subject.reference_image.affine) == ("R", "A", "S")
    assert grids_match(subject.reference_image, subject.label)
    assert set(np.unique(np.asarray(subject.label.dataobj))) == {0, 1}


def test_subject_16_resamples_only_its_label_to_the_image_grid() -> None:
    record = record_for("16")
    raw_label = nib.load(DATASET_ROOT / record["label"])
    raw_reference = nib.load(DATASET_ROOT / record["modalities"]["fat"])
    assert not grids_match(raw_reference, raw_label)

    subject = load_canonical_subject(record, DATASET_ROOT)

    assert subject.label_was_resampled is True
    assert grids_match(subject.reference_image, subject.label)
    assert set(np.unique(np.asarray(subject.label.dataobj))) == {0, 1}
    assert not np.array_equal(raw_label.affine, raw_reference.affine)


def test_resampling_uses_linear_images_and_nearest_binary_labels() -> None:
    source_data = np.zeros((4, 4, 4), dtype=np.float32)
    source_data[1, 1, 1] = 1.0
    source = nib.Nifti1Image(source_data, np.eye(4))
    target = nib.Nifti1Image(np.zeros_like(source_data), np.diag([1.0, 1.0, 1.0, 1.0]))
    target.affine[0, 3] = 0.5

    image = resample_image_to_reference(source, target)
    label = resample_label_to_reference(source, target)

    image_values = np.asarray(image.dataobj)
    label_values = np.asarray(label.dataobj)
    assert np.any((image_values > 0.0) & (image_values < 1.0))
    assert set(np.unique(label_values)) <= {0.0, 1.0}


def test_geometry_validation_rejects_an_unrecorded_mismatch() -> None:
    reference = nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    mismatch = nib.Nifti1Image(np.zeros((2, 2, 2)), np.diag([2.0, 1.0, 1.0, 1.0]))

    with pytest.raises(GeometryMismatchError, match="does not match the image reference"):
        require_matching_grid(reference, mismatch, candidate_name="synthetic")
