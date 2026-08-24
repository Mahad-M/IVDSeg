"""RAS canonicalization and manifest-governed NIfTI alignment handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from ivdseg.manifest import MODALITY_ORDER


AFFINE_ELEMENT_TOLERANCE = 1e-3


class GeometryMismatchError(ValueError):
    """A volume differs from the image reference beyond header precision."""


@dataclass(frozen=True)
class CanonicalSubject:
    """One subject's in-memory, RAS-oriented volumes on a validated image grid."""

    subject_id: str
    modalities: Mapping[str, nib.spatialimages.SpatialImage]
    label: nib.spatialimages.SpatialImage
    reference_modality: str
    label_was_resampled: bool

    @property
    def reference_image(self) -> nib.spatialimages.SpatialImage:
        return self.modalities[self.reference_modality]


def canonicalize(image: nib.spatialimages.SpatialImage) -> nib.spatialimages.SpatialImage:
    """Return the RAS-oriented view of a NIfTI image without writing source data."""
    canonical = nib.as_closest_canonical(image)
    if nib.aff2axcodes(canonical.affine) != ("R", "A", "S"):
        raise GeometryMismatchError(
            f"canonicalization did not produce RAS orientation: {nib.aff2axcodes(canonical.affine)}"
        )
    return canonical


def max_affine_element_delta(
    first: nib.spatialimages.SpatialImage, second: nib.spatialimages.SpatialImage
) -> float:
    return float(np.max(np.abs(first.affine - second.affine)))


def grids_match(
    first: nib.spatialimages.SpatialImage,
    second: nib.spatialimages.SpatialImage,
    *,
    tolerance: float = AFFINE_ELEMENT_TOLERANCE,
) -> bool:
    """Compare grid shape, RAS orientation, and affine up to qform float precision."""
    return (
        first.shape == second.shape
        and nib.aff2axcodes(first.affine) == nib.aff2axcodes(second.affine)
        and max_affine_element_delta(first, second) <= tolerance
    )


def require_matching_grid(
    reference: nib.spatialimages.SpatialImage,
    candidate: nib.spatialimages.SpatialImage,
    *,
    candidate_name: str,
) -> None:
    """Raise a descriptive error instead of silently training on misregistered data."""
    if grids_match(reference, candidate):
        return
    raise GeometryMismatchError(
        f"{candidate_name} does not match the image reference: "
        f"reference shape={reference.shape}, candidate shape={candidate.shape}, "
        f"reference orientation={nib.aff2axcodes(reference.affine)}, "
        f"candidate orientation={nib.aff2axcodes(candidate.affine)}, "
        f"max affine element delta={max_affine_element_delta(reference, candidate):.6g}"
    )


def resample_image_to_reference(
    image: nib.spatialimages.SpatialImage, reference: nib.spatialimages.SpatialImage
) -> nib.spatialimages.SpatialImage:
    """Linearly resample an intensity image to an audited image reference grid."""
    return resample_from_to(image, reference, order=1, mode="constant", cval=0.0)


def resample_label_to_reference(
    label: nib.spatialimages.SpatialImage, reference: nib.spatialimages.SpatialImage
) -> nib.spatialimages.SpatialImage:
    """Nearest-neighbor resample a binary label to an audited image reference grid."""
    resampled = resample_from_to(label, reference, order=0, mode="constant", cval=0.0)
    values = np.unique(np.asarray(resampled.dataobj))
    if not np.isin(values, (0, 1)).all():
        raise GeometryMismatchError(f"nearest-neighbor resampling produced non-binary values: {values}")
    return resampled


def resolve_record_path(dataset_root: Path, relative_path: str) -> Path:
    """Resolve an artifact's manifest-relative source path."""
    return dataset_root / relative_path


def load_canonical_modalities(
    record: Mapping[str, Any], dataset_root: Path
) -> dict[str, nib.spatialimages.SpatialImage]:
    """Load and validate a record's RAS-oriented intensity images."""
    modality_paths = record["modalities"]
    if tuple(modality_paths) != MODALITY_ORDER:
        raise ValueError(
            f"subject {record['subject_id']} has modality order {tuple(modality_paths)}, "
            f"expected {MODALITY_ORDER}"
        )

    modalities = {
        modality: canonicalize(nib.load(resolve_record_path(dataset_root, modality_paths[modality])))
        for modality in MODALITY_ORDER
    }
    alignment = record["label_alignment"]
    reference_modality = alignment["reference_modality"]
    if reference_modality not in modalities:
        raise ValueError(f"unknown reference modality: {reference_modality}")
    reference = modalities[reference_modality]
    for modality, image in modalities.items():
        require_matching_grid(reference, image, candidate_name=f"{record['subject_id']} {modality}")
    return modalities


def load_canonical_subject(
    record: Mapping[str, Any], dataset_root: Path
) -> CanonicalSubject:
    """Load one manifest record without mutating any authoritative NIfTI file."""
    modalities = load_canonical_modalities(record, dataset_root)
    alignment = record["label_alignment"]
    reference_modality = alignment["reference_modality"]
    reference = modalities[reference_modality]

    label = canonicalize(nib.load(resolve_record_path(dataset_root, record["label"])))
    action = alignment["action"]
    if action == "preserve":
        require_matching_grid(reference, label, candidate_name=f"{record['subject_id']} label")
        label_was_resampled = False
    elif action == "resample_to_image_reference":
        if alignment["interpolation"] != "nearest":
            raise ValueError("binary labels must use nearest-neighbor interpolation")
        label = resample_label_to_reference(label, reference)
        require_matching_grid(reference, label, candidate_name=f"{record['subject_id']} resampled label")
        label_was_resampled = True
    else:
        raise ValueError(f"unknown label alignment action: {action}")

    return CanonicalSubject(
        subject_id=record["subject_id"],
        modalities=modalities,
        label=label,
        reference_modality=reference_modality,
        label_was_resampled=label_was_resampled,
    )
