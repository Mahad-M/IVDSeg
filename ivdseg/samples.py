"""Lazy 2.5D tensors and per-slice RF-DETR instance targets from 3D IVD labels."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy import ndimage

from ivdseg.manifest import MODALITY_ORDER
from ivdseg.normalization import NormalizationProfile, normalize_modalities
from ivdseg.spatial import load_canonical_subject


SLICE_OFFSETS = (-1, 0, 1)
MINIMUM_COMPONENT_SIZE_VOXELS = 100


@dataclass(frozen=True)
class InstanceTarget:
    """One IVD component's target on one 2D coronal center slice."""

    component_id: int
    mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    class_id: int = 0


@dataclass(frozen=True)
class PreparedSubject:
    """Normalized 3D modalities and filtered 3D component IDs for lazy sampling."""

    subject_id: str
    normalized_modalities: Mapping[str, np.ndarray]
    component_labels: np.ndarray
    component_sizes: Mapping[int, int]
    semantic_label: np.ndarray | None = None

    @property
    def slice_count(self) -> int:
        return int(self.component_labels.shape[0])

    def tensor_for_slice(self, slice_index: int) -> np.ndarray:
        return make_2_5d_tensor(self.normalized_modalities, slice_index)

    def targets_for_slice(self, slice_index: int) -> list[InstanceTarget]:
        return instance_targets_for_slice(self.component_labels, slice_index)


def channel_names() -> tuple[str, ...]:
    """Expose the model's modality-first channel contract in a human-readable form."""
    return tuple(f"{modality}[{offset:+d}]" for modality in MODALITY_ORDER for offset in SLICE_OFFSETS)


def neighbour_indices(slice_index: int, slice_count: int) -> tuple[int, int, int]:
    """Return previous/center/next indices with edge-slice replication."""
    if not 0 <= slice_index < slice_count:
        raise IndexError(f"slice index {slice_index} is outside [0, {slice_count})")
    return tuple(min(max(slice_index + offset, 0), slice_count - 1) for offset in SLICE_OFFSETS)


def make_2_5d_tensor(modalities: Mapping[str, np.ndarray], slice_index: int) -> np.ndarray:
    """Stack three neighboring coronal slices per modality in fixed 12-channel order."""
    if tuple(modalities) != MODALITY_ORDER:
        raise ValueError(f"modalities must use manifest order {MODALITY_ORDER}")
    slice_count = modalities[MODALITY_ORDER[0]].shape[0]
    if any(volume.shape != modalities[MODALITY_ORDER[0]].shape for volume in modalities.values()):
        raise ValueError("all normalized modalities must share one 3D grid")
    indices = neighbour_indices(slice_index, slice_count)
    return np.stack(
        [modalities[modality][index] for modality in MODALITY_ORDER for index in indices],
        axis=0,
        dtype=np.float32,
    )


def connected_components_3d(
    binary_label: np.ndarray, *, minimum_size_voxels: int = MINIMUM_COMPONENT_SIZE_VOXELS
) -> tuple[np.ndarray, dict[int, int]]:
    """Retain 26-connected IVD components at or above the fixed size threshold."""
    if binary_label.ndim != 3:
        raise ValueError(f"expected a 3D label, got {binary_label.ndim} dimensions")
    if minimum_size_voxels < 1:
        raise ValueError("minimum component size must be positive")
    labeled, component_count = ndimage.label(binary_label.astype(bool), structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labeled.ravel(), minlength=component_count + 1)
    retained = sizes >= minimum_size_voxels
    retained[0] = False
    remap = np.zeros(component_count + 1, dtype=np.int32)
    remap[retained] = np.arange(1, np.count_nonzero(retained) + 1, dtype=np.int32)
    filtered = remap[labeled]
    component_sizes = {
        int(remap[old_component_id]): int(sizes[old_component_id])
        for old_component_id in np.flatnonzero(retained)
    }
    return filtered, component_sizes


def instance_targets_for_slice(component_labels: np.ndarray, slice_index: int) -> list[InstanceTarget]:
    """Extract 2D masks and exclusive-upper-bound boxes for retained 3D components."""
    if not 0 <= slice_index < component_labels.shape[0]:
        raise IndexError(f"slice index {slice_index} is outside [0, {component_labels.shape[0]})")
    component_slice = component_labels[slice_index]
    targets: list[InstanceTarget] = []
    for component_id in np.unique(component_slice):
        if component_id == 0:
            continue
        mask = component_slice == component_id
        rows, columns = np.nonzero(mask)
        targets.append(
            InstanceTarget(
                component_id=int(component_id),
                mask=mask,
                bbox_xyxy=(
                    int(columns.min()),
                    int(rows.min()),
                    int(columns.max()) + 1,
                    int(rows.max()) + 1,
                ),
            )
        )
    return targets


def prepare_subject(
    record: Mapping[str, Any], dataset_root: Path, profile: NormalizationProfile
) -> PreparedSubject:
    """Load one subject under the established spatial and normalization contracts."""
    subject = load_canonical_subject(record, dataset_root)
    modalities = {
        modality: np.asarray(image.get_fdata(dtype=np.float32))
        for modality, image in subject.modalities.items()
    }
    normalized_modalities = normalize_modalities(modalities, profile)
    semantic_label = np.asarray(subject.label.get_fdata(dtype=np.float32)) > 0.5
    component_labels, component_sizes = connected_components_3d(semantic_label)
    return PreparedSubject(
        subject_id=subject.subject_id,
        normalized_modalities=normalized_modalities,
        component_labels=component_labels,
        component_sizes=component_sizes,
        semantic_label=semantic_label,
    )


def build_sample_index(
    records: Iterable[Mapping[str, Any]], dataset_root: Path, profile: NormalizationProfile
) -> dict[str, Any]:
    """Materialize compact sample metadata while leaving 12-channel tensors lazy."""
    selected_records = list(records)
    selected_ids = tuple(record["subject_id"] for record in selected_records)
    if selected_ids != profile.fitted_subject_ids:
        raise ValueError(
            "sample records must match the normalization profile's fitted subject IDs exactly"
        )
    subjects = []
    for record in selected_records:
        prepared = prepare_subject(record, dataset_root, profile)
        instance_counts = [len(prepared.targets_for_slice(index)) for index in range(prepared.slice_count)]
        subjects.append(
            {
                "subject_id": prepared.subject_id,
                "slice_count": prepared.slice_count,
                "positive_slice_count": sum(count > 0 for count in instance_counts),
                "instance_counts_by_slice": instance_counts,
                "retained_component_sizes_voxels": prepared.component_sizes,
            }
        )
    return {
        "schema_version": 1,
        "subject_ids": list(selected_ids),
        "channel_order": list(channel_names()),
        "minimum_component_size_voxels": MINIMUM_COMPONENT_SIZE_VOXELS,
        "subjects": subjects,
    }


def write_sample_index(index: Mapping[str, Any], output_path: Path) -> None:
    """Write a compact JSON record of sample and target generation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
