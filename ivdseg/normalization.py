"""Training-only nonzero-voxel normalization for canonical IVDM3Seg images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ivdseg.manifest import MODALITY_ORDER
from ivdseg.spatial import load_canonical_modalities


CLIP_RANGE = (-5.0, 5.0)


@dataclass(frozen=True)
class ModalityNormalizationStats:
    """Population statistics fitted from nonzero voxels of training images only."""

    mean: float
    std: float
    nonzero_voxel_count: int


@dataclass(frozen=True)
class NormalizationProfile:
    """A fixed, serializable transform shared by validation, test, and final training."""

    fitted_subject_ids: tuple[str, ...]
    modality_stats: Mapping[str, ModalityNormalizationStats]
    clip_range: tuple[float, float] = CLIP_RANGE

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "training-pool nonzero-voxel z-score",
            "fitted_subject_ids": list(self.fitted_subject_ids),
            "clip_range": list(self.clip_range),
            "zero_voxels": "preserve",
            "modality_stats": {
                modality: asdict(self.modality_stats[modality]) for modality in MODALITY_ORDER
            },
        }


def records_by_id(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {record["subject_id"]: record for record in manifest["subjects"]}


def select_training_records(
    manifest: Mapping[str, Any], subject_ids: Iterable[str]
) -> list[Mapping[str, Any]]:
    """Select explicit records and reject any holdout subject before image loading."""
    by_id = records_by_id(manifest)
    selected: list[Mapping[str, Any]] = []
    for subject_id in subject_ids:
        if subject_id not in by_id:
            raise ValueError(f"normalization requested unknown subject: {subject_id}")
        record = by_id[subject_id]
        if record["partition"] != "train":
            raise ValueError(
                f"normalization statistics may only use training subjects; got {subject_id}"
            )
        selected.append(record)
    if not selected:
        raise ValueError("normalization requires at least one training subject")
    return selected


def fit_normalization_profile(
    records: Iterable[Mapping[str, Any]], dataset_root: Path
) -> NormalizationProfile:
    """Fit one population mean and standard deviation per modality from nonzero voxels."""
    selected_records = list(records)
    if not selected_records:
        raise ValueError("normalization requires at least one training record")
    sums = {modality: 0.0 for modality in MODALITY_ORDER}
    squared_sums = {modality: 0.0 for modality in MODALITY_ORDER}
    counts = {modality: 0 for modality in MODALITY_ORDER}

    for record in selected_records:
        if record["partition"] != "train":
            raise ValueError(
                f"normalization statistics may only use training subjects; got {record['subject_id']}"
            )
        for modality, image in load_canonical_modalities(record, dataset_root).items():
            voxels = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float64)
            nonzero_voxels = voxels[voxels != 0.0]
            if not nonzero_voxels.size:
                raise ValueError(f"{record['subject_id']} {modality} has no nonzero voxels")
            counts[modality] += nonzero_voxels.size
            sums[modality] += float(np.sum(nonzero_voxels, dtype=np.float64))
            squared_sums[modality] += float(np.dot(nonzero_voxels, nonzero_voxels))

    stats: dict[str, ModalityNormalizationStats] = {}
    for modality in MODALITY_ORDER:
        mean = sums[modality] / counts[modality]
        variance = max((squared_sums[modality] / counts[modality]) - (mean * mean), 0.0)
        std = float(np.sqrt(variance))
        if std == 0.0:
            raise ValueError(f"{modality} nonzero training voxels have zero standard deviation")
        stats[modality] = ModalityNormalizationStats(
            mean=mean,
            std=std,
            nonzero_voxel_count=counts[modality],
        )
    return NormalizationProfile(
        fitted_subject_ids=tuple(record["subject_id"] for record in selected_records),
        modality_stats=stats,
    )


def normalize_volume(
    volume: np.ndarray, stats: ModalityNormalizationStats, *, clip_range: tuple[float, float] = CLIP_RANGE
) -> np.ndarray:
    """Standardize nonzero values only, preserving zero background and float32 output."""
    output = np.zeros(np.shape(volume), dtype=np.float32)
    nonzero = np.asarray(volume) != 0
    output[nonzero] = np.clip(
        (np.asarray(volume, dtype=np.float32)[nonzero] - stats.mean) / stats.std,
        clip_range[0],
        clip_range[1],
    )
    return output


def normalize_modalities(
    modalities: Mapping[str, np.ndarray], profile: NormalizationProfile
) -> dict[str, np.ndarray]:
    """Normalize in manifest channel order and reject incomplete modality mappings."""
    if tuple(modalities) != MODALITY_ORDER:
        raise ValueError(f"modalities must use manifest order {MODALITY_ORDER}")
    return {
        modality: normalize_volume(modalities[modality], profile.modality_stats[modality], clip_range=profile.clip_range)
        for modality in MODALITY_ORDER
    }


def write_profile(profile: NormalizationProfile, output_path: Path) -> None:
    """Write a stable normalization artifact for a specific training-subject set."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_profile(profile_path: Path) -> NormalizationProfile:
    """Load a previously fitted profile rather than recomputing any statistics."""
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    try:
        stats = {
            modality: ModalityNormalizationStats(**payload["modality_stats"][modality])
            for modality in MODALITY_ORDER
        }
        return NormalizationProfile(
            fitted_subject_ids=tuple(payload["fitted_subject_ids"]),
            modality_stats=stats,
            clip_range=tuple(payload["clip_range"]),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid normalization profile: {profile_path}") from error
