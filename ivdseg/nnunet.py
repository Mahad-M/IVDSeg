"""Immutable dataset conversion and metrics for the B2 3D nnU-Net baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np

from ivdseg.manifest import FIXED_TEST_SUBJECTS, MODALITY_ORDER
from ivdseg.spatial import grids_match, load_canonical_subject


B2_DATASET_ID = 502
B2_DATASET_NAME = "Dataset502_IVDM3SegB2"
B2_CASE_PREFIX = "ivdm3seg"
B2_DEVELOPMENT_TRAIN_SUBJECT_IDS = ("01", "04", "05", "06", "08", "09", "11", "13", "15", "16")
B2_DEVELOPMENT_VALIDATION_SUBJECT_IDS = ("02", "12")
B2_PERMITTED_SUBJECT_IDS = B2_DEVELOPMENT_TRAIN_SUBJECT_IDS + B2_DEVELOPMENT_VALIDATION_SUBJECT_IDS


def _binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    """Calculate binary Dice, treating two empty masks as a perfect match."""
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(prediction, target).sum() / denominator)


@dataclass(frozen=True)
class B2SubjectDice:
    """One restored nnU-Net development prediction measured in canonical NIfTI geometry."""

    subject_id: str
    dice: float
    prediction_voxels: int
    target_voxels: int
    intersection_voxels: int


@dataclass(frozen=True)
class B2DevelopmentSummary:
    """Macro subject-volume Dice for B2's two locked development subjects."""

    subjects: tuple[B2SubjectDice, ...]

    @property
    def macro_dice(self) -> float:
        return float(np.mean([subject.dice for subject in self.subjects]))

    def to_dict(self) -> dict[str, Any]:
        return {"macro_dice": self.macro_dice, "subjects": [asdict(subject) for subject in self.subjects]}


def b2_case_id(subject_id: str) -> str:
    """Return B2's stable, nnU-Net-compatible case identifier."""
    if subject_id not in B2_PERMITTED_SUBJECT_IDS:
        raise ValueError(f"subject {subject_id} is not permitted in the B2 development dataset")
    return f"{B2_CASE_PREFIX}_{subject_id}"


def b2_dataset_json() -> dict[str, Any]:
    """Describe B2's four-channel binary semantic nnU-Net dataset."""
    return {
        "channel_names": {str(index): modality for index, modality in enumerate(MODALITY_ORDER)},
        "labels": {"background": 0, "IVD": 1},
        "numTraining": len(B2_PERMITTED_SUBJECT_IDS),
        "file_ending": ".nii.gz",
        "name": B2_DATASET_NAME,
        "description": "Four-modality 3D IVDM3Seg semantic segmentation; B2 development dataset",
        "reference": "docs/plans/rf-detr-2-5d-ivd-segmentation.md",
        "release": "1.0",
        "licence": "research-only; authoritative source remains IVDM3Seg",
    }


def b2_splits() -> list[dict[str, list[str]]]:
    """Return B2's only legal 10/2 development split in nnU-Net case IDs."""
    return [
        {
            "train": [b2_case_id(subject_id) for subject_id in B2_DEVELOPMENT_TRAIN_SUBJECT_IDS],
            "val": [b2_case_id(subject_id) for subject_id in B2_DEVELOPMENT_VALIDATION_SUBJECT_IDS],
        }
    ]


def select_b2_development_records(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Select exactly B2's permitted 12 cases and reject split drift before NIfTI I/O."""
    if tuple(manifest.get("fixed_test_subjects", ())) != FIXED_TEST_SUBJECTS:
        raise ValueError("B2 manifest has unexpected fixed-test subjects")
    subjects = manifest.get("subjects")
    if not isinstance(subjects, Sequence):
        raise ValueError("B2 manifest lacks a subject list")
    records: dict[str, Mapping[str, Any]] = {}
    for record in subjects:
        if not isinstance(record, Mapping):
            raise ValueError("B2 manifest contains an invalid subject record")
        subject_id = record.get("subject_id")
        if not isinstance(subject_id, str):
            raise ValueError("B2 manifest subject has no valid ID")
        if subject_id in records:
            raise ValueError(f"B2 manifest duplicates subject {subject_id}")
        records[subject_id] = record
    if set(B2_PERMITTED_SUBJECT_IDS) & set(FIXED_TEST_SUBJECTS):
        raise RuntimeError("B2 development split overlaps the fixed holdout")
    if not set(B2_PERMITTED_SUBJECT_IDS).issubset(records):
        missing = sorted(set(B2_PERMITTED_SUBJECT_IDS) - set(records))
        raise ValueError(f"B2 manifest lacks permitted subjects: {missing}")
    selected = tuple(records[subject_id] for subject_id in B2_PERMITTED_SUBJECT_IDS)
    selected_ids = tuple(record["subject_id"] for record in selected)
    if selected_ids != B2_PERMITTED_SUBJECT_IDS:
        raise RuntimeError("B2 development subject ordering drifted")
    return selected


def _export_nifti(image: nib.spatialimages.SpatialImage, path: Path, *, dtype: np.dtype[Any]) -> None:
    data = np.asarray(image.dataobj, dtype=dtype)
    output = nib.Nifti1Image(data, image.affine, image.header.copy())
    output.set_data_dtype(dtype)
    nib.save(output, str(path))


def _validate_existing_b2_dataset(dataset_dir: Path) -> None:
    """Accept only a complete, immutable B2 export when a run is retried."""
    expected_images = {
        f"{b2_case_id(subject_id)}_{channel:04d}.nii.gz"
        for subject_id in B2_PERMITTED_SUBJECT_IDS
        for channel in range(len(MODALITY_ORDER))
    }
    expected_labels = {f"{b2_case_id(subject_id)}.nii.gz" for subject_id in B2_PERMITTED_SUBJECT_IDS}
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError(f"existing B2 raw dataset has no imagesTr/labelsTr directories: {dataset_dir}")
    if {path.name for path in images_dir.glob("*.nii.gz")} != expected_images:
        raise ValueError("existing B2 raw dataset has unexpected image cases")
    if {path.name for path in labels_dir.glob("*.nii.gz")} != expected_labels:
        raise ValueError("existing B2 raw dataset has unexpected label cases")
    if json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8")) != b2_dataset_json():
        raise ValueError("existing B2 raw dataset has unexpected dataset.json")
    if json.loads((dataset_dir / "splits_final.json").read_text(encoding="utf-8")) != b2_splits():
        raise ValueError("existing B2 raw dataset has unexpected development split")
    provenance = json.loads((dataset_dir / "b2-provenance.json").read_text(encoding="utf-8"))
    if tuple(provenance.get("fixed_test_subjects_excluded", ())) != FIXED_TEST_SUBJECTS:
        raise ValueError("existing B2 raw dataset has unexpected fixed-test exclusions")


def export_b2_nnunet_dataset(
    *,
    manifest_path: Path,
    dataset_root: Path,
    nnunet_raw_root: Path,
) -> Path:
    """Write a canonical four-modality B2 raw dataset without reading fixed-test data.

    The conversion writes only into ``nnunet_raw_root``. It copies no legacy
    PNGs and applies the manifest's already-audited in-memory handling for
    subject 16's misaligned label before it becomes an nnU-Net training label.
    """
    dataset_dir = nnunet_raw_root / B2_DATASET_NAME
    if dataset_dir.exists():
        _validate_existing_b2_dataset(dataset_dir)
        return dataset_dir
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("B2 manifest must be a JSON object")
    records = select_b2_development_records(manifest)
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for record in records:
        subject = load_canonical_subject(record, dataset_root)
        case_id = b2_case_id(subject.subject_id)
        for channel, modality in enumerate(MODALITY_ORDER):
            _export_nifti(
                subject.modalities[modality],
                images_dir / f"{case_id}_{channel:04d}.nii.gz",
                dtype=np.dtype(np.float32),
            )
        _export_nifti(subject.label, labels_dir / f"{case_id}.nii.gz", dtype=np.dtype(np.uint8))
    (dataset_dir / "dataset.json").write_text(json.dumps(b2_dataset_json(), indent=2) + "\n", encoding="utf-8")
    (dataset_dir / "splits_final.json").write_text(json.dumps(b2_splits(), indent=2) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "source_dataset_root": str(dataset_root),
        "dataset_name": B2_DATASET_NAME,
        "dataset_id": B2_DATASET_ID,
        "modality_order": list(MODALITY_ORDER),
        "development_train_subject_ids": list(B2_DEVELOPMENT_TRAIN_SUBJECT_IDS),
        "development_validation_subject_ids": list(B2_DEVELOPMENT_VALIDATION_SUBJECT_IDS),
        "fixed_test_subjects_excluded": list(FIXED_TEST_SUBJECTS),
        "label_geometry": "RAS canonical source-label geometry; manifest-directed nearest label resampling for subject 16",
    }
    (dataset_dir / "b2-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return dataset_dir


def evaluate_b2_development_predictions(
    *,
    raw_dataset_dir: Path,
    prediction_dir: Path,
) -> B2DevelopmentSummary:
    """Measure B2's saved validation segmentations against its exported native labels."""
    subjects: list[B2SubjectDice] = []
    for subject_id in B2_DEVELOPMENT_VALIDATION_SUBJECT_IDS:
        case_id = b2_case_id(subject_id)
        label_path = raw_dataset_dir / "labelsTr" / f"{case_id}.nii.gz"
        prediction_path = prediction_dir / f"{case_id}.nii.gz"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"B2 validation prediction is missing: {prediction_path}")
        target_image = nib.load(str(label_path))
        prediction_image = nib.load(str(prediction_path))
        if not grids_match(target_image, prediction_image):
            raise ValueError(f"B2 validation prediction does not match label geometry for subject {subject_id}")
        target = np.asarray(target_image.dataobj) > 0
        prediction = np.asarray(prediction_image.dataobj) > 0
        subjects.append(
            B2SubjectDice(
                subject_id=subject_id,
                dice=_binary_dice(prediction, target),
                prediction_voxels=int(prediction.sum()),
                target_voxels=int(target.sum()),
                intersection_voxels=int(np.logical_and(prediction, target).sum()),
            )
        )
    return B2DevelopmentSummary(subjects=tuple(subjects))
