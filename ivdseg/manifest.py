"""Deterministic subject manifest for the authoritative IVDM3Seg NIfTI corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = 1
SUBJECT_IDS = tuple(f"{number:02d}" for number in range(1, 17))
FIXED_TEST_SUBJECTS = ("03", "07", "10", "14")
DEVELOPMENT_CANDIDATES = ("02", "12")
MODALITY_ORDER = ("fat", "inn", "opp", "water")
FILE_TOKENS = {"fat": "fat", "inn": "inn", "opp": "opp", "water": "wat"}
SUBJECT_16_RESAMPLING_REASON = (
    "The audit found up to 10.817 mm image-label corner displacement; all image modalities "
    "share the oblique image grid."
)


def source_partition(subject_id: str) -> str:
    """Return the source-directory partition for a known subject."""
    if subject_id not in SUBJECT_IDS:
        raise ValueError(f"unknown subject ID: {subject_id}")
    return "valid" if subject_id in FIXED_TEST_SUBJECTS else "train"


def model_partition(subject_id: str) -> str:
    """Return the immutable model partition for a known subject."""
    return "test" if subject_id in FIXED_TEST_SUBJECTS else "train"


def relative_subject_record(subject_id: str) -> dict[str, Any]:
    """Build one portable record using paths relative to the dataset root."""
    source = source_partition(subject_id)
    modalities = {
        modality: f"{source}/{subject_id}/{subject_id}_{FILE_TOKENS[modality]}.nii"
        for modality in MODALITY_ORDER
    }
    label_alignment: dict[str, Any] = {
        "action": "preserve",
        "reference_modality": "fat",
        "interpolation": None,
        "reason": None,
    }
    if subject_id == "16":
        label_alignment = {
            "action": "resample_to_image_reference",
            "reference_modality": "fat",
            "interpolation": "nearest",
            "reason": SUBJECT_16_RESAMPLING_REASON,
        }
    return {
        "subject_id": subject_id,
        "source_partition": source,
        "partition": model_partition(subject_id),
        "development_candidate": subject_id in DEVELOPMENT_CANDIDATES,
        "modalities": modalities,
        "label": f"{source}/labels/{subject_id}_Labels.nii",
        "label_alignment": label_alignment,
    }


def validate_manifest_files(dataset_root: Path, manifest: dict[str, Any]) -> None:
    """Fail with every missing authoritative source path rather than a partial manifest."""
    missing: list[Path] = []
    for subject in manifest["subjects"]:
        paths = [*subject["modalities"].values(), subject["label"]]
        missing.extend(dataset_root / path for path in paths if not (dataset_root / path).is_file())
    if missing:
        formatted_paths = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"manifest source files are missing:\n{formatted_paths}")


def build_manifest(dataset_root: Path, *, validate_files: bool = True) -> dict[str, Any]:
    """Return the versioned manifest without inspecting or modifying NIfTI content."""
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_name": "IVDM3Seg",
        "dataset_root": str(dataset_root),
        "modality_order": list(MODALITY_ORDER),
        "fixed_test_subjects": list(FIXED_TEST_SUBJECTS),
        "development_candidate_subjects": list(DEVELOPMENT_CANDIDATES),
        "audit_artifact": "artifacts/data-audit/ivdm3seg-nifti-audit.json",
        "subjects": [relative_subject_record(subject_id) for subject_id in SUBJECT_IDS],
    }
    if validate_files:
        validate_manifest_files(dataset_root, manifest)
    return manifest


def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Write a stable, diffable JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("IVDM3Seg"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/manifests/ivdm3seg-v1.json")
    )
    args = parser.parse_args()

    manifest = build_manifest(args.dataset_root)
    write_manifest(manifest, args.output)
    print(
        f"Wrote {args.output}: {len(manifest['subjects'])} subjects, "
        f"{len(manifest['fixed_test_subjects'])} fixed test subjects"
    )


if __name__ == "__main__":
    main()
