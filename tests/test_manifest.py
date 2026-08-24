from pathlib import Path

import pytest

from ivdseg.manifest import (
    DEVELOPMENT_CANDIDATES,
    FIXED_TEST_SUBJECTS,
    MODALITY_ORDER,
    SUBJECT_IDS,
    build_manifest,
)


def records_by_id(manifest: dict) -> dict[str, dict]:
    return {record["subject_id"]: record for record in manifest["subjects"]}


def test_manifest_has_all_subjects_with_immutable_split_and_modalities() -> None:
    manifest = build_manifest(Path("unused"), validate_files=False)
    records = records_by_id(manifest)

    assert tuple(records) == SUBJECT_IDS
    assert manifest["fixed_test_subjects"] == list(FIXED_TEST_SUBJECTS)
    assert manifest["development_candidate_subjects"] == list(DEVELOPMENT_CANDIDATES)
    assert {subject_id for subject_id, record in records.items() if record["partition"] == "test"} == set(
        FIXED_TEST_SUBJECTS
    )
    assert all(tuple(record["modalities"]) == MODALITY_ORDER for record in records.values())
    assert records["01"]["modalities"]["water"] == "train/01/01_wat.nii"
    assert records["03"]["label"] == "valid/labels/03_Labels.nii"


def test_manifest_marks_only_subject_16_for_nearest_label_resampling() -> None:
    records = records_by_id(build_manifest(Path("unused"), validate_files=False))

    exceptions = {
        subject_id: record["label_alignment"]
        for subject_id, record in records.items()
        if record["label_alignment"]["action"] != "preserve"
    }
    assert exceptions == {
        "16": {
            "action": "resample_to_image_reference",
            "reference_modality": "fat",
            "interpolation": "nearest",
            "reason": (
                "The audit found up to 10.817 mm image-label corner displacement; all image "
                "modalities share the oblique image grid."
            ),
        }
    }


def test_manifest_content_is_deterministic() -> None:
    assert build_manifest(Path("IVDM3Seg"), validate_files=False) == build_manifest(
        Path("IVDM3Seg"), validate_files=False
    )


def test_file_validation_reports_missing_sources(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"train/01/01_fat\.nii"):
        build_manifest(tmp_path)
