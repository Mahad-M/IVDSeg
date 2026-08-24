import json
from pathlib import Path

import nibabel as nib
import numpy as np

from scripts.train_b2_development import _finish_run_card, _training_command, _write_run_card
from ivdseg.manifest import MODALITY_ORDER
from ivdseg.nnunet import (
    B2_DATASET_NAME,
    B2_DEVELOPMENT_TRAIN_SUBJECT_IDS,
    B2_DEVELOPMENT_VALIDATION_SUBJECT_IDS,
    b2_case_id,
    b2_splits,
    evaluate_b2_development_predictions,
    export_b2_nnunet_dataset,
)


def _write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def test_b2_export_uses_only_the_locked_development_subjects_and_preserves_geometry(tmp_path: Path) -> None:
    dataset_root = tmp_path / "source"
    subjects = []
    all_subject_ids = B2_DEVELOPMENT_TRAIN_SUBJECT_IDS + B2_DEVELOPMENT_VALIDATION_SUBJECT_IDS
    for subject_id in all_subject_ids:
        modalities = {}
        for channel, modality in enumerate(MODALITY_ORDER):
            relative = f"train/{subject_id}/{subject_id}_{modality}.nii"
            modalities[modality] = relative
            _write_nifti(dataset_root / relative, np.full((3, 4, 5), channel + 1, dtype=np.float32))
        label_relative = f"train/labels/{subject_id}_Labels.nii"
        label = np.zeros((3, 4, 5), dtype=np.uint8)
        label[1, 2, 3] = 1
        _write_nifti(dataset_root / label_relative, label)
        subjects.append(
            {
                "subject_id": subject_id,
                "modalities": modalities,
                "label": label_relative,
                "label_alignment": {
                    "action": "preserve",
                    "reference_modality": "fat",
                    "interpolation": None,
                    "reason": None,
                },
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"fixed_test_subjects": ["03", "07", "10", "14"], "subjects": subjects}),
        encoding="utf-8",
    )

    raw_dataset = export_b2_nnunet_dataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        nnunet_raw_root=tmp_path / "nnunet-raw",
    )

    assert raw_dataset.name == B2_DATASET_NAME
    assert len(list((raw_dataset / "imagesTr").glob("*.nii.gz"))) == len(all_subject_ids) * 4
    assert len(list((raw_dataset / "labelsTr").glob("*.nii.gz"))) == len(all_subject_ids)
    assert json.loads((raw_dataset / "splits_final.json").read_text(encoding="utf-8")) == b2_splits()
    dataset_json = json.loads((raw_dataset / "dataset.json").read_text(encoding="utf-8"))
    assert dataset_json["channel_names"] == {"0": "fat", "1": "inn", "2": "opp", "3": "water"}
    assert (
        export_b2_nnunet_dataset(
            manifest_path=manifest_path,
            dataset_root=dataset_root,
            nnunet_raw_root=tmp_path / "nnunet-raw",
        )
        == raw_dataset
    )

    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    for subject_id in B2_DEVELOPMENT_VALIDATION_SUBJECT_IDS:
        label = nib.load(str(raw_dataset / "labelsTr" / f"{b2_case_id(subject_id)}.nii.gz"))
        nib.save(nib.Nifti1Image(np.asarray(label.dataobj), label.affine), str(prediction_dir / f"{b2_case_id(subject_id)}.nii.gz"))

    summary = evaluate_b2_development_predictions(raw_dataset_dir=raw_dataset, prediction_dir=prediction_dir)

    assert summary.macro_dice == 1.0
    assert [subject.subject_id for subject in summary.subjects] == ["02", "12"]


def test_b2_run_card_resumes_after_a_modal_preemption(tmp_path: Path) -> None:
    card_path = tmp_path / "B2-nnunet-3d-17.md"
    _write_run_card(card_path, experiment_id="B2-nnunet-3d", seed=17, command="first-attempt")
    _finish_run_card(
        card_path,
        status="Interrupted / resumable",
        conclusion="B2 development run was interrupted: KeyboardInterrupt",
    )

    resumed = _write_run_card(
        card_path,
        experiment_id="B2-nnunet-3d",
        seed=17,
        command="restarted-after-preemption",
    )

    content = card_path.read_text(encoding="utf-8")
    assert resumed is True
    assert "**Status:** Running" in content
    assert "**Finished:**\n" in content
    assert "**Outcome:** Running after a resumable interruption." in content

    _finish_run_card(card_path, status="Complete", conclusion="B2 development training completed.")
    completed_content = card_path.read_text(encoding="utf-8")
    assert "**Status:** Complete" in completed_content
    assert "**Outcome:** B2 development training completed." in completed_content


def test_b2_resumed_run_requests_nnunet_checkpoint_continuation() -> None:
    assert "--c" not in _training_command(resume=False)
    assert _training_command(resume=True)[-1] == "--c"
