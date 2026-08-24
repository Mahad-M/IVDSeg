from pathlib import Path

import numpy as np
import pytest
import torch

from ivdseg.datamodule import IVDDataModule, SemanticIVDDataModule
from ivdseg.normalization import ModalityNormalizationStats, NormalizationProfile
from ivdseg.samples import PreparedSubject


def _profile() -> NormalizationProfile:
    stats = ModalityNormalizationStats(mean=0.0, std=1.0, nonzero_voxel_count=1)
    return NormalizationProfile(fitted_subject_ids=("01",), modality_stats={name: stats for name in ("fat", "inn", "opp", "water")})


def _manifest() -> dict:
    def record(subject_id: str, partition: str = "train") -> dict:
        return {
            "subject_id": subject_id,
            "partition": partition,
            "modalities": {},
            "label": "unused",
            "label_alignment": {"reference_modality": "fat"},
        }

    return {"subjects": [record("01"), record("02"), record("03", "test")]}


def _prepared(subject_id: str) -> PreparedSubject:
    modalities = {name: np.ones((2, 8, 8), dtype=np.float32) for name in ("fat", "inn", "opp", "water")}
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 2:6, 3:7] = 1
    return PreparedSubject(subject_id, modalities, labels, {1: 16}, semantic_label=labels.astype(bool))


def test_custom_datamodule_returns_rfdetr_nested_tensor_and_mask_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ivdseg.datamodule.prepare_subject", lambda record, dataset_root, profile: _prepared(record["subject_id"]))
    data_module = IVDDataModule(
        manifest=_manifest(),
        dataset_root=Path("unused"),
        normalization_profile=_profile(),
        train_subject_ids=("01",),
        validation_subject_ids=("02",),
        batch_size=2,
        resolution=24,
        train_augmentation=None,
    )
    data_module.setup("fit")

    samples, targets = next(iter(data_module.val_dataloader()))

    assert samples.tensors.shape == (2, 12, 24, 24)
    assert samples.mask.shape == (2, 24, 24)
    assert len(targets) == 2
    assert targets[0]["boxes"].shape == (1, 4)
    assert targets[0]["boxes"].dtype == torch.float32
    assert targets[0]["masks"].shape == (1, 24, 24)
    assert targets[0]["masks"].dtype == torch.bool
    assert targets[0]["labels"].tolist() == [0]
    assert targets[1]["boxes"].shape == (0, 4)
    assert targets[1]["masks"].shape == (0, 24, 24)


def test_datamodule_rejects_fixed_holdout_subjects_before_data_loading() -> None:
    with pytest.raises(ValueError, match="fixed test subjects"):
        IVDDataModule(
            manifest=_manifest(),
            dataset_root=Path("unused"),
            normalization_profile=_profile(),
            train_subject_ids=("01",),
            validation_subject_ids=("03",),
            batch_size=2,
            resolution=24,
        )


def test_datamodule_requires_profile_to_match_optimization_training_subjects() -> None:
    mismatched_profile = NormalizationProfile(
        fitted_subject_ids=("02",),
        modality_stats=_profile().modality_stats,
    )
    with pytest.raises(ValueError, match="must exactly match"):
        IVDDataModule(
            manifest=_manifest(),
            dataset_root=Path("unused"),
            normalization_profile=mismatched_profile,
            train_subject_ids=("01",),
            validation_subject_ids=("02",),
            batch_size=2,
            resolution=24,
        )


def test_datamodule_supports_train_only_final_retraining(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ivdseg.datamodule.prepare_subject", lambda record, dataset_root, profile: _prepared(record["subject_id"]))
    data_module = IVDDataModule(
        manifest=_manifest(),
        dataset_root=Path("unused"),
        normalization_profile=_profile(),
        train_subject_ids=("01",),
        validation_subject_ids=(),
        batch_size=2,
        resolution=24,
        train_augmentation=None,
    )

    data_module.setup("fit")

    samples, targets = next(iter(data_module.train_dataloader()))
    assert samples.tensors.shape == (2, 12, 24, 24)
    assert len(targets) == 2
    assert data_module.val_dataloader() is None


def test_semantic_datamodule_returns_complete_binary_masks_with_standard_batching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ivdseg.datamodule.prepare_subject", lambda record, dataset_root, profile: _prepared(record["subject_id"]))
    data_module = SemanticIVDDataModule(
        manifest=_manifest(),
        dataset_root=Path("unused"),
        normalization_profile=_profile(),
        train_subject_ids=("01",),
        validation_subject_ids=("02",),
        batch_size=2,
        resolution=32,
        train_augmentation=None,
    )
    data_module.setup("fit")

    images, masks, indices = next(iter(data_module.val_dataloader()))

    assert images.shape == (2, 12, 32, 32)
    assert masks.shape == (2, 32, 32)
    assert masks.dtype == torch.bool
    assert indices.tolist() == [0, 1]
    assert masks[0].any()
    assert not masks[1].any()
