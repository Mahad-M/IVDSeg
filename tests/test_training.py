import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule, LightningModule

from ivdseg.datamodule import SliceReference
from ivdseg.training import (
    CUDA_RUNTIME_VERSION,
    DEVELOPMENT_DICE_METRIC,
    DevelopmentThresholdGrid,
    DevelopmentTrainingConfig,
    DevelopmentVolumeDiceMonitor,
    FINAL_TRAIN_SUBJECT_IDS,
    FinalTrainingConfig,
    PlainTextProgressReporter,
    adapt_rgb_patch_embedding,
    binary_dice,
    make_model_config,
    make_train_config,
    load_development_training_config,
    load_final_training_config,
    build_final_training,
    verify_cuda_runtime,
)
from rfdetr.config import RFDETRSegLargeConfig


def _result(scores: list[float], labels: list[int], masks: list[torch.Tensor]) -> dict:
    return {
        "scores": torch.tensor(scores),
        "labels": torch.tensor(labels),
        "masks": torch.stack(masks).unsqueeze(1).bool(),
    }


def _target(index: int) -> dict:
    return {"image_id": torch.tensor([index])}


def test_development_monitor_reconstructs_subject_volumes_with_fixed_score_threshold() -> None:
    target_02 = np.zeros((2, 3, 3), dtype=bool)
    target_02[0, 1, 1] = True
    target_02[1, 0, 2] = True
    target_12 = np.zeros((1, 3, 3), dtype=bool)
    target_12[0, 2, 0] = True
    monitor = DevelopmentVolumeDiceMonitor(
        references=(SliceReference("02", 0), SliceReference("02", 1), SliceReference("12", 0)),
        semantic_labels={"02": target_02, "12": target_12},
        score_threshold=0.50,
    )
    one = torch.zeros((3, 3), dtype=torch.bool)
    one[1, 1] = True
    wrong_class = torch.zeros((3, 3), dtype=torch.bool)
    wrong_class[0, 0] = True
    monitor.add_batch([_result([0.51, 0.99], [0, 1], [one, wrong_class])], [_target(0)])
    monitor.add_batch([_result([0.50], [0], [one])], [_target(1)])  # Strictly above threshold only.
    last = torch.zeros((3, 3), dtype=torch.bool)
    last[2, 0] = True
    monitor.add_batch([_result([0.9], [0], [last])], [_target(2)])

    summary = monitor.finalize()

    assert summary.score_threshold == 0.50
    assert summary.subjects[0].subject_id == "02"
    assert summary.subjects[0].dice == pytest.approx(2.0 / 3.0)
    assert summary.subjects[1].subject_id == "12"
    assert summary.subjects[1].dice == pytest.approx(1.0)
    assert summary.macro_dice == pytest.approx(5.0 / 6.0)


def test_development_monitor_rejects_incomplete_or_duplicate_slice_reconstructions() -> None:
    monitor = DevelopmentVolumeDiceMonitor(
        references=(SliceReference("02", 0), SliceReference("02", 1)),
        semantic_labels={"02": np.zeros((2, 2, 2), dtype=bool)},
    )
    empty = torch.zeros((2, 2), dtype=torch.bool)
    monitor.add_batch([_result([0.9], [0], [empty])], [_target(0)])

    with pytest.raises(RuntimeError, match="incomplete"):
        monitor.finalize()
    with pytest.raises(ValueError, match="more than once"):
        monitor.add_batch([_result([0.9], [0], [empty])], [_target(0)])


def test_development_threshold_grid_selects_the_best_and_lowest_tied_threshold() -> None:
    target = np.zeros((1, 3, 3), dtype=bool)
    target[0, 1, 1] = True
    grid = DevelopmentThresholdGrid(
        references=(SliceReference("02", 0),),
        semantic_labels={"02": target},
        thresholds=(0.05, 0.50),
    )
    mask = torch.zeros((3, 3), dtype=torch.bool)
    mask[1, 1] = True

    grid.add_batch([_result([0.90], [0], [mask])], [_target(0)])
    selection = grid.finalize()

    assert tuple(summary.score_threshold for summary in selection.summaries) == (0.05, 0.50)
    assert selection.selected.score_threshold == 0.05
    assert selection.selected.macro_dice == pytest.approx(1.0)
    assert selection.to_dict()["selected_score_threshold"] == 0.05


def test_binary_dice_handles_empty_volumes_and_rejects_shape_mismatch() -> None:
    assert binary_dice(np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool)) == 1.0
    with pytest.raises(ValueError, match="matching shapes"):
        binary_dice(np.zeros((2, 2), dtype=bool), np.zeros((2, 3), dtype=bool))


def test_rgb_patch_embedding_adaptation_tiles_and_scales_weights_for_twelve_channels() -> None:
    projection = nn.Conv2d(3, 1, kernel_size=1, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]]))

    adapted = adapt_rgb_patch_embedding(projection, num_channels=12)

    assert adapted.in_channels == 12
    torch.testing.assert_close(
        adapted.weight.flatten(),
        torch.tensor([0.25, 0.50, 0.75, 0.25, 0.50, 0.75, 0.25, 0.50, 0.75, 0.25, 0.50, 0.75]),
    )
    assert projection.in_channels == 3
    assert projection.weight.shape == (1, 3, 1, 1)


def test_primary_training_configuration_locks_effective_batch_and_model_shape() -> None:
    config = DevelopmentTrainingConfig(
        manifest_path=Path("manifest.json"),
        dataset_root=Path("data"),
        normalization_profile=Path("profile.json"),
        run_dir=Path("run"),
        seed=17,
        pretrain_weights=None,
    )

    model_config = make_model_config(config)
    train_config = make_train_config(config)

    assert config.effective_batch_size == 8
    assert model_config.num_channels == 12
    assert model_config.num_classes == 1
    assert model_config.resolution == 384
    assert train_config.epochs == 100
    assert train_config.grad_accum_steps == 4
    assert config.deterministic == "warn"
    assert config.progress_report_interval_batches == 10
    with pytest.raises(ValueError, match="effective batch size 8"):
        DevelopmentTrainingConfig(
            manifest_path=Path("manifest.json"),
            dataset_root=Path("data"),
            normalization_profile=Path("profile.json"),
            run_dir=Path("run"),
            seed=17,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
        )


def test_large_near_native_ablation_changes_model_and_uses_a_valid_264_resolution() -> None:
    config = DevelopmentTrainingConfig(
        manifest_path=Path("manifest.json"),
        dataset_root=Path("data"),
        normalization_profile=Path("profile.json"),
        run_dir=Path("run"),
        seed=17,
        model_variant="large",
        resolution=264,
        pretrain_weights=None,
    )

    model_config = make_model_config(config)

    assert isinstance(model_config, RFDETRSegLargeConfig)
    assert model_config.resolution == 264
    assert model_config.dec_layers == 5
    assert model_config.num_queries == 200
    assert config.effective_batch_size == 8
    with pytest.raises(ValueError, match="multiple of RF-DETR Seg's block size"):
        DevelopmentTrainingConfig(
            manifest_path=Path("manifest.json"),
            dataset_root=Path("data"),
            normalization_profile=Path("profile.json"),
            run_dir=Path("run"),
            seed=17,
            model_variant="large",
            resolution=256,
        )


def test_final_training_configuration_uses_all_permitted_subjects_and_selected_epoch_count() -> None:
    config = FinalTrainingConfig(
        manifest_path=Path("manifest.json"),
        dataset_root=Path("data"),
        normalization_profile=Path("profile.json"),
        run_dir=Path("run"),
        seed=17,
        pretrain_weights=None,
    )

    assert FINAL_TRAIN_SUBJECT_IDS == ("01", "02", "04", "05", "06", "08", "09", "11", "12", "13", "15", "16")
    assert config.max_epochs == 7
    assert config.selected_score_threshold == pytest.approx(0.35)
    assert config.model_variant == "small"
    assert config.effective_batch_size == 8
    assert make_model_config(config).num_channels == 12
    with pytest.raises(ValueError, match="effective batch size 8"):
        FinalTrainingConfig(
            manifest_path=Path("manifest.json"),
            dataset_root=Path("data"),
            normalization_profile=Path("profile.json"),
            run_dir=Path("run"),
            seed=17,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
        )


def test_final_large_configuration_preserves_the_development_selected_model_definition() -> None:
    config = FinalTrainingConfig(
        manifest_path=Path("manifest.json"),
        dataset_root=Path("data"),
        normalization_profile=Path("profile.json"),
        run_dir=Path("run"),
        seed=17,
        model_variant="large",
        max_epochs=19,
        selected_score_threshold=0.45,
        source_development_run="R0-large-384-17",
        source_development_checkpoint="best-epoch=018.ckpt",
        pretrain_weights=None,
    )

    model_config = make_model_config(config)

    assert isinstance(model_config, RFDETRSegLargeConfig)
    assert model_config.resolution == 384
    assert config.max_epochs == 19
    assert config.selected_score_threshold == pytest.approx(0.45)


def test_final_trainer_runs_without_requesting_a_validation_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Final RF-DETR retraining is train-only even though its base model validates."""

    class _TrainOnlyDataModule(LightningDataModule):
        def __init__(self, **_: object) -> None:
            super().__init__()
            self.train_dataset = object()

        def setup(self, stage: str | None = None) -> None:
            del stage

        def train_dataloader(self) -> DataLoader[tuple[torch.Tensor]]:
            return DataLoader(TensorDataset(torch.ones((4, 1))), batch_size=2)

        def val_dataloader(self) -> None:
            return None

    class _FinalModuleWithInheritedValidation(LightningModule):
        def __init__(self, *_: object) -> None:
            super().__init__()
            self.layer = nn.Linear(1, 1)

        def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
            del batch_idx
            return self.layer(batch[0]).sum()

        def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
            del batch_idx
            return self.layer(batch[0]).sum()

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.SGD(self.parameters(), lr=0.1)

    monkeypatch.setattr("ivdseg.training.IVDDataModule", _TrainOnlyDataModule)
    monkeypatch.setattr("ivdseg.training.FinalTrainingModule", _FinalModuleWithInheritedValidation)
    config = FinalTrainingConfig(
        manifest_path=Path("manifest.json"),
        dataset_root=Path("data"),
        normalization_profile=Path("profile.json"),
        run_dir=tmp_path / "run",
        seed=17,
        accelerator="cpu",
        pretrain_weights=None,
    )

    trainer, module, data_module, _last_checkpoint = build_final_training(config)

    trainer.fit(module, datamodule=data_module)


def test_load_development_training_config_accepts_a_pre_progress_run_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "development_training": {
                    "manifest_path": "/data/manifest.json",
                    "dataset_root": "/data/IVDM3Seg",
                    "normalization_profile": "/data/profile.json",
                    "run_dir": "/runs/R0-modal-17",
                    "seed": 17,
                    "experiment_id": "R0-modal",
                    "pretrain_weights": "/data/weights.pt",
                    "accelerator": "gpu",
                    "deterministic": "warn",
                    "effective_batch_size": 8,
                    "fixed_test_subjects_excluded": ["03", "07", "10", "14"],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_development_training_config(path)

    assert config.manifest_path == Path("/data/manifest.json")
    assert config.run_dir == Path("/runs/R0-modal-17")
    assert config.progress_report_interval_batches == 10


def test_load_final_training_config_restores_selected_test_evaluation_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "final_training": {
                    "manifest_path": "/data/manifest.json",
                    "dataset_root": "/data/IVDM3Seg",
                    "normalization_profile": "/data/profile.json",
                    "run_dir": "/runs/R0-final-retry-17",
                    "seed": 17,
                    "experiment_id": "R0-final-retry",
                    "max_epochs": 7,
                    "selected_score_threshold": 0.35,
                    "source_development_run": "R0-modal-17",
                    "source_development_checkpoint": "best-epoch=006.ckpt",
                    "pretrain_weights": "/data/weights.pt",
                    "accelerator": "gpu",
                    "deterministic": "warn",
                    "train_subject_ids": ["01", "02"],
                    "validation_subject_ids": [],
                    "fixed_test_subjects_excluded": ["03", "07", "10", "14"],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_final_training_config(path)

    assert config.run_dir == Path("/runs/R0-final-retry-17")
    assert config.max_epochs == 7
    assert config.selected_score_threshold == pytest.approx(0.35)


def test_plain_text_progress_reporter_emits_modal_visible_updates(capsys: pytest.CaptureFixture[str]) -> None:
    reporter = PlainTextProgressReporter(every_n_train_batches=10)

    class _Trainer:
        is_global_zero = True
        current_epoch = 2
        max_epochs = 100
        num_training_batches = 180
        global_step = 7
        callback_metrics = {DEVELOPMENT_DICE_METRIC: torch.tensor(0.912345)}

    trainer = _Trainer()
    reporter.on_train_batch_end(trainer, None, None, None, batch_idx=8)
    assert capsys.readouterr().out == ""

    reporter.on_train_batch_end(trainer, None, None, None, batch_idx=9)
    reporter.on_validation_end(trainer, None)

    output = capsys.readouterr().out
    assert "[ivdseg-progress] epoch=3/100 train_batch=10/180 optimizer_step=7" in output
    assert "[ivdseg-progress] epoch=3/100 validation_complete macro_3d_dice=0.912345" in output


def test_cuda_preflight_rejects_an_unexpected_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "cuda", "13.0")

    with pytest.raises(RuntimeError, match="expected CUDA 12.6, found 13.0"):
        verify_cuda_runtime()


def test_cuda_preflight_checks_a_real_tensor_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    allocations: list[tuple[object, object]] = []

    class _Tensor:
        def zero_(self) -> "_Tensor":
            return self

    def _empty(shape: object, *, device: object) -> _Tensor:
        allocations.append((shape, device))
        return _Tensor()

    monkeypatch.setattr(torch.version, "cuda", CUDA_RUNTIME_VERSION)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "empty", _empty)

    verify_cuda_runtime()

    assert allocations == [(1, "cuda")]
