import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from ivdseg.datamodule import SliceReference
from ivdseg.resunet import ResUNet34
from ivdseg.swin_unet import SwinV2SmallUNet, SwinV2TinyUNet, adapt_pretrained_patch_projection
from ivdseg.unet_training import (
    B1DevelopmentConfig,
    B1FinalTrainingConfig,
    B3DevelopmentConfig,
    B3FinalTrainingConfig,
    B4DevelopmentConfig,
    SemanticDevelopmentVolumeDiceMonitor,
    SemanticDevelopmentThresholdGrid,
    binary_dice_loss,
    load_b1_final_training_config,
    load_b3_development_config,
    load_b3_final_training_config,
    load_b4_development_config,
)


def test_resunet34_preserves_the_segmentation_plane_and_requires_residual_grid_alignment() -> None:
    model = ResUNet34()

    logits = model(torch.zeros((2, 12, 64, 64)))

    assert logits.shape == (2, 1, 64, 64)
    assert model.stage_depths == (3, 4, 6, 3)
    with pytest.raises(ValueError, match="divisible by 16"):
        model(torch.zeros((1, 12, 62, 64)))


def test_semantic_development_monitor_reconstructs_native_subject_volumes_from_logits() -> None:
    target = np.zeros((2, 2, 2), dtype=bool)
    target[0, 0, 1] = True
    monitor = SemanticDevelopmentVolumeDiceMonitor(
        references=(SliceReference("02", 0), SliceReference("02", 1)),
        semantic_labels={"02": target},
        probability_threshold=0.5,
    )
    logits = torch.tensor(
        [
            [[[-10.0, 10.0], [-10.0, -10.0]]],
            [[[-10.0, -10.0], [-10.0, -10.0]]],
        ]
    )

    monitor.add_batch(logits, torch.tensor([0, 1]))
    summary = monitor.finalize()

    assert summary.macro_dice == pytest.approx(1.0)
    assert summary.subjects[0].prediction_voxels == 1
    assert summary.subjects[0].target_voxels == 1
    assert binary_dice_loss(logits, target=torch.zeros((2, 2, 2))).isfinite()


def test_b1_configuration_locks_residual_baseline_protocol() -> None:
    config = B1DevelopmentConfig(
        manifest_path=Path("manifest.json"),
        dataset_root=Path("data"),
        normalization_profile=Path("profile.json"),
        run_dir=Path("run"),
        seed=17,
    )

    assert config.effective_batch_size == 8
    assert config.resolution == 384
    assert config.probability_threshold == pytest.approx(0.5)
    assert config.deterministic == "warn"
    with pytest.raises(ValueError, match="effective batch size 8"):
        B1DevelopmentConfig(
            manifest_path=Path("manifest.json"),
            dataset_root=Path("data"),
            normalization_profile=Path("profile.json"),
            run_dir=Path("run"),
            seed=17,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
        )


def test_semantic_threshold_grid_selects_the_lower_cutoff_for_an_exact_tie() -> None:
    target = np.zeros((1, 2, 2), dtype=bool)
    target[0, 0, 0] = True
    grid = SemanticDevelopmentThresholdGrid(
        references=(SliceReference("02", 0),),
        semantic_labels={"02": target},
        thresholds=(0.25, 0.50),
    )

    grid.add_batch(torch.full((1, 1, 2, 2), 10.0), torch.tensor([0]))
    selection = grid.finalize()

    assert selection.selected.probability_threshold == pytest.approx(0.25)
    assert selection.selected.macro_dice == pytest.approx(0.4)


def test_b1_final_configuration_loader_rejects_any_split_drift(tmp_path: Path) -> None:
    config = B1FinalTrainingConfig(
        manifest_path=Path("manifest.json"),
        dataset_root=Path("data"),
        normalization_profile=Path("profile.json"),
        run_dir=Path("run"),
        seed=17,
    )
    payload = {
        "schema_version": 1,
        "b1_final_training": {
            **config.__dict__,
            "manifest_path": str(config.manifest_path),
            "dataset_root": str(config.dataset_root),
            "normalization_profile": str(config.normalization_profile),
            "run_dir": str(config.run_dir),
            "train_subject_ids": ["01", "02", "04", "05", "06", "08", "09", "11", "12", "13", "15", "16"],
            "validation_subject_ids": [],
            "fixed_test_subjects_excluded": ["03", "07", "10", "14"],
            "effective_batch_size": 8,
            "initialization": "from_scratch",
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_b1_final_training_config(path) == config
    payload["b1_final_training"]["fixed_test_subjects_excluded"] = ["03"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed-test exclusions"):
        load_b1_final_training_config(path)


def test_b1_final_evaluator_uses_the_evaluation_only_fixed_test_selector() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_b1_final.py"
    spec = importlib.util.spec_from_file_location("evaluate_b1_final_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = {
        "subjects": [
            {"subject_id": subject_id, "partition": "test"}
            for subject_id in ("03", "07", "10", "14")
        ]
    }

    records = module._select_b1_final_test_records(manifest)

    assert [record["subject_id"] for record in records] == ["03", "07", "10", "14"]


def test_pretrained_swin_patch_projection_preserves_an_equal_modality_group_response() -> None:
    projection = torch.nn.Conv2d(3, 2, kernel_size=1, bias=True)
    with torch.no_grad():
        projection.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3, 1, 1))
        projection.bias.copy_(torch.tensor([0.25, -0.5]))
    adapted = adapt_pretrained_patch_projection(projection, in_channels=12)
    rgb = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])

    torch.testing.assert_close(adapted(rgb.repeat(1, 4, 1, 1)), projection(rgb))
    assert adapted.in_channels == 12


def test_swin_v2_tiny_unet_preserves_segmentation_plane_without_downloading_weights() -> None:
    model = SwinV2TinyUNet(pretrained=False)

    logits = model(torch.zeros((1, 12, 64, 64)))

    assert logits.shape == (1, 1, 64, 64)
    with pytest.raises(ValueError, match="divisible by 32"):
        model(torch.zeros((1, 12, 62, 64)))


def test_swin_v2_small_unet_preserves_segmentation_plane_without_downloading_weights() -> None:
    model = SwinV2SmallUNet(pretrained=False)

    logits = model(torch.zeros((1, 12, 64, 64)))

    assert logits.shape == (1, 1, 64, 64)
    assert model.architecture_name == "Swin V2 Small"
    assert model.pretrained_weights_name == "Swin_V2_S_Weights.IMAGENET1K_V1"


def test_b3_configuration_locks_pretrained_initialization_and_batch_contract() -> None:
    config = B3DevelopmentConfig(
        manifest_path=Path("manifest.json"),
        dataset_root=Path("data"),
        normalization_profile=Path("profile.json"),
        run_dir=Path("run"),
        seed=17,
    )

    assert config.pretrained is True
    assert config.effective_batch_size == 8
    with pytest.raises(ValueError, match="ImageNet-pretrained"):
        B3DevelopmentConfig(
            manifest_path=Path("manifest.json"),
            dataset_root=Path("data"),
            normalization_profile=Path("profile.json"),
            run_dir=Path("run"),
            seed=17,
            pretrained=False,
        )


def test_b3_development_configuration_loader_rejects_split_drift(tmp_path: Path) -> None:
    config = B3DevelopmentConfig(
        manifest_path=Path("manifest.json"),
        dataset_root=Path("data"),
        normalization_profile=Path("profile.json"),
        run_dir=Path("run"),
        seed=17,
    )
    payload = {
        "schema_version": 1,
        "b3_development": {
            **config.__dict__,
            "manifest_path": str(config.manifest_path),
            "dataset_root": str(config.dataset_root),
            "normalization_profile": str(config.normalization_profile),
            "run_dir": str(config.run_dir),
            "train_subject_ids": ["01", "04", "05", "06", "08", "09", "11", "13", "15", "16"],
            "validation_subject_ids": ["02", "12"],
            "fixed_test_subjects_excluded": ["03", "07", "10", "14"],
            "effective_batch_size": 8,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_b3_development_config(path) == config
    payload["b3_development"]["validation_subject_ids"] = ["02"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="development subject IDs"):
        load_b3_development_config(path)


def test_b4_configuration_locks_small_pretraining_and_development_split(tmp_path: Path) -> None:
    config = B4DevelopmentConfig(
        manifest_path=Path("manifest.json"),
        dataset_root=Path("data"),
        normalization_profile=Path("profile.json"),
        run_dir=Path("run"),
        seed=17,
    )
    payload = {
        "schema_version": 1,
        "b4_development": {
            **config.__dict__,
            "manifest_path": str(config.manifest_path),
            "dataset_root": str(config.dataset_root),
            "normalization_profile": str(config.normalization_profile),
            "run_dir": str(config.run_dir),
            "train_subject_ids": ["01", "04", "05", "06", "08", "09", "11", "13", "15", "16"],
            "validation_subject_ids": ["02", "12"],
            "fixed_test_subjects_excluded": ["03", "07", "10", "14"],
            "effective_batch_size": 8,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert config.micro_batch_size == 1
    assert config.gradient_accumulation_steps == 8
    assert config.encoder_learning_rate == pytest.approx(5e-5)
    assert load_b4_development_config(path) == config
    payload["b4_development"]["pretrained"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ImageNet-pretrained"):
        load_b4_development_config(path)


def test_b3_final_configuration_carries_only_development_selected_decisions() -> None:
    config = B3FinalTrainingConfig(
        manifest_path=Path("manifest.json"),
        dataset_root=Path("data"),
        normalization_profile=Path("profile.json"),
        run_dir=Path("run"),
        seed=17,
    )

    assert config.max_epochs == 18
    assert config.selected_probability_threshold == pytest.approx(0.35)
    assert config.pretrained is True
    with pytest.raises(ValueError, match="ImageNet-pretrained"):
        B3FinalTrainingConfig(
            manifest_path=Path("manifest.json"),
            dataset_root=Path("data"),
            normalization_profile=Path("profile.json"),
            run_dir=Path("run"),
            seed=17,
            pretrained=False,
        )


def test_b3_final_configuration_loader_rejects_initialization_drift(tmp_path: Path) -> None:
    config = B3FinalTrainingConfig(
        manifest_path=Path("manifest.json"),
        dataset_root=Path("data"),
        normalization_profile=Path("profile.json"),
        run_dir=Path("run"),
        seed=17,
    )
    payload = {
        "schema_version": 1,
        "b3_final_training": {
            **config.__dict__,
            "manifest_path": str(config.manifest_path),
            "dataset_root": str(config.dataset_root),
            "normalization_profile": str(config.normalization_profile),
            "run_dir": str(config.run_dir),
            "train_subject_ids": ["01", "02", "04", "05", "06", "08", "09", "11", "12", "13", "15", "16"],
            "validation_subject_ids": [],
            "fixed_test_subjects_excluded": ["03", "07", "10", "14"],
            "effective_batch_size": 8,
            "initialization": "ImageNet Swin_V2_T_Weights.IMAGENET1K_V1",
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_b3_final_training_config(path) == config
    payload["b3_final_training"]["initialization"] = "from_scratch"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected initialization"):
        load_b3_final_training_config(path)
