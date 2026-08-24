"""Train the primary R0 development configuration with 3D-Dice early stopping.

The script creates its experiment card and immutable config before it constructs
the model or calls ``Trainer.fit``.  It must be run on a CUDA-capable training
host for a practical 100-epoch RF-DETR Seg Small run.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shlex
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ivdseg.training import (
    DEVELOPMENT_DICE_METRIC,
    RGB_PRETRAINED_LARGE_WEIGHTS,
    RGB_PRETRAINED_WEIGHTS,
    DevelopmentTrainingConfig,
    build_development_training,
    verify_cuda_runtime,
    write_run_configuration,
)


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _write_run_card(path: Path, config: DevelopmentTrainingConfig, command: str) -> None:
    if path.exists():
        raise FileExistsError(f"experiment card already exists: {path}; use a new experiment ID or seed")
    model_name = f"RFDETRSeg{config.model_variant.title()}"
    is_primary_small = config.model_variant == "small" and config.resolution == 384
    is_capacity_only_large = config.model_variant == "large" and config.resolution == 384
    hypothesis = (
        "RGB-pretrained RF-DETR Seg Small can learn one-class IVD instance masks from 12-channel four-modality "
        "2.5D MRI input."
        if is_primary_small
        else "At R0's unchanged 384 x 384 input resolution, RGB-pretrained RF-DETR Seg Large improves development "
        "3D IVD Dice through additional decoder capacity and queries."
        if is_capacity_only_large
        else f"RGB-pretrained RF-DETR Seg {config.model_variant.title()} at {config.resolution} x {config.resolution} "
        "improves development 3D IVD Dice through its changed architecture and near-native input scale."
    )
    comparison = (
        "Primary R0 development run; no ablation factor is changed."
        if is_primary_small
        else "Capacity-only ablation against completed `R0-modal-17`: Seg Large replaces Seg Small while resolution, "
        "split, seed, optimizer settings, effective batch size, augmentation, and selection rule remain unchanged."
        if is_capacity_only_large
        else "Combined architecture-and-resolution ablation against completed `R0-modal-17`: the Seg variant and "
        "model resize change, while split, seed, optimizer settings, effective batch size, augmentation, and selection "
        "rule remain unchanged."
    )
    content = f"""# {config.experiment_id} - RF-DETR {config.model_variant.title()} 12-channel 2.5D IVD segmentation - Seed {config.seed}

**Status:** Running  
**Started:** {_timestamp()}  
**Finished:**

## Purpose

- **Hypothesis:** {hypothesis}
- **Comparison:** {comparison}
- **Plan reference:** `docs/plans/rf-detr-2-5d-ivd-segmentation.md`

## Immutable Run Definition

- **Code revision:** Workspace has no Git repository; exact tracked implementation files and `uv.lock` are recorded with this run.
- **Environment / package lock:** Python 3.12 via uv; `rfdetr==1.9.1`; `uv.lock`.
- **Dataset manifest and split:** `artifacts/manifests/ivdm3seg-v1.json`; train `01,04,05,06,08,09,11,13,15,16`; development validation `02,12`; fixed test subjects `03,07,10,14` excluded.
- **Model and input:** `{model_name}(num_classes=1, num_channels=12, resolution={config.resolution})`; RGB pretrained patch embedding tiled/scaled to 12 channels after weight load.
- **Seed:** {config.seed}
- **Training command:** `{command}`
- **Selection procedure:** Up to {config.max_epochs} epochs; effective batch size {config.effective_batch_size}; early stop and best checkpoint monitor macro reconstructed development 3D Dice at fixed detection score threshold {config.development_score_threshold:.2f}. The final threshold grid is selected only after training from the best checkpoint.

## Artifacts

- **Configuration:** `{config.run_dir / 'config.json'}`
- **Logs:** `{config.run_dir / 'logs'}`
- **Checkpoint:** `{config.run_dir / 'checkpoints'}`
- **Predictions:** Development reconstructed-metric records at `{config.run_dir / 'metrics/development-3d'}`; final development/test predictions are produced in the later threshold-selection and evaluation stages.
- **Qualitative outputs:** Not generated during this training stage.

## Results

| Subject | Dice | ASD (mm) | Localization distance (mm) | HD95 (mm) | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 03 | | | | | Fixed holdout; not evaluated in this development run. |
| 07 | | | | | Fixed holdout; not evaluated in this development run. |
| 10 | | | | | Fixed holdout; not evaluated in this development run. |
| 14 | | | | | Fixed holdout; not evaluated in this development run. |
| Mean | | | | | |

## Conclusion

- **Outcome:** Running.
- **Failure or deviation from plan:**
- **Decision / next action:** After completion, use the saved best development checkpoint to select the score threshold on `02`/`12` before final retraining.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _finish_run_card(path: Path, *, status: str, conclusion: str) -> None:
    content = path.read_text(encoding="utf-8")
    content = content.replace("**Status:** Running", f"**Status:** {status}")
    content = content.replace("**Finished:**", f"**Finished:** {_timestamp()}")
    content = content.replace("- **Outcome:** Running.", f"- **Outcome:** {conclusion}")
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="R0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model-variant", choices=("small", "large"), default="small")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/ivdm3seg-v1.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("IVDM3Seg"))
    parser.add_argument(
        "--normalization-profile",
        type=Path,
        default=Path("artifacts/normalization/ivdm3seg-development-train-v1.json"),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--pretrain-weights", type=Path)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--experiment-card-dir",
        type=Path,
        default=Path("docs/research/experiments"),
        help="Directory in which to create and update the experiment card.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or Path("artifacts/runs") / f"{args.experiment_id}-{args.seed}"
    default_pretrain_weights = (
        RGB_PRETRAINED_LARGE_WEIGHTS if args.model_variant == "large" else RGB_PRETRAINED_WEIGHTS
    )
    config = DevelopmentTrainingConfig(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        normalization_profile=args.normalization_profile,
        run_dir=run_dir,
        experiment_id=args.experiment_id,
        seed=args.seed,
        model_variant=args.model_variant,
        resolution=args.resolution,
        accelerator=args.accelerator,
        num_workers=args.num_workers,
        pretrain_weights=args.pretrain_weights or default_pretrain_weights,
    )
    card_path = args.experiment_card_dir / f"{config.experiment_id}-{config.seed}.md"
    command = shlex.join([sys.executable, *sys.argv])
    _write_run_card(card_path, config, command)
    try:
        verify_cuda_runtime()
        trainer, module, data_module, best_checkpoint = build_development_training(config)
        config_path = write_run_configuration(config, module.model_config, module.train_config)
        trainer.fit(module, datamodule=data_module, ckpt_path=str(args.resume_from) if args.resume_from else None)
        best_score = best_checkpoint.best_model_score
        best_score_text = "unavailable" if best_score is None else f"{float(best_score):.6f}"
        _finish_run_card(
            card_path,
            status="Complete",
            conclusion=(
                f"Training completed. Best {DEVELOPMENT_DICE_METRIC}={best_score_text}; "
                f"configuration is saved at `{config_path}` and the best checkpoint is "
                f"`{best_checkpoint.best_model_path}`."
            ),
        )
    except BaseException as error:
        _finish_run_card(card_path, status="Failed", conclusion=f"Training stopped: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
