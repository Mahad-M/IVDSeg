"""Train B1's residual U-Net on the development split only."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shlex
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ivdseg.unet_training import (
    B1_DEVELOPMENT_DICE_METRIC,
    B1DevelopmentConfig,
    build_b1_development_training,
    write_b1_development_configuration,
)
from ivdseg.training import verify_cuda_runtime


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _write_run_card(path: Path, config: B1DevelopmentConfig, command: str) -> None:
    if path.exists():
        raise FileExistsError(f"experiment card already exists: {path}; use a new experiment ID or seed")
    content = f"""# {config.experiment_id} - 12-channel ResUNet-34 semantic baseline - Seed {config.seed}

**Status:** Running  
**Started:** {_timestamp()}  
**Finished:**

## Purpose

- **Hypothesis:** A residual U-Net semantic segmenter trained on the same 12-channel 2.5D MRI tensors provides a strong non-detection baseline for IVD segmentation.
- **Comparison:** B1 baseline against RF-DETR R0. The subject split, normalization, input tensor, resize, seed, synchronized augmentation, effective batch size, and development-volume selection rule are held fixed; architecture, binary semantic loss, and optimizer differ.
- **Plan reference:** `docs/plans/rf-detr-2-5d-ivd-segmentation.md`

## Immutable Run Definition

- **Code revision:** Workspace has no Git repository; exact implementation files and `uv.lock` are retained with this run configuration.
- **Environment / package lock:** Python 3.12 via uv; `torch==2.8.0+cu126`; `torchvision==0.23.0`; `uv.lock`.
- **Dataset manifest and split:** Train `01,04,05,06,08,09,11,13,15,16`; development validation `02,12`; fixed test subjects `03,07,10,14` excluded.
- **Model and initialization:** From-scratch `ResUNet34(in_channels=12, out_channels=1, base_channels={config.base_channels})`, a ResNet-34-style residual encoder with stage depths `(3,4,6,3)`, GroupNorm, and U-Net decoder. No ImageNet or task pretraining is used.
- **Input and labels:** The existing modality-first 12-channel 2.5D tensor is resized to `{config.resolution} x {config.resolution}`. Unlike instance-target generation, the binary target uses the complete authoritative semantic IVD label, including components below the detector-specific filtering threshold.
- **Loss and optimizer:** Equal-weight BCE-with-logits plus soft Dice loss; AdamW `lr={config.learning_rate}`, weight decay `{config.weight_decay}`, cosine schedule.
- **Seed:** `{config.seed}`.
- **Training command:** `{command}`.
- **Selection procedure:** Up to `{config.max_epochs}` epochs; effective batch size `{config.effective_batch_size}`; best checkpoint and early stopping monitor macro reconstructed development 3D Dice on subjects `02,12` after sigmoid threshold `>= {config.probability_threshold:.2f}`. No fixed holdout data is loaded.

## Artifacts

- **Configuration:** `{config.run_dir / 'config.json'}`.
- **Logs:** `{config.run_dir / 'logs'}` and Modal `[ivdseg-progress]` output every 10 training batches.
- **Checkpoint:** `{config.run_dir / 'checkpoints'}`.
- **Predictions:** Per-epoch development-volume metric records at `{config.run_dir / 'metrics/development-3d'}`; no test prediction is produced.
- **Qualitative outputs:** None during development training.

## Results

| Subject | Dice | Notes |
| --- | ---: | --- |
| 02 | | Development-only subject. |
| 12 | | Development-only subject. |
| Macro mean | | |

## Conclusion

- **Outcome:** Running.
- **Failure or deviation from plan:**
- **Decision / next action:** Select a sigmoid threshold from the best development checkpoint using only `02` and `12`; final retraining and fixed-holdout evaluation are prohibited until that decision is recorded.
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
    parser.add_argument("--experiment-id", default="B1-resunet34")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/ivdm3seg-v1.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("IVDM3Seg"))
    parser.add_argument(
        "--normalization-profile",
        type=Path,
        default=Path("artifacts/normalization/ivdm3seg-development-train-v1.json"),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--experiment-card-dir", type=Path, default=Path("docs/research/experiments"))
    args = parser.parse_args()

    run_dir = args.run_dir or Path("artifacts/runs") / f"{args.experiment_id}-{args.seed}"
    config = B1DevelopmentConfig(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        normalization_profile=args.normalization_profile,
        run_dir=run_dir,
        experiment_id=args.experiment_id,
        seed=args.seed,
        accelerator=args.accelerator,
        num_workers=args.num_workers,
    )
    card_path = args.experiment_card_dir / f"{config.experiment_id}-{config.seed}.md"
    command = shlex.join([sys.executable, *sys.argv])
    _write_run_card(card_path, config, command)
    try:
        verify_cuda_runtime()
        trainer, module, data_module, checkpoint = build_b1_development_training(config)
        config_path = write_b1_development_configuration(config, module.model)
        trainer.fit(module, datamodule=data_module)
        score = checkpoint.best_model_score
        score_text = "unavailable" if score is None else f"{float(score):.6f}"
        _finish_run_card(
            card_path,
            status="Complete",
            conclusion=(
                f"Development training completed. Best {B1_DEVELOPMENT_DICE_METRIC}={score_text}; configuration is "
                f"`{config_path}` and best checkpoint is `{checkpoint.best_model_path}`."
            ),
        )
    except BaseException as error:
        _finish_run_card(path=card_path, status="Failed", conclusion=f"Training stopped: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
