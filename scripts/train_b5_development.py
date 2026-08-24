"""Train B5's small-budget four-modality 3D pretrained Swin UNETR on development data only."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shlex
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ivdseg.swinunetr_3d import (
    B5_DEVELOPMENT_DICE_METRIC,
    B5_FALLBACK_ROI_SIZE,
    B5_PRIMARY_ROI_SIZE,
    B5DevelopmentConfig,
    build_b5_development_training,
    load_b5_development_config,
    write_b5_development_configuration,
)
from ivdseg.training import verify_cuda_runtime


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _roi_size(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace("x", ",").split(",")
    try:
        roi = tuple(int(part.strip()) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("ROI must have three integer dimensions, e.g. 32x256x256") from error
    if len(roi) != 3:
        raise argparse.ArgumentTypeError("ROI must have three dimensions")
    return roi  # validated again by B5DevelopmentConfig


def _write_run_card(path: Path, config: B5DevelopmentConfig, command: str, *, resume_from: Path | None) -> None:
    if path.exists():
        if resume_from is None:
            raise FileExistsError(f"experiment card already exists: {path}; use a new experiment ID or seed")
        content = path.read_text(encoding="utf-8")
        expected_heading = f"# {config.experiment_id} - Four-modality 3D pretrained Swin UNETR - Seed {config.seed}"
        if not content.startswith(expected_heading + "\n"):
            raise FileExistsError(f"experiment card has a different run definition: {path}")
        if "**Status:** Complete" in content:
            raise FileExistsError(f"completed B5 card may not be resumed: {path}")
        content = content.replace("**Status:** Failed", "**Status:** Running").replace(
            "**Status:** Interrupted / resumable", "**Status:** Running"
        )
        resume_record = f"- **Resume record:** {_timestamp()} resumed from `{resume_from}` after an interrupted remote client.\n"
        if resume_record not in content:
            content = content.replace("- **Decision / next action:**", resume_record + "- **Decision / next action:**", 1)
        path.write_text(content, encoding="utf-8")
        return
    content = f"""# {config.experiment_id} - Four-modality 3D pretrained Swin UNETR - Seed {config.seed}

**Status:** Running  
**Started:** {_timestamp()}  
**Finished:**

## Purpose

- **Hypothesis:** A pretrained 3D Swin UNETR can improve IVD semantic segmentation by combining all four native NIfTI modalities with genuine through-plane context.
- **Comparison:** B5 is a development-only, deliberately finite alternative to B2's stopped default nnU-Net schedule. It does not reuse B3 weights or B3's 2.5D input.
- **Plan reference:** `docs/plans/rf-detr-2-5d-ivd-segmentation.md`

## Immutable Run Definition

- **Dataset and split:** Canonical NIfTI only; train `01,04,05,06,08,09,11,13,15,16`, development validation `02,12`, fixed holdout `03,07,10,14` excluded before data loading. Legacy PNGs are not read.
- **Model:** MONAI `SwinUNETR(in_channels=4, out_channels=1, feature_size=48, use_checkpoint=True)`, approximately 62.2M parameters. Public one-channel self-supervised encoder weights are loaded into `swinViT`; the patch-projection kernel is repeated across `fat,inn,opp,water` and divided by four. The segmentation decoder remains newly initialized.
- **Patch and inference geometry:** Training ROI `{config.roi_size[0]} x {config.roi_size[1]} x {config.roi_size[2]}`; native-grid sliding-window inference with overlap `{config.sliding_window_overlap}` validates each full subject volume.
- **Budget:** At most `{config.max_epochs}` epochs x `{config.patches_per_epoch}` deterministic foreground-biased patches (batch one, BF16 on the L4). Early stopping may finish earlier only after `{config.early_stopping_patience}` non-improving development epochs.
- **Augmentation:** Modality-wise intensity gain/bias and foreground-biased crops only; no anatomical-axis flips.
- **Loss and optimizer:** Equal BCE-with-logits plus soft Dice; AdamW with SSL-encoder LR `{config.encoder_learning_rate}`, decoder LR `{config.decoder_learning_rate}`, weight decay `{config.weight_decay}`, cosine schedule, gradient clipping `0.1`.
- **Selection:** Checkpoint and early stopping monitor macro native subject-volume Dice on `02,12` at the pre-specified sigmoid cutoff `>= {config.probability_threshold:.2f}`. A later development-only `0.05`–`0.95` grid selects a cutoff. B5 requires at least `{0.003:.3f}` improvement over B3's `{0.931567:.6f}` (>= `{0.934567:.6f}`) before final retraining or holdout access.
- **Training command:** `{command}`.

## Artifacts

- **Configuration:** `{config.run_dir / 'config.json'}`.
- **Logs:** `{config.run_dir / 'logs'}` and visible `[ivdseg-progress]` every `{config.progress_report_interval_batches}` patches.
- **Pretrained encoder cache:** `{config.pretrained_weights_path}`; downloaded from MONAI's public release only if absent.
- **Checkpoint:** `{config.run_dir / 'checkpoints'}`.
- **Per-epoch development metrics:** `{config.run_dir / 'metrics/development-3d'}`; no fixed-test prediction is produced.

## Results

| Subject | Dice | Notes |
| --- | ---: | --- |
| 02 | | Development-only subject. |
| 12 | | Development-only subject. |
| Macro mean | | Fixed cutoff during training; threshold grid follows separately. |

## Conclusion

- **Outcome:** Running.
- **Failure or deviation from plan:**
- **Decision / next action:** Complete the development-only threshold grid on the best checkpoint. Do not final-retrain or access the fixed holdout unless the predeclared B3 improvement criterion is met.
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
    parser.add_argument("--experiment-id", default="B5-swinunetr-3d-pretrained")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/ivdm3seg-v1.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("IVDM3Seg"))
    parser.add_argument(
        "--normalization-profile",
        type=Path,
        default=Path("artifacts/normalization/ivdm3seg-development-train-v1.json"),
    )
    parser.add_argument("--pretrained-weights", type=Path, default=Path("artifacts/model-weights/ssl_pretrained_weights.pth"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--roi-size", type=_roi_size, default=B5_PRIMARY_ROI_SIZE)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--experiment-card-dir", type=Path, default=Path("docs/research/experiments"))
    args = parser.parse_args()

    run_dir = args.run_dir or Path("artifacts/runs") / f"{args.experiment_id}-{args.seed}"
    config = B5DevelopmentConfig(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        normalization_profile=args.normalization_profile,
        run_dir=run_dir,
        pretrained_weights_path=args.pretrained_weights,
        experiment_id=args.experiment_id,
        seed=args.seed,
        roi_size=args.roi_size,
        accelerator=args.accelerator,
    )
    card_path = args.experiment_card_dir / f"{config.experiment_id}-{config.seed}.md"
    command = shlex.join([sys.executable, *sys.argv])
    _write_run_card(card_path, config, command, resume_from=args.resume_from)
    try:
        verify_cuda_runtime()
        trainer, module, data_module, checkpoint = build_b5_development_training(config)
        if args.resume_from is None:
            config_path = write_b5_development_configuration(config, module)
        else:
            if not args.resume_from.is_file():
                raise FileNotFoundError(f"B5 resume checkpoint does not exist: {args.resume_from}")
            existing_config = load_b5_development_config(config.run_dir / "config.json")
            if existing_config != config:
                raise ValueError("B5 resume settings do not exactly match the immutable original configuration")
            config_path = config.run_dir / "config.json"
        trainer.fit(module, datamodule=data_module, ckpt_path=str(args.resume_from) if args.resume_from else None)
        score = checkpoint.best_model_score
        score_text = "unavailable" if score is None else f"{float(score):.6f}"
        _finish_run_card(
            card_path,
            status="Complete",
            conclusion=(
                f"Development training completed. Best {B5_DEVELOPMENT_DICE_METRIC}={score_text}; configuration is "
                f"`{config_path}` and best checkpoint is `{checkpoint.best_model_path}`."
            ),
        )
    except BaseException as error:
        _finish_run_card(path=card_path, status="Failed", conclusion=f"Training stopped: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
