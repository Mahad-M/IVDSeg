"""Fresh train-only B3 final retraining from ImageNet Swin V2 Tiny weights."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shlex
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ivdseg.training import verify_cuda_runtime
from ivdseg.unet_training import (
    B3FinalTrainingConfig,
    build_b3_final_training,
    write_b3_final_training_configuration,
)


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _write_run_card(path: Path, config: B3FinalTrainingConfig, command: str) -> None:
    if path.exists():
        raise FileExistsError(f"experiment card already exists: {path}; use a new experiment ID or seed")
    content = f"""# {config.experiment_id} - Final pretrained Swin V2 Tiny U-Net retraining - Seed {config.seed}

**Status:** Running  
**Started:** {_timestamp()}  
**Finished:**

## Purpose

- **Hypothesis:** A fresh all-permitted-subject B3 run is the sole pretrained Swin U-Net checkpoint eligible for fixed-holdout evaluation after development selected its duration and sigmoid cutoff.
- **Comparison:** Final retraining only. It must restart from ImageNet Swin V2 Tiny weights, rather than warm-starting the development checkpoint that selected duration and threshold.
- **Plan reference:** `docs/plans/rf-detr-2-5d-ivd-segmentation.md`

## Immutable Run Definition

- **Dataset and split:** Train `01,02,04,05,06,08,09,11,12,13,15,16`; no validation set; fixed holdout `03,07,10,14` excluded.
- **Model and initialization:** Pretrained `SwinV2TinyUNet(in_channels=12, out_channels=1)`. The official ImageNet Swin V2 Tiny encoder is adapted from RGB to four 3-channel modality groups by replicated-and-scaled patch-projection weights; the U-Net decoder is new.
- **Input and labels:** Full semantic binary labels with modality-first 12-channel 2.5D tensors resized to `{config.resolution} x {config.resolution}`.
- **Loss and optimizer:** Equal BCE-with-logits plus soft Dice; AdamW encoder lr `{config.encoder_learning_rate}`, decoder lr `{config.decoder_learning_rate}`, weight decay `{config.weight_decay}`, cosine schedule.
- **Selected decisions:** Exactly `{config.max_epochs}` epochs (development best epoch 17 means 18 completed epochs) and later sigmoid threshold `>= {config.selected_probability_threshold:.2f}`. No validation or threshold selection occurs here.
- **Training command:** `{command}`.

## Artifacts

- **Configuration / logs / checkpoint:** `{config.run_dir}`; final checkpoint `{config.run_dir / 'checkpoints/last.ckpt'}`.
- **Pretrained encoder cache:** `ivdseg-runs:/artifacts/model-weights/hub/checkpoints/swin_v2_t-b137f0e2.pth`.
- **Predictions / qualitative outputs:** None during final training.

## Conclusion

- **Outcome:** Running; no holdout data is accessed during final training.
- **Failure or deviation from plan:**
- **Decision / next action:** On normal completion, evaluate this new checkpoint exactly once on `03,07,10,14` with sigmoid `>= {config.selected_probability_threshold:.2f}` and native-geometry metrics.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _finish_run_card(path: Path, *, status: str, conclusion: str) -> None:
    content = path.read_text(encoding="utf-8")
    content = content.replace("**Status:** Running", f"**Status:** {status}")
    content = content.replace("**Finished:**", f"**Finished:** {_timestamp()}")
    content = content.replace("- **Outcome:** Running; no holdout data is accessed during final training.", f"- **Outcome:** {conclusion}")
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="B3-swinv2-tiny-pretrained-final")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--normalization-profile", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--experiment-card-dir", type=Path, default=Path("docs/research/experiments"))
    args = parser.parse_args()
    config = B3FinalTrainingConfig(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        normalization_profile=args.normalization_profile,
        run_dir=args.run_dir,
        experiment_id=args.experiment_id,
        seed=args.seed,
        accelerator=args.accelerator,
        num_workers=args.num_workers,
    )
    card_path = args.experiment_card_dir / f"{config.experiment_id}-{config.seed}.md"
    command = shlex.join([sys.executable, *sys.argv])
    _write_run_card(card_path, config, command)
    try:
        config_path = write_b3_final_training_configuration(config)
        verify_cuda_runtime()
        trainer, module, data_module, _checkpoint = build_b3_final_training(config)
        trainer.fit(module, datamodule=data_module)
        _finish_run_card(
            card_path,
            status="Complete",
            conclusion=(
                f"Completed all {config.max_epochs} train-only epochs and persisted `{config.run_dir / 'checkpoints/last.ckpt'}`. "
                f"Configuration is `{config_path}`; no holdout subject was loaded."
            ),
        )
    except BaseException as error:
        _finish_run_card(path=card_path, status="Failed", conclusion=f"Training stopped: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
