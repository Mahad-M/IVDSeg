"""Retrain the selected R0 seed on every non-holdout subject.

This launcher is deliberately train-only: the supplied development run chose
the epoch duration and reconstruction threshold already. It restarts from the
matching published RGB-pretrained RF-DETR checkpoint and never loads validation
or fixed-test data.
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
    FINAL_TRAIN_SUBJECT_IDS,
    RGB_PRETRAINED_LARGE_WEIGHTS,
    RGB_PRETRAINED_WEIGHTS,
    FinalTrainingConfig,
    build_final_training,
    verify_cuda_runtime,
    write_final_training_configuration,
)


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _write_run_card(path: Path, config: FinalTrainingConfig, command: str) -> None:
    if path.exists():
        raise FileExistsError(f"experiment card already exists: {path}; use a new experiment ID or seed")
    train_ids = ",".join(FINAL_TRAIN_SUBJECT_IDS)
    model_name = f"RFDETRSeg{config.model_variant.title()}"
    content = f"""# {config.experiment_id} - RF-DETR {config.model_variant.title()} final retraining - Seed {config.seed}

**Status:** Running  
**Started:** {_timestamp()}  
**Finished:**

## Purpose

- **Hypothesis:** The selected seed can use all 12 permitted training subjects to produce the final {model_name} model without data leakage.
- **Comparison:** Final retraining after development run `{config.source_development_run}` and its threshold-selection stage; this is not a new model-selection experiment.
- **Plan reference:** `docs/plans/rf-detr-2-5d-ivd-segmentation.md`

## Immutable Run Definition

- **Code revision:** Workspace has no Git repository; exact implementation files and `uv.lock` are recorded with this run.
- **Environment / package lock:** Python 3.12 via uv; `rfdetr==1.9.1`; `uv.lock`.
- **Dataset manifest and split:** Train `{train_ids}`; no validation set; fixed test subjects `03,07,10,14` excluded.
- **Normalization:** The all-12-subject training-pool profile `{config.normalization_profile}` is required to match exactly the optimization subjects.
- **Model and input:** `{model_name}(num_classes=1, num_channels=12, resolution=384)`; initialized from the published RGB-pretrained checkpoint `{config.pretrain_weights}`, then its patch embedding is tiled/scaled to 12 channels. The development checkpoint `{config.source_development_checkpoint}` is not used to initialize this model.
- **Seed:** {config.seed}
- **Training command:** `{command}`
- **Selected development decisions:** `{config.source_development_run}` selected {config.max_epochs} completed epochs; score threshold `{config.selected_score_threshold:.2f}` is recorded for later test reconstruction, not optimized during this train-only run.
- **Batching and determinism:** Effective batch size {config.effective_batch_size}; Lightning `deterministic="warn"` keeps seeded supported operations fixed while allowing CUDA `grid_sampler` backward.

## Artifacts

- **Configuration:** `{config.run_dir / 'config.json'}`
- **Logs:** `{config.run_dir / 'logs'}` and Modal plain `[ivdseg-progress]` output every 10 training batches.
- **Checkpoint:** `{config.run_dir / 'checkpoints/last.ckpt'}`
- **Predictions:** None in this training stage. Test reconstruction is a later, separate stage.
- **Qualitative outputs:** Not generated during this training stage.

## Results

| Subject | Dice | ASD (mm) | Localization distance (mm) | HD95 (mm) | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 03 | | | | | Fixed holdout; not loaded or evaluated. |
| 07 | | | | | Fixed holdout; not loaded or evaluated. |
| 10 | | | | | Fixed holdout; not loaded or evaluated. |
| 14 | | | | | Fixed holdout; not loaded or evaluated. |
| Mean | | | | | |

## Conclusion

- **Outcome:** Running.
- **Failure or deviation from plan:**
- **Decision / next action:** Retain the final checkpoint. Reconstruct one native-geometry NIfTI prediction for each fixed holdout subject only in the planned final evaluation stage, using score threshold `{config.selected_score_threshold:.2f}`.
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
    parser.add_argument("--experiment-id", default="R0-final")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model-variant", choices=("small", "large"), default="small")
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--selected-score-threshold", type=float, default=0.35)
    parser.add_argument("--source-development-run", default="R0-modal-17")
    parser.add_argument("--source-development-checkpoint", default="best-epoch=006.ckpt")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/ivdm3seg-v1.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("IVDM3Seg"))
    parser.add_argument(
        "--normalization-profile",
        type=Path,
        default=Path("artifacts/normalization/ivdm3seg-training-pool-v1.json"),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--pretrain-weights", type=Path)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
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
    config = FinalTrainingConfig(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        normalization_profile=args.normalization_profile,
        run_dir=run_dir,
        experiment_id=args.experiment_id,
        seed=args.seed,
        model_variant=args.model_variant,
        max_epochs=args.epochs,
        selected_score_threshold=args.selected_score_threshold,
        source_development_run=args.source_development_run,
        source_development_checkpoint=args.source_development_checkpoint,
        accelerator=args.accelerator,
        num_workers=args.num_workers,
        pretrain_weights=args.pretrain_weights or default_pretrain_weights,
    )
    card_path = args.experiment_card_dir / f"{config.experiment_id}-{config.seed}.md"
    command = shlex.join([sys.executable, *sys.argv])
    _write_run_card(card_path, config, command)
    try:
        verify_cuda_runtime()
        trainer, module, data_module, _last_checkpoint = build_final_training(config)
        config_path = write_final_training_configuration(config, module.model_config, module.train_config)
        trainer.fit(module, datamodule=data_module)
        _finish_run_card(
            card_path,
            status="Complete",
            conclusion=(
                f"Final training completed after {config.max_epochs} epochs; configuration is saved at "
                f"`{config_path}` and the final checkpoint is `{config.run_dir / 'checkpoints/last.ckpt'}`."
            ),
        )
    except BaseException as error:
        _finish_run_card(path=card_path, status="Failed", conclusion=f"Training stopped: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
