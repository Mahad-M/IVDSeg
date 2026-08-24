"""Select the RF-DETR detection-score threshold from a completed development run.

This evaluates only the source run's held-back development subjects. It never
loads the fixed test subjects and writes the full 0.05--0.95 grid before
recording the selected score in a dedicated experiment card.
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
    DevelopmentThresholdSelection,
    load_development_training_config,
    select_development_threshold,
    verify_cuda_runtime,
)


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _write_run_card(
    path: Path,
    *,
    experiment_id: str,
    source_experiment_id: str,
    seed: int,
    source_run_config: Path,
    checkpoint: Path,
    output: Path,
    command: str,
) -> None:
    if path.exists():
        raise FileExistsError(f"experiment card already exists: {path}; use a new experiment ID")
    content = f"""# {experiment_id} - Development threshold selection - Seed {seed}

**Status:** Running  
**Started:** {_timestamp()}  
**Finished:**

## Purpose

- **Hypothesis:** Selecting the detection score on the completed development partition improves the semantic 3D operating point without using test data.
- **Comparison:** Post-training operating-point selection for `{source_experiment_id}-{seed}`; checkpoint, data, model, and mask binarization are fixed.
- **Plan reference:** `docs/plans/rf-detr-2-5d-ivd-segmentation.md`

## Immutable Run Definition

- **Code revision:** Workspace has no Git repository; exact implementation files and `uv.lock` are recorded with this run.
- **Environment / package lock:** Python 3.12 via uv; `rfdetr==1.9.1`; `uv.lock`.
- **Dataset manifest and split:** Restored from `{source_run_config}`; development subjects `02,12` only. Fixed test subjects `03,07,10,14` excluded.
- **Model and input:** Restored from source checkpoint `{checkpoint}`; no architecture or preprocessing factor changes.
- **Seed:** {seed}
- **Training command:** `{command}`
- **Selection procedure:** Reconstruct development predictions once for each detection score in `0.05,0.10,...,0.95`; choose maximum macro 3D Dice, breaking exact ties toward the lower score.

## Artifacts

- **Configuration:** Source immutable configuration `{source_run_config}`
- **Logs:** Standard output contains plain `[ivdseg-threshold]` progress lines.
- **Checkpoint:** `{checkpoint}`
- **Predictions:** The complete threshold grid and per-subject Dice records are written to `{output}`. No test prediction is produced.
- **Qualitative outputs:** Not generated during this selection stage.

## Results

| Subject | Dice | ASD (mm) | Localization distance (mm) | HD95 (mm) | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 03 | | | | | Fixed holdout; not evaluated. |
| 07 | | | | | Fixed holdout; not evaluated. |
| 10 | | | | | Fixed holdout; not evaluated. |
| 14 | | | | | Fixed holdout; not evaluated. |
| Mean | | | | | |

## Conclusion

- **Outcome:** Running.
- **Failure or deviation from plan:**
- **Decision / next action:** Retain the fixed holdout untouched. Use the selected score for final retraining and later test reconstruction only if this development model is accepted over its comparison run.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _finish_run_card(path: Path, selection: DevelopmentThresholdSelection) -> None:
    content = path.read_text(encoding="utf-8")
    content = content.replace("**Status:** Running", "**Status:** Complete")
    content = content.replace("**Finished:**", f"**Finished:** {_timestamp()}")
    outcome = (
        "Threshold selection completed. Selected "
        f"`{selection.selected.score_threshold:.2f}` with development macro 3D Dice "
        f"`{selection.selected.macro_dice:.6f}`."
    )
    content = content.replace("- **Outcome:** Running.", f"- **Outcome:** {outcome}")
    path.write_text(content, encoding="utf-8")


def _fail_run_card(path: Path, error: BaseException) -> None:
    content = path.read_text(encoding="utf-8")
    content = content.replace("**Status:** Running", "**Status:** Failed")
    content = content.replace("**Finished:**", f"**Finished:** {_timestamp()}")
    content = content.replace("- **Outcome:** Running.", f"- **Outcome:** Threshold selection stopped: {type(error).__name__}: {error}")
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experiment-id", default="R0-modal-threshold")
    parser.add_argument("--experiment-card-dir", type=Path, default=Path("docs/research/experiments"))
    args = parser.parse_args()

    config = load_development_training_config(args.source_run_config)
    output = args.output or config.run_dir / "metrics" / "development-threshold-selection.json"
    if output.exists():
        raise FileExistsError(f"threshold-selection output already exists: {output}")
    card_path = args.experiment_card_dir / f"{args.experiment_id}-{config.seed}.md"
    command = shlex.join([sys.executable, *sys.argv])
    _write_run_card(
        card_path,
        experiment_id=args.experiment_id,
        source_experiment_id=config.experiment_id,
        seed=config.seed,
        source_run_config=args.source_run_config,
        checkpoint=args.checkpoint,
        output=output,
        command=command,
    )
    try:
        verify_cuda_runtime()
        selection = select_development_threshold(config, checkpoint_path=args.checkpoint, output_path=output)
        _finish_run_card(card_path, selection)
    except BaseException as error:
        _fail_run_card(card_path, error)
        raise


if __name__ == "__main__":
    main()
