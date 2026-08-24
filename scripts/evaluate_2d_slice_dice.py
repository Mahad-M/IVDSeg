"""Measure 2D slice Dice from the fixed final evaluation's saved predictions.

This is a post-hoc reporting diagnostic only: it does not load a model, rerun
inference, alter the saved NIfTI predictions, or change the primary 3D metric.
It reports both target-foreground-only mean Dice and the potentially inflated
all-slice mean for transparent comparison with earlier 2D work.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ivdseg.evaluation import SliceDiceEvaluationSummary, evaluate_saved_final_slice_dice
from ivdseg.manifest import FIXED_TEST_SUBJECTS
from ivdseg.training import load_final_training_config
from ivdseg.unet_training import load_b1_final_training_config, load_b3_final_training_config


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _write_config(
    output_dir: Path,
    *,
    experiment_id: str,
    seed: int,
    source_run_config: Path,
    prediction_dir: Path,
) -> Path:
    path = output_dir / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": experiment_id,
        "seed": seed,
        "source_final_run_config": str(source_run_config),
        "source_prediction_dir": str(prediction_dir),
        "fixed_test_subjects": list(FIXED_TEST_SUBJECTS),
        "metric": {
            "plane": "canonical RAS axis 0 (the coronal 2.5D model input plane)",
            "foreground_only_mean_dice": "mean binary Dice over target-foreground slices only",
            "all_slice_mean_dice": "mean binary Dice over all slices, including true-negative empty slices",
        },
        "inference": "none; evaluates the immutable saved predictions from final evaluation",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_run_card(
    path: Path,
    *,
    experiment_id: str,
    seed: int,
    source_run_config: Path,
    prediction_dir: Path,
    output_dir: Path,
    command: str,
) -> None:
    if path.exists():
        raise FileExistsError(f"experiment card already exists: {path}; use a new experiment ID")
    content = f"""# {experiment_id} - Final saved-prediction 2D slice-Dice diagnostic - Seed {seed}

**Status:** Running  
**Started:** {_timestamp()}  
**Finished:**

## Purpose

- **Hypothesis:** A foreground-only 2D Dice reveals segmentation quality without the inflation caused by target-empty image slices, while the all-slice version makes any comparison with earlier 2D reporting explicit.
- **Comparison:** Reporting diagnostic only. It reads immutable native predictions produced by final evaluation and never reruns model inference.
- **Plan reference:** `docs/plans/rf-detr-2-5d-ivd-segmentation.md`

## Immutable Run Definition

- **Source final configuration:** `{source_run_config}`
- **Source predictions:** `{prediction_dir}`
- **Dataset and split:** Fixed test subjects `03,07,10,14` only. Saved predictions and source labels are canonicalized to the same RAS grid before per-plane measurement.
- **Plane and metrics:** Canonical RAS axis 0, the coronal 2.5D model input plane. Report foreground-only mean binary Dice over target-positive slices and all-slice mean binary Dice, where an empty target and empty prediction score 1.0.
- **Evaluation command:** `{command}`
- **Inference / postprocessing:** None. The supplied final-evaluation predictions are read unchanged.

## Artifacts

- **Configuration:** `{output_dir / 'config.json'}`
- **Metrics:** `{output_dir / 'metrics/2d-slice-dice.json'}`

## Results

| Subject | Foreground-only 2D Dice | All-slice 2D Dice | Foreground slices | Empty target slices | Empty true-negative slices |
| --- | ---: | ---: | ---: | ---: | ---: |
<!-- RESULTS -->

## Conclusion

- **Outcome:** Running.
- **Failure or deviation from plan:**
- **Decision / next action:** Retain 3D subject-volume Dice as the primary metric; use this diagnostic only with its explicit slice inclusion rule.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _metric(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.6f}"


def _rows(summary: SliceDiceEvaluationSummary) -> str:
    rows = [
        "| "
        f"{subject.subject_id} | {_metric(subject.foreground_only_mean_dice)} | "
        f"{subject.all_slice_mean_dice:.6f} | {subject.foreground_slice_count} | "
        f"{subject.empty_target_slice_count} | {subject.empty_target_true_negative_slice_count} |"
        for subject in summary.subjects
    ]
    metrics = summary.to_dict()
    macro = metrics["macro_subject_mean"]
    pooled = metrics["pooled_slices"]
    rows.extend(
        (
            "| Macro subject mean | "
            f"{_metric(macro['foreground_only_mean_dice'])} | {_metric(macro['all_slice_mean_dice'])} | — | — | — |",
            "| Pooled slices | "
            f"{_metric(pooled['foreground_only_mean_dice'])} | {_metric(pooled['all_slice_mean_dice'])} | "
            f"{pooled['foreground_slice_count']} | {pooled['empty_target_slice_count']} | "
            f"{pooled['empty_target_true_negative_slice_count']} |",
        )
    )
    return "\n".join(rows)


def _finish_run_card(path: Path, *, status: str, conclusion: str, rows: str | None = None) -> None:
    content = path.read_text(encoding="utf-8")
    content = content.replace("**Status:** Running", f"**Status:** {status}")
    content = content.replace("**Finished:**", f"**Finished:** {_timestamp()}")
    content = content.replace("- **Outcome:** Running.", f"- **Outcome:** {conclusion}")
    if rows is not None:
        content = content.replace("<!-- RESULTS -->", rows)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-config", type=Path, required=True)
    parser.add_argument(
        "--source-config-kind",
        choices=("rf-detr", "b1", "b3"),
        default="rf-detr",
        help="Immutable source-training configuration format.",
    )
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", default="R0-final-2d-slice-dice")
    parser.add_argument("--experiment-card-dir", type=Path, default=Path("docs/research/experiments"))
    args = parser.parse_args()

    if args.source_config_kind == "rf-detr":
        config = load_final_training_config(args.source_run_config)
    elif args.source_config_kind == "b1":
        config = load_b1_final_training_config(args.source_run_config)
    else:
        config = load_b3_final_training_config(args.source_run_config)
    card_path = args.experiment_card_dir / f"{args.experiment_id}-{config.seed}.md"
    command = shlex.join([sys.executable, *sys.argv])
    _write_run_card(
        card_path,
        experiment_id=args.experiment_id,
        seed=config.seed,
        source_run_config=args.source_run_config,
        prediction_dir=args.prediction_dir,
        output_dir=args.output_dir,
        command=command,
    )
    try:
        config_path = _write_config(
            args.output_dir,
            experiment_id=args.experiment_id,
            seed=config.seed,
            source_run_config=args.source_run_config,
            prediction_dir=args.prediction_dir,
        )
        summary = evaluate_saved_final_slice_dice(config, prediction_dir=args.prediction_dir)
        metrics_path = args.output_dir / "metrics" / "2d-slice-dice.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
        pooled = summary.to_dict()["pooled_slices"]
        _finish_run_card(
            card_path,
            status="Complete",
            rows=_rows(summary),
            conclusion=(
                "Saved-prediction diagnostic completed. Pooled foreground-only 2D Dice="
                f"{_metric(pooled['foreground_only_mean_dice'])}; pooled all-slice 2D Dice="
                f"{_metric(pooled['all_slice_mean_dice'])}. Configuration is `{config_path}` and metrics are "
                f"`{metrics_path}`."
            ),
        )
    except BaseException as error:
        _finish_run_card(
            card_path,
            status="Failed",
            conclusion=f"Evaluation stopped: {type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
