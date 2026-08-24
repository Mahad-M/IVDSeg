"""Reconstruct and evaluate final RF-DETR predictions on the fixed holdout.

The evaluation is a separate, immutable stage. It restores the completed final
run's definition and checkpoint, applies its already-selected score threshold,
and writes native-geometry NIfTI predictions, fixed central overlays, and
subject-level metrics without changing any training decision.
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

from ivdseg.evaluation import EvaluationSummary, evaluate_final_checkpoint
from ivdseg.manifest import FIXED_TEST_SUBJECTS
from ivdseg.training import load_final_training_config, verify_cuda_runtime


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _write_evaluation_config(
    output_dir: Path,
    *,
    source_run_config: Path,
    checkpoint: Path,
    experiment_id: str,
    seed: int,
    score_threshold: float,
) -> Path:
    path = output_dir / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": experiment_id,
        "seed": seed,
        "source_final_run_config": str(source_run_config),
        "source_checkpoint": str(checkpoint),
        "score_threshold": score_threshold,
        "fixed_test_subjects": list(FIXED_TEST_SUBJECTS),
        "postprocessing": "strict score > threshold; class-0 mask union; mask threshold 0.5; no primary filtering",
        "metrics": {
            "dice": "binary semantic 3D Dice",
            "asd_mm": "symmetric mean surface distance in world coordinates",
            "hd95_mm": "symmetric 95th-percentile surface distance in world coordinates",
            "localization_distance_mm": "world-coordinate binary-volume centroid distance",
            "component_errors": "26-connected predicted/target components without any overlap",
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_run_card(
    path: Path,
    *,
    experiment_id: str,
    seed: int,
    source_run_config: Path,
    checkpoint: Path,
    output_dir: Path,
    score_threshold: float,
    model_variant: str,
    resolution: int,
    command: str,
) -> None:
    if path.exists():
        raise FileExistsError(f"experiment card already exists: {path}; use a new experiment ID")
    content = f"""# {experiment_id} - Final fixed-holdout evaluation - Seed {seed}

**Status:** Running  
**Started:** {_timestamp()}  
**Finished:**

## Purpose

- **Hypothesis:** The final checkpoint provides a valid, reproducible semantic IVD segmentation on the untouched fixed holdout.
- **Comparison:** Evaluation only; checkpoint `{checkpoint}` and selected score threshold `{score_threshold:.2f}` are fixed before opening holdout data.
- **Plan reference:** `docs/plans/rf-detr-2-5d-ivd-segmentation.md`

## Immutable Run Definition

- **Source final configuration:** `{source_run_config}`
- **Source checkpoint:** `{checkpoint}`
- **Dataset and split:** Fixed test subjects `03,07,10,14` only; their authoritative NIfTI labels are used solely for post-inference metrics.
- **Model and preprocessing:** Restored exactly from the source final configuration: 12-channel 2.5D RF-DETR Seg {model_variant.title()} at `{resolution} x {resolution}`, all-training-subject normalization profile, and RGB-pretrained-derived final weights.
- **Operating point:** Strict class-0 detection score `> {score_threshold:.2f}`; model masks are thresholded at 0.5 and unioned per slice. No primary postprocessing or component filtering.
- **Evaluation command:** `{command}`
- **Geometry and metrics:** One binary NIfTI prediction matches each source label's native shape and affine. Report 3D Dice, symmetric ASD/HD95 in mm, physical centroid localization distance, and 26-connected component FP/FN. A distance is undefined if exactly one semantic mask is empty.
- **Qualitative outputs:** One fixed central coronal slice per subject (not selected from prediction or metric performance), target contour green and prediction contour red.

## Artifacts

- **Configuration:** `{output_dir / 'config.json'}`
- **Logs:** Modal standard output contains `[ivdseg-evaluation]` per-subject lines.
- **Checkpoint:** Source only; evaluation does not create a new model checkpoint.
- **Predictions:** `{output_dir / 'predictions'}` (`uint8`, binary NIfTI, native source-label geometry).
- **Metrics:** `{output_dir / 'metrics/test-subject-metrics.json'}`.
- **Qualitative outputs:** `{output_dir / 'overlays'}`.

## Results

| Subject | Dice | ASD (mm) | Localization distance (mm) | HD95 (mm) | Component FP / FN |
| --- | ---: | ---: | ---: | ---: | ---: |
<!-- RESULTS -->

## Conclusion

- **Outcome:** Running.
- **Failure or deviation from plan:**
- **Decision / next action:** Interpret metrics only from the saved artifacts; retain predictions and overlays for later paper evidence.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _metric_text(value: float | None, *, precision: int = 3) -> str:
    return "undefined" if value is None else f"{value:.{precision}f}"


def _result_rows(summary: EvaluationSummary) -> str:
    rows = [
        "| "
        f"{subject.subject_id} | {subject.dice:.6f} | {_metric_text(subject.asd_mm)} | "
        f"{_metric_text(subject.localization_distance_mm)} | {_metric_text(subject.hd95_mm)} | "
        f"{subject.component_false_positives} / {subject.component_false_negatives} |"
        for subject in summary.subjects
    ]
    mean = summary.to_dict()["mean"]
    rows.append(
        "| Mean | "
        f"{mean['dice']:.6f} | {_metric_text(mean['asd_mm'])} | "
        f"{_metric_text(mean['localization_distance_mm'])} | {_metric_text(mean['hd95_mm'])} | "
        f"{mean['component_false_positives']:.3f} / {mean['component_false_negatives']:.3f} |"
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", default="R0-final-eval")
    parser.add_argument("--experiment-card-dir", type=Path, default=Path("docs/research/experiments"))
    args = parser.parse_args()

    config = load_final_training_config(args.source_run_config)
    card_path = args.experiment_card_dir / f"{args.experiment_id}-{config.seed}.md"
    command = shlex.join([sys.executable, *sys.argv])
    _write_run_card(
        card_path,
        experiment_id=args.experiment_id,
        seed=config.seed,
        source_run_config=args.source_run_config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        score_threshold=config.selected_score_threshold,
        model_variant=config.model_variant,
        resolution=config.resolution,
        command=command,
    )
    try:
        config_path = _write_evaluation_config(
            args.output_dir,
            source_run_config=args.source_run_config,
            checkpoint=args.checkpoint,
            experiment_id=args.experiment_id,
            seed=config.seed,
            score_threshold=config.selected_score_threshold,
        )
        verify_cuda_runtime()
        summary = evaluate_final_checkpoint(config, checkpoint_path=args.checkpoint, output_dir=args.output_dir)
        mean = summary.to_dict()["mean"]
        _finish_run_card(
            card_path,
            status="Complete",
            rows=_result_rows(summary),
            conclusion=(
                f"Evaluation completed. Mean Dice={mean['dice']:.6f}; configuration is `{config_path}` and "
                f"subject metrics are `{args.output_dir / 'metrics/test-subject-metrics.json'}`."
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
