"""Calculate final held-out 2D one-class mask AP from the fixed checkpoint."""

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

from ivdseg.evaluation import INSTANCE_AP_IOU_THRESHOLDS, FinalInstanceAPSummary, evaluate_final_instance_ap
from ivdseg.manifest import FIXED_TEST_SUBJECTS
from ivdseg.training import load_final_training_config, verify_cuda_runtime


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _metric_text(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.6f}"


def _write_config(
    output_path: Path,
    *,
    source_run_config: Path,
    checkpoint: Path,
    experiment_id: str,
    seed: int,
    selected_score_threshold: float,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": experiment_id,
        "seed": seed,
        "source_final_run_config": str(source_run_config),
        "source_checkpoint": str(checkpoint),
        "fixed_test_subjects": list(FIXED_TEST_SUBJECTS),
        "metric": "one-class 2D instance-mask AP",
        "iou_thresholds": list(INSTANCE_AP_IOU_THRESHOLDS),
        "matching": "descending score, greedy unmatched target mask with maximum IoU",
        "primary_reconstruction_unchanged": (
            f"the saved NIfTI predictions retain strict score > {selected_score_threshold:.2f}"
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def _write_run_card(
    path: Path,
    *,
    experiment_id: str,
    seed: int,
    source_run_config: Path,
    checkpoint: Path,
    output_path: Path,
    selected_score_threshold: float,
    command: str,
) -> None:
    if path.exists():
        raise FileExistsError(f"experiment card already exists: {path}; use a new experiment ID")
    content = f"""# {experiment_id} - Final 2D instance-mask AP - Seed {seed}

**Status:** Running  
**Started:** {_timestamp()}  
**Finished:**

## Purpose

- **Hypothesis:** The completed final checkpoint has strong ranked 2D instance-mask detection quality on the untouched fixed holdout.
- **Comparison:** Secondary diagnostic only; the checkpoint, test split, final normalization profile, and primary NIfTI outputs are fixed.
- **Plan reference:** `docs/plans/rf-detr-2-5d-ivd-segmentation.md`

## Immutable Run Definition

- **Source final configuration:** `{source_run_config}`
- **Source checkpoint:** `{checkpoint}`
- **Dataset and split:** Fixed test subjects `03,07,10,14` only. Per-slice target instances use the same 26-connected, minimum-100-voxel component rule as training.
- **Metric:** Class-0 mask AP at IoU `0.50`, plus mean AP over `0.50,0.55,...,0.95`. Each slice matches detections in descending score order to the highest-IoU unmatched ground-truth mask; slice events are then globally ranked for AP.
- **Primary reconstruction:** Unchanged. This diagnostic considers ranked detections at every score and does not modify the persisted strict-`>{selected_score_threshold:.2f}` semantic NIfTI predictions.
- **Evaluation command:** `{command}`

## Artifacts

- **Configuration:** `{output_path.parent / 'config.json'}`
- **Logs:** Modal standard output contains `[ivdseg-instance-ap]` per-subject lines.
- **Metrics:** `{output_path}`.
- **Predictions / qualitative outputs:** This diagnostic writes neither predictions nor qualitative outputs; it does not modify the source final-evaluation artifacts.

## Results

| Subject | AP50 | mAP 50–95 | Target instances | Ranked detections |
| --- | ---: | ---: | ---: | ---: |
<!-- RESULTS -->

## Conclusion

- **Outcome:** Running.
- **Failure or deviation from plan:**
- **Decision / next action:** Combine this diagnostic with the already-saved native-geometry volume metrics before preparing evidence tables and panels.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _result_rows(summary: FinalInstanceAPSummary) -> str:
    rows = [
        "| "
        f"{subject_id} | {_metric_text(subject.ap50)} | {_metric_text(subject.map_50_95)} | "
        f"{subject.target_instance_count} | {subject.detection_count} |"
        for subject_id, subject in summary.subjects.items()
    ]
    overall = summary.overall
    rows.append(
        "| Overall | "
        f"{_metric_text(overall.ap50)} | {_metric_text(overall.map_50_95)} | "
        f"{overall.target_instance_count} | {overall.detection_count} |"
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", default="R0-final-instance-ap")
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
        output_path=args.output,
        selected_score_threshold=config.selected_score_threshold,
        command=command,
    )
    try:
        config_path = _write_config(
            args.output.parent / "config.json",
            source_run_config=args.source_run_config,
            checkpoint=args.checkpoint,
            experiment_id=args.experiment_id,
            seed=config.seed,
            selected_score_threshold=config.selected_score_threshold,
        )
        verify_cuda_runtime()
        summary = evaluate_final_instance_ap(config, checkpoint_path=args.checkpoint, output_path=args.output)
        _finish_run_card(
            card_path,
            status="Complete",
            rows=_result_rows(summary),
            conclusion=(
                f"2D instance-mask AP completed. Overall AP50={_metric_text(summary.overall.ap50)} and "
                f"mAP50-95={_metric_text(summary.overall.map_50_95)}; metrics are `{args.output}` "
                f"and configuration is `{config_path}`."
            ),
        )
    except BaseException as error:
        _finish_run_card(
            card_path,
            status="Failed",
            conclusion=f"2D instance-mask AP stopped: {type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
