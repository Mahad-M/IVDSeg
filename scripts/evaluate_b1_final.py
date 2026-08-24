"""Evaluate a completed semantic U-Net final checkpoint once on the fixed holdout.

This separate immutable stage restores the persisted all-training-subject
definition, applies its development-selected sigmoid threshold, and writes
native-grid predictions and metrics without altering a selection decision.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import shlex
import sys
from typing import Any

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ivdseg.datamodule import IVD2p5DSemanticDataset
from ivdseg.evaluation import (
    EvaluationSummary,
    _write_fixed_central_overlay,
    compute_subject_metrics,
    select_fixed_test_records,
    write_native_prediction,
)
from ivdseg.manifest import FIXED_TEST_SUBJECTS
from ivdseg.normalization import load_profile
from ivdseg.spatial import load_canonical_subject, resolve_record_path
from ivdseg.training import verify_cuda_runtime
from ivdseg.unet_training import (
    B1_FINAL_TRAIN_SUBJECT_IDS,
    B1FinalTrainingConfig,
    B3FinalTrainingConfig,
    load_b1_final_training_config,
    load_b1_model,
    load_b3_final_model,
    load_b3_final_training_config,
)


SemanticFinalConfig = B1FinalTrainingConfig | B3FinalTrainingConfig


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _select_b1_final_test_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Select exactly the fixed holdout through the evaluation-only guard.

    The similarly named data-module selector deliberately rejects these records
    because it protects training.  Final evaluation must instead reuse the
    invariant that requires precisely the declared test split.
    """
    return select_fixed_test_records(manifest)


def _model_description(config_kind: str, config: SemanticFinalConfig) -> str:
    if config_kind == "b1":
        return "from-scratch 12-channel 2.5D ResUNet-34 semantic segmenter"
    if config_kind == "b3":
        return "ImageNet-pretrained 12-channel 2.5D Swin V2 Tiny U-Net semantic segmenter"
    raise ValueError(f"unknown semantic final configuration kind: {config_kind}")


def _write_evaluation_config(
    output_dir: Path,
    *,
    source_run_config: Path,
    checkpoint: Path,
    experiment_id: str,
    config: SemanticFinalConfig,
    model_description: str,
) -> Path:
    """Record every fixed inference and metric decision before evaluation."""
    path = output_dir / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": experiment_id,
        "seed": config.seed,
        "source_final_run_config": str(source_run_config),
        "source_checkpoint": str(checkpoint),
        "selected_probability_threshold": config.selected_probability_threshold,
        "fixed_test_subjects": list(FIXED_TEST_SUBJECTS),
        "model": model_description,
        "postprocessing": "sigmoid probability >= selected threshold; no component filtering",
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
    config: SemanticFinalConfig,
    model_description: str,
    source_run_config: Path,
    checkpoint: Path,
    output_dir: Path,
    command: str,
) -> None:
    if path.exists():
        raise FileExistsError(f"experiment card already exists: {path}; use a new experiment ID")
    content = f"""# {experiment_id} - Final fixed-holdout evaluation - Seed {config.seed}

**Status:** Running  
**Started:** {_timestamp()}  
**Finished:**

## Purpose

- **Hypothesis:** The final semantic checkpoint provides a reproducible IVD segmentation on the untouched fixed holdout.
- **Comparison:** Evaluation only; checkpoint `{checkpoint}` and sigmoid threshold `{config.selected_probability_threshold:.2f}` are fixed before opening holdout labels.
- **Plan reference:** `docs/plans/rf-detr-2-5d-ivd-segmentation.md`

## Immutable Run Definition

- **Source final configuration:** `{source_run_config}`
- **Source checkpoint:** `{checkpoint}`
- **Dataset and split:** Fixed test subjects `03,07,10,14` only. Labels do not enter model inference; their native geometry restores output masks and their contents are used only for post-inference metrics.
- **Model and preprocessing:** {model_description.title()} at `{config.resolution} x {config.resolution}`, with the all-12-subject normalization profile and full binary semantic labels.
- **Operating point:** Sigmoid probability `>= {config.selected_probability_threshold:.2f}`. There is no component filtering or metric-driven postprocessing.
- **Evaluation command:** `{command}`
- **Geometry and metrics:** Each binary NIfTI prediction has its source label's native shape and affine. Report semantic 3D Dice, symmetric ASD/HD95 in mm, physical centroid localization distance, and 26-connected component FP/FN. A distance is undefined if exactly one mask is empty.
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


def evaluate_semantic_final_checkpoint(
    config: SemanticFinalConfig,
    *,
    checkpoint_path: Path,
    output_dir: Path,
    config_kind: str,
) -> EvaluationSummary:
    """Restore one semantic final checkpoint, reconstruct test volumes, and save outputs."""
    if not torch.cuda.is_available():
        raise RuntimeError("semantic final evaluation requires a CUDA device")
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    records = _select_b1_final_test_records(manifest)
    profile = load_profile(config.normalization_profile)
    if profile.fitted_subject_ids != B1_FINAL_TRAIN_SUBJECT_IDS:
        raise ValueError("semantic final evaluation requires the all-training-subject normalization profile")
    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics" / "test-subject-metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"evaluation metrics already exist: {metrics_path}")

    dataset = IVD2p5DSemanticDataset(
        records=records,
        dataset_root=config.dataset_root,
        profile=profile,
        resolution=config.resolution,
        augmentation=None,
    )
    predictions = {
        subject_id: np.zeros_like(dataset.semantic_label_for_subject(subject_id), dtype=bool)
        for subject_id in FIXED_TEST_SUBJECTS
    }
    loader = DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    device = torch.device("cuda")
    model = (
        load_b1_model(config, checkpoint_path, device=device)
        if config_kind == "b1"
        else load_b3_final_model(config, checkpoint_path, device=device)
    )
    seen_indices: set[int] = set()
    with torch.inference_mode():
        for images, _masks, sample_indices in loader:
            indices = [int(index) for index in sample_indices.tolist()]
            if any(index in seen_indices for index in indices):
                raise RuntimeError("semantic final evaluation observed a slice more than once")
            seen_indices.update(indices)
            precision_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if torch.cuda.is_bf16_supported()
                else nullcontext()
            )
            with precision_context:
                logits = model(images.to(device, non_blocking=True))
            probabilities = torch.sigmoid(logits)
            for batch_index, sample_index in enumerate(indices):
                reference = dataset.references[sample_index]
                native_plane_shape = predictions[reference.subject_id].shape[1:]
                native_probability = functional.interpolate(
                    probabilities[batch_index : batch_index + 1],
                    size=native_plane_shape,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                predictions[reference.subject_id][reference.slice_index] = (
                    native_probability >= config.selected_probability_threshold
                ).cpu().numpy()
    if seen_indices != set(range(len(dataset))):
        missing = sorted(set(range(len(dataset))) - seen_indices)
        raise RuntimeError(f"semantic final evaluation did not reconstruct every holdout slice; missing={missing}")

    subject_metrics = []
    for record in records:
        subject_id = str(record["subject_id"])
        canonical_subject = load_canonical_subject(record, config.dataset_root)
        target_canonical = dataset.semantic_label_for_subject(subject_id)
        prediction_canonical = predictions[subject_id]
        raw_label_path = resolve_record_path(config.dataset_root, str(record["label"]))
        prediction_path = write_native_prediction(
            prediction_canonical,
            raw_label_path=raw_label_path,
            output_path=output_dir / "predictions" / f"subject-{subject_id}-prediction.nii.gz",
        )
        raw_label = nib.load(raw_label_path)
        native_prediction = np.asarray(nib.load(prediction_path).dataobj) > 0
        native_target = np.asarray(raw_label.dataobj) > 0
        subject_metrics.append(
            compute_subject_metrics(subject_id, native_prediction, native_target, raw_label.affine)
        )
        _write_fixed_central_overlay(
            subject_id=subject_id,
            reference_data=np.asarray(canonical_subject.reference_image.get_fdata(dtype=np.float32)),
            prediction=prediction_canonical,
            target=target_canonical,
            output_path=output_dir / "overlays" / f"subject-{subject_id}-central.png",
        )
        print(
            "[ivdseg-evaluation] "
            f"subject={subject_id} prediction={prediction_path.name} dice={subject_metrics[-1].dice:.6f}",
            flush=True,
        )
    summary = EvaluationSummary(
        source_checkpoint=str(checkpoint_path),
        score_threshold=config.selected_probability_threshold,
        subjects=tuple(subject_metrics),
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-config", type=Path, required=True)
    parser.add_argument("--source-config-kind", choices=("b1", "b3"), default="b1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", default="B1-resunet34-final-eval")
    parser.add_argument("--experiment-card-dir", type=Path, default=Path("docs/research/experiments"))
    args = parser.parse_args()

    config = (
        load_b1_final_training_config(args.source_run_config)
        if args.source_config_kind == "b1"
        else load_b3_final_training_config(args.source_run_config)
    )
    model_description = _model_description(args.source_config_kind, config)
    card_path = args.experiment_card_dir / f"{args.experiment_id}-{config.seed}.md"
    command = shlex.join([sys.executable, *sys.argv])
    _write_run_card(
        card_path,
        experiment_id=args.experiment_id,
        config=config,
        source_run_config=args.source_run_config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        command=command,
        model_description=model_description,
    )
    try:
        config_path = _write_evaluation_config(
            args.output_dir,
            source_run_config=args.source_run_config,
            checkpoint=args.checkpoint,
            experiment_id=args.experiment_id,
            config=config,
            model_description=model_description,
        )
        verify_cuda_runtime()
        summary = evaluate_semantic_final_checkpoint(
            config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            config_kind=args.source_config_kind,
        )
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
