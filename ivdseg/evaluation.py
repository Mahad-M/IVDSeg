"""Held-out native-geometry inference and subject-level IVD evaluation.

This module is intentionally separate from development model selection.  It
accepts only the final model definition, its persisted checkpoint, and the
fixed test split.  Predictions are a raw union of class-0 masks whose detection
score is strictly above the already-selected operating point; no component
filtering or metric-driven postprocessing is applied.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import torch
from torch import Tensor
from torch.nn import functional as torch_functional

from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

from ivdseg.manifest import FIXED_TEST_SUBJECTS, MODALITY_ORDER
from ivdseg.normalization import load_profile, normalize_modalities
from ivdseg.samples import connected_components_3d, make_2_5d_tensor
from ivdseg.spatial import canonicalize, load_canonical_subject, resolve_record_path
from ivdseg.training import (
    FINAL_TRAIN_SUBJECT_IDS,
    FinalTrainingConfig,
    FinalTrainingModule,
    make_model_config,
    make_train_config,
)


INSTANCE_AP_IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


@dataclass(frozen=True)
class SubjectEvaluation:
    """Primary and component diagnostics for one held-out subject volume."""

    subject_id: str
    dice: float
    asd_mm: float | None
    hd95_mm: float | None
    localization_distance_mm: float | None
    prediction_voxels: int
    target_voxels: int
    intersection_voxels: int
    component_false_positives: int
    component_false_negatives: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationSummary:
    """All fixed-holdout metrics from one immutable final checkpoint."""

    source_checkpoint: str
    score_threshold: float
    subjects: tuple[SubjectEvaluation, ...]

    @staticmethod
    def _mean_optional(values: Sequence[float | None]) -> float | None:
        numeric = [value for value in values if value is not None]
        return None if not numeric else float(np.mean(numeric))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_checkpoint": self.source_checkpoint,
            "score_threshold": self.score_threshold,
            "subjects": [subject.to_dict() for subject in self.subjects],
            "mean": {
                "dice": float(np.mean([subject.dice for subject in self.subjects])),
                "asd_mm": self._mean_optional([subject.asd_mm for subject in self.subjects]),
                "hd95_mm": self._mean_optional([subject.hd95_mm for subject in self.subjects]),
                "localization_distance_mm": self._mean_optional(
                    [subject.localization_distance_mm for subject in self.subjects]
                ),
                "component_false_positives": float(
                    np.mean([subject.component_false_positives for subject in self.subjects])
                ),
                "component_false_negatives": float(
                    np.mean([subject.component_false_negatives for subject in self.subjects])
                ),
            },
        }


@dataclass(frozen=True)
class SubjectSliceDice:
    """Slice-level Dice diagnostics on the model's canonical coronal planes.

    ``foreground_only_mean_dice`` excludes target-empty slices.  This is the
    meaningful 2D segmentation measure because true-negative background slices
    receive Dice 1.0.  ``all_slice_mean_dice`` is retained solely to make that
    possible empty-slice inflation explicit when comparing older 2D reports.
    """

    subject_id: str
    foreground_only_mean_dice: float | None
    all_slice_mean_dice: float
    foreground_slice_count: int
    empty_target_slice_count: int
    empty_target_true_negative_slice_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SliceDiceEvaluationSummary:
    """Fixed-holdout 2D Dice diagnostic from already-saved final predictions."""

    source_prediction_dir: str
    subjects: tuple[SubjectSliceDice, ...]

    @staticmethod
    def _mean_optional(values: Sequence[float | None]) -> float | None:
        numeric = [value for value in values if value is not None]
        return None if not numeric else float(np.mean(numeric))

    def to_dict(self) -> dict[str, Any]:
        foreground_values = [subject.foreground_only_mean_dice for subject in self.subjects]
        all_values = [subject.all_slice_mean_dice for subject in self.subjects]
        return {
            "source_prediction_dir": self.source_prediction_dir,
            "slice_plane": "canonical RAS axis 0 (the coronal 2.5D model input plane)",
            "metric_definition": {
                "foreground_only_mean_dice": (
                    "mean binary Dice over slices whose target contains at least one foreground voxel"
                ),
                "all_slice_mean_dice": (
                    "mean binary Dice over every slice; target-empty/prediction-empty slices score 1.0"
                ),
            },
            "subjects": [subject.to_dict() for subject in self.subjects],
            "macro_subject_mean": {
                "foreground_only_mean_dice": self._mean_optional(foreground_values),
                "all_slice_mean_dice": float(np.mean(all_values)),
            },
            "pooled_slices": {
                "foreground_only_mean_dice": self._pooled_foreground_dice(),
                "all_slice_mean_dice": self._pooled_all_slice_dice(),
                "foreground_slice_count": int(sum(subject.foreground_slice_count for subject in self.subjects)),
                "empty_target_slice_count": int(sum(subject.empty_target_slice_count for subject in self.subjects)),
                "empty_target_true_negative_slice_count": int(
                    sum(subject.empty_target_true_negative_slice_count for subject in self.subjects)
                ),
            },
        }

    def _pooled_foreground_dice(self) -> float | None:
        count = sum(subject.foreground_slice_count for subject in self.subjects)
        if count == 0:
            return None
        return float(
            sum(
                subject.foreground_only_mean_dice * subject.foreground_slice_count
                for subject in self.subjects
                if subject.foreground_only_mean_dice is not None
            )
            / count
        )

    def _pooled_all_slice_dice(self) -> float:
        count = sum(subject.foreground_slice_count + subject.empty_target_slice_count for subject in self.subjects)
        if count == 0:
            raise ValueError("slice Dice requires at least one slice")
        return float(
            sum(
                subject.all_slice_mean_dice * (subject.foreground_slice_count + subject.empty_target_slice_count)
                for subject in self.subjects
            )
            / count
        )


@dataclass(frozen=True)
class InstanceAPSummary:
    """One-class 2D mask AP summary over a set of independent slices."""

    ap_by_iou: Mapping[float, float | None]
    target_instance_count: int
    detection_count: int

    @property
    def ap50(self) -> float | None:
        return self.ap_by_iou.get(0.50)

    @property
    def map_50_95(self) -> float | None:
        values = [value for value in self.ap_by_iou.values() if value is not None]
        return None if not values else float(np.mean(values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ap_by_iou": {f"{threshold:.2f}": value for threshold, value in self.ap_by_iou.items()},
            "ap50": self.ap50,
            "map_50_95": self.map_50_95,
            "target_instance_count": self.target_instance_count,
            "detection_count": self.detection_count,
        }


@dataclass(frozen=True)
class FinalInstanceAPSummary:
    """Per-subject and pooled 2D instance-mask AP for the fixed test split."""

    source_checkpoint: str
    subjects: Mapping[str, InstanceAPSummary]
    overall: InstanceAPSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_checkpoint": self.source_checkpoint,
            "iou_thresholds": list(INSTANCE_AP_IOU_THRESHOLDS),
            "subjects": {subject_id: summary.to_dict() for subject_id, summary in self.subjects.items()},
            "overall": self.overall.to_dict(),
        }


class InstanceAPAccumulator:
    """Accumulate one-class mask matches and calculate ranked AP deterministically.

    Matches are greedy within each slice after descending score ordering, which
    is equivalent to global ranked matching because detections from different
    slices never compete for the same ground-truth instance.
    """

    def __init__(self, *, iou_thresholds: Sequence[float] = INSTANCE_AP_IOU_THRESHOLDS) -> None:
        self.iou_thresholds = tuple(float(threshold) for threshold in iou_thresholds)
        if not self.iou_thresholds or tuple(sorted(self.iou_thresholds)) != self.iou_thresholds:
            raise ValueError("IoU thresholds must be a non-empty ascending sequence")
        if any(not 0.0 < threshold <= 1.0 for threshold in self.iou_thresholds):
            raise ValueError("IoU thresholds must be in (0, 1]")
        self._events: dict[float, list[tuple[float, bool]]] = {threshold: [] for threshold in self.iou_thresholds}
        self.target_instance_count = 0
        self.detection_count = 0

    def add_image(
        self,
        *,
        scores: Tensor,
        labels: Tensor,
        masks: Tensor,
        target_masks: Tensor,
    ) -> None:
        """Add all ranked class-0 detections and targets from one 2D slice."""
        if masks.ndim != 4 or masks.shape[1] != 1:
            raise ValueError("prediction masks must have shape [N, 1, H, W]")
        if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape or scores.numel() != masks.shape[0]:
            raise ValueError("prediction scores, labels, and masks must align")
        if target_masks.ndim != 3 or tuple(target_masks.shape[-2:]) != tuple(masks.shape[-2:]):
            raise ValueError("target masks must have shape [M, H, W] on the prediction plane")
        target_masks = target_masks.to(device=masks.device, dtype=torch.bool)
        self.target_instance_count += int(target_masks.shape[0])
        keep = labels == 0
        if not bool(keep.any()):
            return
        kept_scores = scores[keep]
        kept_masks = masks[keep, 0].bool()
        order = torch.argsort(kept_scores, descending=True, stable=True)
        kept_scores = kept_scores[order]
        kept_masks = kept_masks[order]
        self.detection_count += int(kept_scores.numel())
        if target_masks.shape[0]:
            prediction_flat = kept_masks.flatten(1).to(torch.float32)
            target_flat = target_masks.flatten(1).to(torch.float32)
            intersections = prediction_flat @ target_flat.transpose(0, 1)
            prediction_sizes = prediction_flat.sum(dim=1, keepdim=True)
            target_sizes = target_flat.sum(dim=1).unsqueeze(0)
            iou = intersections / (prediction_sizes + target_sizes - intersections).clamp_min(1.0)
        else:
            iou = kept_scores.new_zeros((kept_scores.numel(), 0))
        score_values = kept_scores.detach().cpu().tolist()
        for threshold in self.iou_thresholds:
            matched_targets = torch.zeros((target_masks.shape[0],), dtype=torch.bool, device=masks.device)
            for detection_index, score in enumerate(score_values):
                true_positive = False
                if target_masks.shape[0]:
                    candidate_iou = iou[detection_index].masked_fill(matched_targets, -1.0)
                    best_iou, best_target = candidate_iou.max(dim=0)
                    if float(best_iou) >= threshold:
                        matched_targets[best_target] = True
                        true_positive = True
                self._events[threshold].append((float(score), true_positive))

    @staticmethod
    def _average_precision(events: Sequence[tuple[float, bool]], target_count: int) -> float | None:
        if target_count == 0:
            return None
        ordered = sorted(events, key=lambda event: event[0], reverse=True)
        true_positive = np.asarray([event[1] for event in ordered], dtype=float)
        false_positive = 1.0 - true_positive
        cumulative_true_positive = np.cumsum(true_positive)
        cumulative_false_positive = np.cumsum(false_positive)
        recall = cumulative_true_positive / target_count
        precision = cumulative_true_positive / np.maximum(cumulative_true_positive + cumulative_false_positive, 1.0)
        recall_envelope = np.concatenate(([0.0], recall, [1.0]))
        precision_envelope = np.concatenate(([0.0], precision, [0.0]))
        precision_envelope = np.maximum.accumulate(precision_envelope[::-1])[::-1]
        changes = np.flatnonzero(recall_envelope[1:] != recall_envelope[:-1])
        return float(np.sum((recall_envelope[changes + 1] - recall_envelope[changes]) * precision_envelope[changes + 1]))

    def finalize(self) -> InstanceAPSummary:
        return InstanceAPSummary(
            ap_by_iou={
                threshold: self._average_precision(events, self.target_instance_count)
                for threshold, events in self._events.items()
            },
            target_instance_count=self.target_instance_count,
            detection_count=self.detection_count,
        )


def select_fixed_test_records(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return exactly the locked test records, rejecting any split drift."""
    records = {str(record["subject_id"]): record for record in manifest["subjects"]}
    actual_test_ids = tuple(sorted(str(record["subject_id"]) for record in manifest["subjects"] if record["partition"] == "test"))
    if actual_test_ids != FIXED_TEST_SUBJECTS:
        raise ValueError(
            "manifest test partition must contain exactly the fixed test subjects "
            f"{FIXED_TEST_SUBJECTS}, got {actual_test_ids}"
        )
    try:
        return tuple(records[subject_id] for subject_id in FIXED_TEST_SUBJECTS)
    except KeyError as error:
        raise ValueError(f"manifest lacks fixed test subject {error.args[0]}") from error


def _validate_result(result: Mapping[str, Tensor], *, plane_shape: tuple[int, int]) -> tuple[Tensor, Tensor, Tensor]:
    scores = result.get("scores")
    labels = result.get("labels")
    masks = result.get("masks")
    if not isinstance(scores, Tensor) or not isinstance(labels, Tensor) or not isinstance(masks, Tensor):
        raise ValueError("RF-DETR segmentation result must contain tensor scores, labels, and masks")
    if masks.ndim != 4 or masks.shape[1] != 1 or tuple(masks.shape[-2:]) != plane_shape:
        raise ValueError(f"RF-DETR masks must have shape [N, 1, {plane_shape[0]}, {plane_shape[1]}]")
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape or scores.shape[0] != masks.shape[0]:
        raise ValueError("RF-DETR result fields have inconsistent detection dimensions")
    return scores, labels, masks


def reconstruct_canonical_prediction(
    slice_results: Sequence[Mapping[str, Tensor]],
    *,
    volume_shape: tuple[int, int, int],
    score_threshold: float,
) -> np.ndarray:
    """Union score-filtered class-0 masks into one canonical binary volume."""
    if len(volume_shape) != 3 or any(dimension < 1 for dimension in volume_shape):
        raise ValueError(f"volume shape must be three positive dimensions, got {volume_shape}")
    if len(slice_results) != volume_shape[0]:
        raise ValueError(f"expected {volume_shape[0]} slice results, got {len(slice_results)}")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score threshold must be in [0, 1]")
    prediction = np.zeros(volume_shape, dtype=bool)
    plane_shape = volume_shape[1:]
    for slice_index, result in enumerate(slice_results):
        scores, labels, masks = _validate_result(result, plane_shape=plane_shape)
        keep = (scores > score_threshold) & (labels == 0)
        if bool(keep.any()):
            prediction[slice_index] = masks[keep, 0].any(dim=0).detach().cpu().numpy().astype(bool, copy=False)
    return prediction


def _binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(prediction, target).sum() / denominator)


def compute_subject_slice_dice(subject_id: str, prediction: np.ndarray, target: np.ndarray) -> SubjectSliceDice:
    """Calculate transparent 2D Dice variants on the canonical model planes."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must be matching 3D volumes")
    per_slice_dice = np.asarray(
        [_binary_dice(prediction[slice_index], target[slice_index]) for slice_index in range(prediction.shape[0])],
        dtype=float,
    )
    foreground = target.reshape(target.shape[0], -1).any(axis=1)
    empty_target = ~foreground
    empty_target_true_negative = empty_target & ~prediction.reshape(prediction.shape[0], -1).any(axis=1)
    return SubjectSliceDice(
        subject_id=subject_id,
        foreground_only_mean_dice=(float(per_slice_dice[foreground].mean()) if bool(foreground.any()) else None),
        all_slice_mean_dice=float(per_slice_dice.mean()),
        foreground_slice_count=int(foreground.sum()),
        empty_target_slice_count=int(empty_target.sum()),
        empty_target_true_negative_slice_count=int(empty_target_true_negative.sum()),
    )


def _surface_points(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    surface = mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1))
    return nib.affines.apply_affine(affine, np.argwhere(surface))


def _surface_metrics(prediction: np.ndarray, target: np.ndarray, affine: np.ndarray) -> tuple[float | None, float | None]:
    if not prediction.any() and not target.any():
        return 0.0, 0.0
    if not prediction.any() or not target.any():
        return None, None
    prediction_points = _surface_points(prediction, affine)
    target_points = _surface_points(target, affine)
    target_to_prediction = cKDTree(prediction_points).query(target_points, k=1)[0]
    prediction_to_target = cKDTree(target_points).query(prediction_points, k=1)[0]
    distances = np.concatenate((target_to_prediction, prediction_to_target))
    return float(np.mean(distances)), float(np.percentile(distances, 95))


def _localization_distance(prediction: np.ndarray, target: np.ndarray, affine: np.ndarray) -> float | None:
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return None
    prediction_centroid = nib.affines.apply_affine(affine, np.argwhere(prediction)).mean(axis=0)
    target_centroid = nib.affines.apply_affine(affine, np.argwhere(target)).mean(axis=0)
    return float(np.linalg.norm(prediction_centroid - target_centroid))


def _component_errors(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int]:
    structure = np.ones((3, 3, 3), dtype=bool)
    prediction_labels, prediction_count = ndimage.label(prediction, structure=structure)
    target_labels, target_count = ndimage.label(target, structure=structure)
    false_positives = sum(
        not np.any(target_labels[prediction_labels == component_id])
        for component_id in range(1, prediction_count + 1)
    )
    false_negatives = sum(
        not np.any(prediction_labels[target_labels == component_id])
        for component_id in range(1, target_count + 1)
    )
    return int(false_positives), int(false_negatives)


def compute_subject_metrics(
    subject_id: str, prediction: np.ndarray, target: np.ndarray, affine: np.ndarray
) -> SubjectEvaluation:
    """Compute semantic overlap and world-space geometry metrics for one volume.

    ASD is the mean of both directed surface-to-surface distance sets; HD95 is
    their combined 95th percentile. Localization is the world-coordinate
    distance between binary-volume centroids. A distance is ``None`` if exactly
    one mask is empty, avoiding a fabricated finite value.
    """
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    affine = np.asarray(affine, dtype=float)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must be matching 3D volumes")
    if affine.shape != (4, 4):
        raise ValueError("affine must have shape [4, 4]")
    asd_mm, hd95_mm = _surface_metrics(prediction, target, affine)
    component_false_positives, component_false_negatives = _component_errors(prediction, target)
    return SubjectEvaluation(
        subject_id=subject_id,
        dice=_binary_dice(prediction, target),
        asd_mm=asd_mm,
        hd95_mm=hd95_mm,
        localization_distance_mm=_localization_distance(prediction, target, affine),
        prediction_voxels=int(prediction.sum()),
        target_voxels=int(target.sum()),
        intersection_voxels=int(np.logical_and(prediction, target).sum()),
        component_false_positives=component_false_positives,
        component_false_negatives=component_false_negatives,
    )


def write_native_prediction(canonical_prediction: np.ndarray, *, raw_label_path: Path, output_path: Path) -> Path:
    """Write a canonical prediction on the source label's exact native grid."""
    raw_label = nib.load(Path(raw_label_path))
    canonical_label = canonicalize(raw_label)
    canonical_prediction = np.asarray(canonical_prediction, dtype=bool)
    if canonical_prediction.shape != canonical_label.shape:
        raise ValueError(
            "canonical prediction must match canonical source-label shape; "
            f"got prediction={canonical_prediction.shape}, label={canonical_label.shape}"
        )
    source_orientation = nib.orientations.io_orientation(canonical_label.affine)
    destination_orientation = nib.orientations.io_orientation(raw_label.affine)
    transform = nib.orientations.ornt_transform(source_orientation, destination_orientation)
    native_prediction = nib.orientations.apply_orientation(canonical_prediction.astype(np.uint8), transform)
    if native_prediction.shape != raw_label.shape:
        raise RuntimeError("orientation restoration did not recover the source label shape")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(native_prediction, raw_label.affine, header=raw_label.header.copy())
    image.set_data_dtype(np.uint8)
    image.header.set_slope_inter(None, None)
    nib.save(image, output_path)
    written = nib.load(output_path)
    if written.shape != raw_label.shape or not np.allclose(written.affine, raw_label.affine):
        raise RuntimeError("written prediction does not preserve the source label geometry")
    values = np.unique(np.asarray(written.dataobj))
    if not np.isin(values, (0, 1)).all():
        raise RuntimeError(f"written prediction is not binary: {values}")
    return output_path


def _load_final_model(config: FinalTrainingConfig, checkpoint_path: Path) -> FinalTrainingModule:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"final checkpoint does not exist: {checkpoint_path}")
    inference_config = FinalTrainingConfig(
        **{**asdict(config), "pretrain_weights": None}
    )
    module = FinalTrainingModule(make_model_config(inference_config), make_train_config(inference_config))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) else None
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"checkpoint has no Lightning state_dict: {checkpoint_path}")
    module.load_state_dict(state_dict, strict=True)
    return module


def _infer_canonical_subject(
    module: FinalTrainingModule,
    *,
    normalized_modalities: Mapping[str, np.ndarray],
    volume_shape: tuple[int, int, int],
    resolution: int,
    batch_size: int,
    score_threshold: float,
    device: torch.device,
) -> np.ndarray:
    """Infer one subject at native slice planes and reconstruct its raw union."""
    if tuple(normalized_modalities) != MODALITY_ORDER:
        raise ValueError("normalized modalities must use the locked modality order")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    slice_count, height, width = volume_shape
    slice_results: list[Mapping[str, Tensor]] = []
    with torch.inference_mode():
        for start in range(0, slice_count, batch_size):
            indices = range(start, min(start + batch_size, slice_count))
            images = [
                torch_functional.interpolate(
                    torch.from_numpy(make_2_5d_tensor(normalized_modalities, slice_index)).unsqueeze(0),
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                for slice_index in indices
            ]
            samples = nested_tensor_from_tensor_list(images, block_size=24).to(device, non_blocking=True)
            target_sizes = torch.tensor([[height, width]] * len(images), dtype=torch.int64, device=device)
            precision_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if torch.cuda.is_bf16_supported()
                else nullcontext()
            )
            with precision_context:
                outputs = module.model(samples)
                results = module.postprocess(outputs, target_sizes, score_threshold=score_threshold)
            slice_results.extend(results)
    return reconstruct_canonical_prediction(
        slice_results,
        volume_shape=volume_shape,
        score_threshold=score_threshold,
    )


def _write_fixed_central_overlay(
    *,
    subject_id: str,
    reference_data: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    output_path: Path,
) -> Path:
    """Write one non-cherry-picked central-slice overlay for a subject."""
    import matplotlib.pyplot as plot

    slice_index = reference_data.shape[0] // 2
    plane = np.asarray(reference_data[slice_index], dtype=float)
    nonzero = plane[plane != 0]
    lower, upper = (np.percentile(nonzero, (1, 99)) if nonzero.size else (0.0, 1.0))
    figure, axis = plot.subplots(figsize=(6, 6), constrained_layout=True)
    axis.imshow(plane, cmap="gray", vmin=lower, vmax=upper)
    if target[slice_index].any():
        axis.contour(target[slice_index], levels=[0.5], colors=["lime"], linewidths=1.0)
    if prediction[slice_index].any():
        axis.contour(prediction[slice_index], levels=[0.5], colors=["red"], linewidths=1.0)
    axis.set_title(f"Subject {subject_id}, fixed central coronal slice {slice_index}: target=green, prediction=red")
    axis.set_axis_off()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plot.close(figure)
    return output_path


def _slice_instance_masks(component_labels: np.ndarray, slice_index: int) -> Tensor:
    """Return the training-contract component masks for one native slice."""
    component_slice = component_labels[slice_index]
    component_ids = [component_id for component_id in np.unique(component_slice) if component_id != 0]
    if not component_ids:
        return torch.zeros((0, *component_slice.shape), dtype=torch.bool)
    return torch.stack([torch.from_numpy(component_slice == component_id) for component_id in component_ids]).bool()


def evaluate_final_instance_ap(
    config: FinalTrainingConfig,
    *,
    checkpoint_path: Path,
    output_path: Path,
) -> FinalInstanceAPSummary:
    """Calculate held-out 2D one-class mask AP without changing primary output.

    AP uses ranked class-0 detections at every score, with target components
    generated by the same 26-connected, 100-voxel-minimum rule as RF-DETR
    training. It is a diagnostic only: the primary saved NIfTI prediction keeps
    its separately selected score threshold and is never rewritten here.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("2D instance AP evaluation requires a CUDA device")
    manifest = json.loads(Path(config.manifest_path).read_text(encoding="utf-8"))
    records = select_fixed_test_records(manifest)
    profile = load_profile(config.normalization_profile)
    if profile.fitted_subject_ids != FINAL_TRAIN_SUBJECT_IDS:
        raise ValueError("2D instance AP evaluation requires the all-training-subject normalization profile")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"2D instance AP output already exists: {output_path}")

    module = _load_final_model(config, checkpoint_path)
    device = torch.device("cuda")
    module.to(device).eval()
    overall = InstanceAPAccumulator()
    per_subject: dict[str, InstanceAPSummary] = {}
    for record in records:
        subject_id = str(record["subject_id"])
        subject = load_canonical_subject(record, config.dataset_root)
        normalized_modalities = normalize_modalities(
            {
                modality: np.asarray(subject.modalities[modality].get_fdata(dtype=np.float32))
                for modality in MODALITY_ORDER
            },
            profile,
        )
        semantic_target = np.asarray(subject.label.get_fdata(dtype=np.float32)) > 0.5
        component_labels, _component_sizes = connected_components_3d(semantic_target)
        slice_count, height, width = (int(dimension) for dimension in semantic_target.shape)
        subject_accumulator = InstanceAPAccumulator()
        with torch.inference_mode():
            for start in range(0, slice_count, config.micro_batch_size):
                indices = tuple(range(start, min(start + config.micro_batch_size, slice_count)))
                images = [
                    torch_functional.interpolate(
                        torch.from_numpy(make_2_5d_tensor(normalized_modalities, slice_index)).unsqueeze(0),
                        size=(config.resolution, config.resolution),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    for slice_index in indices
                ]
                samples = nested_tensor_from_tensor_list(images, block_size=24).to(device, non_blocking=True)
                target_sizes = torch.tensor([[height, width]] * len(images), dtype=torch.int64, device=device)
                precision_context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if torch.cuda.is_bf16_supported()
                    else nullcontext()
                )
                with precision_context:
                    outputs = module.model(samples)
                    results = module.postprocess(outputs, target_sizes)
                for slice_index, result in zip(indices, results):
                    scores, labels, masks = _validate_result(result, plane_shape=(height, width))
                    target_masks = _slice_instance_masks(component_labels, slice_index)
                    subject_accumulator.add_image(
                        scores=scores,
                        labels=labels,
                        masks=masks,
                        target_masks=target_masks,
                    )
                    overall.add_image(
                        scores=scores,
                        labels=labels,
                        masks=masks,
                        target_masks=target_masks,
                    )
        per_subject[subject_id] = subject_accumulator.finalize()
        print(
            "[ivdseg-instance-ap] "
            f"subject={subject_id} ap50={per_subject[subject_id].ap50:.6f} "
            f"map_50_95={per_subject[subject_id].map_50_95:.6f}",
            flush=True,
        )
    summary = FinalInstanceAPSummary(
        source_checkpoint=str(checkpoint_path),
        subjects=per_subject,
        overall=overall.finalize(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
    return summary


def evaluate_final_checkpoint(
    config: FinalTrainingConfig,
    *,
    checkpoint_path: Path,
    output_dir: Path,
) -> EvaluationSummary:
    """Run the locked final model once on each fixed holdout and persist outputs."""
    if not torch.cuda.is_available():
        raise RuntimeError("final evaluation requires a CUDA device")
    manifest = json.loads(Path(config.manifest_path).read_text(encoding="utf-8"))
    records = select_fixed_test_records(manifest)
    profile = load_profile(config.normalization_profile)
    if profile.fitted_subject_ids != FINAL_TRAIN_SUBJECT_IDS:
        raise ValueError("final evaluation requires the all-training-subject normalization profile")
    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics" / "test-subject-metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"evaluation metrics already exist: {metrics_path}")

    module = _load_final_model(config, checkpoint_path)
    device = torch.device("cuda")
    module.to(device).eval()
    subject_metrics: list[SubjectEvaluation] = []
    for record in records:
        subject = load_canonical_subject(record, config.dataset_root)
        reference_data = np.asarray(subject.reference_image.get_fdata(dtype=np.float32))
        normalized_modalities = normalize_modalities(
            {
                modality: np.asarray(subject.modalities[modality].get_fdata(dtype=np.float32))
                for modality in MODALITY_ORDER
            },
            profile,
        )
        target_canonical = np.asarray(subject.label.get_fdata(dtype=np.float32)) > 0.5
        prediction_canonical = _infer_canonical_subject(
            module,
            normalized_modalities=normalized_modalities,
            volume_shape=tuple(int(dimension) for dimension in target_canonical.shape),
            resolution=config.resolution,
            batch_size=config.micro_batch_size,
            score_threshold=config.selected_score_threshold,
            device=device,
        )
        raw_label_path = resolve_record_path(config.dataset_root, str(record["label"]))
        prediction_path = write_native_prediction(
            prediction_canonical,
            raw_label_path=raw_label_path,
            output_path=output_dir / "predictions" / f"subject-{record['subject_id']}-prediction.nii.gz",
        )
        raw_label = nib.load(raw_label_path)
        native_prediction = np.asarray(nib.load(prediction_path).dataobj) > 0
        native_target = np.asarray(raw_label.dataobj) > 0
        subject_metrics.append(
            compute_subject_metrics(str(record["subject_id"]), native_prediction, native_target, raw_label.affine)
        )
        _write_fixed_central_overlay(
            subject_id=str(record["subject_id"]),
            reference_data=reference_data,
            prediction=prediction_canonical,
            target=target_canonical,
            output_path=output_dir / "overlays" / f"subject-{record['subject_id']}-central.png",
        )
        print(
            "[ivdseg-evaluation] "
            f"subject={record['subject_id']} prediction={prediction_path.name} "
            f"dice={subject_metrics[-1].dice:.6f}",
            flush=True,
        )
    summary = EvaluationSummary(
        source_checkpoint=str(checkpoint_path),
        score_threshold=config.selected_score_threshold,
        subjects=tuple(subject_metrics),
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
    return summary


def evaluate_saved_final_slice_dice(
    config: FinalTrainingConfig,
    *,
    prediction_dir: Path,
) -> SliceDiceEvaluationSummary:
    """Evaluate existing native predictions without loading a model or rerunning inference."""
    manifest = json.loads(Path(config.manifest_path).read_text(encoding="utf-8"))
    records = select_fixed_test_records(manifest)
    prediction_dir = Path(prediction_dir)
    subject_metrics: list[SubjectSliceDice] = []
    for record in records:
        subject_id = str(record["subject_id"])
        raw_label_path = resolve_record_path(config.dataset_root, str(record["label"]))
        prediction_path = prediction_dir / f"subject-{subject_id}-prediction.nii.gz"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"missing saved prediction for subject {subject_id}: {prediction_path}")
        raw_label = nib.load(raw_label_path)
        saved_prediction = nib.load(prediction_path)
        if saved_prediction.shape != raw_label.shape or not np.allclose(saved_prediction.affine, raw_label.affine):
            raise ValueError(f"saved prediction does not match label geometry for subject {subject_id}")
        target_canonical = np.asarray(canonicalize(raw_label).dataobj) > 0
        prediction_canonical = np.asarray(canonicalize(saved_prediction).dataobj) > 0
        subject_metrics.append(compute_subject_slice_dice(subject_id, prediction_canonical, target_canonical))
    return SliceDiceEvaluationSummary(
        source_prediction_dir=str(prediction_dir),
        subjects=tuple(subject_metrics),
    )
