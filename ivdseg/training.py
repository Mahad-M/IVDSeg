"""Development training harness for 12-channel RF-DETR IVD segmentation.

This module deliberately uses RF-DETR's model and criterion implementation but
not its COCO/PIL DataModule or COCO metric callbacks.  Validation is accumulated
back into subject volumes and the only model-selection metric is macro 3D Dice
across the two held-back development subjects.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
from torch import Tensor, nn

from rfdetr.config import ModelConfig, RFDETRSegLargeConfig, RFDETRSegSmallConfig, SegmentationTrainConfig
from rfdetr.training.module_model import RFDETRModelModule

from ivdseg.datamodule import IVD2p5DDataset, IVDDataModule, SliceReference


DEVELOPMENT_DICE_METRIC = "val/development_dice_3d"
DEVELOPMENT_TRAIN_SUBJECT_IDS = ("01", "04", "05", "06", "08", "09", "11", "13", "15", "16")
DEVELOPMENT_VALIDATION_SUBJECT_IDS = ("02", "12")
FINAL_TRAIN_SUBJECT_IDS = ("01", "02", "04", "05", "06", "08", "09", "11", "12", "13", "15", "16")
RGB_PRETRAINED_WEIGHTS = Path("artifacts/model-weights/rf-detr-seg-small.pt")
RGB_PRETRAINED_LARGE_WEIGHTS = Path("artifacts/model-weights/rf-detr-seg-large.pt")
RFDETR_SEGMENTATION_RESOLUTION_BLOCK_SIZE = 24
CUDA_RUNTIME_VERSION = "12.6"
DEVELOPMENT_THRESHOLD_GRID = tuple(round(index * 0.05, 2) for index in range(1, 20))


@dataclass(frozen=True)
class SubjectDice:
    """One reconstructed subject's binary 3D overlap statistics."""

    subject_id: str
    dice: float
    prediction_voxels: int
    target_voxels: int
    intersection_voxels: int


@dataclass(frozen=True)
class DevelopmentDiceSummary:
    """Epoch-level macro development Dice and its per-subject constituents."""

    score_threshold: float
    macro_dice: float
    subjects: tuple[SubjectDice, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_threshold": self.score_threshold,
            "macro_dice": self.macro_dice,
            "subjects": [asdict(subject) for subject in self.subjects],
        }


@dataclass(frozen=True)
class DevelopmentThresholdSelection:
    """All development-threshold scores and the deterministic selected score."""

    summaries: tuple[DevelopmentDiceSummary, ...]
    selected: DevelopmentDiceSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": [summary.to_dict() for summary in self.summaries],
            "selected_score_threshold": self.selected.score_threshold,
            "selected_macro_dice": self.selected.macro_dice,
        }


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute binary Dice, defining two empty masks as a perfect match."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(f"Dice inputs must have matching shapes, got {prediction.shape} and {target.shape}")
    prediction_count = int(prediction.sum())
    target_count = int(target.sum())
    denominator = prediction_count + target_count
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(prediction, target).sum() / denominator)


class DevelopmentVolumeDiceMonitor:
    """Reconstruct validation predictions and calculate macro subject-level 3D Dice.

    Inputs to :meth:`add_batch` are RF-DETR postprocessed slice predictions.
    Masks are already thresholded at probability 0.5 by RF-DETR's segmentation
    postprocessor.  This monitor applies only the fixed *detection-score*
    threshold used for early stopping, then unions kept masks per center slice.
    """

    def __init__(
        self,
        *,
        references: Sequence[SliceReference],
        semantic_labels: Mapping[str, np.ndarray],
        score_threshold: float = 0.50,
    ) -> None:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if not references:
            raise ValueError("development Dice requires at least one validation slice")
        self.references = tuple(references)
        self.semantic_labels = {
            subject_id: np.asarray(label, dtype=bool) for subject_id, label in semantic_labels.items()
        }
        self.score_threshold = score_threshold
        subject_ids = {reference.subject_id for reference in self.references}
        if set(self.semantic_labels) != subject_ids:
            raise ValueError("semantic labels must cover exactly the validation subject IDs")
        for reference in self.references:
            label = self.semantic_labels[reference.subject_id]
            if not 0 <= reference.slice_index < label.shape[0]:
                raise ValueError(f"invalid slice reference: {reference}")
        self._predictions: dict[str, np.ndarray] = {}
        self._seen_indices: set[int] = set()
        self.reset()

    @classmethod
    def from_validation_dataset(
        cls, dataset: IVD2p5DDataset, *, score_threshold: float = 0.50
    ) -> "DevelopmentVolumeDiceMonitor":
        """Bind the monitor to the data module's exact validation index and labels."""
        subject_ids = tuple(dict.fromkeys(reference.subject_id for reference in dataset.references))
        return cls(
            references=dataset.references,
            semantic_labels={subject_id: dataset.semantic_label_for_subject(subject_id) for subject_id in subject_ids},
            score_threshold=score_threshold,
        )

    def reset(self) -> None:
        """Start an empty reconstruction for the next complete validation epoch."""
        self._predictions = {
            subject_id: np.zeros_like(label, dtype=bool) for subject_id, label in self.semantic_labels.items()
        }
        self._seen_indices = set()

    def add_batch(self, results: Sequence[Mapping[str, Tensor]], targets: Sequence[Mapping[str, Any]]) -> None:
        """Union a validation batch's score-filtered instance masks into volumes."""
        if len(results) != len(targets):
            raise ValueError("validation results and targets must have equal batch size")
        for result, target in zip(results, targets):
            image_id = target.get("image_id")
            if not isinstance(image_id, Tensor) or image_id.numel() != 1:
                raise ValueError("validation target image_id must be a one-element tensor")
            index = int(image_id.detach().cpu().item())
            if not 0 <= index < len(self.references):
                raise ValueError(f"validation target image_id is outside the slice index: {index}")
            if index in self._seen_indices:
                raise ValueError(f"validation slice {index} was observed more than once")
            self._seen_indices.add(index)
            reference = self.references[index]
            subject_prediction = self._predictions[reference.subject_id]

            scores = result.get("scores")
            labels = result.get("labels")
            masks = result.get("masks")
            if not isinstance(scores, Tensor) or not isinstance(labels, Tensor) or not isinstance(masks, Tensor):
                raise ValueError("RF-DETR segmentation result must contain tensor scores, labels, and masks")
            if masks.ndim != 4 or masks.shape[1] != 1:
                raise ValueError(f"RF-DETR masks must have shape [N, 1, H, W], got {tuple(masks.shape)}")
            if scores.ndim != 1 or labels.ndim != 1 or masks.shape[0] != scores.shape[0] or labels.shape != scores.shape:
                raise ValueError("RF-DETR result fields have inconsistent detection dimensions")
            expected_plane = subject_prediction.shape[1:]
            if tuple(masks.shape[-2:]) != expected_plane:
                raise ValueError(
                    f"prediction plane {tuple(masks.shape[-2:])} does not match {reference.subject_id}'s "
                    f"canonical label plane {expected_plane}"
                )
            keep = (scores > self.score_threshold) & (labels == 0)
            if bool(keep.any()):
                union = masks[keep, 0].any(dim=0).detach().cpu().numpy().astype(bool, copy=False)
                subject_prediction[reference.slice_index] |= union

    def finalize(self) -> DevelopmentDiceSummary:
        """Return macro Dice only after every expected validation slice is reconstructed."""
        expected_indices = set(range(len(self.references)))
        if self._seen_indices != expected_indices:
            missing = sorted(expected_indices - self._seen_indices)
            unexpected = sorted(self._seen_indices - expected_indices)
            raise RuntimeError(f"incomplete development reconstruction; missing={missing}, unexpected={unexpected}")
        subjects: list[SubjectDice] = []
        for subject_id in sorted(self.semantic_labels):
            prediction = self._predictions[subject_id]
            target = self.semantic_labels[subject_id]
            intersection = int(np.logical_and(prediction, target).sum())
            subjects.append(
                SubjectDice(
                    subject_id=subject_id,
                    dice=binary_dice(prediction, target),
                    prediction_voxels=int(prediction.sum()),
                    target_voxels=int(target.sum()),
                    intersection_voxels=intersection,
                )
            )
        return DevelopmentDiceSummary(
            score_threshold=self.score_threshold,
            macro_dice=float(np.mean([subject.dice for subject in subjects])),
            subjects=tuple(subjects),
        )

    def write_summary(self, output_dir: Path, *, epoch: int) -> Path:
        """Write the current epoch's subject-level development metric record."""
        summary = self.finalize()
        output_path = Path(output_dir) / "metrics" / "development-3d" / f"epoch-{epoch:03d}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
        return output_path


class DevelopmentThresholdGrid:
    """Score a fixed threshold grid from one shared development inference pass.

    Each monitor receives identical detections and reconstructs the same native
    development volumes. This prevents threshold selection from changing any
    other factor of the experiment. Ties select the lower threshold so the rule
    is deterministic and documented before inspecting the final grid.
    """

    def __init__(
        self,
        *,
        references: Sequence[SliceReference],
        semantic_labels: Mapping[str, np.ndarray],
        thresholds: Sequence[float] = DEVELOPMENT_THRESHOLD_GRID,
    ) -> None:
        self.thresholds = tuple(float(threshold) for threshold in thresholds)
        if not self.thresholds:
            raise ValueError("threshold grid must not be empty")
        if tuple(sorted(self.thresholds)) != self.thresholds:
            raise ValueError("threshold grid must be sorted ascending")
        if len(set(self.thresholds)) != len(self.thresholds):
            raise ValueError("threshold grid must not contain duplicates")
        if any(not 0.0 <= threshold <= 1.0 for threshold in self.thresholds):
            raise ValueError("threshold grid values must be in [0, 1]")
        self._monitors = tuple(
            DevelopmentVolumeDiceMonitor(
                references=references,
                semantic_labels=semantic_labels,
                score_threshold=threshold,
            )
            for threshold in self.thresholds
        )

    @classmethod
    def from_validation_dataset(
        cls, dataset: IVD2p5DDataset, *, thresholds: Sequence[float] = DEVELOPMENT_THRESHOLD_GRID
    ) -> "DevelopmentThresholdGrid":
        subject_ids = tuple(dict.fromkeys(reference.subject_id for reference in dataset.references))
        return cls(
            references=dataset.references,
            semantic_labels={subject_id: dataset.semantic_label_for_subject(subject_id) for subject_id in subject_ids},
            thresholds=thresholds,
        )

    def add_batch(self, results: Sequence[Mapping[str, Tensor]], targets: Sequence[Mapping[str, Any]]) -> None:
        for monitor in self._monitors:
            monitor.add_batch(results, targets)

    def finalize(self) -> DevelopmentThresholdSelection:
        summaries = tuple(monitor.finalize() for monitor in self._monitors)
        selected = max(summaries, key=lambda summary: (summary.macro_dice, -summary.score_threshold))
        return DevelopmentThresholdSelection(summaries=summaries, selected=selected)


def write_development_threshold_selection(selection: DevelopmentThresholdSelection, output_path: Path) -> Path:
    """Persist the entire threshold grid and selected operating point."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selection.to_dict(), indent=2) + "\n", encoding="utf-8")
    return output_path


def adapt_rgb_patch_embedding(projection: nn.Conv2d, *, num_channels: int) -> nn.Conv2d:
    """Tile-and-scale a RGB patch projection using RF-DETR's channel rule.

    RF-DETR's high-level facade applies this exact adaptation for non-RGB input.
    The reusable Lightning module builds the RGB projection directly, so this
    helper closes that gap for the custom tensor training path.
    """
    if num_channels < 1:
        raise ValueError("num_channels must be positive")
    if projection.in_channels == num_channels:
        return projection
    if projection.in_channels != 3 or projection.groups != 1:
        raise ValueError("only a standard RGB patch projection can be adapted")
    adapted = copy.deepcopy(projection)
    if num_channels == 1:
        weight = projection.weight.detach().mean(dim=1, keepdim=True)
    else:
        repeats = (num_channels + 2) // 3
        weight = torch.cat([projection.weight.detach()] * repeats, dim=1)[:, :num_channels]
        weight = weight * (3.0 / num_channels)
    adapted.in_channels = num_channels
    adapted.weight = nn.Parameter(weight.clone(), requires_grad=projection.weight.requires_grad)
    return adapted


def adapt_model_input_channels(model: nn.Module, *, num_channels: int) -> None:
    """Adapt RF-DETR's DINOv2 patch projection after RGB checkpoint loading."""
    try:
        embeddings = model.backbone[0].encoder.encoder.embeddings.patch_embeddings
        projection = embeddings.projection
    except (AttributeError, IndexError, TypeError) as error:
        raise RuntimeError("could not locate RF-DETR's DINOv2 patch embedding projection") from error
    if not isinstance(projection, nn.Conv2d):
        raise RuntimeError("RF-DETR's patch embedding projection is not a Conv2d")
    embeddings.projection = adapt_rgb_patch_embedding(projection, num_channels=num_channels)
    embeddings.num_channels = num_channels


class IVDTrainingModule(RFDETRModelModule):
    """RF-DETR Lightning module with development-volume Dice model selection."""

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: SegmentationTrainConfig,
        development_dice_monitor: DevelopmentVolumeDiceMonitor,
    ) -> None:
        super().__init__(model_config, train_config)
        adapt_model_input_channels(self.model, num_channels=model_config.num_channels)
        self.development_dice_monitor = development_dice_monitor

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.development_dice_monitor.reset()

    def validation_step(self, batch: tuple[Any, Any], batch_idx: int) -> dict[str, Any]:
        output = super().validation_step(batch, batch_idx)
        self.development_dice_monitor.add_batch(output["results"], output["targets"])
        return output

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        summary = self.development_dice_monitor.finalize()
        if self.trainer.is_global_zero:
            self.development_dice_monitor.write_summary(Path(self.train_config.output_dir), epoch=self.current_epoch)
        metric = torch.tensor(summary.macro_dice, dtype=torch.float32, device=self.device)
        self.log(DEVELOPMENT_DICE_METRIC, metric, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        for subject in summary.subjects:
            self.log(
                f"val/development_dice_3d_subject_{subject.subject_id}",
                torch.tensor(subject.dice, dtype=torch.float32, device=self.device),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )


class FinalTrainingModule(RFDETRModelModule):
    """RF-DETR module for train-only final retraining on all permitted subjects."""

    def __init__(self, model_config: ModelConfig, train_config: SegmentationTrainConfig) -> None:
        super().__init__(model_config, train_config)
        adapt_model_input_channels(self.model, num_channels=model_config.num_channels)


class PlainTextProgressReporter(Callback):
    """Emit progress lines for log collectors that cannot render ``tqdm`` redraws."""

    def __init__(self, *, every_n_train_batches: int) -> None:
        if every_n_train_batches < 1:
            raise ValueError("every_n_train_batches must be positive")
        self.every_n_train_batches = every_n_train_batches

    @staticmethod
    def _epoch_label(trainer: Trainer) -> str:
        return f"{trainer.current_epoch + 1}/{trainer.max_epochs}"

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: RFDETRModelModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch
        completed_batches = batch_idx + 1
        if not trainer.is_global_zero or completed_batches % self.every_n_train_batches:
            return
        print(
            "[ivdseg-progress] "
            f"epoch={self._epoch_label(trainer)} "
            f"train_batch={completed_batches}/{trainer.num_training_batches} "
            f"optimizer_step={trainer.global_step}",
            flush=True,
        )

    def on_validation_end(self, trainer: Trainer, pl_module: RFDETRModelModule) -> None:
        del pl_module
        if not trainer.is_global_zero:
            return
        metric = trainer.callback_metrics.get(DEVELOPMENT_DICE_METRIC)
        metric_text = "unavailable" if metric is None else f"{float(metric):.6f}"
        print(
            "[ivdseg-progress] "
            f"epoch={self._epoch_label(trainer)} "
            f"validation_complete macro_3d_dice={metric_text}",
            flush=True,
        )


@dataclass(frozen=True)
class DevelopmentTrainingConfig:
    """Immutable settings for one primary R0 development run."""

    manifest_path: Path
    dataset_root: Path
    normalization_profile: Path
    run_dir: Path
    seed: int
    experiment_id: str = "R0"
    model_variant: Literal["small", "large"] = "small"
    max_epochs: int = 100
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    resolution: int = 384
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    development_score_threshold: float = 0.50
    accelerator: str = "auto"
    num_workers: int = 2
    pretrain_weights: Path | None = RGB_PRETRAINED_WEIGHTS
    deterministic: Literal["warn"] = "warn"
    progress_report_interval_batches: int = 10

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.micro_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("epochs, micro batch size, and gradient accumulation must be positive")
        if self.micro_batch_size * self.gradient_accumulation_steps != 8:
            raise ValueError("the primary development run requires effective batch size 8 per device")
        if self.resolution < RFDETR_SEGMENTATION_RESOLUTION_BLOCK_SIZE or (
            self.resolution % RFDETR_SEGMENTATION_RESOLUTION_BLOCK_SIZE
        ):
            raise ValueError(
                "development resolution must be a positive multiple of RF-DETR Seg's "
                f"block size {RFDETR_SEGMENTATION_RESOLUTION_BLOCK_SIZE}"
            )
        if self.early_stopping_patience < 1 or self.early_stopping_min_delta < 0.0:
            raise ValueError("early stopping settings must be non-negative and meaningful")
        if not 0.0 <= self.development_score_threshold <= 1.0:
            raise ValueError("development score threshold must be in [0, 1]")
        if self.deterministic != "warn":
            raise ValueError("the primary run uses warn-only CUDA determinism for grid-sampler backward")
        if self.progress_report_interval_batches < 1:
            raise ValueError("progress report interval must be positive")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class FinalTrainingConfig:
    """Immutable settings for selected-seed final retraining.

    This run intentionally has no validation partition and starts again from
    the published RGB-pretrained RF-DETR checkpoint.  The development
    checkpoint chooses the epoch count and score threshold; it is not used to
    initialise final-training weights.
    """

    manifest_path: Path
    dataset_root: Path
    normalization_profile: Path
    run_dir: Path
    seed: int
    experiment_id: str = "R0-final"
    model_variant: Literal["small", "large"] = "small"
    max_epochs: int = 7
    selected_score_threshold: float = 0.35
    source_development_run: str = "R0-modal-17"
    source_development_checkpoint: str = "best-epoch=006.ckpt"
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    resolution: int = 384
    accelerator: str = "auto"
    num_workers: int = 2
    pretrain_weights: Path | None = RGB_PRETRAINED_WEIGHTS
    deterministic: Literal["warn"] = "warn"
    progress_report_interval_batches: int = 10

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.micro_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("epochs, micro batch size, and gradient accumulation must be positive")
        if self.micro_batch_size * self.gradient_accumulation_steps != 8:
            raise ValueError("the primary final run requires effective batch size 8 per device")
        if self.resolution != 384:
            raise ValueError("the locked final configuration uses resolution 384")
        if not 0.0 <= self.selected_score_threshold <= 1.0:
            raise ValueError("selected score threshold must be in [0, 1]")
        if self.deterministic != "warn":
            raise ValueError("the final run uses warn-only CUDA determinism for grid-sampler backward")
        if self.progress_report_interval_batches < 1:
            raise ValueError("progress report interval must be positive")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


def make_model_config(config: DevelopmentTrainingConfig | FinalTrainingConfig) -> ModelConfig:
    """Return the selected 12-channel, one-class RF-DETR segmentation architecture."""
    model_config_class: type[ModelConfig]
    if config.model_variant == "large":
        model_config_class = RFDETRSegLargeConfig
    else:
        model_config_class = RFDETRSegSmallConfig
    return model_config_class(
        num_classes=1,
        num_channels=12,
        resolution=config.resolution,
        pretrain_weights=str(config.pretrain_weights) if config.pretrain_weights is not None else None,
        amp=True,
    )


def make_train_config(config: DevelopmentTrainingConfig | FinalTrainingConfig) -> SegmentationTrainConfig:
    """Return RF-DETR optimizer settings while disabling its stock data callbacks."""
    return SegmentationTrainConfig(
        dataset_dir=str(config.dataset_root),
        output_dir=str(config.run_dir),
        batch_size=config.micro_batch_size,
        grad_accum_steps=config.gradient_accumulation_steps,
        epochs=config.max_epochs,
        lr=1e-4,
        lr_encoder=1.5e-4,
        weight_decay=1e-4,
        clip_max_norm=0.1,
        lr_scheduler="cosine",
        warmup_epochs=0.0,
        multi_scale=False,
        use_ema=False,
        compute_val_loss=True,
        num_workers=config.num_workers,
        seed=config.seed,
        accelerator=config.accelerator,
        progress_bar="tqdm",
        tensorboard=False,
        wandb=False,
        mlflow=False,
    )


def _resolve_precision(accelerator: str) -> str:
    """Choose RF-DETR's preferred mixed precision only on a usable CUDA device."""
    if accelerator == "cpu" or not torch.cuda.is_available():
        return "32-true"
    return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"


def verify_cuda_runtime() -> None:
    """Reject an incompatible CUDA wheel before a development run builds its model.

    The training host's GTX 1050 is Pascal (sm_61). Its CUDA 12.9 driver can
    run the CUDA 12.6 wheel, which retains Pascal kernels. Keeping this check
    at the launcher boundary turns a late, opaque PyTorch initialisation error
    into a short actionable preflight failure.
    """
    installed_runtime = torch.version.cuda
    if installed_runtime != CUDA_RUNTIME_VERSION:
        raise RuntimeError(
            "CUDA preflight rejected the installed PyTorch runtime: expected "
            f"CUDA {CUDA_RUNTIME_VERSION}, found {installed_runtime or 'CPU-only'}. "
            "Run `uv sync` so the explicit pytorch-cu126 source pins are installed."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA preflight could not access a CUDA device. Verify the NVIDIA driver and "
            "container/device passthrough before starting the training run."
        )
    try:
        torch.empty(1, device="cuda").zero_()
    except Exception as error:
        raise RuntimeError(
            "CUDA preflight could not allocate a CUDA tensor. Verify that the installed "
            "CUDA 12.6 PyTorch runtime is compatible with the host driver and GPU."
        ) from error


def build_development_training(
    config: DevelopmentTrainingConfig,
) -> tuple[Trainer, IVDTrainingModule, IVDDataModule, ModelCheckpoint]:
    """Assemble the custom data module, adapted RF-DETR module, and PL trainer.

    The monitor reconstructs only a single process's validation stream.  The
    primary protocol currently targets one accelerator; multi-device training
    is deliberately refused rather than silently reporting an incomplete Dice.
    """
    seed_everything(config.seed, workers=True)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    if config.pretrain_weights is not None:
        config.pretrain_weights.parent.mkdir(parents=True, exist_ok=True)
    data_module = IVDDataModule(
        manifest=config.manifest_path,
        dataset_root=config.dataset_root,
        normalization_profile=config.normalization_profile,
        train_subject_ids=DEVELOPMENT_TRAIN_SUBJECT_IDS,
        validation_subject_ids=DEVELOPMENT_VALIDATION_SUBJECT_IDS,
        batch_size=config.micro_batch_size,
        resolution=config.resolution,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=torch.cuda.is_available(),
    )
    data_module.setup("fit")
    if data_module.validation_dataset is None:
        raise RuntimeError("validation dataset was not created")
    monitor = DevelopmentVolumeDiceMonitor.from_validation_dataset(
        data_module.validation_dataset,
        score_threshold=config.development_score_threshold,
    )
    model_config = make_model_config(config)
    train_config = make_train_config(config)
    module = IVDTrainingModule(model_config, train_config, monitor)

    checkpoint_dir = config.run_dir / "checkpoints"
    best_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-epoch={epoch:03d}",
        monitor=DEVELOPMENT_DICE_METRIC,
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [
        PlainTextProgressReporter(every_n_train_batches=config.progress_report_interval_batches),
        best_checkpoint,
        EarlyStopping(
            monitor=DEVELOPMENT_DICE_METRIC,
            mode="max",
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            check_finite=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    logger = CSVLogger(save_dir=str(config.run_dir / "logs"), name="", version="")
    trainer = Trainer(
        default_root_dir=str(config.run_dir),
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=1,
        strategy="auto",
        precision=_resolve_precision(config.accelerator),
        accumulate_grad_batches=config.gradient_accumulation_steps,
        gradient_clip_val=0.1,
        callbacks=callbacks,
        logger=logger,
        deterministic=config.deterministic,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    return trainer, module, data_module, best_checkpoint


def build_final_training(
    config: FinalTrainingConfig,
) -> tuple[Trainer, FinalTrainingModule, IVDDataModule, ModelCheckpoint]:
    """Assemble train-only final retraining without loading the fixed holdout."""
    seed_everything(config.seed, workers=True)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    if config.pretrain_weights is not None:
        config.pretrain_weights.parent.mkdir(parents=True, exist_ok=True)
    data_module = IVDDataModule(
        manifest=config.manifest_path,
        dataset_root=config.dataset_root,
        normalization_profile=config.normalization_profile,
        train_subject_ids=FINAL_TRAIN_SUBJECT_IDS,
        validation_subject_ids=(),
        batch_size=config.micro_batch_size,
        resolution=config.resolution,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=torch.cuda.is_available(),
    )
    data_module.setup("fit")
    model_config = make_model_config(config)
    train_config = make_train_config(config)
    module = FinalTrainingModule(model_config, train_config)

    last_checkpoint = ModelCheckpoint(
        dirpath=config.run_dir / "checkpoints",
        save_top_k=0,
        save_last=True,
    )
    callbacks = [
        PlainTextProgressReporter(every_n_train_batches=config.progress_report_interval_batches),
        last_checkpoint,
        LearningRateMonitor(logging_interval="step"),
    ]
    logger = CSVLogger(save_dir=str(config.run_dir / "logs"), name="", version="")
    trainer = Trainer(
        default_root_dir=str(config.run_dir),
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=1,
        strategy="auto",
        precision=_resolve_precision(config.accelerator),
        accumulate_grad_batches=config.gradient_accumulation_steps,
        gradient_clip_val=0.1,
        callbacks=callbacks,
        logger=logger,
        deterministic=config.deterministic,
        log_every_n_steps=10,
        # RF-DETR's base Lightning module implements validation hooks. Final
        # retraining has no validation partition, so explicitly disable the
        # loop rather than returning a sentinel loader that could be mistaken
        # for a measured validation result.
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    return trainer, module, data_module, last_checkpoint


def load_development_training_config(run_config_path: Path) -> DevelopmentTrainingConfig:
    """Restore the immutable training settings recorded with a completed run."""
    payload = json.loads(Path(run_config_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported run configuration schema: {payload.get('schema_version')!r}")
    raw_config = payload.get("development_training")
    if not isinstance(raw_config, Mapping):
        raise ValueError("run configuration lacks development_training")
    supported_fields = {field.name for field in fields(DevelopmentTrainingConfig) if field.init}
    config_values = {key: value for key, value in raw_config.items() if key in supported_fields}
    required_paths = ("manifest_path", "dataset_root", "normalization_profile", "run_dir", "pretrain_weights")
    for field_name in required_paths:
        value = config_values.get(field_name)
        if value is None and field_name == "pretrain_weights":
            continue
        if not isinstance(value, str):
            raise ValueError(f"run configuration has no valid {field_name}")
        config_values[field_name] = Path(value)
    try:
        return DevelopmentTrainingConfig(**config_values)
    except TypeError as error:
        raise ValueError("run configuration is incomplete for threshold selection") from error


def load_final_training_config(run_config_path: Path) -> FinalTrainingConfig:
    """Restore the immutable final-training settings for held-out evaluation."""
    payload = json.loads(Path(run_config_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported run configuration schema: {payload.get('schema_version')!r}")
    raw_config = payload.get("final_training")
    if not isinstance(raw_config, Mapping):
        raise ValueError("run configuration lacks final_training")
    supported_fields = {field.name for field in fields(FinalTrainingConfig) if field.init}
    config_values = {key: value for key, value in raw_config.items() if key in supported_fields}
    required_paths = ("manifest_path", "dataset_root", "normalization_profile", "run_dir", "pretrain_weights")
    for field_name in required_paths:
        value = config_values.get(field_name)
        if value is None and field_name == "pretrain_weights":
            continue
        if not isinstance(value, str):
            raise ValueError(f"run configuration has no valid {field_name}")
        config_values[field_name] = Path(value)
    try:
        return FinalTrainingConfig(**config_values)
    except TypeError as error:
        raise ValueError("run configuration is incomplete for final evaluation") from error


def _move_targets_to_device(targets: Sequence[Mapping[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    """Move tensor target fields required by RF-DETR postprocessing to ``device``."""
    return [
        {key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value for key, value in target.items()}
        for target in targets
    ]


def select_development_threshold(
    config: DevelopmentTrainingConfig,
    *,
    checkpoint_path: Path,
    output_path: Path,
) -> DevelopmentThresholdSelection:
    """Select a detection score only from the saved development partition.

    The model inference pass is shared by all thresholds. It never loads the
    fixed test subjects and it preserves the development preprocessing profile
    recorded with the source run.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"development checkpoint does not exist: {checkpoint_path}")
    if config.accelerator not in {"auto", "gpu", "cuda"}:
        raise ValueError("threshold selection requires the saved GPU development configuration")
    if not torch.cuda.is_available():
        raise RuntimeError("threshold selection requires a CUDA device")

    seed_everything(config.seed, workers=True)
    data_module = IVDDataModule(
        manifest=config.manifest_path,
        dataset_root=config.dataset_root,
        normalization_profile=config.normalization_profile,
        train_subject_ids=DEVELOPMENT_TRAIN_SUBJECT_IDS,
        validation_subject_ids=DEVELOPMENT_VALIDATION_SUBJECT_IDS,
        batch_size=config.micro_batch_size,
        resolution=config.resolution,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=True,
    )
    data_module.setup("validate")
    if data_module.validation_dataset is None:
        raise RuntimeError("development validation dataset was not created")
    threshold_grid = DevelopmentThresholdGrid.from_validation_dataset(data_module.validation_dataset)

    monitor = DevelopmentVolumeDiceMonitor.from_validation_dataset(
        data_module.validation_dataset,
        score_threshold=config.development_score_threshold,
    )
    inference_config = replace(config, pretrain_weights=None)
    module = IVDTrainingModule(make_model_config(inference_config), make_train_config(inference_config), monitor)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) else None
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"checkpoint has no Lightning state_dict: {checkpoint_path}")
    module.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda")
    module.to(device).eval()
    use_bfloat16 = torch.cuda.is_bf16_supported()
    validation_loader = data_module.val_dataloader()
    with torch.inference_mode():
        for batch_index, (samples, targets) in enumerate(validation_loader, start=1):
            device_samples = samples.to(device, non_blocking=True)
            device_targets = _move_targets_to_device(targets, device)
            precision_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()
            with precision_context:
                outputs = module.model(device_samples)
                original_sizes = torch.stack([target["orig_size"] for target in device_targets])
                results = module.postprocess(outputs, original_sizes)
            threshold_grid.add_batch(results, device_targets)
            if batch_index % 10 == 0:
                print(f"[ivdseg-threshold] validation_batch={batch_index}/{len(validation_loader)}", flush=True)

    selection = threshold_grid.finalize()
    write_development_threshold_selection(selection, output_path)
    print(
        "[ivdseg-threshold] "
        f"selected_score_threshold={selection.selected.score_threshold:.2f} "
        f"macro_3d_dice={selection.selected.macro_dice:.6f}",
        flush=True,
    )
    return selection


def sha256_if_file(path: Path | None) -> str | None:
    """Return an artifact hash without requiring absent pretrained weights to exist."""
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_configuration(
    config: DevelopmentTrainingConfig,
    model_config: ModelConfig,
    train_config: SegmentationTrainConfig,
) -> Path:
    """Persist the exact development run definition before calling ``Trainer.fit``."""
    output_path = config.run_dir / "config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": config.experiment_id,
        "development_training": {
            **asdict(config),
            "manifest_path": str(config.manifest_path),
            "dataset_root": str(config.dataset_root),
            "normalization_profile": str(config.normalization_profile),
            "run_dir": str(config.run_dir),
            "pretrain_weights": str(config.pretrain_weights) if config.pretrain_weights is not None else None,
            "effective_batch_size": config.effective_batch_size,
            "fixed_test_subjects_excluded": ["03", "07", "10", "14"],
        },
        "model_config": model_config.model_dump(mode="json"),
        "rf_detr_train_config": train_config.model_dump(mode="json"),
        "pretrain_weights_sha256": sha256_if_file(config.pretrain_weights),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def write_final_training_configuration(
    config: FinalTrainingConfig,
    model_config: ModelConfig,
    train_config: SegmentationTrainConfig,
) -> Path:
    """Persist the final train-only definition before calling ``Trainer.fit``."""
    output_path = config.run_dir / "config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": config.experiment_id,
        "final_training": {
            **asdict(config),
            "manifest_path": str(config.manifest_path),
            "dataset_root": str(config.dataset_root),
            "normalization_profile": str(config.normalization_profile),
            "run_dir": str(config.run_dir),
            "pretrain_weights": str(config.pretrain_weights) if config.pretrain_weights is not None else None,
            "train_subject_ids": list(FINAL_TRAIN_SUBJECT_IDS),
            "validation_subject_ids": [],
            "effective_batch_size": config.effective_batch_size,
            "fixed_test_subjects_excluded": ["03", "07", "10", "14"],
        },
        "model_config": model_config.model_dump(mode="json"),
        "rf_detr_train_config": train_config.model_dump(mode="json"),
        "pretrain_weights_sha256": sha256_if_file(config.pretrain_weights),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
