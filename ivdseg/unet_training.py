"""Development-only training harness for the B1 12-channel residual U-Net."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
from torch import Tensor
from torch.nn import functional as functional

from ivdseg.datamodule import IVD2p5DSemanticDataset, SemanticIVDDataModule, SliceReference
from ivdseg.resunet import ResUNet34, residual_encoder_depths
from ivdseg.swin_unet import SwinV2SmallUNet, SwinV2TinyUNet


B1_DEVELOPMENT_DICE_METRIC = "val/development_dice_3d"
B1_DEVELOPMENT_TRAIN_SUBJECT_IDS = ("01", "04", "05", "06", "08", "09", "11", "13", "15", "16")
B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS = ("02", "12")
B1_FIXED_TEST_SUBJECTS = ("03", "07", "10", "14")
B1_FINAL_TRAIN_SUBJECT_IDS = ("01", "02", "04", "05", "06", "08", "09", "11", "12", "13", "15", "16")


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    """Calculate binary Dice, treating two empty masks as a perfect match."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(f"Dice inputs must have matching shapes, got {prediction.shape} and {target.shape}")
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(prediction, target).sum() / denominator)


@dataclass(frozen=True)
class SemanticSubjectDice:
    """The reconstructed binary semantic overlap for one development subject."""

    subject_id: str
    dice: float
    prediction_voxels: int
    target_voxels: int
    intersection_voxels: int


@dataclass(frozen=True)
class SemanticDevelopmentDiceSummary:
    """Per-epoch semantic 3D Dice used to select the B1 checkpoint."""

    probability_threshold: float
    macro_dice: float
    subjects: tuple[SemanticSubjectDice, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability_threshold": self.probability_threshold,
            "macro_dice": self.macro_dice,
            "subjects": [asdict(subject) for subject in self.subjects],
        }


class SemanticDevelopmentVolumeDiceMonitor:
    """Reconstruct semantic validation volumes from U-Net logits at a fixed cutoff."""

    def __init__(
        self,
        *,
        references: Sequence[SliceReference],
        semantic_labels: Mapping[str, np.ndarray],
        probability_threshold: float = 0.50,
    ) -> None:
        if not 0.0 <= probability_threshold <= 1.0:
            raise ValueError("probability threshold must be in [0, 1]")
        if not references:
            raise ValueError("development Dice requires at least one validation slice")
        self.references = tuple(references)
        self.semantic_labels = {
            subject_id: np.asarray(label, dtype=bool) for subject_id, label in semantic_labels.items()
        }
        referenced_subjects = {reference.subject_id for reference in self.references}
        if set(self.semantic_labels) != referenced_subjects:
            raise ValueError("semantic labels must cover exactly the validation subject IDs")
        self.probability_threshold = probability_threshold
        self._predictions: dict[str, np.ndarray] = {}
        self._seen_indices: set[int] = set()
        self.reset()

    @classmethod
    def from_validation_dataset(
        cls, dataset: IVD2p5DSemanticDataset, *, probability_threshold: float = 0.50
    ) -> "SemanticDevelopmentVolumeDiceMonitor":
        subject_ids = tuple(dict.fromkeys(reference.subject_id for reference in dataset.references))
        return cls(
            references=dataset.references,
            semantic_labels={subject_id: dataset.semantic_label_for_subject(subject_id) for subject_id in subject_ids},
            probability_threshold=probability_threshold,
        )

    def reset(self) -> None:
        self._predictions = {
            subject_id: np.zeros_like(label, dtype=bool) for subject_id, label in self.semantic_labels.items()
        }
        self._seen_indices = set()

    def add_batch(self, logits: Tensor, sample_indices: Tensor) -> None:
        """Resize logits to native planes and add one prediction for every validation slice."""
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"semantic logits must have shape [B, 1, H, W], got {tuple(logits.shape)}")
        if sample_indices.ndim != 1 or sample_indices.shape[0] != logits.shape[0]:
            raise ValueError("sample indices must contain one value per semantic logit")
        probabilities = torch.sigmoid(logits.detach())
        for batch_index, sample_index in enumerate(sample_indices.detach().cpu().tolist()):
            index = int(sample_index)
            if not 0 <= index < len(self.references):
                raise ValueError(f"validation sample index is outside the slice index: {index}")
            if index in self._seen_indices:
                raise ValueError(f"validation slice {index} was observed more than once")
            self._seen_indices.add(index)
            reference = self.references[index]
            target_volume = self.semantic_labels[reference.subject_id]
            native_plane_shape = target_volume.shape[1:]
            native_probability = functional.interpolate(
                probabilities[batch_index : batch_index + 1],
                size=native_plane_shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            self._predictions[reference.subject_id][reference.slice_index] = (
                native_probability >= self.probability_threshold
            ).cpu().numpy()

    def finalize(self) -> SemanticDevelopmentDiceSummary:
        expected_indices = set(range(len(self.references)))
        if self._seen_indices != expected_indices:
            missing = sorted(expected_indices - self._seen_indices)
            unexpected = sorted(self._seen_indices - expected_indices)
            raise RuntimeError(f"incomplete development reconstruction; missing={missing}, unexpected={unexpected}")
        subjects: list[SemanticSubjectDice] = []
        for subject_id in sorted(self.semantic_labels):
            prediction = self._predictions[subject_id]
            target = self.semantic_labels[subject_id]
            subjects.append(
                SemanticSubjectDice(
                    subject_id=subject_id,
                    dice=binary_dice(prediction, target),
                    prediction_voxels=int(prediction.sum()),
                    target_voxels=int(target.sum()),
                    intersection_voxels=int(np.logical_and(prediction, target).sum()),
                )
            )
        return SemanticDevelopmentDiceSummary(
            probability_threshold=self.probability_threshold,
            macro_dice=float(np.mean([subject.dice for subject in subjects])),
            subjects=tuple(subjects),
        )

    def write_summary(self, output_dir: Path, *, epoch: int) -> Path:
        output_path = Path(output_dir) / "metrics" / "development-3d" / f"epoch-{epoch:03d}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.finalize().to_dict(), indent=2) + "\n", encoding="utf-8")
        return output_path


@dataclass(frozen=True)
class SemanticThresholdSelection:
    """Development-only sigmoid-grid results and their deterministic winner."""

    summaries: tuple[SemanticDevelopmentDiceSummary, ...]
    selected: SemanticDevelopmentDiceSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": [summary.to_dict() for summary in self.summaries],
            "selected_probability_threshold": self.selected.probability_threshold,
            "selected_macro_dice": self.selected.macro_dice,
        }


class SemanticDevelopmentThresholdGrid:
    """Score fixed sigmoid cutoffs from identical B1 validation logits."""

    def __init__(
        self,
        *,
        references: Sequence[SliceReference],
        semantic_labels: Mapping[str, np.ndarray],
        thresholds: Sequence[float] = tuple(round(index * 0.05, 2) for index in range(1, 20)),
    ) -> None:
        thresholds = tuple(float(threshold) for threshold in thresholds)
        if not thresholds or tuple(sorted(thresholds)) != thresholds or len(set(thresholds)) != len(thresholds):
            raise ValueError("threshold grid must be nonempty, sorted, and unique")
        self.monitors = tuple(
            SemanticDevelopmentVolumeDiceMonitor(
                references=references,
                semantic_labels=semantic_labels,
                probability_threshold=threshold,
            )
            for threshold in thresholds
        )

    @classmethod
    def from_validation_dataset(cls, dataset: IVD2p5DSemanticDataset) -> "SemanticDevelopmentThresholdGrid":
        subject_ids = tuple(dict.fromkeys(reference.subject_id for reference in dataset.references))
        return cls(
            references=dataset.references,
            semantic_labels={subject_id: dataset.semantic_label_for_subject(subject_id) for subject_id in subject_ids},
        )

    def add_batch(self, logits: Tensor, sample_indices: Tensor) -> None:
        for monitor in self.monitors:
            monitor.add_batch(logits, sample_indices)

    def finalize(self) -> SemanticThresholdSelection:
        summaries = tuple(monitor.finalize() for monitor in self.monitors)
        selected = max(summaries, key=lambda summary: (summary.macro_dice, -summary.probability_threshold))
        return SemanticThresholdSelection(summaries=summaries, selected=selected)


@dataclass(frozen=True)
class B1DevelopmentConfig:
    """Locked development settings for the residual U-Net baseline."""

    manifest_path: Path
    dataset_root: Path
    normalization_profile: Path
    run_dir: Path
    seed: int
    experiment_id: str = "B1-resunet34"
    max_epochs: int = 100
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    resolution: int = 384
    base_channels: int = 32
    probability_threshold: float = 0.50
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    accelerator: str = "auto"
    num_workers: int = 2
    deterministic: str = "warn"
    progress_report_interval_batches: int = 10

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.micro_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("epochs, micro batch size, and gradient accumulation must be positive")
        if self.micro_batch_size * self.gradient_accumulation_steps != 8:
            raise ValueError("B1 uses effective batch size 8 per device")
        if self.resolution < 16 or self.resolution % 16:
            raise ValueError("ResUNet-34 input resolution must be a positive multiple of 16")
        if self.base_channels < 8:
            raise ValueError("B1 base_channels must be at least 8")
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("probability threshold must be in [0, 1]")
        if self.early_stopping_patience < 1 or self.early_stopping_min_delta < 0.0:
            raise ValueError("early stopping settings must be non-negative and meaningful")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer settings must be non-negative and meaningful")
        if self.deterministic != "warn":
            raise ValueError("B1 uses warn-only CUDA determinism for synchronized affine augmentation")
        if self.progress_report_interval_batches < 1:
            raise ValueError("progress report interval must be positive")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class B1FinalTrainingConfig:
    manifest_path: Path
    dataset_root: Path
    normalization_profile: Path
    run_dir: Path
    seed: int
    experiment_id: str = "B1-resunet34-final"
    max_epochs: int = 39
    selected_probability_threshold: float = 0.50
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    resolution: int = 384
    base_channels: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    accelerator: str = "auto"
    num_workers: int = 2
    deterministic: str = "warn"
    progress_report_interval_batches: int = 10

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def __post_init__(self) -> None:
        B1DevelopmentConfig(
            manifest_path=self.manifest_path, dataset_root=self.dataset_root,
            normalization_profile=self.normalization_profile, run_dir=self.run_dir, seed=self.seed,
            max_epochs=self.max_epochs, micro_batch_size=self.micro_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps, resolution=self.resolution,
            base_channels=self.base_channels, probability_threshold=self.selected_probability_threshold,
            learning_rate=self.learning_rate, weight_decay=self.weight_decay, accelerator=self.accelerator,
            num_workers=self.num_workers, deterministic=self.deterministic,
            progress_report_interval_batches=self.progress_report_interval_batches,
        )


def binary_dice_loss(logits: Tensor, target: Tensor, *, smoothing: float = 1.0) -> Tensor:
    """Soft Dice loss over each image, including target-empty slices stably."""
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"logits must have shape [B, 1, H, W], got {tuple(logits.shape)}")
    if target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("semantic target must have shape [B, H, W] matching logits")
    probability = torch.sigmoid(logits).flatten(1)
    target_float = target.to(dtype=probability.dtype).flatten(1)
    numerator = 2.0 * (probability * target_float).sum(dim=1) + smoothing
    denominator = probability.sum(dim=1) + target_float.sum(dim=1) + smoothing
    return 1.0 - (numerator / denominator).mean()


class B1ResUNetModule(LightningModule):
    """Binary semantic ResUNet with BCE-plus-Dice training loss."""

    def __init__(self, config: B1DevelopmentConfig, monitor: SemanticDevelopmentVolumeDiceMonitor) -> None:
        super().__init__()
        self.config = config
        self.model = ResUNet34(in_channels=12, base_channels=config.base_channels, out_channels=1)
        self.monitor = monitor
        self.save_hyperparameters(
            {
                "experiment_id": config.experiment_id,
                "seed": config.seed,
                "resolution": config.resolution,
                "base_channels": config.base_channels,
                "encoder_stage_depths": list(residual_encoder_depths(self.model)),
                "initialization": "from_scratch",
            }
        )

    @staticmethod
    def _unpack_batch(batch: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        images, masks, indices = batch
        if images.ndim != 4 or images.shape[1] != 12:
            raise ValueError(f"B1 images must have shape [B, 12, H, W], got {tuple(images.shape)}")
        if masks.ndim != 3 or masks.shape[0] != images.shape[0] or masks.shape[1:] != images.shape[-2:]:
            raise ValueError("B1 masks must have shape [B, H, W] matching images")
        return images, masks.bool(), indices

    def _loss(self, logits: Tensor, masks: Tensor) -> Tensor:
        bce = functional.binary_cross_entropy_with_logits(logits[:, 0], masks.to(dtype=logits.dtype))
        dice = binary_dice_loss(logits, masks)
        self.log("train/bce_loss", bce, on_step=False, on_epoch=True, batch_size=logits.shape[0])
        self.log("train/dice_loss", dice, on_step=False, on_epoch=True, batch_size=logits.shape[0])
        return bce + dice

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        images, masks, _indices = self._unpack_batch(batch)
        loss = self._loss(self.model(images), masks)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=images.shape[0])
        return loss

    def on_validation_epoch_start(self) -> None:
        self.monitor.reset()

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> None:
        del batch_idx
        images, masks, indices = self._unpack_batch(batch)
        logits = self.model(images)
        loss = functional.binary_cross_entropy_with_logits(logits[:, 0], masks.to(dtype=logits.dtype)) + binary_dice_loss(
            logits, masks
        )
        self.monitor.add_batch(logits, indices)
        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=images.shape[0])

    def on_validation_epoch_end(self) -> None:
        summary = self.monitor.finalize()
        if self.trainer.is_global_zero:
            self.monitor.write_summary(Path(self.config.run_dir), epoch=self.current_epoch)
        metric = torch.tensor(summary.macro_dice, dtype=torch.float32, device=self.device)
        self.log(B1_DEVELOPMENT_DICE_METRIC, metric, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        for subject in summary.subjects:
            self.log(
                f"val/development_dice_3d_subject_{subject.subject_id}",
                torch.tensor(subject.dice, dtype=torch.float32, device=self.device),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class B1FinalResUNetModule(B1ResUNetModule):
    """Train-only B1 module; validation is deliberately disabled by its trainer."""

    def __init__(self, config: B1FinalTrainingConfig) -> None:
        LightningModule.__init__(self)
        self.config = config
        self.model = ResUNet34(in_channels=12, base_channels=config.base_channels, out_channels=1)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs), "interval": "epoch"}}


class B1PlainTextProgressReporter(Callback):
    """Modal-visible batch/validation progress for the non-RF-DETR baseline."""

    def __init__(self, *, every_n_train_batches: int) -> None:
        if every_n_train_batches < 1:
            raise ValueError("progress interval must be positive")
        self.every_n_train_batches = every_n_train_batches

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        del pl_module, outputs, batch
        completed = batch_idx + 1
        if trainer.is_global_zero and completed % self.every_n_train_batches == 0:
            print(
                f"[ivdseg-progress] epoch={trainer.current_epoch + 1}/{trainer.max_epochs} "
                f"train_batch={completed}/{trainer.num_training_batches} optimizer_step={trainer.global_step}",
                flush=True,
            )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if trainer.is_global_zero:
            value = trainer.callback_metrics.get(B1_DEVELOPMENT_DICE_METRIC)
            text = "unavailable" if value is None else f"{float(value):.6f}"
            print(
                f"[ivdseg-progress] epoch={trainer.current_epoch + 1}/{trainer.max_epochs} "
                f"validation_complete macro_3d_dice={text}",
                flush=True,
            )


def _resolve_precision(accelerator: str) -> str:
    if accelerator == "cpu" or not torch.cuda.is_available():
        return "32-true"
    return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"


def build_b1_development_training(
    config: B1DevelopmentConfig,
) -> tuple[Trainer, B1ResUNetModule, SemanticIVDDataModule, ModelCheckpoint]:
    """Assemble B1 without touching any fixed-holdout subject."""
    seed_everything(config.seed, workers=True)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    data_module = SemanticIVDDataModule(
        manifest=config.manifest_path,
        dataset_root=config.dataset_root,
        normalization_profile=config.normalization_profile,
        train_subject_ids=B1_DEVELOPMENT_TRAIN_SUBJECT_IDS,
        validation_subject_ids=B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS,
        batch_size=config.micro_batch_size,
        resolution=config.resolution,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=torch.cuda.is_available(),
    )
    data_module.setup("fit")
    if data_module.validation_dataset is None:
        raise RuntimeError("B1 validation dataset was not created")
    monitor = SemanticDevelopmentVolumeDiceMonitor.from_validation_dataset(
        data_module.validation_dataset,
        probability_threshold=config.probability_threshold,
    )
    module = B1ResUNetModule(config, monitor)
    checkpoint = ModelCheckpoint(
        dirpath=config.run_dir / "checkpoints",
        filename="best-epoch={epoch:03d}",
        monitor=B1_DEVELOPMENT_DICE_METRIC,
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks: list[Callback] = [
        B1PlainTextProgressReporter(every_n_train_batches=config.progress_report_interval_batches),
        checkpoint,
        EarlyStopping(
            monitor=B1_DEVELOPMENT_DICE_METRIC,
            mode="max",
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            check_finite=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
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
        logger=CSVLogger(save_dir=str(config.run_dir / "logs"), name="", version=""),
        deterministic=config.deterministic,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    return trainer, module, data_module, checkpoint


def build_b1_final_training(config: B1FinalTrainingConfig) -> tuple[Trainer, B1FinalResUNetModule, SemanticIVDDataModule, ModelCheckpoint]:
    seed_everything(config.seed, workers=True)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    data_module = SemanticIVDDataModule(manifest=config.manifest_path, dataset_root=config.dataset_root,
        normalization_profile=config.normalization_profile, train_subject_ids=B1_FINAL_TRAIN_SUBJECT_IDS,
        validation_subject_ids=(), batch_size=config.micro_batch_size, resolution=config.resolution,
        num_workers=config.num_workers, seed=config.seed, pin_memory=torch.cuda.is_available())
    data_module.setup("fit")
    module = B1FinalResUNetModule(config)
    checkpoint = ModelCheckpoint(dirpath=config.run_dir / "checkpoints", save_top_k=0, save_last=True)
    trainer = Trainer(default_root_dir=str(config.run_dir), max_epochs=config.max_epochs, accelerator=config.accelerator,
        devices=1, precision=_resolve_precision(config.accelerator), accumulate_grad_batches=config.gradient_accumulation_steps,
        gradient_clip_val=0.1, callbacks=[B1PlainTextProgressReporter(every_n_train_batches=config.progress_report_interval_batches), checkpoint],
        logger=CSVLogger(save_dir=str(config.run_dir / "logs"), name="", version=""), deterministic=config.deterministic,
        log_every_n_steps=10, limit_val_batches=0, num_sanity_val_steps=0)
    return trainer, module, data_module, checkpoint


def write_b1_development_configuration(config: B1DevelopmentConfig, model: ResUNet34) -> Path:
    """Persist every B1 decision before the trainer can access a batch."""
    output_path = config.run_dir / "config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": config.experiment_id,
        "b1_development": {
            **asdict(config),
            "manifest_path": str(config.manifest_path),
            "dataset_root": str(config.dataset_root),
            "normalization_profile": str(config.normalization_profile),
            "run_dir": str(config.run_dir),
            "train_subject_ids": list(B1_DEVELOPMENT_TRAIN_SUBJECT_IDS),
            "validation_subject_ids": list(B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS),
            "fixed_test_subjects_excluded": list(B1_FIXED_TEST_SUBJECTS),
            "effective_batch_size": config.effective_batch_size,
            "initialization": "from_scratch",
        },
        "model": {
            "name": "ResUNet34",
            "in_channels": model.in_channels,
            "out_channels": model.out_channels,
            "encoder_channels": list(model.encoder_channels),
            "encoder_stage_depths": list(residual_encoder_depths(model)),
            "normalization": "GroupNorm",
        },
        "loss": "BCEWithLogits + soft Dice (equal weights)",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "scheduler": "CosineAnnealingLR",
        },
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def load_b1_development_config(path: Path) -> B1DevelopmentConfig:
    """Restore one immutable B1 development definition for post-training scoring."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("b1_development")
    if payload.get("schema_version") != 1 or not isinstance(raw, Mapping):
        raise ValueError("run configuration lacks supported b1_development settings")
    supported = {field.name for field in fields(B1DevelopmentConfig) if field.init}
    values = {key: value for key, value in raw.items() if key in supported}
    for field_name in ("manifest_path", "dataset_root", "normalization_profile", "run_dir"):
        value = values.get(field_name)
        if not isinstance(value, str):
            raise ValueError(f"run configuration has no valid {field_name}")
        values[field_name] = Path(value)
    config = B1DevelopmentConfig(**values)
    if tuple(raw.get("train_subject_ids", ())) != B1_DEVELOPMENT_TRAIN_SUBJECT_IDS:
        raise ValueError("B1 run configuration has unexpected training subject IDs")
    if tuple(raw.get("validation_subject_ids", ())) != B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS:
        raise ValueError("B1 run configuration has unexpected development subject IDs")
    if tuple(raw.get("fixed_test_subjects_excluded", ())) != B1_FIXED_TEST_SUBJECTS:
        raise ValueError("B1 run configuration has unexpected fixed-test exclusions")
    return config


def load_b1_final_training_config(path: Path) -> B1FinalTrainingConfig:
    """Restore the immutable train-only B1 definition for holdout inference."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("b1_final_training")
    if payload.get("schema_version") != 1 or not isinstance(raw, Mapping):
        raise ValueError("run configuration lacks supported b1_final_training settings")
    supported = {field.name for field in fields(B1FinalTrainingConfig) if field.init}
    values = {key: value for key, value in raw.items() if key in supported}
    for field_name in ("manifest_path", "dataset_root", "normalization_profile", "run_dir"):
        value = values.get(field_name)
        if not isinstance(value, str):
            raise ValueError(f"run configuration has no valid {field_name}")
        values[field_name] = Path(value)
    config = B1FinalTrainingConfig(**values)
    if tuple(raw.get("train_subject_ids", ())) != B1_FINAL_TRAIN_SUBJECT_IDS:
        raise ValueError("B1 final configuration has unexpected training subject IDs")
    if tuple(raw.get("validation_subject_ids", ())) != ():
        raise ValueError("B1 final configuration unexpectedly has validation subject IDs")
    if tuple(raw.get("fixed_test_subjects_excluded", ())) != B1_FIXED_TEST_SUBJECTS:
        raise ValueError("B1 final configuration has unexpected fixed-test exclusions")
    if raw.get("effective_batch_size") != config.effective_batch_size:
        raise ValueError("B1 final configuration has an unexpected effective batch size")
    if raw.get("initialization") != "from_scratch":
        raise ValueError("B1 final configuration has an unexpected initialization")
    return config


def load_b1_model(
    config: B1DevelopmentConfig | B1FinalTrainingConfig,
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> ResUNet34:
    """Restore the exact B1 semantic model without creating a trainer or data loader."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, Mapping):
        raise ValueError("B1 checkpoint lacks a Lightning state_dict")
    state = {key.removeprefix("model."): value for key, value in raw_state.items() if key.startswith("model.")}
    model = ResUNet34(in_channels=12, base_channels=config.base_channels, out_channels=1).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


B3_DEVELOPMENT_DICE_METRIC = "val/development_dice_3d"


@dataclass(frozen=True)
class B3DevelopmentConfig:
    """Locked development settings for the pretrained B3 Swin V2 Tiny U-Net."""

    manifest_path: Path
    dataset_root: Path
    normalization_profile: Path
    run_dir: Path
    seed: int
    experiment_id: str = "B3-swinv2-tiny-pretrained"
    max_epochs: int = 100
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    resolution: int = 384
    encoder_learning_rate: float = 1e-4
    decoder_learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    accelerator: str = "auto"
    num_workers: int = 2
    deterministic: str = "warn"
    progress_report_interval_batches: int = 10
    pretrained: bool = True

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.micro_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("epochs, micro batch size, and gradient accumulation must be positive")
        if self.micro_batch_size * self.gradient_accumulation_steps != 8:
            raise ValueError("B3 uses effective batch size 8 per device")
        if self.resolution < 32 or self.resolution % 32:
            raise ValueError("Swin V2 Tiny input resolution must be a positive multiple of 32")
        if self.encoder_learning_rate <= 0.0 or self.decoder_learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("B3 optimizer settings must be positive except non-negative weight decay")
        if self.early_stopping_patience < 1 or self.early_stopping_min_delta < 0.0:
            raise ValueError("B3 early stopping settings must be non-negative and meaningful")
        if self.deterministic != "warn":
            raise ValueError("B3 uses warn-only CUDA determinism for synchronized affine augmentation")
        if self.progress_report_interval_batches < 1:
            raise ValueError("progress report interval must be positive")
        if not self.pretrained:
            raise ValueError("B3 is locked to ImageNet-pretrained Swin V2 Tiny initialization")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


class B3SwinUNetModule(LightningModule):
    """Fine-tune a pretrained Swin V2 Tiny U-Net with B1's semantic objective."""

    def __init__(
        self,
        config: B3DevelopmentConfig,
        monitor: SemanticDevelopmentVolumeDiceMonitor,
        *,
        model_class: type[SwinV2TinyUNet] | type[SwinV2SmallUNet] = SwinV2TinyUNet,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model_class(pretrained=config.pretrained)
        self.monitor = monitor
        self.save_hyperparameters(
            {
                "experiment_id": config.experiment_id,
                "seed": config.seed,
                "resolution": config.resolution,
                "encoder": f"torchvision {self.model.architecture_name}",
                "pretrained_weights": self.model.pretrained_weights_name,
                "input_stem": "3-to-12 channel replicated-and-scaled pretrained patch projection",
            }
        )

    def _loss(self, logits: Tensor, masks: Tensor) -> Tensor:
        bce = functional.binary_cross_entropy_with_logits(logits[:, 0], masks.to(dtype=logits.dtype))
        dice = binary_dice_loss(logits, masks)
        self.log("train/bce_loss", bce, on_step=False, on_epoch=True, batch_size=logits.shape[0])
        self.log("train/dice_loss", dice, on_step=False, on_epoch=True, batch_size=logits.shape[0])
        return bce + dice

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        images, masks, _indices = B1ResUNetModule._unpack_batch(batch)
        loss = self._loss(self.model(images), masks)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=images.shape[0])
        return loss

    def on_validation_epoch_start(self) -> None:
        self.monitor.reset()

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> None:
        del batch_idx
        images, masks, indices = B1ResUNetModule._unpack_batch(batch)
        logits = self.model(images)
        loss = functional.binary_cross_entropy_with_logits(logits[:, 0], masks.to(dtype=logits.dtype)) + binary_dice_loss(
            logits, masks
        )
        self.monitor.add_batch(logits, indices)
        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=images.shape[0])

    def on_validation_epoch_end(self) -> None:
        summary = self.monitor.finalize()
        if self.trainer.is_global_zero:
            self.monitor.write_summary(Path(self.config.run_dir), epoch=self.current_epoch)
        metric = torch.tensor(summary.macro_dice, dtype=torch.float32, device=self.device)
        self.log(B3_DEVELOPMENT_DICE_METRIC, metric, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        for subject in summary.subjects:
            self.log(
                f"val/development_dice_3d_subject_{subject.subject_id}",
                torch.tensor(subject.dice, dtype=torch.float32, device=self.device),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    def configure_optimizers(self) -> dict[str, Any]:
        encoder_parameters = list(self.model.encoder.parameters()) + list(self.model.encoder_norm.parameters())
        encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
        decoder_parameters = [
            parameter for parameter in self.model.parameters() if id(parameter) not in encoder_parameter_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": self.config.encoder_learning_rate},
                {"params": decoder_parameters, "lr": self.config.decoder_learning_rate},
            ],
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def build_b3_development_training(
    config: B3DevelopmentConfig,
) -> tuple[Trainer, B3SwinUNetModule, SemanticIVDDataModule, ModelCheckpoint]:
    """Assemble pretrained B3 while preserving the fixed holdout exclusion."""
    seed_everything(config.seed, workers=True)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    data_module = SemanticIVDDataModule(
        manifest=config.manifest_path,
        dataset_root=config.dataset_root,
        normalization_profile=config.normalization_profile,
        train_subject_ids=B1_DEVELOPMENT_TRAIN_SUBJECT_IDS,
        validation_subject_ids=B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS,
        batch_size=config.micro_batch_size,
        resolution=config.resolution,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=torch.cuda.is_available(),
    )
    data_module.setup("fit")
    if data_module.validation_dataset is None:
        raise RuntimeError("B3 validation dataset was not created")
    monitor = SemanticDevelopmentVolumeDiceMonitor.from_validation_dataset(
        data_module.validation_dataset,
        probability_threshold=0.50,
    )
    module = B3SwinUNetModule(config, monitor)
    checkpoint = ModelCheckpoint(
        dirpath=config.run_dir / "checkpoints",
        filename="best-epoch={epoch:03d}",
        monitor=B3_DEVELOPMENT_DICE_METRIC,
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks: list[Callback] = [
        B1PlainTextProgressReporter(every_n_train_batches=config.progress_report_interval_batches),
        checkpoint,
        EarlyStopping(
            monitor=B3_DEVELOPMENT_DICE_METRIC,
            mode="max",
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            check_finite=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
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
        logger=CSVLogger(save_dir=str(config.run_dir / "logs"), name="", version=""),
        deterministic=config.deterministic,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    return trainer, module, data_module, checkpoint


def write_b3_development_configuration(config: B3DevelopmentConfig, model: SwinV2TinyUNet) -> Path:
    """Persist B3's pretrained initialization before training accesses a batch."""
    output_path = config.run_dir / "config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": config.experiment_id,
        "b3_development": {
            **asdict(config),
            "manifest_path": str(config.manifest_path),
            "dataset_root": str(config.dataset_root),
            "normalization_profile": str(config.normalization_profile),
            "run_dir": str(config.run_dir),
            "train_subject_ids": list(B1_DEVELOPMENT_TRAIN_SUBJECT_IDS),
            "validation_subject_ids": list(B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS),
            "fixed_test_subjects_excluded": list(B1_FIXED_TEST_SUBJECTS),
            "effective_batch_size": config.effective_batch_size,
        },
        "model": {
            "name": "SwinV2TinyUNet",
            "encoder": "torchvision Swin V2 Tiny",
            "pretrained_weights": "Swin_V2_T_Weights.IMAGENET1K_V1",
            "pretrained_weight_url": "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth",
            "in_channels": model.in_channels,
            "out_channels": model.out_channels,
            "encoder_channels": list(model.encoder_channels),
            "input_stem": "3-to-12 channel replicated-and-scaled pretrained patch projection",
            "decoder": "three Swin skip-fusion stages plus two full-resolution upsampling stages",
        },
        "loss": "BCEWithLogits + soft Dice (equal weights)",
        "optimizer": {
            "name": "AdamW",
            "encoder_learning_rate": config.encoder_learning_rate,
            "decoder_learning_rate": config.decoder_learning_rate,
            "weight_decay": config.weight_decay,
            "scheduler": "CosineAnnealingLR",
        },
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def load_b3_development_config(path: Path) -> B3DevelopmentConfig:
    """Restore one immutable pretrained B3 development definition for scoring."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("b3_development")
    if payload.get("schema_version") != 1 or not isinstance(raw, Mapping):
        raise ValueError("run configuration lacks supported b3_development settings")
    supported = {field.name for field in fields(B3DevelopmentConfig) if field.init}
    values = {key: value for key, value in raw.items() if key in supported}
    for field_name in ("manifest_path", "dataset_root", "normalization_profile", "run_dir"):
        value = values.get(field_name)
        if not isinstance(value, str):
            raise ValueError(f"run configuration has no valid {field_name}")
        values[field_name] = Path(value)
    config = B3DevelopmentConfig(**values)
    if tuple(raw.get("train_subject_ids", ())) != B1_DEVELOPMENT_TRAIN_SUBJECT_IDS:
        raise ValueError("B3 run configuration has unexpected training subject IDs")
    if tuple(raw.get("validation_subject_ids", ())) != B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS:
        raise ValueError("B3 run configuration has unexpected development subject IDs")
    if tuple(raw.get("fixed_test_subjects_excluded", ())) != B1_FIXED_TEST_SUBJECTS:
        raise ValueError("B3 run configuration has unexpected fixed-test exclusions")
    if raw.get("effective_batch_size") != config.effective_batch_size:
        raise ValueError("B3 run configuration has an unexpected effective batch size")
    return config


def load_b3_model(config: B3DevelopmentConfig, checkpoint_path: Path, *, device: torch.device) -> SwinV2TinyUNet:
    """Restore B3 from its checkpoint without re-downloading ImageNet weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, Mapping):
        raise ValueError("B3 checkpoint lacks a Lightning state_dict")
    state = {key.removeprefix("model."): value for key, value in raw_state.items() if key.startswith("model.")}
    model = SwinV2TinyUNet(pretrained=False).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


B4_DEVELOPMENT_DICE_METRIC = B3_DEVELOPMENT_DICE_METRIC


@dataclass(frozen=True)
class B4DevelopmentConfig:
    """Locked development settings for the pretrained B4 Swin V2 Small U-Net."""

    manifest_path: Path
    dataset_root: Path
    normalization_profile: Path
    run_dir: Path
    seed: int
    experiment_id: str = "B4-swinv2-small-pretrained"
    max_epochs: int = 100
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    resolution: int = 384
    encoder_learning_rate: float = 5e-5
    decoder_learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    accelerator: str = "auto"
    num_workers: int = 2
    deterministic: str = "warn"
    progress_report_interval_batches: int = 10
    pretrained: bool = True

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.micro_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("epochs, micro batch size, and gradient accumulation must be positive")
        if self.micro_batch_size * self.gradient_accumulation_steps != 8:
            raise ValueError("B4 uses effective batch size 8 per device")
        if self.resolution < 32 or self.resolution % 32:
            raise ValueError("Swin V2 Small input resolution must be a positive multiple of 32")
        if self.encoder_learning_rate <= 0.0 or self.decoder_learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("B4 optimizer settings must be positive except non-negative weight decay")
        if self.early_stopping_patience < 1 or self.early_stopping_min_delta < 0.0:
            raise ValueError("B4 early stopping settings must be non-negative and meaningful")
        if self.deterministic != "warn":
            raise ValueError("B4 uses warn-only CUDA determinism for synchronized affine augmentation")
        if self.progress_report_interval_batches < 1:
            raise ValueError("progress report interval must be positive")
        if not self.pretrained:
            raise ValueError("B4 is locked to ImageNet-pretrained Swin V2 Small initialization")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


class B4SwinUNetModule(B3SwinUNetModule):
    """Fine-tune a pretrained Swin V2 Small U-Net with B3's semantic objective."""

    def __init__(self, config: B4DevelopmentConfig, monitor: SemanticDevelopmentVolumeDiceMonitor) -> None:
        super().__init__(config, monitor, model_class=SwinV2SmallUNet)  # type: ignore[arg-type]


def build_b4_development_training(
    config: B4DevelopmentConfig,
) -> tuple[Trainer, B4SwinUNetModule, SemanticIVDDataModule, ModelCheckpoint]:
    """Assemble pretrained B4 while preserving the fixed holdout exclusion."""
    seed_everything(config.seed, workers=True)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    data_module = SemanticIVDDataModule(
        manifest=config.manifest_path,
        dataset_root=config.dataset_root,
        normalization_profile=config.normalization_profile,
        train_subject_ids=B1_DEVELOPMENT_TRAIN_SUBJECT_IDS,
        validation_subject_ids=B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS,
        batch_size=config.micro_batch_size,
        resolution=config.resolution,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=torch.cuda.is_available(),
    )
    data_module.setup("fit")
    if data_module.validation_dataset is None:
        raise RuntimeError("B4 validation dataset was not created")
    monitor = SemanticDevelopmentVolumeDiceMonitor.from_validation_dataset(
        data_module.validation_dataset,
        probability_threshold=0.50,
    )
    module = B4SwinUNetModule(config, monitor)
    checkpoint = ModelCheckpoint(
        dirpath=config.run_dir / "checkpoints",
        filename="best-epoch={epoch:03d}",
        monitor=B4_DEVELOPMENT_DICE_METRIC,
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks: list[Callback] = [
        B1PlainTextProgressReporter(every_n_train_batches=config.progress_report_interval_batches),
        checkpoint,
        EarlyStopping(
            monitor=B4_DEVELOPMENT_DICE_METRIC,
            mode="max",
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            check_finite=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
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
        logger=CSVLogger(save_dir=str(config.run_dir / "logs"), name="", version=""),
        deterministic=config.deterministic,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    return trainer, module, data_module, checkpoint


def write_b4_development_configuration(config: B4DevelopmentConfig, model: SwinV2SmallUNet) -> Path:
    """Persist B4's pretrained initialization before training accesses a batch."""
    output_path = config.run_dir / "config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": config.experiment_id,
        "b4_development": {
            **asdict(config),
            "manifest_path": str(config.manifest_path),
            "dataset_root": str(config.dataset_root),
            "normalization_profile": str(config.normalization_profile),
            "run_dir": str(config.run_dir),
            "train_subject_ids": list(B1_DEVELOPMENT_TRAIN_SUBJECT_IDS),
            "validation_subject_ids": list(B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS),
            "fixed_test_subjects_excluded": list(B1_FIXED_TEST_SUBJECTS),
            "effective_batch_size": config.effective_batch_size,
        },
        "model": {
            "name": "SwinV2SmallUNet",
            "encoder": "torchvision Swin V2 Small",
            "pretrained_weights": model.pretrained_weights_name,
            "pretrained_weight_url": model.pretrained_weight_url,
            "in_channels": model.in_channels,
            "out_channels": model.out_channels,
            "encoder_channels": list(model.encoder_channels),
            "encoder_depths": [2, 2, 18, 2],
            "input_stem": "3-to-12 channel replicated-and-scaled pretrained patch projection",
            "decoder": "three Swin skip-fusion stages plus two full-resolution upsampling stages",
        },
        "loss": "BCEWithLogits + soft Dice (equal weights)",
        "optimizer": {
            "name": "AdamW",
            "encoder_learning_rate": config.encoder_learning_rate,
            "decoder_learning_rate": config.decoder_learning_rate,
            "weight_decay": config.weight_decay,
            "scheduler": "CosineAnnealingLR",
        },
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def load_b4_development_config(path: Path) -> B4DevelopmentConfig:
    """Restore one immutable pretrained B4 development definition for scoring."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("b4_development")
    if payload.get("schema_version") != 1 or not isinstance(raw, Mapping):
        raise ValueError("run configuration lacks supported b4_development settings")
    supported = {field.name for field in fields(B4DevelopmentConfig) if field.init}
    values = {key: value for key, value in raw.items() if key in supported}
    for field_name in ("manifest_path", "dataset_root", "normalization_profile", "run_dir"):
        value = values.get(field_name)
        if not isinstance(value, str):
            raise ValueError(f"run configuration has no valid {field_name}")
        values[field_name] = Path(value)
    config = B4DevelopmentConfig(**values)
    if tuple(raw.get("train_subject_ids", ())) != B1_DEVELOPMENT_TRAIN_SUBJECT_IDS:
        raise ValueError("B4 run configuration has unexpected training subject IDs")
    if tuple(raw.get("validation_subject_ids", ())) != B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS:
        raise ValueError("B4 run configuration has unexpected development subject IDs")
    if tuple(raw.get("fixed_test_subjects_excluded", ())) != B1_FIXED_TEST_SUBJECTS:
        raise ValueError("B4 run configuration has unexpected fixed-test exclusions")
    if raw.get("effective_batch_size") != config.effective_batch_size:
        raise ValueError("B4 run configuration has an unexpected effective batch size")
    return config


def load_b4_model(config: B4DevelopmentConfig, checkpoint_path: Path, *, device: torch.device) -> SwinV2SmallUNet:
    """Restore B4 from its checkpoint without re-downloading ImageNet weights."""
    del config
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, Mapping):
        raise ValueError("B4 checkpoint lacks a Lightning state_dict")
    state = {key.removeprefix("model."): value for key, value in raw_state.items() if key.startswith("model.")}
    model = SwinV2SmallUNet(pretrained=False).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@dataclass(frozen=True)
class B3FinalTrainingConfig:
    """Immutable all-permitted-subject final definition for selected B3."""

    manifest_path: Path
    dataset_root: Path
    normalization_profile: Path
    run_dir: Path
    seed: int
    experiment_id: str = "B3-swinv2-tiny-pretrained-final"
    max_epochs: int = 18
    selected_probability_threshold: float = 0.35
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    resolution: int = 384
    encoder_learning_rate: float = 1e-4
    decoder_learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    accelerator: str = "auto"
    num_workers: int = 2
    deterministic: str = "warn"
    progress_report_interval_batches: int = 10
    pretrained: bool = True

    def __post_init__(self) -> None:
        B3DevelopmentConfig(
            manifest_path=self.manifest_path,
            dataset_root=self.dataset_root,
            normalization_profile=self.normalization_profile,
            run_dir=self.run_dir,
            seed=self.seed,
            experiment_id=self.experiment_id,
            max_epochs=self.max_epochs,
            micro_batch_size=self.micro_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            resolution=self.resolution,
            encoder_learning_rate=self.encoder_learning_rate,
            decoder_learning_rate=self.decoder_learning_rate,
            weight_decay=self.weight_decay,
            accelerator=self.accelerator,
            num_workers=self.num_workers,
            deterministic=self.deterministic,
            progress_report_interval_batches=self.progress_report_interval_batches,
            pretrained=self.pretrained,
        )
        if not 0.0 <= self.selected_probability_threshold <= 1.0:
            raise ValueError("B3 selected probability threshold must be in [0, 1]")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


def _b3_optimizer_parameters(
    model: SwinV2TinyUNet,
    *,
    encoder_learning_rate: float,
    decoder_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Create B3's explicit pretrained-encoder/new-decoder AdamW groups."""
    encoder_parameters = list(model.encoder.parameters()) + list(model.encoder_norm.parameters())
    encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_parameter_ids]
    return torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": encoder_learning_rate},
            {"params": decoder_parameters, "lr": decoder_learning_rate},
        ],
        weight_decay=weight_decay,
    )


class B3FinalSwinUNetModule(LightningModule):
    """Train-only B3 module that restarts from the cached ImageNet encoder."""

    def __init__(self, config: B3FinalTrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.model = SwinV2TinyUNet(pretrained=config.pretrained)

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        images, masks, _indices = B1ResUNetModule._unpack_batch(batch)
        logits = self.model(images)
        bce = functional.binary_cross_entropy_with_logits(logits[:, 0], masks.to(dtype=logits.dtype))
        dice = binary_dice_loss(logits, masks)
        self.log("train/bce_loss", bce, on_step=False, on_epoch=True, batch_size=images.shape[0])
        self.log("train/dice_loss", dice, on_step=False, on_epoch=True, batch_size=images.shape[0])
        loss = bce + dice
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=images.shape[0])
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = _b3_optimizer_parameters(
            self.model,
            encoder_learning_rate=self.config.encoder_learning_rate,
            decoder_learning_rate=self.config.decoder_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def build_b3_final_training(
    config: B3FinalTrainingConfig,
) -> tuple[Trainer, B3FinalSwinUNetModule, SemanticIVDDataModule, ModelCheckpoint]:
    """Build fresh B3 ImageNet-initialized final training with no validation loop."""
    seed_everything(config.seed, workers=True)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    data_module = SemanticIVDDataModule(
        manifest=config.manifest_path,
        dataset_root=config.dataset_root,
        normalization_profile=config.normalization_profile,
        train_subject_ids=B1_FINAL_TRAIN_SUBJECT_IDS,
        validation_subject_ids=(),
        batch_size=config.micro_batch_size,
        resolution=config.resolution,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=torch.cuda.is_available(),
    )
    data_module.setup("fit")
    module = B3FinalSwinUNetModule(config)
    checkpoint = ModelCheckpoint(dirpath=config.run_dir / "checkpoints", save_top_k=0, save_last=True)
    trainer = Trainer(
        default_root_dir=str(config.run_dir),
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=1,
        strategy="auto",
        precision=_resolve_precision(config.accelerator),
        accumulate_grad_batches=config.gradient_accumulation_steps,
        gradient_clip_val=0.1,
        callbacks=[B1PlainTextProgressReporter(every_n_train_batches=config.progress_report_interval_batches), checkpoint],
        logger=CSVLogger(save_dir=str(config.run_dir / "logs"), name="", version=""),
        deterministic=config.deterministic,
        log_every_n_steps=10,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    return trainer, module, data_module, checkpoint


def write_b3_final_training_configuration(config: B3FinalTrainingConfig) -> Path:
    """Persist B3's all-training-subject final definition before fitting."""
    output_path = config.run_dir / "config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": config.experiment_id,
        "b3_final_training": {
            **asdict(config),
            "manifest_path": str(config.manifest_path),
            "dataset_root": str(config.dataset_root),
            "normalization_profile": str(config.normalization_profile),
            "run_dir": str(config.run_dir),
            "train_subject_ids": list(B1_FINAL_TRAIN_SUBJECT_IDS),
            "validation_subject_ids": [],
            "fixed_test_subjects_excluded": list(B1_FIXED_TEST_SUBJECTS),
            "effective_batch_size": config.effective_batch_size,
            "initialization": "ImageNet Swin_V2_T_Weights.IMAGENET1K_V1",
        },
        "model": {
            "name": "SwinV2TinyUNet",
            "encoder": "torchvision Swin V2 Tiny",
            "pretrained_weights": "Swin_V2_T_Weights.IMAGENET1K_V1",
            "pretrained_weight_url": "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth",
            "input_stem": "3-to-12 channel replicated-and-scaled pretrained patch projection",
        },
        "loss": "BCEWithLogits + soft Dice (equal weights)",
        "optimizer": {
            "name": "AdamW",
            "encoder_learning_rate": config.encoder_learning_rate,
            "decoder_learning_rate": config.decoder_learning_rate,
            "weight_decay": config.weight_decay,
            "scheduler": "CosineAnnealingLR",
        },
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def load_b3_final_training_config(path: Path) -> B3FinalTrainingConfig:
    """Restore B3's immutable all-training-subject definition for evaluation."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("b3_final_training")
    if payload.get("schema_version") != 1 or not isinstance(raw, Mapping):
        raise ValueError("run configuration lacks supported b3_final_training settings")
    supported = {field.name for field in fields(B3FinalTrainingConfig) if field.init}
    values = {key: value for key, value in raw.items() if key in supported}
    for field_name in ("manifest_path", "dataset_root", "normalization_profile", "run_dir"):
        value = values.get(field_name)
        if not isinstance(value, str):
            raise ValueError(f"run configuration has no valid {field_name}")
        values[field_name] = Path(value)
    config = B3FinalTrainingConfig(**values)
    if tuple(raw.get("train_subject_ids", ())) != B1_FINAL_TRAIN_SUBJECT_IDS:
        raise ValueError("B3 final configuration has unexpected training subject IDs")
    if tuple(raw.get("validation_subject_ids", ())) != ():
        raise ValueError("B3 final configuration unexpectedly has validation subject IDs")
    if tuple(raw.get("fixed_test_subjects_excluded", ())) != B1_FIXED_TEST_SUBJECTS:
        raise ValueError("B3 final configuration has unexpected fixed-test exclusions")
    if raw.get("effective_batch_size") != config.effective_batch_size:
        raise ValueError("B3 final configuration has an unexpected effective batch size")
    if raw.get("initialization") != "ImageNet Swin_V2_T_Weights.IMAGENET1K_V1":
        raise ValueError("B3 final configuration has an unexpected initialization")
    return config


def load_b3_final_model(
    config: B3FinalTrainingConfig,
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> SwinV2TinyUNet:
    """Restore the final B3 checkpoint without a pretrained-weight download."""
    del config
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, Mapping):
        raise ValueError("B3 final checkpoint lacks a Lightning state_dict")
    state = {key.removeprefix("model."): value for key, value in raw_state.items() if key.startswith("model.")}
    model = SwinV2TinyUNet(pretrained=False).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
