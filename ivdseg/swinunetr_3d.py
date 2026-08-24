"""Small-budget, four-modality 3D pretrained Swin UNETR development harness.

The module deliberately owns the volumetric seam: it reads only canonical NIfTI
volumes, samples deterministic foreground-biased 3D patches for training, and
uses sliding-window inference for subject-volume development selection.  It is
separate from the 2.5D U-Net code so the two protocols cannot be mixed by
accident.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
from torch import Tensor
from torch.nn import functional as functional
from torch.utils.data import DataLoader, Dataset

from ivdseg.manifest import FIXED_TEST_SUBJECTS, MODALITY_ORDER
from ivdseg.normalization import NormalizationProfile, load_profile, normalize_modalities
from ivdseg.spatial import load_canonical_subject
from ivdseg.unet_training import B1_DEVELOPMENT_TRAIN_SUBJECT_IDS, B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS, binary_dice


B5_DEVELOPMENT_DICE_METRIC = "val/development_dice_3d"
B5_PRETRAINED_WEIGHTS_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/ssl_pretrained_weights.pth"
)
B5_PRETRAINED_WEIGHTS_NAME = "ssl_pretrained_weights.pth"
B5_PRIMARY_ROI_SIZE = (32, 256, 256)
B5_FALLBACK_ROI_SIZE = (64, 128, 128)
B5_ALLOWED_ROI_SIZES = (B5_PRIMARY_ROI_SIZE, B5_FALLBACK_ROI_SIZE)
B5_REFERENCE_DEVELOPMENT_DICE = 0.931567
B5_REQUIRED_DEVELOPMENT_GAIN = 0.003


def _resolve_precision(accelerator: str) -> str:
    if accelerator == "cpu" or not torch.cuda.is_available():
        return "32-true"
    return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"


@dataclass(frozen=True)
class B5DevelopmentConfig:
    """Immutable development-only definition for the B5 volumetric experiment."""

    manifest_path: Path
    dataset_root: Path
    normalization_profile: Path
    run_dir: Path
    pretrained_weights_path: Path
    seed: int
    experiment_id: str = "B5-swinunetr-3d-pretrained"
    roi_size: tuple[int, int, int] = B5_PRIMARY_ROI_SIZE
    max_epochs: int = 20
    patches_per_epoch: int = 80
    micro_batch_size: int = 1
    foreground_patch_probability: float = 0.75
    encoder_learning_rate: float = 2e-5
    decoder_learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 0.001
    probability_threshold: float = 0.50
    sliding_window_overlap: float = 0.50
    accelerator: str = "auto"
    num_workers: int = 0
    deterministic: str = "warn"
    progress_report_interval_batches: int = 20
    feature_size: int = 48
    use_checkpoint: bool = True
    pretrained: bool = True

    def __post_init__(self) -> None:
        if tuple(self.roi_size) not in B5_ALLOWED_ROI_SIZES:
            raise ValueError(f"B5 ROI must be one of {B5_ALLOWED_ROI_SIZES}, got {self.roi_size}")
        if any(dimension % 32 for dimension in self.roi_size):
            raise ValueError("Swin UNETR ROI dimensions must all be divisible by 32")
        if self.max_epochs < 1 or self.patches_per_epoch < 1 or self.micro_batch_size != 1:
            raise ValueError("B5 uses positive epoch/patch counts and a fixed micro-batch size of one")
        if not 0.0 < self.foreground_patch_probability <= 1.0:
            raise ValueError("foreground patch probability must be in (0, 1]")
        if self.encoder_learning_rate <= 0 or self.decoder_learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("B5 optimizer settings are invalid")
        if self.early_stopping_patience < 1 or self.early_stopping_min_delta < 0:
            raise ValueError("B5 early stopping settings are invalid")
        if not 0.0 <= self.probability_threshold <= 1.0 or not 0.0 <= self.sliding_window_overlap < 1.0:
            raise ValueError("B5 probability and sliding-window settings are invalid")
        if self.num_workers != 0:
            raise ValueError("B5 preloads its small development cohort and intentionally uses num_workers=0")
        if self.deterministic != "warn":
            raise ValueError("B5 uses warn-only CUDA determinism")
        if self.progress_report_interval_batches < 1 or self.feature_size != 48:
            raise ValueError("B5 is locked to progress reporting and the MONAI Base feature size of 48")
        if not self.use_checkpoint or not self.pretrained:
            raise ValueError("B5 is locked to checkpointed, SSL-pretrained Swin UNETR")


@dataclass(frozen=True)
class B5Volume:
    """One normalized native-grid four-modality volume held in CPU memory."""

    subject_id: str
    image: np.ndarray
    target: np.ndarray

    def __post_init__(self) -> None:
        if self.image.ndim != 4 or self.image.shape[0] != len(MODALITY_ORDER):
            raise ValueError("B5 volumes must have [4, D, H, W] images")
        if self.target.ndim != 3 or self.image.shape[1:] != self.target.shape:
            raise ValueError("B5 image and semantic target grids must match")


def _read_manifest(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or tuple(payload.get("fixed_test_subjects", ())) != FIXED_TEST_SUBJECTS:
        raise ValueError("B5 manifest is missing the immutable fixed-test definition")
    return payload


def _records_for_ids(manifest: Mapping[str, Any], subject_ids: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
    raw_subjects = manifest.get("subjects")
    if not isinstance(raw_subjects, list):
        raise ValueError("B5 manifest is missing subject records")
    records = {record.get("subject_id"): record for record in raw_subjects if isinstance(record, Mapping)}
    if len(records) != len(raw_subjects):
        raise ValueError("B5 manifest has missing or duplicate subject IDs")
    if set(subject_ids) & set(FIXED_TEST_SUBJECTS):
        raise ValueError("B5 development selection may not include fixed-test subjects")
    selected: list[Mapping[str, Any]] = []
    for subject_id in subject_ids:
        record = records.get(subject_id)
        if not isinstance(record, Mapping) or record.get("partition") != "train":
            raise ValueError(f"B5 subject {subject_id} is unavailable in the training partition")
        selected.append(record)
    return tuple(selected)


def load_b5_volume(record: Mapping[str, Any], dataset_root: Path, profile: NormalizationProfile) -> B5Volume:
    """Load canonical NIfTI data and stack normalized modalities in manifest order."""
    subject = load_canonical_subject(record, dataset_root)
    modalities = {
        modality: np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
        for modality, image in subject.modalities.items()
    }
    normalized = normalize_modalities(modalities, profile)
    image = np.stack([normalized[modality] for modality in MODALITY_ORDER], axis=0, dtype=np.float32)
    target = np.asarray(subject.label.get_fdata(dtype=np.float32), dtype=np.float32) > 0.5
    return B5Volume(subject_id=subject.subject_id, image=image, target=target)


def _pad_volume(image: np.ndarray, target: np.ndarray, roi_size: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    padding = tuple(max(roi_size[index] - target.shape[index], 0) for index in range(3))
    if not any(padding):
        return image, target
    image_padding = tuple((amount // 2, amount - amount // 2) for amount in padding)
    return (
        np.pad(image, ((0, 0), *image_padding), mode="constant"),
        np.pad(target, image_padding, mode="constant"),
    )


class B5PatchDataset(Dataset[dict[str, Tensor]]):
    """Deterministic foreground-biased patches over the ten permitted train volumes."""

    def __init__(
        self,
        volumes: Sequence[B5Volume],
        *,
        roi_size: tuple[int, int, int],
        patches_per_epoch: int,
        foreground_probability: float,
        seed: int,
    ) -> None:
        if not volumes:
            raise ValueError("B5 patch dataset requires at least one volume")
        self.volumes = tuple(volumes)
        self.roi_size = tuple(roi_size)
        self.patches_per_epoch = patches_per_epoch
        self.foreground_probability = foreground_probability
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.patches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence((self.seed, self.epoch, index)))

    def _crop(self, volume: B5Volume, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        image, target = _pad_volume(volume.image, volume.target, self.roi_size)
        if rng.random() < self.foreground_probability:
            foreground = np.argwhere(target)
        else:
            foreground = np.empty((0, 3), dtype=np.int64)
        if foreground.size:
            center = foreground[int(rng.integers(len(foreground)))]
            starts = [
                int(np.clip(center[axis] - self.roi_size[axis] // 2, 0, target.shape[axis] - self.roi_size[axis]))
                for axis in range(3)
            ]
        else:
            starts = [int(rng.integers(target.shape[axis] - self.roi_size[axis] + 1)) for axis in range(3)]
        slices = tuple(slice(starts[axis], starts[axis] + self.roi_size[axis]) for axis in range(3))
        return image[(slice(None), *slices)].copy(), target[slices].copy()

    @staticmethod
    def _intensity_jitter(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        output = image.copy()
        for channel in range(output.shape[0]):
            nonzero = output[channel] != 0.0
            gain = float(rng.uniform(0.90, 1.10))
            bias = float(rng.uniform(-0.10, 0.10))
            output[channel, nonzero] = output[channel, nonzero] * gain + bias
        return output

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        rng = self._rng(index)
        volume = self.volumes[(self.epoch * len(self) + index) % len(self.volumes)]
        image, target = self._crop(volume, rng)
        return {
            "image": torch.from_numpy(self._intensity_jitter(image, rng)).to(torch.float32),
            "target": torch.from_numpy(target.copy()).to(torch.bool),
        }


class B5VolumeDataset(Dataset[dict[str, Any]]):
    """Native validation volumes for sliding-window subject-level scoring."""

    def __init__(self, volumes: Sequence[B5Volume]) -> None:
        self.volumes = tuple(volumes)

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        volume = self.volumes[index]
        return {
            "image": torch.from_numpy(volume.image.copy()).to(torch.float32),
            "target": torch.from_numpy(volume.target.copy()).to(torch.bool),
            "subject_id": volume.subject_id,
        }


class B5VolumeDataModule(LightningDataModule):
    """Small in-memory canonical NIfTI cohort with immutable split guards."""

    def __init__(self, config: B5DevelopmentConfig) -> None:
        super().__init__()
        self.config = config
        self.manifest = _read_manifest(config.manifest_path)
        self.profile = load_profile(config.normalization_profile)
        if self.profile.fitted_subject_ids != B1_DEVELOPMENT_TRAIN_SUBJECT_IDS:
            raise ValueError("B5 normalization profile must be fitted only on its ten train subjects")
        self.train_records = _records_for_ids(self.manifest, B1_DEVELOPMENT_TRAIN_SUBJECT_IDS)
        self.validation_records = _records_for_ids(self.manifest, B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS)
        self.train_dataset: B5PatchDataset | None = None
        self.validation_dataset: B5VolumeDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit", "validate"):
            return
        if self.train_dataset is None:
            train_volumes = tuple(load_b5_volume(record, self.config.dataset_root, self.profile) for record in self.train_records)
            validation_volumes = tuple(
                load_b5_volume(record, self.config.dataset_root, self.profile) for record in self.validation_records
            )
            self.train_dataset = B5PatchDataset(
                train_volumes,
                roi_size=self.config.roi_size,
                patches_per_epoch=self.config.patches_per_epoch,
                foreground_probability=self.config.foreground_patch_probability,
                seed=self.config.seed,
            )
            self.validation_dataset = B5VolumeDataset(validation_volumes)

    def set_epoch(self, epoch: int) -> None:
        if self.train_dataset is None:
            raise RuntimeError("B5 training dataset is not ready")
        self.train_dataset.set_epoch(epoch)

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        if self.train_dataset is None:
            raise RuntimeError("B5 training dataset is not ready")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        if self.validation_dataset is None:
            raise RuntimeError("B5 validation dataset is not ready")
        return DataLoader(
            self.validation_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )


@dataclass(frozen=True)
class B5SubjectDice:
    subject_id: str
    dice: float
    prediction_voxels: int
    target_voxels: int
    intersection_voxels: int


@dataclass(frozen=True)
class B5DevelopmentDiceSummary:
    probability_threshold: float
    macro_dice: float
    subjects: tuple[B5SubjectDice, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability_threshold": self.probability_threshold,
            "macro_dice": self.macro_dice,
            "subjects": [asdict(subject) for subject in self.subjects],
        }


@dataclass(frozen=True)
class B5ThresholdSelection:
    summaries: tuple[B5DevelopmentDiceSummary, ...]
    selected: B5DevelopmentDiceSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": [summary.to_dict() for summary in self.summaries],
            "selected_probability_threshold": self.selected.probability_threshold,
            "selected_macro_dice": self.selected.macro_dice,
        }


def b5_development_summary(
    probabilities: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray], *, probability_threshold: float
) -> B5DevelopmentDiceSummary:
    """Score exactly the two canonical development subjects at a fixed sigmoid threshold."""
    expected_subjects = B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS
    if tuple(sorted(probabilities)) != tuple(sorted(expected_subjects)) or set(probabilities) != set(targets):
        raise ValueError("B5 probabilities and targets must cover exactly the two development subjects")
    subjects: list[B5SubjectDice] = []
    for subject_id in expected_subjects:
        probability = np.asarray(probabilities[subject_id], dtype=np.float32)
        target = np.asarray(targets[subject_id], dtype=bool)
        if probability.shape != target.shape:
            raise ValueError(f"B5 development shapes differ for subject {subject_id}")
        prediction = probability >= probability_threshold
        subjects.append(
            B5SubjectDice(
                subject_id=subject_id,
                dice=binary_dice(prediction, target),
                prediction_voxels=int(prediction.sum()),
                target_voxels=int(target.sum()),
                intersection_voxels=int(np.logical_and(prediction, target).sum()),
            )
        )
    return B5DevelopmentDiceSummary(
        probability_threshold=probability_threshold,
        macro_dice=float(np.mean([subject.dice for subject in subjects])),
        subjects=tuple(subjects),
    )


def b5_threshold_grid(
    probabilities: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    thresholds: Sequence[float] = tuple(round(index * 0.05, 2) for index in range(1, 20)),
) -> B5ThresholdSelection:
    """Select the deterministic lower-cutoff winner from B5's development-only grid."""
    summaries = tuple(
        b5_development_summary(probabilities, targets, probability_threshold=float(threshold)) for threshold in thresholds
    )
    if not summaries:
        raise ValueError("B5 threshold grid must not be empty")
    return B5ThresholdSelection(
        summaries=summaries,
        selected=max(summaries, key=lambda summary: (summary.macro_dice, -summary.probability_threshold)),
    )


def create_b5_model(*, use_checkpoint: bool = True) -> SwinUNETR:
    """Construct MONAI's Base Swin UNETR before optional SSL encoder loading."""
    return SwinUNETR(
        in_channels=len(MODALITY_ORDER),
        out_channels=1,
        feature_size=48,
        norm_name="instance",
        use_checkpoint=use_checkpoint,
        spatial_dims=3,
    )


def _adapt_ssl_patch_projection(weight: Tensor, target_shape: torch.Size) -> Tensor:
    if weight.ndim != 5 or tuple(weight.shape[1:]) != (1, 2, 2, 2):
        raise ValueError(f"unexpected SSL patch projection shape: {tuple(weight.shape)}")
    if tuple(target_shape) != (weight.shape[0], len(MODALITY_ORDER), 2, 2, 2):
        raise ValueError(f"B5 patch projection target is incompatible: {tuple(target_shape)}")
    return weight.repeat(1, len(MODALITY_ORDER), 1, 1, 1) / len(MODALITY_ORDER)


def _ssl_target_key(source_key: str, target_state: Mapping[str, Tensor]) -> str | None:
    """Map MONAI's public SSL encoder checkpoint layouts onto current Swin UNETR."""
    if not source_key.startswith("encoder."):
        return None
    suffix = source_key.removeprefix("encoder.")
    candidates = (
        f"swinViT.{suffix}",
        f"swinViT.{source_key[8:18]}{source_key[20:]}",  # MONAI's published conversion rule.
    )
    return next((candidate for candidate in candidates if candidate in target_state), None)


def load_b5_ssl_encoder(model: SwinUNETR, checkpoint_path: Path) -> dict[str, Any]:
    """Load only compatible public SSL encoder weights and adapt its one-channel stem."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = payload.get("model") if isinstance(payload, Mapping) else None
    if not isinstance(source_state, Mapping):
        raise ValueError("B5 SSL checkpoint lacks its expected model state")
    target_state = model.state_dict()
    adapted_state: dict[str, Tensor] = {}
    skipped = 0
    for raw_key, raw_value in source_state.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, Tensor):
            continue
        target_key = _ssl_target_key(raw_key, target_state)
        if target_key is None:
            skipped += 1
            continue
        value = raw_value
        if target_key == "swinViT.patch_embed.proj.weight":
            value = _adapt_ssl_patch_projection(value, target_state[target_key].shape)
        if tuple(value.shape) != tuple(target_state[target_key].shape):
            skipped += 1
            continue
        adapted_state[target_key] = value
    if "swinViT.patch_embed.proj.weight" not in adapted_state or len(adapted_state) < 20:
        raise RuntimeError("B5 SSL conversion did not load the expected Swin encoder weights")
    target_state.update(adapted_state)
    model.load_state_dict(target_state, strict=True)
    return {
        "checkpoint_path": str(checkpoint_path),
        "source_key_count": len(source_state),
        "loaded_encoder_key_count": len(adapted_state),
        "skipped_source_key_count": skipped,
        "input_stem": "one-channel SSL patch projection repeated over four modalities and divided by four",
    }


def ensure_b5_ssl_checkpoint(path: Path) -> Path:
    """Use a persisted public checkpoint, downloading it into the run Volume only once."""
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = torch.hub.load_state_dict_from_url(
        B5_PRETRAINED_WEIGHTS_URL,
        model_dir=str(path.parent),
        file_name=path.name,
        map_location="cpu",
        progress=True,
    )
    del downloaded
    if not path.is_file():
        raise FileNotFoundError(f"B5 SSL checkpoint download did not create {path}")
    return path


class B5SwinUNETRModule(LightningModule):
    """Lightning integration for 3D BCE-plus-soft-Dice Swin UNETR fine tuning."""

    def __init__(self, config: B5DevelopmentConfig) -> None:
        super().__init__()
        self.config = config
        self.model = create_b5_model(use_checkpoint=config.use_checkpoint)
        checkpoint_path = ensure_b5_ssl_checkpoint(config.pretrained_weights_path)
        self.pretrained_load = load_b5_ssl_encoder(self.model, checkpoint_path)
        self._probabilities: dict[str, np.ndarray] = {}
        self._targets: dict[str, np.ndarray] = {}
        self.save_hyperparameters(
            {
                "experiment_id": config.experiment_id,
                "seed": config.seed,
                "roi_size": list(config.roi_size),
                "modalities": list(MODALITY_ORDER),
                "encoder": "MONAI Swin UNETR Base feature_size=48",
                "pretrained_weights_url": B5_PRETRAINED_WEIGHTS_URL,
                "input_stem": self.pretrained_load["input_stem"],
            }
        )

    @staticmethod
    def _unpack_batch(batch: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
        image = batch.get("image")
        target = batch.get("target")
        if not isinstance(image, Tensor) or not isinstance(target, Tensor):
            raise TypeError("B5 batches require tensor image and target fields")
        if image.ndim != 5 or image.shape[1] != len(MODALITY_ORDER):
            raise ValueError(f"B5 images must have shape [B, 4, D, H, W], got {tuple(image.shape)}")
        if target.ndim != 4 or target.shape != (image.shape[0], *image.shape[2:]):
            raise ValueError("B5 targets must have shape [B, D, H, W] matching their images")
        return image, target.bool()

    @staticmethod
    def _soft_dice_loss(logits: Tensor, targets: Tensor) -> Tensor:
        probabilities = torch.sigmoid(logits[:, 0])
        target_float = targets.to(dtype=logits.dtype)
        numerator = 2.0 * (probabilities * target_float).sum(dim=(1, 2, 3)) + 1.0
        denominator = probabilities.sum(dim=(1, 2, 3)) + target_float.sum(dim=(1, 2, 3)) + 1.0
        return 1.0 - (numerator / denominator).mean()

    def _loss(self, logits: Tensor, targets: Tensor, *, stage: str) -> Tensor:
        bce = functional.binary_cross_entropy_with_logits(logits[:, 0], targets.to(dtype=logits.dtype))
        dice = self._soft_dice_loss(logits, targets)
        self.log(f"{stage}/bce_loss", bce, on_step=False, on_epoch=True, batch_size=logits.shape[0])
        self.log(f"{stage}/dice_loss", dice, on_step=False, on_epoch=True, batch_size=logits.shape[0])
        return bce + dice

    def on_train_epoch_start(self) -> None:
        data_module = self.trainer.datamodule
        if not isinstance(data_module, B5VolumeDataModule):
            raise TypeError("B5 requires B5VolumeDataModule")
        data_module.set_epoch(self.current_epoch)

    def training_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        images, targets = self._unpack_batch(batch)
        loss = self._loss(self.model(images), targets, stage="train")
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=images.shape[0])
        return loss

    def on_validation_epoch_start(self) -> None:
        self._probabilities = {}
        self._targets = {}

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        del batch_idx
        images, targets = self._unpack_batch(batch)
        logits = sliding_window_inference(
            images,
            roi_size=self.config.roi_size,
            sw_batch_size=1,
            predictor=self.model,
            overlap=self.config.sliding_window_overlap,
        )
        loss = self._loss(logits, targets, stage="val")
        subject_ids = batch.get("subject_id")
        if not isinstance(subject_ids, Sequence) or len(subject_ids) != images.shape[0]:
            raise ValueError("B5 validation batch is missing subject IDs")
        for index, subject_id in enumerate(subject_ids):
            if not isinstance(subject_id, str) or subject_id in self._probabilities:
                raise ValueError("B5 validation subjects must be unique strings")
            self._probabilities[subject_id] = torch.sigmoid(logits[index, 0]).detach().cpu().numpy()
            self._targets[subject_id] = targets[index].detach().cpu().numpy()
        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=images.shape[0])

    def on_validation_epoch_end(self) -> None:
        summary = b5_development_summary(
            self._probabilities,
            self._targets,
            probability_threshold=self.config.probability_threshold,
        )
        if self.trainer.is_global_zero:
            output_path = Path(self.config.run_dir) / "metrics" / "development-3d" / f"epoch-{self.current_epoch:03d}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
        metric = torch.tensor(summary.macro_dice, dtype=torch.float32, device=self.device)
        self.log(B5_DEVELOPMENT_DICE_METRIC, metric, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        for subject in summary.subjects:
            self.log(
                f"val/development_dice_3d_subject_{subject.subject_id}",
                torch.tensor(subject.dice, dtype=torch.float32, device=self.device),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    def configure_optimizers(self) -> dict[str, Any]:
        encoder_parameters = list(self.model.swinViT.parameters())
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        decoder_parameters = [parameter for parameter in self.model.parameters() if id(parameter) not in encoder_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": self.config.encoder_learning_rate},
                {"params": decoder_parameters, "lr": self.config.decoder_learning_rate},
            ],
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class B5PlainTextProgressReporter(Callback):
    """Stable line-oriented B5 progress suitable for Modal's logs."""

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        del pl_module, outputs, batch
        completed = batch_idx + 1
        interval = trainer.lightning_module.config.progress_report_interval_batches
        if trainer.is_global_zero and completed % interval == 0:
            print(
                f"[ivdseg-progress] epoch={trainer.current_epoch + 1}/{trainer.max_epochs} "
                f"train_patch={completed}/{trainer.num_training_batches} optimizer_step={trainer.global_step}",
                flush=True,
            )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if trainer.is_global_zero:
            value = trainer.callback_metrics.get(B5_DEVELOPMENT_DICE_METRIC)
            text = "unavailable" if value is None else f"{float(value):.6f}"
            print(
                f"[ivdseg-progress] epoch={trainer.current_epoch + 1}/{trainer.max_epochs} "
                f"validation_complete macro_3d_dice={text}",
                flush=True,
            )


def build_b5_development_training(
    config: B5DevelopmentConfig,
) -> tuple[Trainer, B5SwinUNETRModule, B5VolumeDataModule, ModelCheckpoint]:
    """Build B5's development-only training loop without loading the fixed holdout."""
    seed_everything(config.seed, workers=True)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    data_module = B5VolumeDataModule(config)
    data_module.setup("fit")
    module = B5SwinUNETRModule(config)
    checkpoint = ModelCheckpoint(
        dirpath=config.run_dir / "checkpoints",
        filename="best-epoch={epoch:03d}",
        monitor=B5_DEVELOPMENT_DICE_METRIC,
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks: list[Callback] = [
        B5PlainTextProgressReporter(),
        checkpoint,
        EarlyStopping(
            monitor=B5_DEVELOPMENT_DICE_METRIC,
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


def write_b5_development_configuration(config: B5DevelopmentConfig, module: B5SwinUNETRModule) -> Path:
    """Persist B5's split, budget, model and exact SSL conversion before training."""
    output_path = config.run_dir / "config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": config.experiment_id,
        "b5_development": {
            **asdict(config),
            "manifest_path": str(config.manifest_path),
            "dataset_root": str(config.dataset_root),
            "normalization_profile": str(config.normalization_profile),
            "run_dir": str(config.run_dir),
            "pretrained_weights_path": str(config.pretrained_weights_path),
            "train_subject_ids": list(B1_DEVELOPMENT_TRAIN_SUBJECT_IDS),
            "validation_subject_ids": list(B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS),
            "fixed_test_subjects_excluded": list(FIXED_TEST_SUBJECTS),
            "max_training_patches": config.max_epochs * config.patches_per_epoch,
            "early_stopping": "may terminate before the finite maximum when the fixed development metric stalls",
        },
        "model": {
            "name": "MONAI SwinUNETR",
            "feature_size": config.feature_size,
            "in_channels": len(MODALITY_ORDER),
            "out_channels": 1,
            "gradient_checkpointing": config.use_checkpoint,
            "pretrained_encoder": module.pretrained_load,
        },
        "input": {
            "modalities": list(MODALITY_ORDER),
            "canonical_grid": "native NIfTI grid; random ROI patches for training, sliding-window native volumes for validation",
            "roi_size": list(config.roi_size),
            "augmentation": "foreground-biased deterministic crop plus modality-wise intensity gain/bias; no anatomical flips",
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


def load_b5_development_config(path: Path) -> B5DevelopmentConfig:
    """Restore the immutable B5 definition for development-only threshold scoring."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("b5_development")
    if payload.get("schema_version") != 1 or not isinstance(raw, Mapping):
        raise ValueError("run configuration lacks supported b5_development settings")
    supported = {field.name for field in fields(B5DevelopmentConfig) if field.init}
    values = {key: value for key, value in raw.items() if key in supported}
    for field_name in (
        "manifest_path",
        "dataset_root",
        "normalization_profile",
        "run_dir",
        "pretrained_weights_path",
    ):
        value = values.get(field_name)
        if not isinstance(value, str):
            raise ValueError(f"B5 run configuration has no valid {field_name}")
        values[field_name] = Path(value)
    roi_size = values.get("roi_size")
    if not isinstance(roi_size, list) or len(roi_size) != 3:
        raise ValueError("B5 run configuration has no valid ROI")
    values["roi_size"] = tuple(int(dimension) for dimension in roi_size)
    config = B5DevelopmentConfig(**values)
    if tuple(raw.get("train_subject_ids", ())) != B1_DEVELOPMENT_TRAIN_SUBJECT_IDS:
        raise ValueError("B5 run configuration has unexpected training subject IDs")
    if tuple(raw.get("validation_subject_ids", ())) != B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS:
        raise ValueError("B5 run configuration has unexpected development subject IDs")
    if tuple(raw.get("fixed_test_subjects_excluded", ())) != FIXED_TEST_SUBJECTS:
        raise ValueError("B5 run configuration has unexpected fixed-test exclusions")
    return config


def load_b5_model(config: B5DevelopmentConfig, checkpoint_path: Path, *, device: torch.device) -> SwinUNETR:
    """Restore B5's saved model without downloading or reapplying SSL initialization."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_state = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) else None
    if not isinstance(raw_state, Mapping):
        raise ValueError("B5 checkpoint lacks a Lightning state_dict")
    state = {key.removeprefix("model."): value for key, value in raw_state.items() if key.startswith("model.")}
    model = create_b5_model(use_checkpoint=config.use_checkpoint).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.inference_mode()
def evaluate_b5_development_checkpoint(config: B5DevelopmentConfig, checkpoint_path: Path, *, device: torch.device) -> B5ThresholdSelection:
    """Run fresh native-volume development inference and choose only its cutoff."""
    data_module = B5VolumeDataModule(config)
    data_module.setup("validate")
    model = load_b5_model(config, checkpoint_path, device=device)
    probabilities: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for batch in data_module.val_dataloader():
        images, target = B5SwinUNETRModule._unpack_batch(batch)
        subject_ids = batch.get("subject_id")
        if not isinstance(subject_ids, Sequence) or len(subject_ids) != 1 or not isinstance(subject_ids[0], str):
            raise ValueError("B5 evaluation batch has no single subject ID")
        logits = sliding_window_inference(
            images.to(device),
            roi_size=config.roi_size,
            sw_batch_size=1,
            predictor=model,
            overlap=config.sliding_window_overlap,
        )
        probabilities[subject_ids[0]] = torch.sigmoid(logits[0, 0]).cpu().numpy()
        targets[subject_ids[0]] = target[0].cpu().numpy()
    return b5_threshold_grid(probabilities, targets)
