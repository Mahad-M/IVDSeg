"""Custom NIfTI-to-tensor Lightning data module for RF-DETR IVD segmentation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
from pytorch_lightning import LightningDataModule
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from rfdetr.utilities.tensors import make_collate_fn

from ivdseg.augmentations import (
    AugmentationConfig,
    Synchronized2p5DAugmentation,
    resize_tensor_and_target,
    target_from_masks,
)
from ivdseg.manifest import FIXED_TEST_SUBJECTS
from ivdseg.normalization import NormalizationProfile, load_profile
from ivdseg.samples import PreparedSubject, prepare_subject


RFDETR_SEG_SMALL_BLOCK_SIZE = 24


@dataclass(frozen=True)
class SliceReference:
    """One center slice of an explicitly selected subject."""

    subject_id: str
    slice_index: int


def _worker_init_fn(worker_id: int) -> None:
    """Seed NumPy and stdlib augmentation helpers consistently per worker."""
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _read_manifest(manifest: Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(manifest, Path):
        return json.loads(manifest.read_text(encoding="utf-8"))
    return manifest


def _records_for_subject_ids(
    manifest: Mapping[str, Any], subject_ids: Sequence[str]
) -> list[Mapping[str, Any]]:
    records = {str(record["subject_id"]): record for record in manifest["subjects"]}
    selected: list[Mapping[str, Any]] = []
    for subject_id in subject_ids:
        try:
            record = records[subject_id]
        except KeyError as error:
            raise ValueError(f"unknown subject ID in data module: {subject_id}") from error
        if record["partition"] != "train":
            raise ValueError(f"data module may not use fixed test subject {subject_id}")
        selected.append(record)
    return selected


class IVD2p5DDataset(Dataset[tuple[Tensor, dict[str, Any]]]):
    """Lazy NIfTI-backed center slices with RF-DETR segmentation targets.

    This class intentionally does not inherit from RF-DETR's COCO/PIL datasets.
    It loads subjects on demand and holds a bounded per-worker LRU cache of
    prepared canonical NIfTI subjects.  The default comfortably covers the
    10-subject development and 12-subject final-training pools, avoiding
    repeated NIfTI decoding once slice-level shuffling starts.
    """

    def __init__(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        dataset_root: Path,
        profile: NormalizationProfile,
        resolution: int = 384,
        augmentation: Synchronized2p5DAugmentation | None = None,
        subject_cache_size: int = 16,
        resolution_block_size: int = RFDETR_SEG_SMALL_BLOCK_SIZE,
    ) -> None:
        if resolution_block_size < 1:
            raise ValueError("resolution block size must be positive")
        if resolution % resolution_block_size != 0:
            raise ValueError(
                f"resolution {resolution} must be divisible by the configured model block size {resolution_block_size}"
            )
        if subject_cache_size < 1:
            raise ValueError("subject_cache_size must be positive")
        self.records = {str(record["subject_id"]): record for record in records}
        if len(self.records) != len(records):
            raise ValueError("dataset records must have distinct subject IDs")
        self.dataset_root = Path(dataset_root)
        self.profile = profile
        self.resolution = resolution
        self.augmentation = augmentation
        self.subject_cache_size = subject_cache_size
        self._prepared_cache: OrderedDict[str, PreparedSubject] = OrderedDict()
        self.references = self._build_references()

    def _build_references(self) -> tuple[SliceReference, ...]:
        """Read each label once to make a fixed, explicit slice index."""
        references: list[SliceReference] = []
        for record in self.records.values():
            prepared = self._prepared_subject(str(record["subject_id"]))
            references.extend(
                SliceReference(prepared.subject_id, slice_index)
                for slice_index in range(prepared.slice_count)
            )
        return tuple(references)

    def _prepared_subject(self, subject_id: str) -> PreparedSubject:
        cached = self._prepared_cache.pop(subject_id, None)
        if cached is not None:
            self._prepared_cache[subject_id] = cached
            return cached
        prepared = prepare_subject(self.records[subject_id], self.dataset_root, self.profile)
        self._prepared_cache[subject_id] = prepared
        while len(self._prepared_cache) > self.subject_cache_size:
            self._prepared_cache.popitem(last=False)
        return prepared

    def __len__(self) -> int:
        return len(self.references)

    def semantic_label_for_subject(self, subject_id: str) -> np.ndarray:
        """Return the authoritative canonical binary label used for volume metrics.

        Component filtering is an RF-DETR target-generation detail.  Development
        volume Dice must compare predictions with the original binary semantic
        annotation (after the manifest-governed in-memory subject-16 resample),
        so it deliberately does not use ``component_labels`` here.
        """
        semantic_label = self._prepared_subject(subject_id).semantic_label
        if semantic_label is None:
            raise RuntimeError(f"prepared subject {subject_id} has no semantic label")
        return np.asarray(semantic_label, dtype=bool)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Any]]:
        reference = self.references[index]
        prepared = self._prepared_subject(reference.subject_id)
        image = torch.from_numpy(prepared.tensor_for_slice(reference.slice_index)).to(torch.float32)
        instances = prepared.targets_for_slice(reference.slice_index)
        height, width = image.shape[-2:]
        masks = (
            torch.stack([torch.from_numpy(instance.mask) for instance in instances]).bool()
            if instances
            else torch.zeros((0, height, width), dtype=torch.bool)
        )
        labels = torch.tensor([instance.class_id for instance in instances], dtype=torch.int64)
        image_id = torch.tensor([index], dtype=torch.int64)
        target = target_from_masks(
            masks,
            labels=labels,
            template={"image_id": image_id},
            original_size=(height, width),
        )
        if self.augmentation is not None:
            image, target = self.augmentation(image, target)
        return resize_tensor_and_target(image, target, resolution=self.resolution)


class IVD2p5DSemanticDataset(IVD2p5DDataset):
    """Lazy 2.5D slices with one authoritative binary semantic mask per plane.

    RF-DETR derives instance targets after filtering small 3D components.  A
    semantic U-Net baseline instead learns the complete binary annotation, so
    this dataset deliberately uses :attr:`PreparedSubject.semantic_label`.
    It still reuses the identical subject preparation, 12-channel construction,
    resize, and synchronized augmentation contracts as the detector pipeline.
    """

    def __init__(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        dataset_root: Path,
        profile: NormalizationProfile,
        resolution: int = 384,
        augmentation: Synchronized2p5DAugmentation | None = None,
        subject_cache_size: int = 16,
    ) -> None:
        super().__init__(
            records=records,
            dataset_root=dataset_root,
            profile=profile,
            resolution=resolution,
            augmentation=augmentation,
            subject_cache_size=subject_cache_size,
            resolution_block_size=16,
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        reference = self.references[index]
        prepared = self._prepared_subject(reference.subject_id)
        if prepared.semantic_label is None:
            raise RuntimeError(f"prepared subject {reference.subject_id} has no semantic label")
        image = torch.from_numpy(prepared.tensor_for_slice(reference.slice_index)).to(torch.float32)
        semantic_mask = torch.from_numpy(prepared.semantic_label[reference.slice_index]).bool()
        height, width = image.shape[-2:]
        target = target_from_masks(
            semantic_mask.unsqueeze(0),
            template={"image_id": torch.tensor([index], dtype=torch.int64)},
            original_size=(height, width),
        )
        if self.augmentation is not None:
            image, target = self.augmentation(image, target)
        image, target = resize_tensor_and_target(image, target, resolution=self.resolution)
        # ``any`` has the correct all-false result for a target-empty slice.
        resized_semantic_mask = target["masks"].any(dim=0)
        return image, resized_semantic_mask, torch.tensor(index, dtype=torch.int64)


class IVDDataModule(LightningDataModule):
    """RF-DETR-compatible data module using only explicit non-holdout IDs.

    Development runs provide a separate validation tuple.  Final retraining
    deliberately provides an empty tuple, which creates a train-only module
    rather than silently reusing a development subject as validation data.
    """

    def __init__(
        self,
        *,
        manifest: Path | Mapping[str, Any],
        dataset_root: Path,
        normalization_profile: Path | NormalizationProfile,
        train_subject_ids: Sequence[str],
        validation_subject_ids: Sequence[str],
        batch_size: int,
        resolution: int = 384,
        num_workers: int = 0,
        seed: int = 17,
        train_augmentation: AugmentationConfig | None = None,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        self.manifest = _read_manifest(manifest)
        self.dataset_root = Path(dataset_root)
        self.profile = (
            load_profile(normalization_profile)
            if isinstance(normalization_profile, Path)
            else normalization_profile
        )
        self.train_subject_ids = tuple(train_subject_ids)
        self.validation_subject_ids = tuple(validation_subject_ids)
        if not self.train_subject_ids:
            raise ValueError("data module requires at least one training subject")
        if len(set(self.train_subject_ids)) != len(self.train_subject_ids):
            raise ValueError("training subject IDs must be distinct")
        if len(set(self.validation_subject_ids)) != len(self.validation_subject_ids):
            raise ValueError("validation subject IDs must be distinct")
        overlap = set(self.train_subject_ids) & set(self.validation_subject_ids)
        if overlap:
            raise ValueError(f"training and validation subjects overlap: {sorted(overlap)}")
        if set(self.train_subject_ids) & set(FIXED_TEST_SUBJECTS):
            raise ValueError("fixed test subjects may not be used for training")
        if set(self.validation_subject_ids) & set(FIXED_TEST_SUBJECTS):
            raise ValueError("fixed test subjects may not be used for validation")
        if self.profile.fitted_subject_ids != self.train_subject_ids:
            raise ValueError(
                "normalization profile fitted subjects must exactly match train_subject_ids; "
                f"got profile={self.profile.fitted_subject_ids}, train={self.train_subject_ids}"
            )
        self.train_records = _records_for_subject_ids(self.manifest, self.train_subject_ids)
        self.validation_records = _records_for_subject_ids(self.manifest, self.validation_subject_ids)
        self.batch_size = batch_size
        self.resolution = resolution
        self.num_workers = num_workers
        self.seed = seed
        self.train_augmentation = Synchronized2p5DAugmentation(train_augmentation)
        self.pin_memory = pin_memory
        self._collate_fn = make_collate_fn(block_size=RFDETR_SEG_SMALL_BLOCK_SIZE)
        self.train_dataset: IVD2p5DDataset | None = None
        self.validation_dataset: IVD2p5DDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create lazily loaded training and, when requested, validation datasets."""
        if stage in (None, "fit", "validate"):
            if self.train_dataset is None:
                self.train_dataset = IVD2p5DDataset(
                    records=self.train_records,
                    dataset_root=self.dataset_root,
                    profile=self.profile,
                    resolution=self.resolution,
                    augmentation=self.train_augmentation,
                )
            if self.validation_subject_ids and self.validation_dataset is None:
                self.validation_dataset = IVD2p5DDataset(
                    records=self.validation_records,
                    dataset_root=self.dataset_root,
                    profile=self.profile,
                    resolution=self.resolution,
                    augmentation=None,
                )

    @staticmethod
    def _require_dataset(dataset: IVD2p5DDataset | None, name: str) -> IVD2p5DDataset:
        if dataset is None:
            raise RuntimeError(f"{name} dataset is not ready; call setup('fit') first")
        return dataset

    def _loader_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "collate_fn": self._collate_fn,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "worker_init_fn": _worker_init_fn,
        }
        if self.num_workers > 0:
            options["persistent_workers"] = True
        return options

    def train_dataloader(self) -> DataLoader[Any]:
        dataset = self._require_dataset(self.train_dataset, "training")
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            **self._loader_options(),
        )

    def val_dataloader(self) -> DataLoader[Any] | None:
        if not self.validation_subject_ids:
            return None
        dataset = self._require_dataset(self.validation_dataset, "validation")
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, **self._loader_options())


class SemanticIVDDataModule(IVDDataModule):
    """Data module for binary semantic segmentation baselines.

    It inherits every split, manifest, normalization, augmentation, and fixed
    holdout guard from :class:`IVDDataModule`, but uses PyTorch's ordinary batch
    collation because each sample has one tensor mask rather than a variable
    number of detection instances.
    """

    train_dataset: IVD2p5DSemanticDataset | None
    validation_dataset: IVD2p5DSemanticDataset | None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit", "validate"):
            if self.train_dataset is None:
                self.train_dataset = IVD2p5DSemanticDataset(
                    records=self.train_records,
                    dataset_root=self.dataset_root,
                    profile=self.profile,
                    resolution=self.resolution,
                    augmentation=self.train_augmentation,
                )
            if self.validation_subject_ids and self.validation_dataset is None:
                self.validation_dataset = IVD2p5DSemanticDataset(
                    records=self.validation_records,
                    dataset_root=self.dataset_root,
                    profile=self.profile,
                    resolution=self.resolution,
                    augmentation=None,
                )

    def _loader_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "worker_init_fn": _worker_init_fn,
        }
        if self.num_workers > 0:
            options["persistent_workers"] = True
        return options
