"""Smoke-test the custom NIfTI/tensor RF-DETR development data module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ivdseg.datamodule import IVDDataModule, RFDETR_SEG_SMALL_BLOCK_SIZE


DEFAULT_TRAIN_SUBJECTS = ("01", "04", "05", "06", "08", "09", "11", "13", "15", "16")
DEFAULT_VALIDATION_SUBJECTS = ("02", "12")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/ivdm3seg-v1.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("IVDM3Seg"))
    parser.add_argument(
        "--normalization-profile",
        type=Path,
        default=Path("artifacts/normalization/ivdm3seg-development-train-v1.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/datamodule-smoke/development-v1.json"))
    args = parser.parse_args()

    data_module = IVDDataModule(
        manifest=args.manifest,
        dataset_root=args.dataset_root,
        normalization_profile=args.normalization_profile,
        train_subject_ids=DEFAULT_TRAIN_SUBJECTS,
        validation_subject_ids=DEFAULT_VALIDATION_SUBJECTS,
        batch_size=2,
        resolution=384,
        seed=17,
    )
    data_module.setup("fit")
    train_samples, train_targets = next(iter(data_module.train_dataloader()))
    samples, targets = next(iter(data_module.val_dataloader()))
    validation_dataset = data_module.validation_dataset
    if validation_dataset is None:  # Defensive: setup above is required to create it.
        raise RuntimeError("validation dataset was not created")
    positive_image, positive_target = next(
        (image, target) for image, target in validation_dataset if target["masks"].shape[0] > 0
    )
    payload = {
        "schema_version": 1,
        "input_source": "custom NIfTI/tensor dataset (no COCO/PIL loader)",
        "train_subject_ids": list(DEFAULT_TRAIN_SUBJECTS),
        "validation_subject_ids": list(DEFAULT_VALIDATION_SUBJECTS),
        "normalization_profile": str(args.normalization_profile),
        "train_slice_count": len(data_module.train_dataset or ()),
        "validation_slice_count": len(data_module.validation_dataset or ()),
        "resolution": 384,
        "rfdetr_seg_small_block_size": RFDETR_SEG_SMALL_BLOCK_SIZE,
        "augmented_train_batch_tensor_shape": list(train_samples.tensors.shape),
        "augmented_train_batch_instance_counts": [
            int(target["masks"].shape[0]) for target in train_targets
        ],
        "batch_tensor_shape": list(samples.tensors.shape),
        "batch_padding_mask_shape": list(samples.mask.shape),
        "first_batch_instance_counts": [int(target["masks"].shape[0]) for target in targets],
        "target_box_format": "normalized cxcywh",
        "positive_sample_tensor_shape": list(positive_image.shape),
        "positive_target_instance_count": int(positive_target["masks"].shape[0]),
        "positive_target_mask_dtype": str(positive_target["masks"].dtype),
        "positive_target_mask_shape": list(positive_target["masks"].shape),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
