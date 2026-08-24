#!/usr/bin/env python3
"""Generate a compact index for lazy 12-channel RF-DETR training samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivdseg.manifest import build_manifest
from ivdseg.normalization import load_profile, select_training_records
from ivdseg.samples import build_sample_index, write_sample_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("IVDM3Seg"))
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("artifacts/normalization/ivdm3seg-training-pool-v1.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/samples/ivdm3seg-training-pool-v1.json")
    )
    args = parser.parse_args()

    manifest = build_manifest(args.dataset_root)
    profile = load_profile(args.profile)
    records = select_training_records(manifest, profile.fitted_subject_ids)
    index = build_sample_index(records, args.dataset_root, profile)
    write_sample_index(index, args.output)
    print(
        f"Wrote {args.output}: {sum(subject['slice_count'] for subject in index['subjects'])} "
        "lazy center-slice samples."
    )


if __name__ == "__main__":
    main()
