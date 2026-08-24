#!/usr/bin/env python3
"""Fit and save the no-test-leakage normalization profile for IVDM3Seg."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivdseg.manifest import build_manifest
from ivdseg.normalization import fit_normalization_profile, select_training_records, write_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("IVDM3Seg"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/normalization/ivdm3seg-training-pool-v1.json")
    )
    parser.add_argument(
        "--subject-id",
        action="append",
        dest="subject_ids",
        help="Training subject to fit; repeat to define a development-training subset.",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.dataset_root)
    subject_ids = args.subject_ids or [
        record["subject_id"] for record in manifest["subjects"] if record["partition"] == "train"
    ]
    profile = fit_normalization_profile(
        select_training_records(manifest, subject_ids), args.dataset_root
    )
    write_profile(profile, args.output)
    print(f"Wrote {args.output} using {len(profile.fitted_subject_ids)} training subjects.")


if __name__ == "__main__":
    main()
