#!/usr/bin/env python3
"""Validate the manifest-governed in-memory spatial transform for every subject."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib

from ivdseg.manifest import build_manifest
from ivdseg.spatial import grids_match, load_canonical_subject


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("IVDM3Seg"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/spatial-validation/ivdm3seg-v1.json")
    )
    args = parser.parse_args()

    manifest = build_manifest(args.dataset_root)
    subjects = []
    for record in manifest["subjects"]:
        subject = load_canonical_subject(record, args.dataset_root)
        subjects.append(
            {
                "subject_id": subject.subject_id,
                "shape": list(subject.reference_image.shape),
                "orientation": list(nib.aff2axcodes(subject.reference_image.affine)),
                "label_was_resampled": subject.label_was_resampled,
                "label_matches_image_grid": grids_match(subject.reference_image, subject.label),
            }
        )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "manifest": "artifacts/manifests/ivdm3seg-v1.json",
        "subject_count": len(subjects),
        "subjects": subjects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Validated {len(subjects)} subjects; "
        f"{sum(item['label_was_resampled'] for item in subjects)} label resample(s)."
    )


if __name__ == "__main__":
    main()
