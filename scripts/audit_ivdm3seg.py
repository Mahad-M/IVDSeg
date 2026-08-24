#!/usr/bin/env python3
"""Audit the IVDM3Seg NIfTI corpus without third-party dependencies.

The audit is deliberately stdlib-only so it can be run before the training
environment is created.  It inspects all source images and labels, verifies
the fixed evaluation split, compares image/label geometry, and writes a
machine-readable JSON report plus a concise Markdown summary.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import struct
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO


MODALITIES = ("fat", "inn", "opp", "wat")
FIXED_TEST_SUBJECTS = ("03", "07", "10", "14")
EXPECTED_SUBJECTS = tuple(f"{number:02d}" for number in range(1, 17))
LABEL_DATATYPE_UINT8 = 2
# Qform values are stored as float32, which creates sub-micrometre translation
# differences between otherwise identical image and label headers. A 1 µm
# element-wise tolerance accepts that storage precision while exposing real
# registration differences (subject 16 is >10 mm at volume corners).
AFFINE_ELEMENT_TOLERANCE = 1e-3


class AuditError(Exception):
    """A malformed or unreadable NIfTI file."""


@dataclass(frozen=True)
class NiftiHeader:
    path: str
    endianness: str
    nifti_version: int
    shape: tuple[int, ...]
    datatype: int
    datatype_name: str
    bitpix: int
    voxel_offset: int
    pixdim: tuple[float, float, float]
    qform_code: int
    sform_code: int
    affine_source: str
    affine: tuple[tuple[float, float, float, float], ...]
    spacing_mm: tuple[float, float, float]
    orientation: str
    scl_slope: float
    scl_inter: float


DATATYPES: dict[int, tuple[str, str, int]] = {
    2: ("uint8", "B", 1),
    4: ("int16", "h", 2),
    8: ("int32", "i", 4),
    16: ("float32", "f", 4),
    64: ("float64", "d", 8),
    256: ("int8", "b", 1),
    512: ("uint16", "H", 2),
    768: ("uint32", "I", 4),
    1024: ("int64", "q", 8),
    1280: ("uint64", "Q", 8),
}


def open_nifti(path: Path) -> BinaryIO:
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def unpack_at(header: bytes, fmt: str, offset: int, endian: str):
    return struct.unpack_from(endian + fmt, header, offset)


def qform_affine(header: bytes, endian: str, pixdim: tuple[float, float, float]):
    b, c, d = unpack_at(header, "3f", 256, endian)
    qx, qy, qz = unpack_at(header, "3f", 268, endian)
    qfac = unpack_at(header, "f", 76, endian)[0]
    a_squared = 1.0 - (b * b + c * c + d * d)
    a = math.sqrt(a_squared) if a_squared > 1e-7 else 0.0
    if a == 0.0:
        length = math.sqrt(b * b + c * c + d * d)
        if length == 0.0:
            raise AuditError("invalid qform quaternion")
        b, c, d = b / length, c / length, d / length

    rotation = (
        (a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)),
        (2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)),
        (2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b),
    )
    scales = (pixdim[0], pixdim[1], pixdim[2] * (-1.0 if qfac < 0 else 1.0))
    return tuple(
        tuple(rotation[row][column] * scales[column] for column in range(3)) + (offset,)
        for row, offset in enumerate((qx, qy, qz))
    ) + ((0.0, 0.0, 0.0, 1.0),)


def sform_affine(header: bytes, endian: str):
    rows = tuple(unpack_at(header, "4f", offset, endian) for offset in (280, 296, 312))
    return rows + ((0.0, 0.0, 0.0, 1.0),)


def axis_orientation(affine: tuple[tuple[float, float, float, float], ...]) -> str:
    positive = (("L", "R"), ("P", "A"), ("I", "S"))
    codes: list[str] = []
    used_axes: set[int] = set()
    for column in range(3):
        vector = [affine[row][column] for row in range(3)]
        axis = max(range(3), key=lambda row: abs(vector[row]))
        if axis in used_axes or abs(vector[axis]) < 1e-6:
            return "INVALID"
        used_axes.add(axis)
        codes.append(positive[axis][1] if vector[axis] > 0 else positive[axis][0])
    return "".join(codes)


def spacing_from_affine(affine: tuple[tuple[float, float, float, float], ...]):
    return tuple(
        math.sqrt(sum(affine[row][column] ** 2 for row in range(3)))
        for column in range(3)
    )


def read_header(path: Path) -> NiftiHeader:
    with open_nifti(path) as stream:
        header = stream.read(352)
    if len(header) < 352:
        raise AuditError("file is shorter than a NIfTI-1 header")

    little = struct.unpack_from("<i", header, 0)[0] == 348
    big = struct.unpack_from(">i", header, 0)[0] == 348
    if not little and not big:
        raise AuditError("not a NIfTI-1 single-file header (sizeof_hdr != 348)")
    endian = "<" if little else ">"
    dims = unpack_at(header, "8h", 40, endian)
    dimension_count = dims[0]
    if not 3 <= dimension_count <= 7:
        raise AuditError(f"unsupported dimension count: {dimension_count}")
    shape = tuple(dims[1 : dimension_count + 1])
    if any(size <= 0 for size in shape):
        raise AuditError(f"invalid dimensions: {shape}")

    datatype = unpack_at(header, "h", 70, endian)[0]
    bitpix = unpack_at(header, "h", 72, endian)[0]
    if datatype not in DATATYPES:
        raise AuditError(f"unsupported NIfTI datatype code: {datatype}")
    datatype_name, _, datatype_size = DATATYPES[datatype]
    if bitpix != datatype_size * 8:
        raise AuditError(f"bitpix {bitpix} does not match datatype {datatype_name}")

    raw_pixdim = unpack_at(header, "8f", 76, endian)
    pixdim = tuple(abs(value) for value in raw_pixdim[1:4])
    if any(value <= 0.0 for value in pixdim):
        raise AuditError(f"invalid voxel dimensions: {pixdim}")
    voxel_offset = int(round(unpack_at(header, "f", 108, endian)[0]))
    if voxel_offset < 352:
        raise AuditError(f"invalid voxel offset: {voxel_offset}")

    qform_code = unpack_at(header, "h", 252, endian)[0]
    sform_code = unpack_at(header, "h", 254, endian)[0]
    if sform_code > 0:
        affine, affine_source = sform_affine(header, endian), "sform"
    elif qform_code > 0:
        affine, affine_source = qform_affine(header, endian, pixdim), "qform"
    else:
        affine = (
            (pixdim[0], 0.0, 0.0, 0.0),
            (0.0, pixdim[1], 0.0, 0.0),
            (0.0, 0.0, pixdim[2], 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        affine_source = "fallback-pixdim"

    return NiftiHeader(
        path=str(path),
        endianness="little" if endian == "<" else "big",
        nifti_version=1,
        shape=shape,
        datatype=datatype,
        datatype_name=datatype_name,
        bitpix=bitpix,
        voxel_offset=voxel_offset,
        pixdim=pixdim,
        qform_code=qform_code,
        sform_code=sform_code,
        affine_source=affine_source,
        affine=affine,
        spacing_mm=spacing_from_affine(affine),
        orientation=axis_orientation(affine),
        scl_slope=unpack_at(header, "f", 112, endian)[0],
        scl_inter=unpack_at(header, "f", 116, endian)[0],
    )


def read_label_summary(path: Path, header: NiftiHeader) -> dict[str, int]:
    if header.datatype not in DATATYPES:
        raise AuditError(f"cannot decode datatype {header.datatype}")
    _, value_format, value_size = DATATYPES[header.datatype]
    voxel_count = math.prod(header.shape)
    byte_count = voxel_count * value_size
    with open_nifti(path) as stream:
        stream.seek(header.voxel_offset)
        payload = stream.read(byte_count)
    if len(payload) != byte_count:
        raise AuditError(
            f"truncated payload: expected {byte_count} bytes, found {len(payload)} bytes"
        )
    values = struct.iter_unpack(("<" if header.endianness == "little" else ">") + value_format, payload)
    counts = Counter(value[0] for value in values)
    return {str(value): count for value, count in sorted(counts.items())}


def close_enough(
    left: float, right: float, tolerance: float = AFFINE_ELEMENT_TOLERANCE
) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def affine_matches(left: NiftiHeader, right: NiftiHeader) -> bool:
    return all(
        close_enough(left.affine[row][column], right.affine[row][column])
        for row in range(4)
        for column in range(4)
    )


def max_affine_element_delta(left: NiftiHeader, right: NiftiHeader) -> float:
    return max(
        abs(left.affine[row][column] - right.affine[row][column])
        for row in range(4)
        for column in range(4)
    )


def max_corner_displacement_mm(left: NiftiHeader, right: NiftiHeader) -> float:
    if left.shape != right.shape or len(left.shape) != 3:
        return math.nan
    largest = 0.0
    for voxel in itertools.product(*((0, size - 1) for size in left.shape)):
        left_world = tuple(
            sum(left.affine[row][column] * voxel[column] for column in range(3))
            + left.affine[row][3]
            for row in range(3)
        )
        right_world = tuple(
            sum(right.affine[row][column] * voxel[column] for column in range(3))
            + right.affine[row][3]
            for row in range(3)
        )
        largest = max(
            largest,
            math.sqrt(sum((first - second) ** 2 for first, second in zip(left_world, right_world))),
        )
    return largest


def find_subjects(root: Path) -> dict[str, tuple[Path, Path]]:
    subjects: dict[str, tuple[Path, Path]] = {}
    for source_partition in ("train", "valid"):
        image_root = root / source_partition
        label_root = image_root / "labels"
        if not image_root.is_dir() or not label_root.is_dir():
            continue
        for subject_dir in sorted(image_root.iterdir()):
            if not subject_dir.is_dir() or subject_dir.name == "labels":
                continue
            subject_id = subject_dir.name
            label_path = label_root / f"{subject_id}_Labels.nii"
            if subject_id in subjects:
                raise AuditError(f"subject {subject_id} occurs in more than one source partition")
            subjects[subject_id] = (subject_dir, label_path)
    return subjects


def subject_report(subject_id: str, image_dir: Path, label_path: Path) -> dict:
    issues: list[str] = []
    images: dict[str, NiftiHeader] = {}
    geometry_alignment: dict[str, dict[str, float]] = {}
    for modality in MODALITIES:
        path = image_dir / f"{subject_id}_{modality}.nii"
        if not path.is_file():
            issues.append(f"missing {modality} modality: {path}")
            continue
        try:
            images[modality] = read_header(path)
        except AuditError as error:
            issues.append(f"invalid {modality} modality: {error}")

    label: NiftiHeader | None = None
    label_values: dict[str, int] = {}
    if not label_path.is_file():
        issues.append(f"missing label: {label_path}")
    else:
        try:
            label = read_header(label_path)
            label_values = read_label_summary(label_path, label)
        except AuditError as error:
            issues.append(f"invalid label: {error}")

    if label is not None:
        if label.datatype != LABEL_DATATYPE_UINT8:
            issues.append(f"label datatype is {label.datatype_name}, expected uint8")
        if label.scl_slope not in (0.0, 1.0) or label.scl_inter != 0.0:
            issues.append("label has non-identity intensity scaling")
        if set(label_values) - {"0", "1"}:
            issues.append(f"label is not binary; values are {sorted(label_values)}")
        if label_values.get("1", 0) == 0:
            issues.append("label has no positive IVD voxels")

        for modality, image in images.items():
            if image.shape != label.shape:
                issues.append(f"{modality} shape {image.shape} != label shape {label.shape}")
            if image.orientation != label.orientation:
                issues.append(
                    f"{modality} orientation {image.orientation} != label orientation {label.orientation}"
                )
            max_element_delta = max_affine_element_delta(image, label)
            max_corner_delta = max_corner_displacement_mm(image, label)
            geometry_alignment[modality] = {
                "max_affine_element_delta": max_element_delta,
                "max_corner_displacement_mm": max_corner_delta,
            }
            if not affine_matches(image, label):
                issues.append(
                    f"{modality} affine differs from label (max element delta "
                    f"{max_element_delta:.6g}; maximum corner displacement "
                    f"{max_corner_delta:.6g} mm)"
                )

    geometry = None
    if label is not None:
        geometry = {
            "shape": list(label.shape),
            "spacing_mm": list(label.spacing_mm),
            "orientation": label.orientation,
            "affine_source": label.affine_source,
        }
    return {
        "subject_id": subject_id,
        "source_partition": "valid" if subject_id in FIXED_TEST_SUBJECTS else "train",
        "model_partition": "test" if subject_id in FIXED_TEST_SUBJECTS else "train",
        "development_candidate": subject_id in {"02", "12"},
        "image_dir": str(image_dir),
        "label_path": str(label_path),
        "geometry": geometry,
        "modalities": {modality: asdict(header) for modality, header in images.items()},
        "label": asdict(label) if label is not None else None,
        "label_value_counts": label_values,
        "geometry_alignment": geometry_alignment,
        "issues": issues,
        "status": "PASS" if not issues else "FAIL",
    }


def markdown_report(report: dict) -> str:
    status = report["summary"]["status"]
    lines = [
        "# IVDM3Seg NIfTI Alignment Audit",
        "",
        f"- **Generated:** {report['generated_at_utc']}",
        f"- **Dataset root:** `{report['dataset_root']}`",
        f"- **Status:** **{status}**",
        f"- **Subjects audited:** {report['summary']['subject_count']}",
        f"- **Fixed test subjects:** {', '.join(FIXED_TEST_SUBJECTS)}",
        "- **Geometry comparator:** shape, axis orientation, and affine "
        f"(absolute element tolerance `{AFFINE_ELEMENT_TOLERANCE:g}`).",
        "",
        "| Subject | Model partition | Shape | Spacing (mm) | Orientation | Label voxels (0 / 1) | Max corner Δ (mm) | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for subject in report["subjects"]:
        geometry = subject["geometry"] or {}
        shape = " × ".join(str(value) for value in geometry.get("shape", ("—",)))
        spacing = ", ".join(f"{value:g}" for value in geometry.get("spacing_mm", ())) or "—"
        counts = subject["label_value_counts"]
        label_counts = f"{counts.get('0', 0)} / {counts.get('1', 0)}" if counts else "—"
        max_corner_delta = max(
            (item["max_corner_displacement_mm"] for item in subject["geometry_alignment"].values()),
            default=math.nan,
        )
        corner_delta = f"{max_corner_delta:.6g}" if not math.isnan(max_corner_delta) else "—"
        lines.append(
            f"| {subject['subject_id']} | {subject['model_partition']} | {shape} | {spacing} | "
            f"{geometry.get('orientation', '—')} | {label_counts} | {corner_delta} | {subject['status']} |"
        )
    lines.extend(["", "## Validation results", ""])
    if status == "PASS":
        lines.append(
            "All expected subjects, modalities, and labels are present. Every modality aligns with its "
            "label, and all labels are unscaled binary `uint8` volumes with positive IVD voxels."
        )
    else:
        lines.append("The following checks failed:")
        lines.append("")
        for subject in report["subjects"]:
            for issue in subject["issues"]:
                lines.append(f"- Subject `{subject['subject_id']}`: {issue}")
        for issue in report["dataset_issues"]:
            lines.append(f"- Dataset: {issue}")
    lines.append("")
    return "\n".join(lines)


def run_audit(dataset_root: Path) -> dict:
    discovered = find_subjects(dataset_root)
    dataset_issues: list[str] = []
    discovered_ids = tuple(sorted(discovered))
    if discovered_ids != EXPECTED_SUBJECTS:
        missing = sorted(set(EXPECTED_SUBJECTS) - set(discovered_ids))
        unexpected = sorted(set(discovered_ids) - set(EXPECTED_SUBJECTS))
        if missing:
            dataset_issues.append(f"missing expected subject IDs: {', '.join(missing)}")
        if unexpected:
            dataset_issues.append(f"unexpected subject IDs: {', '.join(unexpected)}")
    for subject_id in FIXED_TEST_SUBJECTS:
        if subject_id in discovered and discovered[subject_id][0].parent.name != "valid":
            dataset_issues.append(f"fixed test subject {subject_id} is not under valid/")
    for subject_id in set(discovered) - set(FIXED_TEST_SUBJECTS):
        if discovered[subject_id][0].parent.name != "train":
            dataset_issues.append(f"training subject {subject_id} is not under train/")

    subjects = [subject_report(subject_id, *discovered[subject_id]) for subject_id in discovered_ids]
    failures = sum(subject["status"] == "FAIL" for subject in subjects) + bool(dataset_issues)
    return {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_root": str(dataset_root),
        "audit_version": 1,
        "expected_subjects": list(EXPECTED_SUBJECTS),
        "fixed_test_subjects": list(FIXED_TEST_SUBJECTS),
        "dataset_issues": dataset_issues,
        "subjects": subjects,
        "summary": {
            "subject_count": len(subjects),
            "passing_subject_count": sum(subject["status"] == "PASS" for subject in subjects),
            "failing_subject_count": sum(subject["status"] == "FAIL" for subject in subjects),
            "status": "PASS" if not failures else "FAIL",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("IVDM3Seg"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data-audit"))
    args = parser.parse_args()

    report = run_audit(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "ivdm3seg-nifti-audit.json"
    markdown_path = args.output_dir / "ivdm3seg-nifti-audit.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(f"{report['summary']['status']}: {report['summary']['passing_subject_count']}/"
          f"{report['summary']['subject_count']} subjects passed")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0 if report["summary"]["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
