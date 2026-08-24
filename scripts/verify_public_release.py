#!/usr/bin/env python3
"""Reject restricted IVDM3Seg artifacts from a staged public release directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PROHIBITED_DIRECTORY_NAMES = frozenset(
    {".uv-python", ".venv", "__pycache__", "artifacts", "data", "ivdm3seg"}
)
PROHIBITED_FILENAMES = frozenset({"ivdm3seg.zip"})
PROHIBITED_SUFFIXES = (
    ".nii.gz",
    ".tar.gz",
    ".ckpt",
    ".dcm",
    ".jpeg",
    ".mha",
    ".mhd",
    ".nrrd",
    ".onnx",
    ".pth",
    ".tiff",
    ".tif",
    ".jpg",
    ".nii",
    ".png",
    ".pt",
    ".tar",
    ".zip",
)


def relative_display_path(root: Path, candidate: Path) -> str:
    """Return a stable path suitable for a release-review error message."""
    return candidate.relative_to(root).as_posix()


def find_prohibited_paths(root: Path) -> list[str]:
    """Return every known restricted path without descending into bad directories."""
    findings: list[str] = []
    for current, directory_names, file_names in os.walk(root, topdown=True):
        current_path = Path(current)

        retained_directories: list[str] = []
        for directory_name in sorted(directory_names):
            directory_path = current_path / directory_name
            if directory_name.casefold() in PROHIBITED_DIRECTORY_NAMES:
                findings.append(f"directory: {relative_display_path(root, directory_path)}")
            else:
                retained_directories.append(directory_name)
        directory_names[:] = retained_directories

        for file_name in sorted(file_names):
            file_path = current_path / file_name
            normalized_name = file_name.casefold()
            if normalized_name in PROHIBITED_FILENAMES or normalized_name.endswith(
                PROHIBITED_SUFFIXES
            ):
                findings.append(f"file: {relative_display_path(root, file_path)}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_root", type=Path, help="fresh directory staged for public release")
    args = parser.parse_args()
    release_root = args.release_root.resolve()

    if not release_root.is_dir():
        parser.error(f"release root is not a directory: {release_root}")

    prohibited_paths = find_prohibited_paths(release_root)
    if prohibited_paths:
        print("restricted artifacts found; do not publish this directory:")
        for prohibited_path in prohibited_paths:
            print(f"- {prohibited_path}")
        return 1

    print(f"release check passed: {release_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

