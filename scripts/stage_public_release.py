#!/usr/bin/env python3
"""Create a fresh, code-only IVD Segmentation release staging directory."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Match the project's standalone script launch convention.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_public_release import find_prohibited_paths


ROOT_FILES = (".python-version", "pyproject.toml", "uv.lock", "LICENSE", "CITATION.cff")
SOURCE_DIRECTORIES = ("ivdseg", "tests")
EXCLUDED_SCRIPT_PREFIXES = ("modal_",)
EXCLUDED_SCRIPTS = {"render_manuscript_figures.py", "render_paper_docx.py"}
IGNORED_NAMES = {"__pycache__", ".pytest_cache"}
PUBLIC_README = Path("docs/reproducibility/public-code-readme.md")
RELEASE_CHECKLIST = Path("docs/reproducibility/release-checklist.md")


def ignore_generated(directory: str, names: list[str]) -> set[str]:
    """Omit caches and bytecode while retaining all public source files."""
    del directory
    return {name for name in names if name in IGNORED_NAMES or name.endswith((".pyc", ".pyo"))}


def public_script_paths(source_root: Path) -> list[Path]:
    """Return local pipeline scripts without cloud-specific or manuscript renderers."""
    return sorted(
        path
        for path in (source_root / "scripts").glob("*.py")
        if not path.name.startswith(EXCLUDED_SCRIPT_PREFIXES) and path.name not in EXCLUDED_SCRIPTS
    )


def stage_release(source_root: Path, destination: Path) -> None:
    """Copy the reviewed public subset into a new destination and validate it."""
    source_root = source_root.resolve()
    destination = destination.resolve()
    if destination == source_root or source_root in destination.parents:
        raise ValueError("release destination must not be the source root or nested inside it")
    if destination.exists():
        raise FileExistsError(f"release destination already exists: {destination}")

    destination.mkdir(parents=True)
    for relative_path in ROOT_FILES:
        source_path = source_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"required release file is missing: {source_path}")
        shutil.copy2(source_path, destination / relative_path)

    shutil.copy2(source_root / PUBLIC_README, destination / "README.md")
    shutil.copy2(source_root / RELEASE_CHECKLIST, destination / "RELEASE-CHECKLIST.md")

    for directory_name in SOURCE_DIRECTORIES:
        shutil.copytree(
            source_root / directory_name,
            destination / directory_name,
            ignore=ignore_generated,
        )

    scripts_destination = destination / "scripts"
    scripts_destination.mkdir()
    for source_path in public_script_paths(source_root):
        shutil.copy2(source_path, scripts_destination / source_path.name)

    prohibited_paths = find_prohibited_paths(destination)
    if prohibited_paths:
        raise RuntimeError(
            "public-release staging produced prohibited paths:\n"
            + "\n".join(f"- {path}" for path in prohibited_paths)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="new empty directory to create for GitHub upload",
    )
    args = parser.parse_args()
    stage_release(Path.cwd(), args.destination)
    print(f"Created validated code-only release stage: {args.destination}")


if __name__ == "__main__":
    main()
