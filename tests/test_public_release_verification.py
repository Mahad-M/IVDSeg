from pathlib import Path

from scripts.verify_public_release import find_prohibited_paths


def test_release_checker_accepts_source_only_tree(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("source-only release\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "train.py").write_text("print('safe')\n", encoding="utf-8")

    assert find_prohibited_paths(tmp_path) == []


def test_release_checker_reports_restricted_directories_and_files(tmp_path: Path) -> None:
    (tmp_path / "IVDM3Seg").mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "result.nii.gz").write_text("not a real image", encoding="utf-8")
    (tmp_path / "model.ckpt").write_text("not a real checkpoint", encoding="utf-8")

    assert find_prohibited_paths(tmp_path) == [
        "directory: IVDM3Seg",
        "directory: artifacts",
        "file: model.ckpt",
        "file: result.nii.gz",
    ]
