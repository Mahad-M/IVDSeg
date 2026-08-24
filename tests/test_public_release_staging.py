from pathlib import Path

from scripts.stage_public_release import stage_release


def test_stage_release_copies_only_the_reviewed_public_subset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for filename in (".python-version", "pyproject.toml", "uv.lock", "LICENSE", "CITATION.cff"):
        (source / filename).write_text(filename, encoding="utf-8")
    (source / "ivdseg").mkdir()
    (source / "ivdseg" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "tests").mkdir()
    (source / "tests" / "test_module.py").write_text("assert True\n", encoding="utf-8")
    (source / "scripts").mkdir()
    (source / "scripts" / "train.py").write_text("print('train')\n", encoding="utf-8")
    (source / "scripts" / "modal_train.py").write_text("print('cloud')\n", encoding="utf-8")
    (source / "scripts" / "render_paper_docx.py").write_text("print('paper')\n", encoding="utf-8")
    (source / "IVDM3Seg").mkdir()
    (source / "IVDM3Seg" / "case.nii").write_text("restricted", encoding="utf-8")
    (source / "docs/reproducibility").mkdir(parents=True)
    (source / "docs/reproducibility/public-code-readme.md").write_text("public README\n", encoding="utf-8")
    (source / "docs/reproducibility/release-checklist.md").write_text("checklist\n", encoding="utf-8")

    destination = tmp_path / "stage"
    stage_release(source, destination)

    assert (destination / "README.md").read_text(encoding="utf-8") == "public README\n"
    assert (destination / "scripts/train.py").is_file()
    assert not (destination / "scripts/modal_train.py").exists()
    assert not (destination / "scripts/render_paper_docx.py").exists()
    assert not (destination / "IVDM3Seg").exists()
