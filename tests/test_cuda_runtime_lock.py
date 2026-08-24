from __future__ import annotations

from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTORCH_CU126_INDEX = "https://download.pytorch.org/whl/cu126"


def _packages(lock_data: dict[str, object], name: str) -> list[dict[str, object]]:
    packages = [package for package in lock_data["package"] if package["name"] == name]  # type: ignore[index]
    if not packages:
        raise AssertionError(f"{name} is absent from uv.lock")
    return packages  # type: ignore[return-value]


def test_project_locks_the_pascal_compatible_pytorch_pair() -> None:
    """Keep uv from silently resolving a wheel that omits GTX 1050 kernels."""
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(pyproject["project"]["dependencies"])
    assert "torch==2.8.0" in dependencies
    assert "torchvision==0.23.0" in dependencies

    sources = pyproject["tool"]["uv"]["sources"]
    assert sources["torch"] == {"index": "pytorch-cu126"}
    assert sources["torchvision"] == {"index": "pytorch-cu126"}
    assert pyproject["tool"]["uv"]["index"] == [
        {"name": "pytorch-cu126", "url": PYTORCH_CU126_INDEX, "explicit": True}
    ]

    lock_data = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8"))
    for package_name, version in (("torch", "2.8.0+cu126"), ("torchvision", "0.23.0+cu126")):
        assert any(
            package["version"] == version
            and package["source"] == {"registry": PYTORCH_CU126_INDEX}
            for package in _packages(lock_data, package_name)
        )
