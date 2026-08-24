"""Select B1's sigmoid threshold on its completed development checkpoint only."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from ivdseg.datamodule import IVD2p5DSemanticDataset, _records_for_subject_ids
from ivdseg.normalization import load_profile
from ivdseg.unet_training import (
    B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS,
    SemanticDevelopmentThresholdGrid,
    load_b4_development_config,
    load_b4_model,
    load_b3_development_config,
    load_b3_model,
    load_b1_development_config,
    load_b1_model,
)
from ivdseg.training import verify_cuda_runtime


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _write_card(path: Path, *, experiment_id: str, source_run: str, checkpoint: Path) -> None:
    if path.exists():
        raise FileExistsError(f"experiment card already exists: {path}; use a new experiment ID")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# {experiment_id} - Development sigmoid-threshold selection - Seed 17

**Status:** Running  
**Started:** {_timestamp()}  
**Finished:**

## Purpose

- **Comparison:** Post-training operating-point selection for `{source_run}` only; no architecture, checkpoint, preprocessing, or data-split factor changes.
- **Immutable definition:** Checkpoint `{checkpoint}`; development subjects `02,12` only; sigmoid thresholds `0.05,0.10,...,0.95`; maximum macro reconstructed 3D Dice with an exact tie resolved to the lower cutoff.
- **Holdout:** Subjects `03,07,10,14` are excluded.

## Results

<!-- RESULTS -->

## Conclusion

- **Outcome:** Running.
""",
        encoding="utf-8",
    )


def _finish_card(path: Path, selection: dict[str, object]) -> None:
    content = path.read_text(encoding="utf-8")
    selected_threshold = float(selection["selected_probability_threshold"])
    selected_dice = float(selection["selected_macro_dice"])
    rows = "\n".join(
        f"| {entry['probability_threshold']:.2f} | {entry['macro_dice']:.6f} |"
        for entry in selection["thresholds"]  # type: ignore[index]
    )
    content = content.replace("<!-- RESULTS -->", "| Sigmoid threshold | Macro development 3D Dice |\n| ---: | ---: |\n" + rows)
    content = content.replace("**Status:** Running", "**Status:** Complete")
    content = content.replace("**Finished:**", f"**Finished:** {_timestamp()}")
    content = content.replace(
        "- **Outcome:** Running.",
        f"- **Outcome:** Selected sigmoid threshold `{selected_threshold:.2f}` with macro development 3D Dice `{selected_dice:.6f}`.",
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-config", type=Path, required=True)
    parser.add_argument("--source-config-kind", choices=("b1", "b3", "b4"), default="b1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", default="B1-resunet34-threshold")
    parser.add_argument("--experiment-card-dir", type=Path, default=Path("docs/research/experiments"))
    args = parser.parse_args()
    if args.source_config_kind == "b1":
        config = load_b1_development_config(args.source_run_config)
    elif args.source_config_kind == "b3":
        config = load_b3_development_config(args.source_run_config)
    else:
        config = load_b4_development_config(args.source_run_config)
    card_path = args.experiment_card_dir / f"{args.experiment_id}-{config.seed}.md"
    _write_card(card_path, experiment_id=args.experiment_id, source_run=config.experiment_id, checkpoint=args.checkpoint)
    try:
        verify_cuda_runtime()
        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
        dataset = IVD2p5DSemanticDataset(
            records=_records_for_subject_ids(manifest, B1_DEVELOPMENT_VALIDATION_SUBJECT_IDS),
            dataset_root=config.dataset_root,
            profile=load_profile(config.normalization_profile),
            resolution=config.resolution,
            augmentation=None,
        )
        loader = DataLoader(dataset, batch_size=config.micro_batch_size, shuffle=False, num_workers=config.num_workers)
        device = torch.device("cuda")
        if args.source_config_kind == "b1":
            model = load_b1_model(config, args.checkpoint, device=device)
        elif args.source_config_kind == "b3":
            model = load_b3_model(config, args.checkpoint, device=device)
        else:
            model = load_b4_model(config, args.checkpoint, device=device)
        grid = SemanticDevelopmentThresholdGrid.from_validation_dataset(dataset)
        with torch.inference_mode():
            for images, _masks, indices in loader:
                grid.add_batch(model(images.to(device, non_blocking=True)), indices)
        selection = grid.finalize().to_dict()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        _finish_card(card_path, selection)
    except BaseException:
        content = card_path.read_text(encoding="utf-8")
        content = content.replace("**Status:** Running", "**Status:** Failed").replace("**Finished:**", f"**Finished:** {_timestamp()}")
        card_path.write_text(content, encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
