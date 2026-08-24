"""Select B5's sigmoid cutoff from the best development checkpoint only."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ivdseg.swinunetr_3d import evaluate_b5_development_checkpoint, load_b5_development_config
from ivdseg.training import verify_cuda_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_b5_development_config(args.config)
    if args.device.startswith("cuda"):
        verify_cuda_runtime()
    device = torch.device(args.device)
    selection = evaluate_b5_development_checkpoint(config, args.checkpoint, device=device)
    output = args.output or config.run_dir / "metrics" / "development-threshold-grid.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": config.experiment_id,
        "checkpoint": str(args.checkpoint),
        "split": {"development_validation_subject_ids": ["02", "12"], "fixed_test_subjects_excluded": ["03", "07", "10", "14"]},
        **selection.to_dict(),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"B5 threshold selection: cutoff={selection.selected.probability_threshold:.2f} "
        f"macro_3d_dice={selection.selected.macro_dice:.6f} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
