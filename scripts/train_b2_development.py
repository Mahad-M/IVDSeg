"""Prepare and train B2's development-only four-modality 3D nnU-Net baseline."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ivdseg.manifest import FIXED_TEST_SUBJECTS
from ivdseg.nnunet import (
    B2_DATASET_ID,
    B2_DATASET_NAME,
    B2_DEVELOPMENT_TRAIN_SUBJECT_IDS,
    B2_DEVELOPMENT_VALIDATION_SUBJECT_IDS,
    evaluate_b2_development_predictions,
    export_b2_nnunet_dataset,
)
from ivdseg.training import verify_cuda_runtime


EXPERIMENT_ID = "B2-nnunet-3d"
TRAINER_NAME = "nnUNetTrainerB2NoMirroring"
CONFIGURATION = "3d_fullres"
FOLD = 0


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _cli(name: str) -> str:
    path = Path(sys.executable).with_name(name)
    if not path.is_file():
        raise FileNotFoundError(f"nnU-Net console command is unavailable: {path}")
    return str(path)


def _environment(
    *, raw_root: Path, preprocessed_root: Path, results_root: Path, run_dir: Path, seed: int
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "nnUNet_raw": str(raw_root),
            "nnUNet_preprocessed": str(preprocessed_root),
            "nnUNet_results": str(results_root),
            "nnUNet_extTrainer": str(PROJECT_ROOT / "ivdseg" / "nnunet_extensions"),
            "nnUNet_n_proc_DA": "4",
            "IVDSEG_B2_SEED": str(seed),
            "MPLCONFIGDIR": str(run_dir / "matplotlib"),
        }
    )
    return environment


def _write_config(
    path: Path,
    *,
    experiment_id: str,
    seed: int,
    manifest: Path,
    dataset_root: Path,
    nnunet_raw_root: Path,
    nnunet_preprocessed_root: Path,
    nnunet_results_root: Path,
    run_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_id": experiment_id,
        "seed": seed,
        "b2_development": {
            "framework": "nnU-Net v2",
            "framework_version": importlib.metadata.version("nnunetv2"),
            "dataset_id": B2_DATASET_ID,
            "dataset_name": B2_DATASET_NAME,
            "manifest_path": str(manifest),
            "dataset_root": str(dataset_root),
            "nnunet_raw_root": str(nnunet_raw_root),
            "nnunet_preprocessed_root": str(nnunet_preprocessed_root),
            "nnunet_results_root": str(nnunet_results_root),
            "run_dir": str(run_dir),
            "trainer": TRAINER_NAME,
            "configuration": CONFIGURATION,
            "fold": FOLD,
            "train_subject_ids": list(B2_DEVELOPMENT_TRAIN_SUBJECT_IDS),
            "validation_subject_ids": list(B2_DEVELOPMENT_VALIDATION_SUBJECT_IDS),
            "fixed_test_subjects_excluded": list(FIXED_TEST_SUBJECTS),
            "channels": ["fat", "inn", "opp", "water"],
            "label": "binary semantic IVD (0 background, 1 IVD)",
            "planning": "nnUNetv2_plan_and_preprocess default ExperimentPlanner; 3d_fullres only",
            "training_length": "nnU-Net default 1000 epochs, 250 iterations per epoch",
            "augmentation": "nnU-Net default self-configured augmentation with mirroring disabled",
            "selection": "nnU-Net checkpoint_best on the explicit fold-0 10/2 split; report native development Dice from its saved validation predictions",
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_run_card(path: Path, *, experiment_id: str, seed: int, command: str) -> bool:
    if path.exists():
        content = path.read_text(encoding="utf-8")
        expected_heading = f"# {experiment_id} - Four-modality 3D nnU-Net v2 development baseline - Seed {seed}"
        if not content.startswith(expected_heading + "\n"):
            raise FileExistsError(f"experiment card already exists with a different run definition: {path}")
        if "**Status:** Complete" in content:
            raise FileExistsError(f"experiment card records a completed run: {path}; use a new experiment ID or seed")
        if not re.search(r"(?m)^\*\*Status:\*\* (?:Running|Failed|Interrupted / resumable)\s*$", content):
            raise ValueError(f"experiment card has an unrecognized status and cannot be resumed: {path}")
        content = re.sub(r"(?m)^\*\*Status:\*\* .+$", "**Status:** Running", content, count=1)
        content = re.sub(r"(?m)^\*\*Finished:\*\*.*$", "**Finished:**", content, count=1)
        content = re.sub(
            r"(?m)^- \*\*Outcome:\*\*.*$",
            "- **Outcome:** Running after a resumable interruption.",
            content,
            count=1,
        )
        if "- **Resume record:**" not in content:
            content = content.replace(
                "- **Decision / next action:**",
                f"- **Resume record:** Restarted at {_timestamp()} after a Modal interruption; nnU-Net will continue from its newest persisted checkpoint when one exists.\n"
                "- **Decision / next action:**",
                1,
            )
        path.write_text(content, encoding="utf-8")
        return True
    content = f"""# {experiment_id} - Four-modality 3D nnU-Net v2 development baseline - Seed {seed}

**Status:** Running  
**Started:** {_timestamp()}  
**Finished:**

## Purpose

- **Hypothesis:** A self-configured volumetric nnU-Net v2 baseline can improve semantic IVD segmentation by using complete four-modality 3D context.
- **Comparison:** Independent volumetric reference. It uses the project's same ten development-training subjects and two selection subjects, but changes the model family and input from 2.5D slices to nnU-Net's planned four-channel 3D volumes.
- **Plan reference:** `docs/plans/rf-detr-2-5d-ivd-segmentation.md`

## Immutable Run Definition

- **Framework:** `nnunetv2=={importlib.metadata.version('nnunetv2')}`; default `3d_fullres` ExperimentPlanner configuration and `nnUNetTrainerB2NoMirroring` (default nnU-Net augmentation except no anatomical-axis mirroring).
- **Source data and split:** Canonical NIfTI only; train `01,04,05,06,08,09,11,13,15,16`, development validation `02,12`, and fixed holdout `03,07,10,14` excluded. No PNGs are read or exported.
- **nnU-Net conversion:** `Dataset502_IVDM3SegB2` contains four aligned NIfTI channels per permitted case (`fat`, `inn`, `opp`, `water`) and binary semantic labels. Manifest-directed nearest-neighbor label resampling applies only to subject 16 before export.
- **Planning and training:** `nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity -c 3d_fullres`, then the standard 1000-epoch / 250-iterations-per-epoch training schedule on explicit custom fold 0. The post-training native Dice reads only saved validation predictions from `checkpoint_best`.
- **Command:** `{command}`.

## Artifacts

- **Configuration:** B2 run `config.json`.
- **Raw / planned data:** `nnunet/raw/Dataset502_IVDM3SegB2/`; `nnunet/preprocessed/Dataset502_IVDM3SegB2/` in the runs Volume.
- **Checkpoints / logs:** `nnunet/results/Dataset502_IVDM3SegB2/nnUNetTrainerB2NoMirroring__nnUNetPlans__3d_fullres/fold_0/`.
- **Development predictions / metrics:** The checkpoint-best `validation/` folder and `metrics/development-3d.json` in the B2 run directory.

## Results

| Subject | Dice | Notes |
| --- | ---: | --- |
| 02 | | Development-only subject. |
| 12 | | Development-only subject. |
| Macro mean | | |

## Conclusion

- **Outcome:** Running.
- **Failure or deviation from plan:**
- **Decision / next action:** Read the immutable planner, checkpoint, and saved validation artifacts before deciding whether B2 is eligible for all-12-subject final retraining. Fixed-holdout access remains prohibited.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return False


def _finish_run_card(path: Path, *, status: str, conclusion: str, rows: str | None = None) -> None:
    content = path.read_text(encoding="utf-8")
    content = re.sub(r"(?m)^\*\*Status:\*\* .+$", f"**Status:** {status}", content, count=1)
    content = re.sub(r"(?m)^\*\*Finished:\*\*.*$", f"**Finished:** {_timestamp()}", content, count=1)
    content = re.sub(r"(?m)^- \*\*Outcome:\*\*.*$", f"- **Outcome:** {conclusion}", content, count=1)
    if rows is not None:
        content = content.replace(
            "| 02 | | Development-only subject. |\n| 12 | | Development-only subject. |\n| Macro mean | | |",
            rows,
        )
    path.write_text(content, encoding="utf-8")


def _checkpoint_epoch(checkpoint_path: Path) -> int | None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("current_epoch")
    return int(epoch) if isinstance(epoch, int) else None


def _training_command(*, resume: bool) -> list[str]:
    command = [
        _cli("nnUNetv2_train"),
        str(B2_DATASET_ID),
        CONFIGURATION,
        str(FOLD),
        "-tr",
        TRAINER_NAME,
        "--val_best",
        "-device",
        "cuda",
    ]
    if resume:
        command.append("--c")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--nnunet-raw-root", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed-root", type=Path, required=True)
    parser.add_argument("--nnunet-results-root", type=Path, required=True)
    parser.add_argument("--experiment-card-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir
    card_path = args.experiment_card_dir / f"{args.experiment_id}-{args.seed}.md"
    command = shlex.join([sys.executable, *sys.argv])
    resumed = _write_run_card(card_path, experiment_id=args.experiment_id, seed=args.seed, command=command)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_config(
            run_dir / "config.json",
            experiment_id=args.experiment_id,
            seed=args.seed,
            manifest=args.manifest,
            dataset_root=args.dataset_root,
            nnunet_raw_root=args.nnunet_raw_root,
            nnunet_preprocessed_root=args.nnunet_preprocessed_root,
            nnunet_results_root=args.nnunet_results_root,
            run_dir=run_dir,
        )
        verify_cuda_runtime()
        raw_dataset_dir = export_b2_nnunet_dataset(
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            nnunet_raw_root=args.nnunet_raw_root,
        )
        environment = _environment(
            raw_root=args.nnunet_raw_root,
            preprocessed_root=args.nnunet_preprocessed_root,
            results_root=args.nnunet_results_root,
            run_dir=run_dir,
            seed=args.seed,
        )
        planning_command = [
            _cli("nnUNetv2_plan_and_preprocess"),
            "-d",
            str(B2_DATASET_ID),
            "--verify_dataset_integrity",
            "-c",
            CONFIGURATION,
            "-np",
            "2",
            "--no_pbar",
        ]
        print(f"[ivdseg-b2] planning command={shlex.join(planning_command)}", flush=True)
        subprocess.run(planning_command, check=True, env=environment)
        training_command = _training_command(resume=resumed)
        print(f"[ivdseg-b2] training command={shlex.join(training_command)}", flush=True)
        subprocess.run(training_command, check=True, env=environment)
        fold_dir = (
            args.nnunet_results_root
            / B2_DATASET_NAME
            / f"{TRAINER_NAME}__nnUNetPlans__{CONFIGURATION}"
            / f"fold_{FOLD}"
        )
        checkpoint_path = fold_dir / "checkpoint_best.pth"
        prediction_dir = fold_dir / "validation"
        summary = evaluate_b2_development_predictions(
            raw_dataset_dir=raw_dataset_dir,
            prediction_dir=prediction_dir,
        )
        metrics_path = run_dir / "metrics" / "development-3d.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
        checkpoint_epoch = _checkpoint_epoch(checkpoint_path)
        rows = "\n".join(
            [
                *(f"| {subject.subject_id} | {subject.dice:.6f} | checkpoint_best native prediction |" for subject in summary.subjects),
                f"| Macro mean | {summary.macro_dice:.6f} | checkpoint_best; stored epoch={checkpoint_epoch} |",
            ]
        )
        _finish_run_card(
            card_path,
            status="Complete",
            rows=rows,
            conclusion=(
                f"Development training and checkpoint-best validation completed. Macro native 3D Dice={summary.macro_dice:.6f}; "
                f"configuration is `{run_dir / 'config.json'}`, checkpoint is `{checkpoint_path}`, and metrics are `{metrics_path}`."
            ),
        )
    except KeyboardInterrupt as error:
        _finish_run_card(
            card_path,
            status="Interrupted / resumable",
            conclusion=f"B2 development run was interrupted: {type(error).__name__}: {error}",
        )
        raise
    except BaseException as error:
        _finish_run_card(
            card_path,
            status="Failed",
            conclusion=f"B2 development run stopped: {type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
