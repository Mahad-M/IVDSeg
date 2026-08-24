"""Measure one full B5 Swin UNETR training step before committing the ROI budget."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.nn import functional as functional

from ivdseg.swinunetr_3d import B5_ALLOWED_ROI_SIZES, B5_FALLBACK_ROI_SIZE, B5_PRIMARY_ROI_SIZE, create_b5_model
from ivdseg.training import verify_cuda_runtime


def _one_step(roi_size: tuple[int, int, int], *, seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = create_b5_model(use_checkpoint=True).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    images = torch.randn((1, 4, *roi_size), device="cuda")
    targets = torch.rand((1, *roi_size), device="cuda") > 0.5
    torch.cuda.synchronize()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(images)
        bce = functional.binary_cross_entropy_with_logits(logits[:, 0], targets.to(dtype=logits.dtype))
        probabilities = torch.sigmoid(logits[:, 0])
        target_float = targets.to(dtype=logits.dtype)
        dice = 1.0 - (
            (2.0 * (probabilities * target_float).sum(dim=(1, 2, 3)) + 1.0)
            / (probabilities.sum(dim=(1, 2, 3)) + target_float.sum(dim=(1, 2, 3)) + 1.0)
        ).mean()
        loss = bce + dice
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_bytes = int(torch.cuda.max_memory_allocated())
    del images, targets, logits, loss, optimizer, model
    torch.cuda.empty_cache()
    return {
        "roi_size": list(roi_size),
        "status": "passed",
        "seconds_per_full_training_step": elapsed,
        "peak_allocated_bytes": peak_bytes,
        "peak_allocated_gib": peak_bytes / (1024**3),
    }


def _is_oom(error: RuntimeError) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    verify_cuda_runtime()
    candidates = (B5_PRIMARY_ROI_SIZE, B5_FALLBACK_ROI_SIZE)
    results: list[dict[str, object]] = []
    selected: tuple[int, int, int] | None = None
    for roi_size in candidates:
        try:
            result = _one_step(roi_size, seed=args.seed)
        except RuntimeError as error:
            if not _is_oom(error):
                raise
            torch.cuda.empty_cache()
            result = {"roi_size": list(roi_size), "status": "out_of_memory", "error": str(error)}
        results.append(result)
        if result["status"] == "passed":
            selected = roi_size
            break
    if selected is None:
        raise RuntimeError(f"B5 preflight exhausted the only allowed ROI sizes: {B5_ALLOWED_ROI_SIZES}")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "synthetic one-step L4 memory/runtime preflight; no NIfTI data or labels loaded",
        "model": "MONAI SwinUNETR Base feature_size=48, 4 inputs, 1 output, gradient checkpointing",
        "precision": "bf16 autocast",
        "candidates": results,
        "selected_roi_size": list(selected),
        "fallback_policy": (
            "Use 32x256x256 when it passes. Only a reproducible memory failure permits the predeclared 64x128x128 fallback."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"B5 preflight selected roi={selected[0]}x{selected[1]}x{selected[2]} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
