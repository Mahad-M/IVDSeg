# IVD Segmentation Research Code

This repository contains the reproducible source code for four-modality 2.5D intervertebral-disc segmentation experiments on IVDM3Seg. It is a **code-only** release.

## What is included

- Python source, training/evaluation scripts, and tests;
- the locked Python 3.12 environment definition (`pyproject.toml` and `uv.lock`);
- the MIT License and citation metadata.

## What is not included

IVDM3Seg images, labels, access information, derived manifests, normalization profiles, predictions, checkpoints, logs, image overlays, and model weights are intentionally absent. Access to IVDM3Seg must be obtained independently from its data provider; this repository is not a route to the data.

## Environment

```bash
uv sync --group dev
uv run pytest -q
```

Run project Python commands through `uv run`. On the historical GTX 1050 host, verify the locked CUDA runtime before training:

```bash
uv run python -c "from ivdseg.training import verify_cuda_runtime; verify_cuda_runtime()"
```

## Reproducing the protocol

Only researchers independently authorized to access IVDM3Seg should create a local manifest and run the pipeline. The fixed final holdout case IDs are `03`, `07`, `10`, and `14`; do not use them in normalization, development selection, or threshold selection. The source data do not publish a case-ID-to-participant/timepoint crosswalk, so this protocol is case-ID-disjoint, not verified participant-disjoint.

## Citation and licence

The code is available under the MIT License. The source repository is `https://github.com/Mahad-M/IVDSeg`. Please cite the versioned archival release described in `CITATION.cff`; its DOI will be added after archival publication.
