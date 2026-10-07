# Kaggle baseline verification

The project is a standalone checkout of `https://github.com/rquq/GECC_VLSP_26.git`.
The baseline is at the repository root on `codex/kaggle-baseline`.
`notebooks/kaggle_smoke.ipynb` defaults to that repository and branch.

## Verified locally on 2026-10-07

- All 13 contract and Kaggle workflow tests passed under both system Python and
  the pinned training environment. Clone tests execute the notebook's source
  cell against real Git repositories, including shallow cloning of the requested
  branch, fetching updates on rerun, refusing local edits or another repository,
  and reporting a missing branch before training.
- Notebook JSON passed `nbformat.validate`, and every Python cell compiled.
- GPU preflight detected 401 records and a leakage-safe 320/81 split. It checked
  dependencies and source identity without creating a training output directory.
- A fresh clone of baseline commit `0f7e8f7c30a9476a1aa2c33e2302794d138ea359`
  completed the real Qwen2.5-0.5B smoke configuration through the Kaggle launcher:
  four optimizer steps, 32 training records, eight validation records, finite
  gradients and parameters, training loss `0.5886649861931801`, and reloading
  the saved adapter for evaluation. All eight generations parsed successfully.
  This is a pipeline check; eight rows do not establish competitive quality.
- The local device was one RTX 4060 Laptop GPU with torch `2.7.1+cu118`,
  transformers `4.51.3`, peft `0.15.2`, and accelerate `1.6.0`.
  Rank-zero peak allocated VRAM was approximately 2.87 GiB.
- The output ZIP passed its integrity check and contains adapter weights,
  tokenizer, predictions, metrics, identity metrics, and both run/source
  manifests. Source identity records the selected branch and exact commit.
- Actual two-process distributed training was checked separately on CPUs with
  a tiny random one-layer Qwen fixture and the same trainer, tokenizer, LoRA
  target modules, and data path. Both ranks completed two optimizer steps;
  rank zero saved/reloaded the adapter and produced reports. The manifest
  records `world_size=2` and effective batch 8. This validates distributed
  coordination, not model quality or the CUDA/NCCL backend.

## Kaggle execution

Enable Internet, choose **GPU T4 x2**, and attach `VGEC 2026_v1.json` privately.
Run all notebook cells with `MODE="smoke"`; inspect the saved metrics and adapter
before switching to `MODE="full"`. The launcher defaults to two CUDA workers.
Its `--num-gpus 1` option exists for local GPU diagnostics only.

A dual-T4 Kaggle session and the three-epoch full run have not been executed
from this machine. The notebook performs device/data/source preflight in that
session before training. Dataset files, the metric PDF, local environments,
training outputs, and archives are excluded from Git.
