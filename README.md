# Qwen, Sailor and Mistral soft voting + Qwen correction

The `kaggle-linux` notebook launches the committed recipe. Experiments are controlled through chat and commits; the notebook has no experiment switches.

1. Upload [VGEC26-grouped-validation-20261015.zip](dataset/kaggle-grouped-validation-20261015/VGEC26-grouped-validation-20261015.zip) as a **private** Kaggle dataset. Attach only that dataset version. Its enclosing upload folder can have any name.
2. Import `notebooks/kaggle_two_phase.ipynb`, enable Internet and GPU T4×2. Set repository-read `GITHUB_TOKEN`; enable `WANDB_API_KEY` for live monitoring. `HF_TOKEN` is optional for these ungated bases.
3. Save & Run All. The launcher clones `kaggle-linux`, installs pinned dependencies while preserving Kaggle CUDA PyTorch, then checks the committed dataset hashes and model placement.
4. Collect `run/results.zip`. W&B's root run is named **run**; its unique group contains the individual model runs.

| Stage | Model | Epoch limit | Seed |
|---|---|---:|---:|
| Classifier1 | Qwen/Qwen2.5-7B-Instruct |5|42|
| Classifier2 | sail/Sailor-7B-Chat |5|42|
| Classifier3 | mistralai/Mistral-7B-Instruct-v0.3 |5|42|
| Corrector | Qwen/Qwen2.5-7B-Instruct |6|42|

**Soft voting:** for each class, average the three member probabilities with weights1/3. The highest mean decides the class; exact ties use the lower class ID. Hard-vote counts are diagnostic only. Every member's saved adapter/head is retained and checked before reuse. Each classifier selects its own epoch on the same78 original validation rows.

The approved grouped validation package provides1145 training exposures per stage,229 per class; canonical populations are778 classifier /766 corrector. Both tasks share split42 and the same78-row validation set. Related source/target/parent/family/template/near-duplicate components are withheld together. Dataset bytes and original annotations are unchanged by this update. This is an internal development partition: earlier experiments exposed some originals, so it is not historically blind. Public inputs remain inference-only.

Standard LoRA uses FP16 frozen bases, FP32 adapters/head, rank16/alpha32 and effective batch8. Each model runs sequentially across both T4s in one process. Quantization, folds and hints are disabled. Fresh child processes, explicit exit/join, release checks and per-load memory budgets protect model transitions. Runtime memory availability can still cause a guarded failure. Strict RNG controls target replay on the same hardware/software; they do not guarantee cross-platform bit identity.

The root W&B run reports live model-prefixed loss/validation metrics, stage outcomes, current stage and elapsed time, with a30-second heartbeat until export/archive finishes. Native system monitoring samples every15seconds: all visible GPUs (utilization, VRAM, temperature/power when supported), process-tree CPU/RAM, free RAM, root/output disk usage/I/O and network traffic. Open the run's **System** section to view these charts. Only numeric summaries and corpus-safe metadata are uploaded. Without a W&B key, local status/artifacts remain available.

The default is `configs/experiment.json`, selecting `configs/validation_baseline.json` and `configs/validation_correction.json`. A separately committed `configs/full_training_experiment.json` preserves the older full-training dataset option; changing experiments requires a committed recipe change, not a notebook toggle.

Verification and limits, exact pinned revisions/counts, historical superseded checks and artifact paths are in `agents/experiments/MODEL_SOFT_VOTING_20261015.md`/.json. Mechanical tests do not establish an ensemble quality gain. No full7B training or new official score is claimed. Corpus, derivatives, run outputs and upload ZIPs remain private.
