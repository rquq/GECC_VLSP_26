# Sailor classifier and Qwen corrector — retained v2

The `kaggle-linux` branch trains the new retained-v2 dataset. The notebook launches the committed recipe; experiments remain chat/commit controlled, with no notebook toggles.

## Kaggle run

1. Download `dataset/kaggle-retained-v2-20261015/VGEC26-sailor-qwen-retained-v2-20261015.zip` and upload it as a **private** Kaggle dataset. Use this complete package, which includes `public_test.json` for inference. The earlier dataset-only ZIP remains preserved.
2. Import the current `notebooks/kaggle_two_phase.ipynb` from `kaggle-linux`. The augmentation notebook targets another branch and is a different workflow.
3. Attach only the new data version, enable Internet and GPU T4 ×2, and enable the repository-read `GITHUB_TOKEN` secret. `WANDB_API_KEY` and `HF_TOKEN` remain optional.
4. Save & Run All. The launcher clones a fresh `kaggle-linux` checkout, installs the pinned environment while preserving Kaggle CUDA PyTorch, and runs `tools.run_experiment`.
5. Collect the exported `run/results.zip`, containing root `results.json` with singleton-list corrections.

The enclosing Kaggle dataset name can be arbitrary. The runner finds the archive or recursively discovers the extracted canonical filenames, then matches the three training hashes pinned in `configs/experiment.json`. Existing runtime integrity, size and memory checks remain enabled.

## Fixed experiment

| Stage | Model | Training exposures | Epochs |
|---|---|---:|---:|
| Classification | sail/Sailor-7B-Chat |1360|5|
| Correction | Qwen/Qwen2.5-7B-Instruct |1360|6|

Both stages have272 exposures per class. Stored populations before repeats are945 classifier /933 corrector rows. All401 original classifier rows are retained unchanged, including the8 notes. Correction retains389 original rows, including112 exact abbreviated organizer references that are never oversampled.11 explanatory-note targets and the inline-annotation ID11 remain excluded from correction. Legacy192 pairs, new code320 pairs and32 external clean controls retain their recorded provenance and licenses.

Standard LoRA uses frozen FP16 bases and FP32 trainable adapters/head, rank16, alpha32 and effective batch8. One process spans both T4 GPUs. Quantization, folds and hints are disabled. Source revisions, learning rates, precision, loss, placement and generation settings are preserved. Final epochs are fixed with zero independent validation; no local quality score is fabricated. Public inputs remain inference-only.

The training loader supports the retained v1/v2 formats through pure provenance/rule helpers. It does not import or launch the augmentation pipeline. Original stage files and targets in the new upload package are copied unchanged; the package adds the existing public inference input, hashes and upload instructions.

## Preparation status

Prepared and committed at the user request. **No tests, smoke runs, model preflight, full-size training or new score verification were run for this branch update**, as explicitly requested. Historical tests of the source v2 dataset remain historical evidence. The Kaggle run itself has not been started.

Read `agents/experiments/KAGGLE_RETAINED_V2_20261015.md`/.json for package hashes, exact changes and the skipped-verification record. Dataset provenance, public-output audit and all extra-data types are recorded in `PUBLIC_ABBREVIATION_20261015.md`/.json and `RETAINED_BALANCED_20261015.md`/.json. Keep corpus data, derivatives and run outputs private.
