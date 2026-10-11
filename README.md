# Kaggle Sailor2 Task 1 → Task 2

This branch keeps the existing Sailor2 Task 1 baseline and restores alpha's
Task 2 correction stage. Task 2 prompts contain the source sentence and Task 1's
predicted label plus error-type description. Training hints come from train-only
out-of-fold (OOF) classifiers; validation hints come from the selected main
classifier. Gold labels are not substituted into prompts. These are independently
initialized QLoRA adapters, not classifier-to-corrector weight transfer. Every
input reaches correction, including predicted class 0.

## Run on Kaggle

Import `notebooks/kaggle_two_phase.ipynb` into a **private** Kaggle notebook.
Enable Internet, select **GPU T4 ×2**, attach the following private files from
the same audited dataset build, and Run All:

The GitHub repository is **private**. Before Run All, create a fine-grained
GitHub token restricted to `rquq/GECC_VLSP_26` with **Contents: Read-only**.
In Kaggle **Add-ons → Secrets**, add it as `GITHUB_TOKEN` and enable notebook
access. The Git bootstrap uses that secret for clone and fetch without storing
it in the remote URL or Git config. This authenticates code access only; datasets
still come from the private Kaggle Dataset attached through Add Input.

If Git access fails, the bootstrap reports sanitized Git stderr and the exit code:
authentication failures require checking the token's repository access and expiry;
DNS/connection failures require checking Kaggle Internet. Never paste tokens into
notebook source, output or chat. Revoke any exposed token and replace the Secret.

- `VGEC 2026_v1.json`: original corpus.
- `VGEC_classifier.json`: capped mixed classifier dataset.
- `VGEC_balanced.json`: correction dataset with split/provenance metadata.

The local Kaggle checkout contains the ready-to-upload package:
`kaggle-input.zip`. Its three JSON files are also in `dataset/kaggle-input/`.
In Kaggle, create a **private Dataset**, upload that ZIP (or the three loose JSON
files), then add the Dataset to the notebook using **Add Input**. Confirm the
three JSON files appear under `/kaggle/input`, then Run All. Do not attach the
ZIP as an unextracted file. No XLSX, audit, manifest or synthetic-pool upload is
needed. The ZIP and corpus files are local-only and intentionally Git-ignored;
a fresh Git clone does not contain restricted corpus data.


The notebook clones `kaggle-linux`, checks the compatible splits and hardware,
trains Task 1 and its OOF classifiers, then trains Task 2. Set explicit attachment
paths in the parameter cell if filenames are ambiguous. Both configs discover
inputs under `/kaggle/input`; outputs use `/kaggle/output`, linked to
`/kaggle/working/output` so Save Version retains artifacts. Keep notebook versions,
datasets, checkpoints and text-containing outputs private.

For a Kaggle terminal, first configure Git authentication for the private
repository; Kaggle Secrets are read automatically only by the notebook bootstrap:

```bash
git clone --single-branch --branch kaggle-linux https://github.com/rquq/GECC_VLSP_26.git /kaggle/working/gecc-two-phase-source
cd /kaggle/working/gecc-two-phase-source
python -m pip install -r requirements-kaggle.txt
mkdir -p /kaggle/working/output
ln -s /kaggle/working/output /kaggle/output  # Only if this path does not already exist.
python -m model.two_phase --classification-config configs/baseline.json \
  --correction-config configs/correction.json
```

If `/kaggle/output` exists, ensure it resolves to `/kaggle/working/output`.
`requirements-kaggle.txt` pins non-torch dependencies and preserves Kaggle's
CUDA-enabled PyTorch. The notebook records the cloned commit and resolved configs.

## Two-T4 training

Both configs default to `run.world_size=2`. The pipeline launches each stage
with `torchrun`, one NF4 model replica per GPU and synchronized adapter gradients.
The main classifier, every OOF fit, and the corrector use both GPUs for training.
Per-GPU microbatch 1 × two GPUs × four accumulation steps = global effective
batch 8. T4 uses FP16; supported newer GPUs may use BF16.

Epoch validation, hint prediction and correction generation run on rank 0 while
the other rank waits. This is data parallelism for training throughput, not model
splitting, and does not guarantee a 2× end-to-end speedup. Each replica must fit
one T4. The default launch fails before downloads if two CUDA GPUs are unavailable.
For deliberate single-GPU runs, set `run.world_size=1` in **both** copied configs.
Direct stage entrypoints require a matching distributed launch:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m model.classification --config configs/baseline.json
```

## Retained recipes

Both stages use pinned `sail/Sailor2-8B-Chat` revision
`577d5b2745863cec11e1b2f1c92586a91cea446e`, NF4 double quantization,
LoRA rank 16 / alpha 32 / dropout 0.05, sequence length 512, effective batch 8,
and seed 42.

- **Task 1:** single-forward soft hierarchical BCE + twice conditional CE,
  square-root class weights, five epochs, LR `5e-5`. The raw corpus is split
  before augmentation: 312 retained original training rows + 192 synthetic rows
  = 504 training rows; 81 raw validation rows. Selection: validation macro F1.
- **OOF hints:** five connected folds, three epochs per fold, no outer-holdout
  selection. Duplicate sources, synthetic parent relationships and contrast/template
  families stay connected. Hints cover the correction training population,
  including correction rows not retained in the main classifier's training set.
- **Task 2:** alpha's response-only JSON supervision, four epochs, LR `2e-5`.
  Prompt tokens are masked from loss; target completions are never truncated.
  Selection: error-only GLEU+, then error-only exact match, then clean preservation;
  ties retain the earlier epoch. Invalid generated JSON is reported with raw
  generation and parse error, not replaced with a fake successful correction.

Validation selection reuses the same holdout; reported scores are exploratory,
not independent test performance. Fresh training need not reproduce historical
scores or identical numerics across hardware/library versions.

## Saved pipeline inference

The run contains `classification/classifier`, `classification/hints.json`,
`correction/adapter`, selected-adapter evaluations, epoch histories, predictions,
metrics, resolved configs and `pipeline_manifest.json`. The notebook retains a
ZIP and training log alongside the run.

```bash
python -m model.pipeline_predict \
  --classifier /kaggle/output/RUN/classification/classifier \
  --adapter /kaggle/output/RUN/correction/adapter \
  --data /kaggle/input/DATA/input.json \
  --output /kaggle/output/predictions.json
```

Use `--evaluate` for labeled records. To reuse a completed compatible Task 1 run,
pass `--classifier-run /kaggle/input/DATA/completed-classification` to
`model.two_phase`; its saved classifier, OOF hints and provenance are validated.

## Local checks and provenance

The local shared virtual environment is in sibling `test`:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false ../test/.venv/bin/python -m unittest discover -s agents/tests -q
```

Local runs need copied configs with explicit paths for all three datasets and
an appropriate `run.world_size`. Dataset authoring/building utilities remain in
`tools/` for provenance; training never authors or rebuilds attached inputs.

Local verification passed 83 regression tests and real connected notebook-runtime
smokes with tiny unquantized Qwen2 models at world sizes 1 and 2. The smokes used
the attached audited 504/81 classifier split, trained two OOF folds with complete
correction hint coverage, updated both adapters, reloaded saved weights, generated
new-input predictions, and retained the output link, ZIP and log. A complete
correction epoch consumed its final partial accumulation window in both modes.


Full Sailor2 training on Kaggle T4 ×2 has not been exercised locally; the local
machine exposes only one CUDA GPU. Distributed CPU smoke evidence does not prove
T4 memory headroom or a measured throughput gain.

## Branch changes

Restored alpha correction, response-only collation, connected OOF handoff,
saved pipeline inference, and the full two-phase notebook. Kept the existing
Task 1 loss and dataset recipe; did not restore alpha experimental losses,
encoder/ensemble paths, alternate experiment configs or duplicate dependencies.
Both stages now default to two-process data-parallel training.
Fixed Transformers 4.51 epoch accounting and rank-local tail consumption so
OOF and correction fits finish every configured epoch rather than silently
dropping the last accumulation window.
