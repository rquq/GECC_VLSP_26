# Kaggle Sailor2 classification baseline

This branch trains and evaluates the Sailor2 classification baseline only. It
contains no correction model, hint-generation stage, or ensemble pipeline.

## Run on Kaggle

Import `notebooks/kaggle_baseline.ipynb` into a **private** Kaggle notebook,
enable Internet and a CUDA GPU, and attach the private inputs
`VGEC 2026_v1.json` and `VGEC_classifier.json`. Run All. The notebook clones the
`kaggle-linux` branch, checks the split, trains, and reloads the selected adapter
for evaluation. Set explicit attachment paths in its parameter cell if matching
filenames are ambiguous.

`configs/baseline.json` discovers uploaded data under `/kaggle/input` and writes
to `/kaggle/output`. The notebook links that output path to
`/kaggle/working/output` so Save Version retains checkpoints, logs, resolved
configs, and the artifact ZIP. Keep both the version and text-containing outputs
private.

For direct execution in a Kaggle terminal:

```bash
git clone --single-branch --branch kaggle-linux https://github.com/rquq/GECC_VLSP_26.git /kaggle/working/gecc-baseline-source
cd /kaggle/working/gecc-baseline-source
python -m pip install -r requirements-kaggle.txt
mkdir -p /kaggle/working/output
ln -s /kaggle/working/output /kaggle/output  # Only if /kaggle/output does not already exist.
python -m model.classification --config configs/baseline.json
```

If `/kaggle/output` already exists, ensure it points to
`/kaggle/working/output` before training; otherwise Save Version may not retain
artifacts. `requirements-kaggle.txt` is the sole dependency recipe. It pins
non-torch dependencies and preserves Kaggle's CUDA-enabled PyTorch installation.
Precision resolves to BF16 on supported GPUs and FP16 otherwise.

## Baseline recipe

- Base model: `sail/Sailor2-8B-Chat`, pinned to revision
  `577d5b2745863cec11e1b2f1c92586a91cea446e`.
- NF4 QLoRA, rank 16; single-forward soft hierarchical BCE plus twice
  conditional CE, with square-root class weights.
- Five epochs, learning rate `5e-5`, effective batch size 8, fixed seed 42.
- Split the 401-row raw corpus before augmentation: 312 retained original
  training rows plus 192 capped synthetic rows = **504 training rows**;
  **81 raw validation rows**.

The best adapter is selected by validation macro F1. Fresh training is not a
guarantee of historical scores or identical numerics across hardware and library
versions. Full Sailor2 training inside Kaggle has not been run here.

## Local checks and data provenance

Run commands from this checkout. The shared local virtual environment is in the
sibling `test` checkout:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false ../test/.venv/bin/python -m unittest discover -s agents/tests -q
../test/.venv/bin/python -m model.classification --config configs/baseline.json --data '/path/to/VGEC 2026_v1.json' --output-dir runs/baseline-reproduction
```

For local training, set `data.augmentation_path` in a local copy of the baseline
config to the classifier dataset path and pass that config with `--config`;
the checked-in config's automatic discovery targets `/kaggle/input`.

Dataset authoring/building tools in `tools/` and the data/metric utilities remain
for provenance and rebuilding private inputs. Training uses the uploaded raw and
classifier datasets; it does not author or rebuild data during the run.

## Cleanup verification

Removed correction/OOF, encoder and ensemble entrypoints, alternate experiment
configs, separate-weighting loss code, stale pipeline guides, and duplicate
dependency recipes. The notebook is now `notebooks/kaggle_baseline.ipynb`;
classification needs no `--skip-hints` switch. Dataset-building utilities and
their provenance/leakage checks remain.

The retained suite passed 74 tests. A local execution of the notebook's config,
training and artifact cells used simulated Kaggle mounts and the actual 504/81
split, performed a tiny Qwen2 CPU optimizer update, changed LoRA and score-head
weights, reloaded the selected adapter, and retained the output link, ZIP and log.
This verifies the launch/artifact path, not full Sailor2 GPU training or Kaggle's
host permissions.

