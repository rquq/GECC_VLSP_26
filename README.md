# Reproducible Sailor classification baseline

## Kaggle Linux branch

`kaggle-linux` runs the existing Sailor2 classification baseline, not the
Task 1 to Task 2 pipeline. Import `notebooks/kaggle_two_phase.ipynb` into
a private Kaggle notebook, enable Internet and a CUDA GPU, attach the same
private `VGEC 2026_v1.json` and `VGEC_classifier.json` used locally, then Run All.
The notebook clones this branch, checks the split before training, trains five
epochs and reloads the selected adapter for evaluation. No correction dataset
or OOF training is needed. Explicit attachment paths are configurable in the
parameter cell when filenames are ambiguous.

`configs/baseline.json` discovers input under `/kaggle/input` and writes to
`/kaggle/output`. The notebook links that path to `/kaggle/working/output`,
so Save Version retains checkpoints, logs, resolved configs and the artifact ZIP.
Save a private version; the corpus and text-containing outputs must stay private.
`requirements-kaggle.txt` pins the tested non-torch dependencies and preserves
Kaggle's CUDA-enabled PyTorch instead of forcing a different CUDA wheel.
GPU precision resolves to BF16 when supported, FP16 otherwise. Fresh GPU fits
are not a guarantee of identical historical scores.

Direct Kaggle terminal execution:

```bash
git clone --single-branch --branch kaggle-linux https://github.com/rquq/GECC_VLSP_26.git /kaggle/working/gecc-baseline-source
cd /kaggle/working/gecc-baseline-source
python -m pip install -r requirements-kaggle.txt
mkdir -p /kaggle/working/output
ln -s /kaggle/working/output /kaggle/output  # Once; skip if already linked.
python -m model.classification --config configs/baseline.json --skip-hints
```

Verification: real local-Git notebook clone/update checks and tiny-model
training/reload tests pass. A clean clone of the committed branch executed the
notebook's configuration, training and artifact cells with simulated Kaggle mounts
and the actual 504/81 baseline split: one tiny-model CPU optimizer update, saved
adapter reload, no OOF hints, and retained ZIP/log. Full Sailor2
training inside Kaggle has not been run here.

## Local baseline reference

Run commands from this checkout. The selected checkpoint remains unchanged;
historical manifests retain their original paths as provenance of the old layout.

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m unittest discover -s agents/tests -q
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m model.classification --config configs/baseline.json --output-dir runs/baseline-reproduction --skip-hints
```

Baseline: Sailor2-8B-Chat at its pinned revision, NF4 QLoRA, rank 16,
single-forward soft hierarchical BCE + twice conditional CE, square-root class
weights, effective batch 8, learning rate 5e-5, five epochs. Epoch 3 was selected
by outer-validation macro F1. The split seed is fixed at 42 before augmentation.

The unchanged raw corpus has 401 rows. Baseline classification uses 312 retained
original training rows plus 192 capped synthetic rows, and 81 raw validation rows.
Selected results: macro F1 0.6327503693, accuracy 0.8024691358 (65/81).
Baseline classification does not train or run a correction model.

The historical checkpoint manifest records Transformers 5.9.0 and PEFT 0.21.2.
`requirements-baseline.txt` retains that dependency recipe. The current tested
environment uses Transformers 4.51.3 and PEFT 0.15.2, pinned in `requirements.txt`.
The reproduction command above runs in the current environment; it is not a
claim of identical training numerics across library versions.

The loader supports the checkpoint's backend-only Qwen tokenizer and its newer
special-token metadata without changing saved files. All 401 original sentences
produce identical token IDs to the saved backend and pinned base tokenizer;
padding and vocabulary size are also verified. Saved weights remain unchanged.
Use `../alpha/` for reviewed data and new model experiments.
