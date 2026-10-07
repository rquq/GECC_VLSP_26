# VLSP 2026 Vietnamese grammatical error correction baseline

Smoke-ready Hugging Face baseline for the supplied VGEC dataset. GitHub holds the code; a Kaggle notebook clones a selected branch and launches two independent GPU workers. Official test data, leaderboard aggregation, and submission format have not been supplied. Prediction JSON is a **local proposal**, not a verified competition submission.

## Dataset and evaluation

See [docs/DATA_METRICS.md](docs/DATA_METRICS.md) for the complete evidence and scoring assumptions. The dataset has 401 rows, five classes, 200 no-error/null-correction rows, and six duplicate-source groups. Labels are sorted in contiguous blocks, so neither a raw prefix smoke sample nor a tail holdout is appropriate. Default split: 320 training / 81 validation rows; class-stratified and duplicate-group-safe. Null corrections for class 0 become the unchanged source; other null gold corrections fail.

Important annotation risk: 11 references begin with parenthesized annotations, and 112 non-null corrections contain ellipses. Gold text is preserved rather than silently inventing full-sentence corrections. Official metric details such as Vietnamese tokenization and ranking weights remain unspecified.

Local evaluation implements Macro-F1, Micro-F1, confusion matrix, strict sentence Exact Match, **source-aware GLEU+** from the PDF, and whitespace-token ROUGE-L. GLEU+ is not NLTK GLEU. Correction metrics are reported on all rows and error-only rows to expose the advantage of copying the 200 clean sentences. Invalid model JSON is counted, never replaced by a successful-looking class-0/source-copy fallback.

## Model and GPU decisions

Start with [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct): public Apache-2.0 model; its card lists Vietnamese support. This is a pipeline baseline, not a claim that 0.5B is sufficient for competitive classification. Model weights/tokenizer are pinned to an immutable revision.

- Joint completion: `{"error_type_id": 0, "corrected_sentence": "..."}`.
- Response-only causal-LM loss; prompt and padding labels are `-100`.
- LoRA rank 8 on attention and MLP projections: 4,399,104 trainable parameters.
- FP32 stored training weights plus FP16 autocast/loss scaling; BF16 is disabled for T4.
- Initial gradient scale 4096 avoids observed early overflows with AMP's default 65536. Nonfinite gradients/adapter parameters fail the run instead of being called a successful smoke test.
- Gradient checkpointing, SDPA, batch size 1 per GPU, accumulation 4; effective batch 8 on two GPUs.
- DDP replicates the model once per GPU. Two 16GB T4s are **not one 32GB device**. No `device_map="auto"`, model sharding, FlashAttention-2, or bitsandbytes dependency.
- Maximum combined sequence length 512; all 401 supplied examples fit (observed maximum 259 model tokens). Oversized labels fail rather than being truncated.

`configs/smoke.json`: class-balanced 32 train / 8 validation samples after the leakage-safe split, four optimizer steps. Metrics on eight rows prove plumbing only. `configs/baseline.json`: full split, three epochs. Hyperparameters are deliberately identical otherwise; no oversampling in the full run.

For a later experiment branch, the conservative next model is Qwen2.5-1.5B-Instruct with the same LoRA pipeline. This model's memory/quality has **not** been verified here; do not call it the working baseline. QLoRA is a later choice for materially larger models, not required for this smoke run.

## GitHub → Kaggle workflow

This is a standalone project in [rquq/GECC_VLSP_26](https://github.com/rquq/GECC_VLSP_26), with the baseline at the repository root. The notebook defaults to branch `codex/kaggle-baseline`. Publish that branch before running the notebook; the notebook checks its existence and records its exact commit.

From this project folder, publish the code when ready:

```bash
git add .gitignore README.md requirements.txt configs vgec tools notebooks docs tests
git commit -m "Add Vietnamese GEC LoRA baseline and Kaggle notebook"
git push -u origin codex/kaggle-baseline
```

The dataset and metric PDF are excluded from Git. Attach `VGEC 2026_v1.json` privately to the Kaggle notebook. Generated adapters, run outputs, and archives are also ignored.

1. Upload `notebooks/kaggle_smoke.ipynb` to Kaggle.
2. Enable **Internet**, select **GPU T4 x2**, and attach the dataset.
3. Set `BRANCH` to the published branch you want to train. `REPO_URL` defaults to `https://github.com/rquq/GECC_VLSP_26.git`, and `PROJECT_SUBDIR="."` runs the code at its root.
4. Leave `MODE="smoke"` for four optimizer steps, saved-adapter evaluation, and artifact packaging. Set `MODE="full"` for three epochs on the complete split.
5. Set `DATA_PATH` only when discovery is ambiguous or the attached filename differs. Dataset paths containing spaces are supported.
6. Run all cells. The notebook preserves Kaggle's compatible CUDA torch, installs the pinned Hugging Face stack, performs preflight checks, and launches `torch.distributed.run --standalone --nproc_per_node=2`.
7. Save a notebook version and download the output ZIP. It includes the adapter/tokenizer, predictions, metrics, and manifests tying the results to the model, data, and Git commit.

Rerunning all cells creates a new output directory and updates a clean clone of the same repository/branch. Kernel restarts are supported: rerun from the parameter cell. Local clone edits are preserved and rejected; select another `CLONE_DIR` to switch branches or recover from a force-push. Existing training outputs are never overwritten.

Public Qwen requires no HF token. Optional gated/private Hugging Face access uses Kaggle Secrets `HF_TOKEN` with `USE_HF_SECRET=True`. The supplied GitHub repository is public; private clones need authentication configured separately. Never embed credentials in `REPO_URL` or notebook cells.

The launcher is also usable directly after cloning and installing requirements:

```bash
python tools/kaggle_run.py --mode smoke --preflight-only
python tools/kaggle_run.py --mode smoke
python tools/kaggle_run.py --mode full
```

`--preflight-only` checks data, dependencies, devices, and source revision without training. The notebook supplies `--expected-commit` to ensure it executes the verified source. Local GPU diagnostics may use `--num-gpus 1`; Kaggle defaults to two GPU workers.

## Local commands

Run from this folder in a Python environment with CUDA-enabled torch already installed:

```bash
python -m pip install -r requirements.txt
python -m vgec.data --data "dataset/VGEC 2026_v1.json" --output-dir runs/analysis
python -m unittest discover -s tests -v
python -m vgec.train --config configs/smoke.json \
  --data "dataset/VGEC 2026_v1.json" --output-dir runs/smoke
```

For two CUDA devices:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 -m vgec.train \
  --config configs/smoke.json --data "dataset/VGEC 2026_v1.json" \
  --output-dir runs/smoke-two-gpu
```

For explicit CPU diagnostics, add `--device cpu`; training disables FP16 on CPU. CPU is not the Kaggle performance target.

### Saved adapter and predictions

Each successful run writes `adapter/` (LoRA weights plus tokenizer), canonical `train.json`/`validation.json`, `trainer_history.json`, `predictions.json` (canonical predictions with raw-generation diagnostics), `metrics.json`, `identity_metrics.json`, and `run_manifest.json`. The manifest records exact data/model hashes, selected IDs, effective batch, package versions, devices, finite training loss, and rank-0 peak allocated VRAM. Training reloads the **saved adapter** to produce validation predictions, rather than evaluating only an in-memory model. The identity classifier's label is estimated from training rows, not validation labels.

To generate the proposed dataset-shaped prediction JSON from raw labeled or unlabeled input:

```bash
python -m vgec.predict --adapter runs/smoke/adapter \
  --data "dataset/VGEC 2026_v1.json" --output runs/predictions.json --limit 8 --evaluate
```

For unseen input, use `[ {"ID": 1, "Input": "..."} ]` and omit `--evaluate`. The command writes dataset-shaped predictions, a `.diagnostics.json` file, and optional `.metrics.json`. It does not enforce source copying from a predicted label 0 or otherwise hide generation errors. Official formatting needs organizer confirmation.

Rescore existing canonical or proposed dataset-shaped predictions:

```bash
python -m vgec.metrics --data "dataset/VGEC 2026_v1.json" \
  --predictions runs/predictions.json
```

For scoring a limited prediction file, supply a raw gold file containing exactly those IDs; the scorer intentionally rejects missing/extra IDs. Canonical split exports are not raw dataset input.
