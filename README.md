# KANO at VLSP 2026: Vietnamese Grammatical Error Correction

Code for our best submitted system: **0.5806 combined score**, ranked **2nd in constrained-resource** and **3rd in unconstrained-resource** evaluation. Both tracks used the same submission.

## Method

Three hierarchical LoRA classifiers (Qwen2.5-7B, Sailor-7B and Mistral-7B) use weighted soft voting **1:1:2**. A separate Qwen2.5-7B LoRA model generates minimal corrections without classifier hints. Classifiers train for 5 epochs; the corrector trains for 4. Checkpoints are selected on the shared 78-sentence development set. Settings are in `configs/experiment.json` and its two stage configurations.

## Data

Use the original **quality** package, extracted: `VGEC 2026_v1.json`, `VGEC_classifier.json`, `VGEC_balanced.json`, and `private_test.json`. It contains 1,128 classifier and 1,011 corrector training exposures. Data and model weights are not distributed here.

## Run on Kaggle

1. Import `notebooks/kaggle_two_phase.ipynb`; attach the quality dataset.
2. Enable Internet, select **T4 ×2**, and enable a repository-read `GITHUB_TOKEN` secret. `HF_TOKEN` and `WANDB_API_KEY` are optional.
3. Run all cells. Download the generated **`run/results.zip`** for submission, not the full notebook output archive.

## Run locally

Use Linux/WSL, CUDA and sufficient GPU memory for the same FP16 LoRA recipe (two GPUs with memory comparable to T4 ×2). This configuration does **not** target a single 8GB GPU.

```bash
python3 -m pip install -r requirements-kaggle.txt
```

Keep your CUDA-compatible PyTorch installation. Put the extracted data in `dataset/`, then open `notebooks/local_two_phase.ipynb` and run all cells. Alternatively:

```bash
python3 -m tools.run_experiment --input-root ./dataset --output-root ./runs
```

Saved runs contain selected adapters, validation metrics and the final submission. Local metrics approximate the official scorer; a fresh run may not reproduce the exact leaderboard score. Historical source is retained in Git tags.
