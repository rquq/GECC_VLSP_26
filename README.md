# Quality-run ablations

The active `kaggle-linux` recipe reproduces the quality classifier settings for three controlled Sailor-7B LoRA fits: quality/hierarchy, organizer-only/hierarchy and quality/flat CE. Each uses 705 optimizer steps, five shared evaluation points, seed42, FP16 LoRA r16/alpha32 and the original LR5e-5 recipe.

Upload **[VGEC26-quality-ablations-20261027.zip](dataset/VGEC26-quality-ablations-20261027.zip)** as one private Kaggle dataset. Attach it to the existing notebook, enable Internet and T4 x2, and press **Save Version / Run All**. The uploaded dataset name is arbitrary; discovery uses the extracted manifest. Repository-read GITHUB_TOKEN is required by the existing launcher; WANDB_API_KEY enables monitoring.

Outputs: `comparison.csv`, `comparison.md`, `ablation_summary.json`, `ablation_results.zip`, and all three selected adapters/learning curves. Historical quality voting comparisons use cached probabilities; the fresh arms share the same 78-row development set and selection budget. These are paper ablations, not a new private-test submission. The original submitted recipe remains in `configs/quality_experiment.json`; the previous larger-model plan remains in `configs/upgraded_training_plan.json`.
