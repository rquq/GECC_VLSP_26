# Fresh upgraded ensemble for track1039

The next default experiment restores the quality system's three-model SOFT voting mechanism using **Qwen2.5-14B + Sailor2-20B + Mistral-Nemo-12B**, then trains a fresh **Qwen2.5-14B corrector**. This is a user-authorized gamble; task quality and full-session runtime are untested.

1. Keep the compact PRIVATE `VGEC26-fast-quality-private-20261020.zip` attached (2.17MB). No new dataset upload is required; arbitrary Kaggle extraction names remain supported. Attach only one dataset version.
2. Use the latest `notebooks/kaggle_two_phase.ipynb`, Internet ON and GPU T4 x2. Repository-read GITHUB_TOKEN is required; WANDB_API_KEY enables monitoring; HF_TOKEN is optional. No notebook toggles.
3. Save Version / Run All. It trains allfour models freshly; no previous checkpoint/output attachment is needed. Download `run/results.zip` after completion and keep the whole output. Nothing is automatically submitted.

|Task|Base|Full epochs|LR|Dropout|Weight decay|Micro/effective|
|---|---|---:|---:|---:|---:|---|
|Classifier|Qwen2.5-14B-Instruct|5|3e-5|.05|.01|4/8|
|Classifier|Sailor2-20B-Chat|5|3e-5|.05|.01|4/8|
|Classifier|Mistral-Nemo-Instruct-2407,12B|5|5e-5|.05|.01|4/8|
|Corrector|Qwen2.5-14B-Instruct|6|2e-5|.075|.02|2/8|

All use pinned pretrained bases, NF4 double-quant QLoRA/FP16 compute/FP32 trainables, LoRA rank16/alpha32, seed42, max512 and nonreentrant checkpointing. Classifiers use weighted binary +lambda2 conditional subtype CE and joint probabilities. Fixed weighted SOFT voting1:1:2 (Qwen:Sailor:Mistral), T1, phrase multiplier1.25 preserves the quality experiment's mechanism. Each selects its own best development Macro-F1 epoch; no new calibration sweep.

Corrector evaluates epochs2/4/5/6 and selects all-GLEU, error-GLEU then clean preservation. It uses the current minimal-edit/no-hint prompt, greedy192, generation batch4 with smaller-batch OOM retry. No classifier-clean source-copy gate. Model preflight/test fits, untrained-baseline generation, saved-checkpoint development replays, folds and decoder sweeps remain disabled.

Execution order: Sailor fit/private prediction → Nemo fit/private prediction → Qwen fit/private prediction → weighted vote → fresh Qwen correction fit → private correction/export. One worker spans bothT4s at a time; Qwen's downloaded base stays cached across its classifier and corrector stages, with independent freshly initialized adapters. Root W&B run `run` monitors member metrics, sanitized console and CPU/RAM/GPU/VRAM/disk resources through export. Only owned temporary base caches are evicted.

The same quality data remains1128 classifier/1011 corrector exposures,78 shared development rows excluded from fitting and2034 private inputs inference-only. No samples or labels changed. The previous one-classifier default is `configs/fast_single_experiment.json`; new default points to `configs/upgraded_training_plan.json`. There is no artificial controller time limit, but four fresh fits may approach Kaggle's session limit. Nemo and Qwen classifier memory fit, full runtime and ensemble quality are not verified. No tests or model runs performed during preparation, as requested.

The already prepared cached candidate `runs/final-private-1039-20261020/results.zip` and original0.5806 quality1038 source/tag/adapters/docs remain intact. Completing Kaggle training does not consume the final submission slot. Review the completed output before uploading to1039. Details and pinned revisions: `agents/experiments/UPGRADED_ENSEMBLE_20261020.md`/.json.
