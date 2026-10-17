**Next expanded trial — 2026-10-18:** user explicitly authorizes expandedV2 plus tuning and requires FIVE FULL epochs for allfour models. Default configs/experiment.json now pins VGEC26-diverse-v2-private-20261017.zip, expanded_classification/expanded_correction. Qwen classifier LR2e-5/warmup70; Sailor/Mistral5e-5/warmup30; corrector1.5e-5/dropout.10. All seed42, standardLoRA, effective8/micro1; no folds/hints/notebook toggles. WeightedSOFT1:2:1, temperature1, phrase1.25 scored60.11% on reused October18 development predictions; no fresh/unseen gain claimed.78development rows/private2034 inputs remain fit-excluded. V2's negative Bami evidence remains; this is an authorized experiment, not a proven improvement. Previous quality recipe preserved as configs/quality_experiment.json. Read agents/experiments/EXPANDED_TUNING_20261018.md/.json; older default/no-promotion descriptions below are historical.

# Fresh private-test retraining: three classifiers and Qwen correction

October17 data research: larger candidates were built and tested locally, but all four-way comparisons favored retaining the existing dataset on Macro-F1. The default recipe below is unchanged. See `agents/experiments/DATA_DIVERSIFICATION_20261017.md`; candidate archives/configs are research artifacts and should not be attached in place of the default data.

The committed `kaggle-linux` recipe fits Qwen2.5-7B, Sailor-7B and Mistral-7B classifiers from fresh pretrained bases, then fits the Qwen2.5-7B corrector. Experiments are controlled through chat and commits; notebooks contain only the checkout/environment/launch steps.

1. Upload **VGEC26-quality-private-20261017.zip** as a PRIVATE Kaggle dataset and attach that version alone. Its extracted folder name can vary. Keep VLSP data and notebook outputs private.
2. Import `notebooks/kaggle_two_phase.ipynb`, enable Internet and GPU T4×2, and keep repository-read `GITHUB_TOKEN`. `HF_TOKEN` is optional for these ungated models; `WANDB_API_KEY` enables live monitoring.
3. Save & Run All. No old notebook output or adapters are needed. The launcher checks exact training/input hashes, model sizes, data partitions and available memory.
4. Collect `run/results.zip`, containing predictions for the **2,034 private-test IDs**. The recipe pins the supplied private_test.json; the older443-row public input is no longer selected.

|Stage|Model|Maximum epochs|LR|Seed|
|---|---|---:|---:|---:|
|Classifier1|Qwen/Qwen2.5-7B-Instruct|5|5e-5|42|
|Classifier2|sail/Sailor-7B-Chat|5|5e-5|42|
|Classifier3|mistralai/Mistral-7B-Instruct-v0.3|5|5e-5|42|
|Corrector|Qwen/Qwen2.5-7B-Instruct|4|2e-5|42|

Standard LoRA: rank16/alpha32/dropout.05, FP16 frozen bases and FP32 trainables, microbatch1/effective batch8, maximum512 tokens, gradient checkpointing. Models run sequentially across both T4s in one process per fit. No quantized training, folds or classifier hints. Existing process isolation/reload verification and memory-release gates remain active; runtime capacity can still cause a guarded failure.

Classification keeps the hierarchical binary-plus-lambda2-subtype loss and selects each member by development Macro-F1. Weighted **soft voting** averages joint class probabilities with weights1:1:2, temperature1, then phrase multiplier1.25 and renormalization. Exact ties choose the lower class ID. These settings were selected on previous development predictions; their gain on fresh fits/new data is unmeasured. Hard-vote counts are diagnostics only. Calibration is serialized, validated and reapplied on inference reload.

Correction uses the existing minimal-edit/preserve-clean prompt, greedy decoding, repetition penalty1.0 and maximum192 new tokens. New checkpoint selection prioritizes all-row GLEU+, then error-only GLEU+, then clean preservation. Epoch limits permit earlier selected checkpoints. Historical configs retain their original selection contracts.

The new quality mixture preserves321 eligible organizer classifier training rows and309 corrector rows verbatim;78 unchanged grouped development rows stay excluded from fitting. Classifier training exposures are229/229/212/229/229; corrector exposures229/229/149/185/219. Independent examples are fewer: repeated exposures are explicitly identified. Caps limit synthetic operator/family shortcuts; abbreviated correction references are retained without oversampling. Source snapshots/capped rows remain in audit material. This is a research candidate, not a human-verified or measured data-quality gain. Prior development exposure is not erased by the current zero-conflict leakage audit.

W&B's root run is named **run**, with model-prefixed metrics, current stage and a30-second heartbeat through export/archive. Native system monitoring samples every15seconds: all visible GPUs, VRAM/utilization/temperature/power when supported, process-tree CPU/RAM, free RAM, root/output disk usage/I/O and network traffic. No corpus text is logged. Without a key, local status/artifacts remain available.

Default recipe: `configs/experiment.json`; stage settings: `configs/retrain_best_classification.json` and `configs/retrain_best_correction.json`. Previous recovery/training/tuning configurations and saved runs are preserved. See `agents/experiments/PRIVATE_RETRAIN_20261017.md`/.json and `EXTERNAL_DATA_REVIEW_20261017.json` for evidence, exact data/config hashes, license/provenance decisions and verification limits.225 tests, actual tiny four-fit/reload/export and native data/model-count preflight passed; full-size T4 training and new official scores remain pending. Raw/derived corpus, model outputs and upload ZIPs are ignored and must not be pushed.
