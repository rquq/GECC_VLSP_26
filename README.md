# Track1039: controlled quality-data comparison

Current configs/experiment.json selects configs/quality_trial_plan.json. The user-reported October19 7B result is0.5307, versus0.5806 for the preserved quality system on1038. Most of this difference comes from classification. The20B/14B challengers have not yet been trained at full size.

1. Upload dataset/kaggle-quality-track1039-private-20261019-json/VGEC26-quality-track1039-private-20261019.zip as a PRIVATE Kaggle dataset. Attach that version alone; its extraction folder name may vary. No older notebook output is needed. The upload contains plain JSON and model files, with no nested ZIP/TAR archives. If the previous quality package is already attached and Kaggle expanded its inner ZIP, keep it: the loader accepts its verified extracted JSON without checking ZIP-container hashes.
2. Import notebooks/kaggle_two_phase.ipynb, enable Internet and GPU T4 x2. Provide repository-read GITHUB_TOKEN; HF_TOKEN is optional and WANDB_API_KEY enables monitoring.
3. Save Version / Run All. Choices are committed in code/configs, with no notebook toggles. Retain the entire output and choose among the four labeled submission ZIPs below. Nothing is automatically submitted.

| Fresh model | Precision | Epochs | LR |
|---|---|---:|---:|
| Sailor2-20B-Chat classifier |NF4 QLoRA / FP16 compute|5 full|3e-5|
| Qwen2.5-14B-Instruct corrector |NF4 QLoRA / FP16 compute|5 full|2e-5|

Both use seed/split42, LoRA rank16/alpha32/dropout.05, max512, microbatch1/effective8, nonreentrant gradient checkpointing and AdamW. Complete allfive epochs, retain the best checkpoint and verify reload. Classifier supervision remains weighted detection + lambda2 subtype loss. Correction uses minimal edits / preserve clean, no hints, greedy192-token decoding. No folds or decoder sweep.

The training dataset is byte-identical to the quality run:1128 classifier and1011 corrector exposures, shared78 development rows excluded from fitting,2034 private inputs inference-only. No synthetic generation, annotation repair or private-test supervision is added. This trial changes model/precision/training recipe, not dataset quality. Existing repeated exposures should not be described as independent new examples.

| ZIP relative to run/ | Classifier | Corrector |
|---|---|---|
|quality_control/results.zip|Exact original quality ensemble predictions|Exact original quality Qwen7B predictions|
|results.zip|New Sailor20B|Frozen quality Qwen7B predictions|
|corrector_challenger/results.zip|Exact quality ensemble predictions|New Qwen14B|
|both_challengers/results.zip|New Sailor20B|New Qwen14B|

The control results.json is byte-identical to the original quality predictions; its ZIP is generated only in the run output. Its0.5806 report belongs to1038; the same ZIP on1039 has not been independently scored. comparison_metrics.json compares allfour pairings on the same reused78-row development set; these scores do not automatically promote a private winner. Root results.zip is the classifier-only challenger. Private corrections are generated once by the14B model and joined by exact typed IDs. Frozen control predictions are bound to the exact private input hash and never enter fitting.

Root selected.json/selected_models contains20B + quality7B. both_challengers/selected.json and its selected_models contain20B +14B. Use tools.replay_focused with the corresponding selection for portable replay. The corrector-only ZIP is an input-bound comparison using frozen full-ensemble labels, not a portable three-model ensemble bundle. Preserve all original quality adapters for that historical system.

Root W&B run run stays active through training, inference and export, monitoring numeric progress/quality and CPU/RAM/GPU/VRAM/temperature/power/disk/network. One process splits each model across both T4s; model stages and release gates are sequential. Only a marked controller-owned temporary1039 base cache is evicted; model changes can redownload weights. Allow about40GiB free cache space. User caches, original runs and quality backups are not cleanup targets.

Native model-placement estimates are about11.2/11.4GiB per T4 including reserves. These are architecture-based estimates with simulated observed T4 capacities, not measured native peaks or an OOM guarantee. Full native training, runtime and new scores remain pending; local CPU smoke tests prove wiring only.

The quality1038 experiment is frozen at archive/track1038-quality-20261019 (338293f), with private backups under backups/track1038-quality-20261019/. Original data/configs/adapters/submission and reporting docs remain preserved. Exact new reported scores are recorded separately in agents/experiments/PRIVATE_REGRESSION_20261019.md. The previous anchored20B/14B plan remains accessible via configs/anchored_unconstrained_experiment.json. Read agents/experiments/QUALITY_TRIAL_20261019.md and its paired JSON for current evidence and limits.
