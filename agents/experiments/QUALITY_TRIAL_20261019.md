**Upload packaging superseded on2026-10-19:** follow PLAIN_CONTROL_UPLOAD_20261019.md. Controls are plain JSON inside datasets; ZIPs exist only in run outputs. Legacy extracted uploads work without reupload. Original prediction JSON bytes remain identical; output ZIP-wrapper byte identity is no longer required. Historical package/hash evidence below belongs to the original implementation.

# Quality-data controlled track1039 trial — 2026-10-19

Intent: respond to the official private classification regression, preserve the exact0.5806 quality experiment, and include the user's explicitly requested14B corrector. Source base3aebe4a. Status: implementation and local verification recorded in paired JSON; native two-T4 fitting remains pending. The earlier anchored20B/14B controller is preserved as configs/anchored_unconstrained_experiment.json.

Default configs/experiment.json selects configs/quality_trial_plan.json and tools/run_quality_trial.py. The notebook remains a launcher with no experiment switches; only introductory Markdown changes. Upload PRIVATE dataset/kaggle-quality-track1039-private-20261019/VGEC26-quality-track1039-private-20261019.zip, attach it alone, Internet + T4x2, Save Version / Run All. Arbitrary extraction folder names are supported, with exact package/file/input hash checks. No previous notebook output attachment is needed.

| Fresh fit | Classifier | Corrector |
|---|---|---|
| Pinned base |sail/Sailor2-20B-Chat@bca071eb3dedd23b4e141f63e25d9456300534eb|Qwen/Qwen2.5-14B-Instruct@cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8|
| Full epochs |5|5|
| LR / schedule / warmup |3e-5 /cosine /30steps|2e-5 /linear /.10ratio|
| Training exposures |1128|1011|
| Precision |NF4/double-quant, FP16 compute, FP32 trainables|same|
| LoRA |rank16/alpha32/dropout.05;7attention/MLP projections; score head saved|same projections; no classifier head|
| Supervision |weighted binary detection + lambda2 subtype loss, sqrt class weights|response-only correction loss|

Both use seed/split42, microbatch1/effective8, max512, AdamW betas.9/.999/epsilon1e-8, decay.01, gradient clip1, nonreentrant checkpointing. Complete allfive epochs, keep best checkpoint and verify reload. No folds, inference hints, decoder search, synthesized rows or target repairs. The corrector uses minimal edits, preserves clean sentences and greedily generates at most192 tokens. Every original quality data byte is retained; the preexisting quality recipe/configs are unchanged. Private2034 inputs are inference-only; the shared78 development rows remain excluded from both fits. This compares model families/size/precision/training recipe, not a size-only causal experiment.

| Output relative to run/ | Classification | Corrections | Purpose |
|---|---|---|---|
|quality_control/results.zip|Exact quality ensemble predictions|Exact quality Qwen7B predictions|Byte-identical original control;0.5806 reported on1038,1039 score unknown|
|results.zip|New Sailor20B|Frozen quality Qwen7B|Classifier change only|
|corrector_challenger/results.zip|Exact quality ensemble predictions|New Qwen14B|Corrector change only|
|both_challengers/results.zip|New Sailor20B|New Qwen14B|Both changes|

Generate classifier private predictions once and14B private corrections once, then combine by exact typed IDs. Frozen private predictions never enter fitting, augmentation or checkpoint selection. Their reuse is permitted only for the committed input SHA256. comparison_metrics.json reports allfour pairings on the same development rows. No automatic private winner or fallback promotion is made from that small reused set. The notebook shows four labeled download links. Root results.zip deliberately remains the classifier-only challenger, rather than silently implying the two-model experiment won.

The root selected_models/ bundle contains20B + quality7B; both_challengers/selected_models/ contains20B +14B. Each has a hash-bound selected.json and replay entrypoint. The correction-only counterfactual reuses frozen full-ensemble labels and is an input-bound output, not a portable replacement for allthree original classifier adapters. The exact quality control and original ensemble adapters remain preserved separately. Control ZIP is copied before fitting so it survives later failure; interrupted/failed candidates are never fabricated or submitted.

Root W&B run named run stays active through training, inference and export. Existing numerical quality/progress and CPU/RAM/GPU/VRAM/temperature/power/disk/network monitoring remain. Model processes run sequentially with release gates. Only the marked controller-owned temporary1039 base cache is evicted. Allow roughly40GiB free for the largest raw base; stage changes can redownload weights. Per-stage36000s and total41400s are caps, not a runtime guarantee.

Native meta placement estimates remain about11.2GiB/T4 for Sailor and11.4GiB/T4 for Qwen, including conservative reserves. This is not a measured native NF4 peak or an OOM guarantee. Local4060/WSL memory cannot run this native recipe. Tiny CPU fits validate wiring/reload/export only, not native QLoRA compatibility, model quality or timing. Test and audit evidence, package hashes and remaining limits are recorded in the paired JSON.

Verification completed:270 regression tests passed; actual final CPU workflow completed both five-epoch fits, selected-checkpoint reloads, allfour development comparisons and correctly paired ZIPs. A clean GitHub clone of implementation commita01b7d79fef739e5796efcdc7c250c9551d0d5a7 passed15 additional focused export/config/launcher/track-isolation checks. Native two-T4 NF4 training, timing and score gains remain unmeasured. The notebook's executable cells are identical to the prior committed launcher.1,259,752,602bytes of duplicate draft ZIP/tensor files created during this turn were removed after checking them against the final package; draft manifests/docs and all preexisting experiments remain.
