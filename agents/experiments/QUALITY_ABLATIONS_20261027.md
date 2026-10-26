# Controlled quality-run ablations — 2026-10-27

Status: prepared and verified with actual tiny native-model fits/reloads; full Sailor7B Kaggle run has NOT started. The user requested one existing-notebook Save Version plus one private dataset upload, keeping the original quality-run code/recipe as closely as possible.

## Committed experiment

Three sequential fresh Sailor-7B-Chat standard-FP16-LoRA arms: quality/hierarchical, organizer-only/hierarchical, quality/flat five-class CE. Source is the protected338293f quality run. Exact frozen model revision, LoRA and training configuration are retained; only objective/composition, explicit705-update budget and141-step evaluation cadence vary as required. Quality gets its original five-epoch budget. Organizer-only retains321 eligible originals and cycles them for the same number of updates, reporting the actual pass/exposure differences. Both head types use the same inverse-square-root weighting formula and accumulation-window row normalization. Each has the same five checkpoint-selection opportunities and final-step reporting.

The same session compares three cached quality member predictions and five predeclared soft-voting rules on the unchanged78 development rows. Cached historical results and fresh fits are separate sources; fixed quality-corrector GLEU/illustrative combined scores do not imply a new corrector fit or private result. No additional models, teacher generation, relabeling, private inputs or submission are included.

## Upload and run

One PRIVATE dataset: dataset/VGEC26-quality-ablations-20261027.zip, about1.22MB. Plain JSON/text payloads, no weights or inner ZIP/TAR. Exact normalized1128-row quality fit data and78dev match saved quality training artifacts. Organizer321 rows are an exact eligible original subset. Original quality provenance, cached predictions, license notice and a content-hash manifest are included. Kaggle's arbitrary extraction folder names work; only extracted file hashes are checked at runtime, never the outer ZIP bytes.

Existing Kaggle notebook CODE CELLS are unchanged. Default configs/experiment.json dispatches to the new plan. After attaching the new dataset, Save Version / Run All with Internet and T4x2; existing repository-read GITHUB_TOKEN remains required. WANDB_API_KEY enables online monitoring. Each fit and selected-checkpoint verification runs in a new process; original GPU-release guards and a run-owned shared Sailor download cache prevent retained model state between arms. No model/controller wall deadline. Root W&B remains named run, mirrors sanitized child console, records CPU/RAM/disk/network/GPU resources, and receives arm-specific live train/validation curves with optimizer-step axes.

Outputs: comparison.csv, comparison.md, ablation_summary.json, ablation_results.zip and all selected adapters/curves. The ZIP is a paper-analysis result, not an organizer submission. The original quality recipe, datasets, adapters, code archive and official0.5806 evidence are unchanged.

## Verification and limits

9 focused regressions,12 legacy classifier tests and3 filename-loaded notebook tests pass. Actual offline tiny Qwen controller completed three fits, three fresh-process adapter reloads, exact common evaluation steps, cached voting, summaries and valid result ZIP. Real offline W&B root stream contains68 history records and24 system-stat keys, including live loss/validation for all three arms. Exact quality model/LoRA/training config, normalized data and notebook code-cell equality verified. AST graph review and independent final review completed; two discovered preflight/live-metric issues were fixed with regressions. No full7B training or native dual-T4 execution was claimed. All scores remain reused-development evidence; only actual later Kaggle results can establish experiment outcomes.

Exact configs, package/source hashes, smoke paths and statuses are in QUALITY_ABLATIONS_20261027.json. Existing unrelated dirty files were preserved and excluded from this implementation's staged content.
