# Working on the constrained Kaggle pipeline

Experiments are requested in chat and expressed as committed changes to `configs/experiment.json` and the stage configs. Keep both notebooks as launchers; do not add experiment toggles or duplicate pipeline logic inside cells. Keep research diagnostics outside the default notebook path.

Keep notebook cells short: source checkout, shared environment setup, then launch. Dependency installation and result handling belong in `tools/notebook.py`. Transport settings are fixed; experiment changes are chat-requested commits.

Preserve original corpus files, the grouped seed-42 holdout, saved adapters, run logs, and branch identity. The canonical data bundle is pinned by hashes; changing a training dataset requires an explicit recipe/hash commit. Repair training annotations only in a private versioned overlay with original ID, evidence, and confidence. Do not silently relabel validation or derive training examples from public/private test inputs.

Every constrained-track component and any synthetic generator must satisfy the 8B parameter limit. Quantization changes storage, not parameter count. The user explicitly authorized the current Qwen2.5-7B-Instruct trial: two fresh stages, mode `both`, OOF disabled, independent no-hint correction, and no saved Sailor adapter reuse. Record measured results before claiming a replacement win. Preserve the previous five-fold system under `../alpha/preserved/kaggle-fold-hints-47f3a23/` and Git tag `alpha-fold-hints-20261013`. Predicted-hint training still requires genuine OOF provenance; never replace it with in-sample predictions or gold labels.

Record every experiment, including those requested by the user and failed/negative runs, in `agents/experiments/`. Use exact configs, data hashes, source commits, budgets, outcomes and artifact locations. A proposed experiment is not a measured result. The launcher journals attempts in ignored `agents/experiments/local.jsonl` and each saved run. Import completed Kaggle results back into the tracked records in a follow-up commit. Read `agents/experiments/HISTORY.md` before repeating old work; recover evidence without rerunning training. Raw recovered archive notes stay in ignored `agents/private/`.

Use `agents/docs/DATA_GUIDELINES.md` and `agents/experiments/DATA_PLAN.md` for curation and local-teacher work. No reviewed flags or synthetic candidates are promoted merely because they pass structural checks.

The initial approved model/data review and budget are in `agents/experiments/MODEL_SELECTION.md` and `MODEL_SELECTION.json`. Completed outcomes are in LOCAL_REVIEW_RESULTS.md / .json. The Bami screen and synthetic pilot were negative; only supported original repairs enter the corrector. Prefer measured promotion over model-card rankings; check historical encoder results before repeating them. Every candidate model needs its exact revision, native architecture/adapter count, tokenizer contract and save/reload verification before it replaces a production stage.

Export `results.zip` containing only root `results.json`; each Fixed Sentence is a singleton list. Do not omit corrections for predicted-clean rows. Invalid labels, IDs, or unresolved generation failures must prevent submission export and retain diagnostics. Malformed sentence JSON may use the bounded schema-constrained retry; valid predictions must be preserved.

Run focused tests for changes, then the local suite and a real tiny-model pipeline smoke for orchestration changes. Distinguish local/CUDA smoke evidence from a complete Kaggle T4 run. Use native BF16 capability checks with emulation disabled. Preserve the selected-adapter frozen-base reload and complete accumulation-window handling.

Keep corpus text out of W&B and external messages. Secrets come from the environment or Kaggle Secrets; do not store or print them. Keep data and text-bearing generated outputs private.
