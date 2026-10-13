# Sailor-7B Vietnamese error classification and correction

The Kaggle and local notebooks run the same committed experiment. Experiment choices belong in `configs/experiment.json`, `configs/baseline.json`, and `configs/correction.json`; notebook cells contain no recipe overrides, training switches, or diagnostic toggles. Ask for an experiment in chat, review its config/code change, commit it, and run the notebook against that published commit.

## Kaggle

Import `notebooks/kaggle_two_phase.ipynb`, enable Internet, select GPU T4 ×2, attach the private canonical dataset and `public_test.json`, and enable the read-only repository secret `GITHUB_TOKEN`. Optional `WANDB_API_KEY` enables online tracking; without it, runs use offline tracking. `HF_TOKEN` is optional for authenticated model access. Run All and Save & Run All use the same path without restarting the kernel.

The notebook clones a fresh `kaggle-linux` checkout and creates an isolated training environment while retaining Kaggle's CUDA PyTorch. It runs `python -m tools.run_experiment`. The runner resolves the committed recipe, checks the input hashes and model parameter limits, adapts placement to one or two visible GPUs, checks the full pipeline, then trains and exports.

Attach the complete private `dataset/kaggle-reviewed-ready-20261013/VGEC26-reviewed-v4.zip` (about 2.47 GiB), or all of its extracted files including `reviewed-classifier/`. It contains the retained classifier, original corpus, pinned classifier bundle, reviewed corrector bundle and public inference input. Keep this Kaggle dataset private. The committed recipe pins all training hashes. Missing, ambiguous, or mismatched inputs stop before expensive training. Public/private inputs are inference-only; missing submission data never silently switches this launcher to a validation submission. Change `submission_filename` in the committed recipe for private testing.

The fresh run contains the exact commit, immutable input hashes, resolved configs, logs, selected adapters, local reports, and `run/results.zip`. Upload that ZIP. It contains only root `results.json` with singleton-list corrections, the format verified by KANO's second public entry. Diagnostics, sources, and gold remain separate files.

The default does not create a second ZIP of the entire training run; Kaggle already retains the run files. This avoids duplicating model weights and corpus-bearing diagnostics in saved outputs.

## Local

Run `notebooks/local_two_phase.ipynb` from the checkout or notebooks directory. It discovers the existing training interpreter and uses the same runner and committed recipe, without installing or cloning. Alternatively:

```bash
python -m tools.run_experiment
```

Inputs default to `dataset/` and outputs to `runs/`. The canonical archive takes precedence over stale loose training JSON. `GECC_TRAINING_PYTHON` selects an existing interpreter for notebook execution. The full NF4 correction placement requires more memory than an 8 GB laptop GPU; use the local machine for data audits and small smoke tests, and Kaggle for full training.

## Pipeline

Both stages use the pinned `sail/Sailor-7B-Chat` revision. The retained classifier was trained with a soft hierarchical clean/error and conditional four-subtype head for five epochs at LR 5e-5, effective batch eight, seed 42, and LoRA rank 16. The independent corrector trains response-only JSON for eight epochs at LR 2e-5, using the source sentence without a label hint. Every input reaches correction, including predicted-clean sentences. The current recipe reuses the completed Sailor classifier and trains only the independent corrector: one fit, no OOF training and no prompt hints. Saved classifier validation predictions supply classification metrics only.

The previous five-fold hint system is preserved in `../alpha/preserved/kaggle-fold-hints-47f3a23/` and Git tag `alpha-fold-hints-20261013`. Its notebook pins that tag. The reviewed corrector dataset adds five verified clean annotations while preserving the original augmentation and 81-row holdout. Its full-size no-hint score remains pending.

Selected corrector reload reuses the frozen base after releasing optimizer/DDP state. Native BF16 support is checked with emulation disabled; T4s use FP16. Valid greedy corrections are preserved. Malformed sentence JSON receives one schema-constrained beam retry; unresolved invalid output keeps diagnostics and fails export rather than inventing a correction.

The lower-level `model.two_phase` and `model.pipeline_predict` CLIs remain available for targeted development, saved-run reuse, and recovery. A task2 recipe can set `classifier_run` and optionally `initial_adapter`; those are committed choices. Hint ablation, edit generation, and reranking live in separate research modules and have no notebook switches. Neither a changed corrector prompt nor a different synthetic pool should be promoted without a measured comparison.

For inexpensive classifier screening, commit `mode="task1"` in the experiment recipe. The notebook displays the classifier report without requiring an inference input or fabricating a full submission. A fresh two-fit experiment uses `mode="both"`; the current corrector-only recipe uses `mode="task2"` and its saved classifier. OOF remains disabled. Predicted-hint correction requires genuine OOF artifacts. Normal notebook runs require a clean Git checkout.

## Evaluation and next experiments

KANO's format-corrected public result is Macro-F1 0.3473, Micro-F1 0.5485, GLEU+ 0.7224, exact match 0.5395, ROUGE-L 0.8262, and combined 0.5349. Classification is the main current bottleneck. The latest local classifier gets 40/40 clean rows right but only 15/41 error subtypes right. Its synthetic rare-type supervision is dominated by single patterns. Prioritize diverse, reviewed training examples and controlled model/data comparisons over more epochs.

The existing 81-row holdout and original files remain unchanged. Repair training annotations in a private versioned overlay with evidence and review confidence. Repaired/full corrections and synthetic pairs must stay inside the training partition; source/parent families remain together during OOF splitting. Report error-only correction metrics alongside all-row metrics, and retain annotation-like reference diagnostics. Local metrics are not an official scorer reproduction.

Every experiment belongs in `agents/experiments/`, including user requests, negative outcomes, failures and historical claims. The launcher writes local events there and into each saved run. `agents/experiments/HISTORY.md` indexes recovered artifacts/notes without rerunning old training; `DATA_PLAN.md` describes subtype curation and a bounded eligible local-teacher pilot. Five supported clean repairs are prepared for correction; five reviewed reference completions remain pending. The Qwen synthetic pilot admitted no examples and the new Bami screen did not beat Sailor. See `agents/experiments/LOCAL_REVIEW_RESULTS.md` for measured outcomes and failure recovery. Root `AGENTS.md` makes this documentation workflow persistent.

## Resource tracks and privacy

[1038](https://aihub.ml/competitions/1038) is the constrained-resource track: each component and each synthetic generator must have at most 8B parameters. The runner counts unquantized architecture plus adapter tensors; 4-bit packing does not reduce parameter count. [1039](https://aihub.ml/competitions/1039) is the unconstrained-resource track and lists no numeric cap, although generic model-limit wording remains. Both evaluate classification and correction using the average of Macro-F1 and GLEU+ for ranking. External resources require disclosure to organizers.

Keep corpus-derived data and text outputs private under the local agreement. W&B uploads compact scalars and optional model artifacts; corpus text and raw console capture are disabled. Credentials remain environment/secret values and are redacted from notebook logs. Full configs and corpus-bearing diagnostics stay in local outputs.

## Checks

```bash
python -m unittest discover -s agents/tests
python -m tools.run_experiment --preflight-only
```

The first command runs local regression and tiny-model tests. The second resolves the actual committed data and checks available training hardware without fitting models. Passing local tests does not establish that an entire Kaggle T4 run has completed; that requires the actual saved Kaggle version.
