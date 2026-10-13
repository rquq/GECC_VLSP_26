# Sailor-7B Vietnamese error classification and correction

The Kaggle and local notebooks run the same committed experiment. Experiment choices belong in `configs/experiment.json`, `configs/baseline.json`, and `configs/correction.json`; notebook cells contain no recipe overrides, training switches, or diagnostic toggles. Ask for an experiment in chat, review its config/code change, commit it, and run the notebook against that published commit.

## Kaggle

Import `notebooks/kaggle_two_phase.ipynb`, enable Internet, select GPU T4 ×2, attach the private canonical dataset and `public_test.json`, and enable the read-only repository secret `GITHUB_TOKEN`. Optional `WANDB_API_KEY` enables online tracking; without it, runs use offline tracking. `HF_TOKEN` is optional for authenticated model access. Run All and Save & Run All use the same path without restarting the kernel.

The notebook clones a fresh `kaggle-linux` checkout and creates an isolated training environment while retaining Kaggle's CUDA PyTorch. It runs `python -m tools.run_experiment`. The runner resolves the committed recipe, checks the input hashes and model parameter limits, adapts placement to one or two visible GPUs, checks the full pipeline, then trains and exports.

Attach either `VGEC26-rebuilt.zip` or its extracted files: `VGEC 2026_v1.json`, `VGEC_classifier.json`, `VGEC_balanced.json`, and the inference input. The recipe pins the training file hashes used by the successful correction submission. Missing, ambiguous, or mismatched inputs stop before expensive training. Public/private inputs are inference-only; missing submission data never silently switches this launcher to a validation submission. Change `submission_filename` in the committed recipe for private testing.

The fresh run contains the exact commit, immutable input hashes, resolved configs, logs, selected adapters, local reports, and `run/results.zip`. Upload that ZIP. It contains only root `results.json` with singleton-list corrections, the format verified by KANO's second public entry. Diagnostics, sources, and gold remain separate files.

The default does not create a second ZIP of the entire training run; Kaggle already retains the run files. This avoids duplicating model weights and corpus-bearing diagnostics in saved outputs.

## Local

Run `notebooks/local_two_phase.ipynb` from the checkout or notebooks directory. It discovers the existing training interpreter and uses the same runner and committed recipe, without installing or cloning. Alternatively:

```bash
python -m tools.run_experiment
```

Inputs default to `dataset/` and outputs to `runs/`. The canonical archive takes precedence over stale loose training JSON. `GECC_TRAINING_PYTHON` selects an existing interpreter for notebook execution. The full NF4 correction placement requires more memory than an 8 GB laptop GPU; use the local machine for data audits and small smoke tests, and Kaggle for full training.

## Pipeline

Both stages use the pinned `sail/Sailor-7B-Chat` revision. The classifier trains a soft hierarchical clean/error and conditional four-subtype head. Its five training epochs use LR 5e-5, effective batch eight, seed 42, and LoRA rank 16. Five grouped OOF fits supply training-only predicted hints; validation hints come from the selected main classifier. The independent corrector trains response-only JSON for eight epochs at LR 2e-5. Every input reaches correction, including predicted-clean sentences.

Selected corrector reload reuses the frozen base after releasing optimizer/DDP state. Native BF16 support is checked with emulation disabled; T4s use FP16. Valid greedy corrections are preserved. Malformed sentence JSON receives one schema-constrained beam retry; unresolved invalid output keeps diagnostics and fails export rather than inventing a correction.

The lower-level `model.two_phase` and `model.pipeline_predict` CLIs remain available for targeted development, saved-run reuse, and recovery. A task2 recipe can set `classifier_run` and optionally `initial_adapter`; those are committed choices. Hint ablation, edit generation, and reranking live in separate research modules and have no notebook switches. Neither a changed corrector prompt nor a different synthetic pool should be promoted without a measured comparison.

For inexpensive classifier screening, commit `mode="task1"` in the experiment recipe and `oof.enabled=false` in the classifier config. The notebook displays the classifier report without requiring an inference input or fabricating a full submission. Once a classifier recipe earns further testing, commit `mode="both"` and restore OOF generation. Normal notebook runs require a clean Git checkout.

## Evaluation and next experiments

KANO's format-corrected public result is Macro-F1 0.3473, Micro-F1 0.5485, GLEU+ 0.7224, exact match 0.5395, ROUGE-L 0.8262, and combined 0.5349. Classification is the main current bottleneck. The latest local classifier gets 40/40 clean rows right but only 15/41 error subtypes right. Its synthetic rare-type supervision is dominated by single patterns. Prioritize diverse, reviewed training examples and controlled model/data comparisons over more epochs.

The existing 81-row holdout and original files remain unchanged. Repair training annotations in a private versioned overlay with evidence and review confidence. Repaired/full corrections and synthetic pairs must stay inside the training partition; source/parent families remain together during OOF splitting. Report error-only correction metrics alongside all-row metrics, and retain annotation-like reference diagnostics. Local metrics are not an official scorer reproduction.

## Resource tracks and privacy

[1038](https://aihub.ml/competitions/1038) is the constrained-resource track: each component and each synthetic generator must have at most 8B parameters. The runner counts unquantized architecture plus adapter tensors; 4-bit packing does not reduce parameter count. [1039](https://aihub.ml/competitions/1039) is the unconstrained-resource track and lists no numeric cap, although generic model-limit wording remains. Both evaluate classification and correction using the average of Macro-F1 and GLEU+ for ranking. External resources require disclosure to organizers.

Keep corpus-derived data and text outputs private under the local agreement. W&B uploads compact scalars and optional model artifacts; corpus text and raw console capture are disabled. Credentials remain environment/secret values and are redacted from notebook logs. Full configs and corpus-bearing diagnostics stay in local outputs.

## Checks

```bash
python -m unittest discover -s agents/tests
python -m tools.run_experiment --preflight-only
```

The first command runs local regression and tiny-model tests. The second resolves the actual committed data and checks available training hardware without fitting models. Passing local tests does not establish that an entire Kaggle T4 run has completed; that requires the actual saved Kaggle version.
