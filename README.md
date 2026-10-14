# Sailor classifier and Qwen corrector with standard LoRA

The Kaggle and local notebooks run the same committed experiment. Experiment choices belong in `configs/experiment.json`, `configs/baseline.json`, and `configs/correction.json`; notebook cells contain no recipe overrides, training switches, or diagnostic toggles. Ask for an experiment in chat, review its config/code change, commit it, and run the notebook against that published commit.

## Kaggle

Import `notebooks/kaggle_two_phase.ipynb`, enable Internet, select GPU T4 ×2, attach the private canonical dataset and `public_test.json`, and enable the read-only repository secret `GITHUB_TOKEN`. Optional `WANDB_API_KEY` enables online tracking; without it, runs use offline tracking. `HF_TOKEN` is optional for authenticated model access. Run All and Save & Run All use the same path without restarting the kernel.

The notebook clones a fresh `kaggle-linux` checkout and creates an isolated training environment while retaining Kaggle's CUDA PyTorch. It runs `python -m tools.run_experiment`. The runner resolves the committed recipe, checks the input hashes and model parameter limits, uses the committed one-process placement across two visible GPUs, checks the full pipeline, then trains and exports.

Attach the private `dataset/kaggle-sailor-qwen-lora-20261014/VGEC26-sailor-qwen-lora-full-training-20261014.zip`, or its extracted files. The ZIP contains the original corpus, verified final-training stage bundles, public inference input, an annotation review queue and a manifest; no model weights. Sailor and Qwen weights download during loading. Dataset names and enclosing upload folders can vary: the runner finds the canonical filenames and verifies the committed SHA256 values. Attach only this data version. Public/private test inputs stay inference-only. Change `submission_filename` through a recipe commit for private testing.

The fresh run contains the exact commit, immutable input hashes, resolved configs, logs, selected adapters, local reports, and `run/results.zip`. Upload that ZIP. It contains only root `results.json` with singleton-list corrections, the format verified by KANO's second public entry. Diagnostics, sources, and gold remain separate files.

The default does not create a second ZIP of the entire training run; Kaggle already retains the run files. This avoids duplicating model weights and corpus-bearing diagnostics in saved outputs.

## Local

Run `notebooks/local_two_phase.ipynb` from the checkout or notebooks directory. It discovers the existing training interpreter and uses the same runner and committed recipe, without installing or cloning. Alternatively:

```bash
python -m tools.run_experiment
```

Inputs default to `dataset/` and outputs to `runs/`. The canonical archive takes precedence over stale loose training JSON. `GECC_TRAINING_PYTHON` selects an existing interpreter for notebook execution. The full FP16 LoRA recipe requires two GPUs and exceeds an 8 GB laptop GPU; use the local machine for data audits and small smoke tests, and Kaggle for full training.

## Pipeline

The active recipe trains fresh standard LoRA adapters, with `load_in_4bit=false` and no k-bit preparation:

| Stage | Base and revision | Training rows | Epochs | Learning rate |
|---|---|---:|---:|---:|
| Classification | `sail/Sailor-7B-Chat` @ `19066fae0a8a3ba029c190d8e3dacccf4d1234b8` | 585 | 5 | 5e-5 |
| Correction | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28` | 495 | 6 | 2e-5 |

Both use rank16/alpha32/dropout0.05, effective batch8, microbatch1, AdamW, nonreentrant gradient checkpointing and seed42. Frozen base weights stay FP16; trainable adapters and the classifier score head use FP32 for stable AMP scaling. Each model is split across T4×2 in one process (`world_size=1`); the stages run sequentially. No folds, hints or notebook switches. The classifier retains its weighted soft hierarchical objective. The corrector receives only the source sentence and learns response-only JSON. Every public input reaches correction, including classifier predictions of clean.

Each training population includes all81 former local validation rows unchanged, as requested. `vgec-final-training-v1` validates the embedded original split bundle and requires its exact train+validation concatenation. `validation_fraction=0` does not resplit raw arrays. Five/six final epochs are saved and reload-probed, with local classification/correction/combined scores explicitly unavailable. The22 flagged added correction targets remain included; their flags do not establish supported repairs. Historical baseline data, runs and holdout bytes remain preserved. This changes model, precision and training population together, so any future public difference cannot be attributed to one change alone.

Selected corrector reload reuses the frozen base after releasing optimizer/DDP state. Native BF16 support is checked with emulation disabled; T4s use FP16. Valid greedy corrections are preserved. Malformed sentence JSON receives one schema-constrained beam retry; unresolved invalid output keeps diagnostics and fails export rather than inventing a correction.

The lower-level `model.two_phase` and `model.pipeline_predict` CLIs remain available for targeted development, saved-run reuse, and recovery. A task2 recipe can set `classifier_run` and optionally `initial_adapter`; those are committed choices. Hint ablation, edit generation, and reranking live in separate research modules and have no notebook switches. Neither a changed corrector prompt nor a different synthetic pool should be promoted without a measured comparison.

For inexpensive classifier screening, commit `mode="task1"` in the experiment recipe. The notebook displays the classifier report without requiring an inference input or fabricating a full submission. The current Sailor/Qwen LoRA recipe uses `mode="both"` for fresh classification and correction. OOF remains disabled. Predicted-hint correction requires genuine OOF artifacts. Normal notebook runs require a clean Git checkout.

## Evaluation and next experiments

The best supported prior public submission is the Sailor7B/Qwen7B hybrid: combined0.5423, Macro-F1 0.3473 and GLEU+0.7373, documented in `agents/experiments/PUBLIC_RESULTS_20261014.md`. Those are historical submission scores, not results of this new LoRA fit. Classification subtype quality and weak correction annotations remain the data bottleneck.

Use a fresh training-only grouped holdout for future controlled screening; never report these81 former validation rows as unseen for the full-training fit. Preserve source/parent families together and require evidence for annotation repairs and eligible generators. Report error-only correction and clean preservation alongside all-row GLEU+ when independent references exist. Local metrics remain an approximation of the official scorer.

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
