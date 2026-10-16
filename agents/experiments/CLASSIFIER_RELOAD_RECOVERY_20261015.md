# Saved Mistral reload recovery — 2026-10-15

**Follow-up 2026-10-16:** the supplied full Kaggle output completed Mistral verification, corrector training and export. See VALIDATION_RESULTS_20261016.md/.json. The pending statements below describe the original preparation date.

The supplied updated output completed all five Mistral epochs (720 optimizer steps, 4336.93 seconds of training). The classifier adapter and tokenizer exist. The next call failed during inference placement preflight: cuda:0 had 7.46 GiB free versus an estimated 8.43 GiB requirement. This happened in the training process before the saved model could load; lowering the training batch size does not address that boundary. Residual training allocations are the working diagnosis; no full-size local allocation trace was taken.

The selected checkpoint is epoch 1, internal Macro-F1 0.5137657474 and Micro-F1 0.7051282051. Epoch 5 fell to Macro-F1 0.4160062160. Retain the existing best checkpoint. Saved Qwen and Sailor Macro-F1 are 0.5003168719 and 0.4988912704. These are selected scores on the same small internal 78-row partition, not public results or an ensemble score.

## Implementation

For one-process, no-hint classifier stages, the coordinator runs fitting and saved-model verification in separate sequential child processes. A training receipt is saved before verification. The training child exits and the existing GPU release gate checks memory before the verifier starts. Only a successful saved reload and matching selected-epoch score publish predictions, metrics and a completed manifest. DDP/historical hint paths retain their prior orchestration. The corrector already reloads its selected adapter into the retained frozen base, rather than loading a second full base.

Recovery binds 51 exact saved files from the supplied latest output, including all three adapters, native tokenizer files, classifier configs, training/validation rows and Mistral training/epoch histories. The committed source proof contains a recovered training receipt, with evaluation still pending. The original artifacts are preserved. The completed-run validator remains strict; pending checkpoints pass a separate tensor/config check and exact recipe/data comparison, then must complete fresh-process inference before ensemble construction. Generated text and dataset rows stay private.

## Next Kaggle run

Attach the **full output of qwen-sailor-mistral-soft-vote-validation-resume-20261015T140358967795Z**, plus the unchanged private grouped-validation dataset containing the four canonical input files. Arbitrary dataset names and nested Kaggle mount paths are supported. Attach one copy of the latest output. The old two-classifier output alone is insufficient for this new default.

The unchanged notebook launcher fetches the committed kaggle-linux branch. It reuses Qwen/Sailor, verifies saved Mistral without fitting, constructs equal-weight soft voting, trains the Qwen corrector for at most six epochs, then runs inference and creates results.zip. No notebook controls or precision/model/data changes. Keep the saved output private.

## Validation and limits

202 unit/regression tests passed, including incomplete-stage recovery, hash tampering, reload failure and selected-score mismatch. Actual tiny-model CPU continuations passed both new-member fitting plus separate verification, and pending-member verification without fitting; all three pending-recovery adapter files remained byte-identical and submission export completed. A real tiny standard-LoRA FP16 CUDA pipeline also passed all three classifier fit/exit/reload transitions, corrector training and valid submission export on the local RTX4060. Native-file preparation matched all 51 bound files and the actual 1145 training/78 validation rows across all three members without loading full-size bases. See the paired JSON for exact artifacts, hashes, CUDA smoke status and limits. A complete T4 x2 continuation has not been run locally; full-size Mistral reload and the new ensemble/corrector scores remain pending.
