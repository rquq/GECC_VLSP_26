# Pipeline review — 2026-10-13

This review describes the preserved five-fold pipeline at commit 47f3a23. The current no-fold/no-hint recipe and its pending full-size result are documented in `agents/experiments/CURRENT.md`; dataset work follows `DATA_PLAN.md` and `agents/docs/DATA_GUIDELINES.md`.

KANO's second entry verifies the submission-format diagnosis: combined 0.5349, GLEU+ 0.7224, exact match 0.5395, ROUGE-L 0.8262, Macro-F1 0.3473, and Micro-F1 0.5485. Classification is now the primary bottleneck. The corrected submission preserved labels and sentences and changed strings to singleton lists.

## Resource track

Competition 1038 evaluates the joint classification/correction task under an 8B limit for each component and synthetic generator. Competition 1039 evaluates the same task as an unconstrained-resource track; its published page does not specify a numerical cap, but retains generic model-limit wording. Both rank by the average of Macro-F1 and GLEU+. Keep the active pipeline on 1038's rules.

The pinned Sailor-7B classifier architecture plus adapter/head/padding allowance is 7,139,016,704 parameters; the corrector plus adapter is 7,761,301,504. The old Sailor2 classifier plus its adapter is 8,048,568,320, above the strict 8B limit. Do not restore the stronger old classifier to the constrained track. The runner counts unquantized meta architecture and actual PEFT adapter tensors; packed NF4 element counts are not eligibility counts.

## Execution changes

- Both notebooks are launchers for `configs/experiment.json`, with no experiment overrides/toggles in cells. Kaggle uses the latest clean committed `kaggle-linux` source and records the commit; the local launcher prioritizes the project environment. Helpers are loaded from the current checkout to avoid stale notebook module caching.
- Stage configs remain the recipe authority. Hardware placement adapts to one/two GPUs while effective batch size stays fixed. `task1` screening, `both`, and saved-classifier `task2` are committed recipe choices.
- The canonical rebuilt archive is preferred over stale loose JSON. All three training hashes are pinned and checked before model loading. Full submissions require the declared inference file; a missing public/private input never silently exports validation.
- Native BF16 checks disable emulation; T4s use FP16. Selected corrector reload preserves the frozen base and releases trainer/optimizer/DDP state.
- Every exported correction is a singleton list. ZIP generation is part of the inference code and includes only root `results.json`. Existing output files cannot be overwritten accidentally; invalid IDs, labels, or unresolved generation errors block export and retain diagnostics.
- Valid greedy outputs are preserved. Malformed sentence JSON receives a schema-constrained four-beam retry. Recovery uses zero length penalty and early stopping to favor completed JSON over strings that grow to the token limit. This is format recovery, not a source/gold fallback.
- Console metric summaries are compact. W&B remains corpus-safe, with offline tracking if no API key is supplied. The default no longer duplicates the entire training folder into another ZIP.

## Score experiments

The selected Sailor-7B classifier has 0.434133 Macro-F1 on the existing 81-row holdout. It gets 40/40 clean samples right and only 15/41 error subtypes right. Phrase and function-word recall are each 1/7. Keep that confusion matrix as the reference for subsequent classifier experiments.

Two train-only grouped-CV lexical baselines were measured on the same holdout: word/character TF-IDF with balanced logistic regression gave 0.271008 Macro-F1 using retained originals only and 0.295297 with the current augmentation. Neither is promoted into production. Their hyperparameter was selected on training-only CV, not the holdout.

A private review queue contains 103 original-training rows flagged for correction-target problems: 30 sentence, 25 phrase, 20 function-word, and 28 punctuation rows. Flags are diagnostic, not proof that every label is wrong. Proposed labels/targets remain empty; no automatic relabeling or holdout changes were applied. The corrector retains only 2 original phrase, 6 function-word, and 1 punctuation correction examples, so repairing well-supported full training corrections is valuable.

Recommended experiments, in order:

1. Screen classifier learning rates and synthetic influence while keeping the same model, original partition, and evaluation protocol. Favor train-only grouped inner validation; measure all five classes. Do not run five OOF fits and retrain the corrector for every preliminary classifier candidate.
2. Replace single-rule rare-type augmentation with reviewed minimal error/clean pairs covering varied phrase order/attachment, function-word scope and usage, and internal punctuation. Keep matched lexical controls and parent groups. Change the bundle policy/validation code deliberately when changing its fixed caps; do not bypass provenance checks.
3. Repair supported original training targets in a versioned overlay. Keep original labels, sources, old targets, evidence and reviewer confidence. Quarantine ambiguity rather than invent missing content. Freeze the 81-row holdout and all public/private inputs.
4. Test Qwen2.5-7B-Instruct as a classifier-only alternative with the same data and loss. Its official model card lists Vietnamese support and 7.61B causal-model parameters; the exact selected architecture plus adapter must still pass the runner's size check. This is an untested candidate, not a claimed improvement. Keep the existing Sailor corrector until a paired evaluation justifies changing it.
5. Evaluate correction with/without predicted hints, then train a fresh no-hint adapter if warranted. Original-corpus OOF function-word and punctuation F1 are zero; those hints can misdirect correction. Reranking/edit representation require separate measured comparisons.

At unchanged GLEU+ 0.7224, Macro-F1 0.70 would yield combined 0.7112; Macro-F1 0.80 would yield 0.7612. These are arithmetic scenarios. A new classifier can change correction hints and GLEU+, so preserve and measure both stages together before promotion.

## Verification scope

The local regression suite, real tiny-model CUDA NF4 pipeline, CUDA FP16 pipeline, and two-process CPU DDP pipeline cover classification, OOF hints, corrector training, selected-adapter reload, public inference, typed IDs, handoffs, and ZIP export. The smoke corpus/tokenizer has a controlled protocol preference and is not a language-quality benchmark. Full-data preflight was also exercised with real pinned inputs/model architectures and simulated two-T4 capabilities/memory.

These checks do not constitute a completed full-size Kaggle T4 ×2 run. The local RTX 4060 laptop has 8 GB VRAM, below the full corrector's conservative placement requirement. No new 7B score or 70–80% Macro-F1 claim is made. Existing successful artifacts and organizer data remain intact.

Primary references: [constrained track](https://aihub.ml/competitions/1038), [unconstrained track](https://aihub.ml/competitions/1039), and [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
