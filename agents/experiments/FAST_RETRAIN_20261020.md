Outcome update: the native run is now completed; see FAST_RESULTS_20261020.md/.json for runtime, official user-reported score and negative development tuning result. The preparation notes below describe what was known before that run.

# Fresh quality-data retrain — October20

Prepared, not executed. User requests fresh classifier → corrector → inference; remove our runtime cap, tune using saved results, use standard LoRA if feasible, and skip verification runs. No tests, model preflight, smoke fits or new native inference/training were run for this change.

## What interrupted the completed training

Source: `runs/20_10_2026/runs/track1039-quality-controlled-20261019-20261019T064115932070Z`. Both five-epoch fits completed. Console ends with `TimeoutError: Stage selected-private-corrector exceeded its committed time budget`, from the four-way trial's stage_timeout/_run_stage path. Its parent wall budget was41,400seconds (11h30m); per-stage ceiling36,000seconds. We imposed these budgets, not the user. New default never passes timeout_seconds to _run_stage. Historical controllers/configs preserve their original budgets for reproducibility and are no longer default. Kaggle can still terminate at its external session limit.

Private decoding progress is856/2034. These2034 are unlabeled private-test sentences being corrected for submission, NOT generated augmentation. The new14B private export is incomplete. Preserve partial predictions and both trained adapters. Existing root results.zip is the20B classifier plus frozen quality7B corrections; quality_control/results.zip is the old system. Neither is a completed20B/14B submission.

Training-loop time alone: classifier12,462.8913s (3h27m43s); corrector15,066.0566s (4h11m6s); combined7h38m49s. Downloads, baseline generation, repeated checkpoint evaluation and private decoding add substantial overhead. Removing the cap alone would not solve the platform session limit.

## Saved evidence and tuning rationale

|Epoch|Classifier Macro-F1|Mean logged classifier loss|Corrector all-GLEU|Error-GLEU|Clean preservation|Mean logged corrector loss|
|---:|---:|---:|---:|---:|---:|---:|
|1|.43415|3.28958|.58331|.13481|1.000|.24964|
|2|.55056|.60119|.60650|.21695|.925|.07114|
|3|.59412|.16722|.63060|.25511|.925|.03665|
|4|.56354|.04777|.64857|.28530|.975|.01606|
|5|.57301|.00887|.66385|.30861|.950|.00699|

Loss values are arithmetic means of logged training losses by epoch, not exact sample-weighted evaluation losses. Classifier best epoch3; clean40/40 but phrase2/6, function2/6 and punctuation3/7 remain weak. Postepoch3 development decline alongside near-zero training loss supports a regularization experiment. Corrector improves through epoch5: low training loss alone does not prove it overfit.

Same reused78-row development: quality control Macro.57277/all-GLEU.60575/combined.58926; new20B/14B Macro.59412/all-GLEU.66385/combined.62899. This is exploratory development evidence, not an official/private score or independent generalization result.

Classifier changes: LR3e-5→2e-5, dropout.05→.10, weightdecay.01→.05, warmup30→20steps, micro1→4/accum2/effective8, eval1→8. Keep five full epochs, all-epoch development selection, hierarchy/lambda2, rank16/alpha32, seeds, model revision and data. These changes are hypotheses, not measured gains.

Corrector: retain LR2e-5 and five full epochs because development improves. Warmup.10→.05, dropout.05→.075, decay.01→.02 are modest regularization hypotheses targeting memorization/clean rewriting. Micro1→2/accum4/effective8. Generate validation only at epochs2/4/5, preserving best all-GLEU/error-GLEU/clean checkpoint among those evaluations; epoch1/3 are intentionally not scored. Logging every10 updates instead of everyupdate. No output truncation, new beam search or reference/prompt changes.

## Precision and runtime plan

Sailor classifier native base18,395,133,952 parameters needs34.26GiB for FP16 weights alone; this exceeds both observed14.56GiB T4s combined. Qwen14B native base14,770,033,664 needs27.51GiB before adapters, gradients, optimizer and activations, leaving unsafe training headroom. NF4 manifests report packed counts, which must not be confused with native parameter counts. Retain NF4 double-quant QLoRA/FP16 compute/FP32 trainables. Standard LoRA would require smaller bases or larger GPUs.

The prior corrector's peak allocated memory was8,137,598,464 and9,160,131,584 bytes. This suggests inference batching headroom, but does not verify these new microbatches or guarantee no OOM. Training source lengths were at most69 classifier tokens and183 corrector tokens (validation encoded222); dynamic padding is already active, so merely reducing max_length512 would not remove most work.

Fresh run order:

1. Train Sailor20B five full epochs, save selected epoch predictions and adapter.
2. Classify2034 private inputs immediately, batch8, while20B base remains cached. This is inference-only and never supplies corrector training prompts.
3. Train Qwen14B five full epochs, evaluate epochs2/4/5, save selected epoch predictions and adapter.
4. Correct private inputs with left-padded greedy batches up to4, then export one final results.zip with exact typed IDs/order. Actual generation OOM reduces batch4→2→1. Malformed full-sentence JSON retains the existing individual schema-constrained retry. Every private row is generated; classifier-clean rows are not copied automatically.

No four-way comparison, baseline generation, decoder sweep, model preflight, extra smoke fit, saved-model score replay, or duplicate selected-model directory. Selected-epoch development predictions feed reports. Reload-verification flags remain false and verification_status explicitly says skipped_user_request; strict historical reuse validators are not weakened. Loading the adapters for necessary private inference is still required.

Private decoding writes an append-only predictions.jsonl journal and a tiny progress.json per batch, instead of rewriting all accumulated predictions after every sentence. Final predictions.json restores original input order. Root W&B run `run` tracks numeric metrics/progress and system resources throughout; log_models=false avoids uploading model artifacts, while local adapters remain in output. Private text is never sent to W&B.

Only marked controller-owned temporary base caches may be evicted; sequence20B→14B avoids downloading20B again after the corrector. Existing dataset upload works; no nested ZIP/TAR files are introduced. Data stays1128/1011 exposures, original quality hash, same78 development rows fit-excluded,2034 private inference-only. No new synthesis or annotation changes.

## Artifacts and limits

Default: configs/experiment.json → configs/fast_training_plan.json → configs/fast_training_recipe.json with fast_classification.json /fast_correction.json. Notebook remains four simple code cells and committed choices, no toggles. Old trial wrapper remains configs/controlled_quality_experiment.json. Original1038 reporting archive/backups and old outputs are untouched; three preexisting edited experiment documents remain outside this commit.

Next run/results.zip means fresh20B classifier + fresh14B corrector. This differs from the old four-way trial's root ZIP meaning. Keep the entire Kaggle output. No automatic upload/submission. Runtime savings, batch memory fit and score changes are unmeasured because the user explicitly skipped verification. Kaggle may still time out.

Technical reference: [Transformers generation guidance](https://huggingface.co/docs/transformers/llm_tutorial) explains decoder left-padding and inference batching's throughput/memory tradeoff; [PEFT quantization guidance](https://huggingface.co/docs/peft/developer_guides/quantization) documents adapters on quantized bases. Native counts, actual metrics and memory values above come from the saved run, not theoretical quality claims.
