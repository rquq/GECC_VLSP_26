# Expanded V2 fresh trial with targeted tuning — 2026-10-18

User explicitly authorizes the next expanded-dataset trial despite negative encoder screens, asks to tune parameters, then requires ALL models to train at least5epochs. The committed default now selects expandedV2, preserving the completed quality recipe as configs/quality_experiment.json and its unchanged retrain_best configs. This is an exploratory trial, not evidence that expanded data is superior. No full-size new training has started.

## Evidence and decision

October18 quality-run Qwen classifier selects epoch1 Macro47.22%; epochs2–5 regress to44.46/43.41/44.16/44.16. Logged training loss21.2503 early→.0015 near completion; fresh reload matches selected score. No loss-normalization or reload bug explains the result. Memorization/mixture sensitivity is plausible, but data/seed/learning dynamics are confounded; do not claim Qwen's architecture is intrinsically bad. Lower its LR5e-5→2e-5 and warmup30→70steps. Final user steering supersedes the initially proposed3/4epoch caps: Qwen, Sailor and Mistral each fit FIVE full epochs, as does the corrector. No early-stopping callback or max-step truncation.

Three predeclared saved-probability probes on the SAME78development rows, temperature1 and phrase_scale1.25: weights1:1:2 Macro.5727738928/Micro.7435897436;1:2:1 .6010678211/.7820512821;1:3:1 .5868253968/.7692307692. Choose1:2:1 as a provisional development-selected soft vote. This score is NOT a fresh fit or independent improvement. Allthree members remain, with no hard gate/vote. All earlier calibration results remain historical evidence.

## Exact next fit

| Component | LR | Full epochs | Warmup | LoRA r/alpha/dropout |
|---|---:|---:|---:|---|
|Qwen7B classifier|2e-5|5|70steps|16/32/.05|
|Sailor7B classifier|5e-5|5|30steps|16/32/.05|
|Mistral7B classifier|5e-5|5|30steps|16/32/.05|
|Qwen7B corrector|1.5e-5|5|10%|16/32/.10|

All seed42/split42, standardLoRA FP16 frozen bases/FP32 trainables, microbatch1/effective8, max512, gradient checkpointing. Hierarchical weightedBCE+2conditionalCE unchanged. Greedy192tokens/penalty1.0 and existing minimal-edit/clean-preservation prompt unchanged; no expensive beam sweep or classifier hints. Corrector selects all-rowGLEU+, then errorGLEU+, then clean preservation. A checkpoint from an earlier epoch may win even though all5epochs execute. W&B root run stays run with resource/stage monitoring. No notebook toggles/cell-code changes.

The corrector was still improving all-rowGLEU at its previous4epoch cap. Test a smaller LR and more LoRA dropout over5epochs; this is an anti-overfitting hypothesis, not a measured gain. Do not change abbreviated original target wording or forbid ellipses globally: some original inputs/references intentionally contain them.

## Data and comparison limits

Upload PRIVATE dataset/kaggle-diverse-v2-private-20261017/VGEC26-diverse-v2-private-20261017.zip. Existing bytes/hashes unchanged.849classifier/833corrector canonical rows,844distinct classifier sources.1394classifier exposures293/290/284/285/242;1289corrector293/257/259/265/215. Cap2 per example, protected annotation/abbreviation policies,64additional pinned licensed humanUD parents. Same78development rows excluded from fitting,2034private inputs inference-only. Original labels/targets and all historical packages retained. No new synthetic generation, inferred repair or relabeling by the large assistant.

V2 beat additiveV3 in the cheap Bami screens but both lost Macro versus quality; the user's new instruction authorizes testing V2 with7B models. Do not call this a promotion supported by measured7B gains. Source/target/parent/template audits remain scoped safeguards, not a guarantee of semantic independence. Changing dataset AND optimization makes next old/quality/expanded comparison descriptive; a causal dataset comparison requires matching settings. Fiveepochs of larger data means more updates, so runtime is likely longer.

## Implementation and verification

A small shared classifier_member_config resolves only committed learning_rate/num_train_epochs/warmup_steps overrides; data, seed, precision and model roles cannot be overridden through that field. Config loading rejects unknown/nonfinite/invalid overrides. The same resolver is used by fit orchestration, native parameter auditing, placement preflight, ensemble build/validation and partial-reuse checks. Members must still use identical populations, LoRA geometry and data identities, and each exact requested training recipe is validated; arbitrary drift is rejected. Meta parameter counting now deep-copies LoRA settings to avoid PEFT mutating shared modules_to_save lists.

237 tests pass, including real meta audit immutability, requested per-model fit resolution, malformed override rejection and ensemble training-drift rejection. Native data/count preflight passes on the actual expanded archive. The actual tinyCPU pipeline completed allfour fresh fits for FIVE FULL epochs each, reloaded allfour selected models, retained1:2:1soft voting and exported a valid2rowtyped-ID ZIP. Paths/proofs are in the pairedJSON. Tiny architecture/rank and Qwenwarmup2 were used for the fixture, not native70; no quality conclusion follows. A preliminary local smoke invocation used unsupported CLI options and stopped before fitting; it was corrected to use the existing programmatic prepare_run entrypoint, with the failed log preserved. Short2step wiring also completed before the final5epoch fixture run. Neither is evidence of7B quality.

All individual native architecture+adapter counts remain below8B. Native balanced-T4 placement/5epoch7B quality remains unmeasured locally. Full originals, runs and private data preserved; pre-existing uncommitted user edits in CURRENT.md/PRIVATE_RETRAIN documents are preserved separately from this change. Record actual Kaggle outcomes before making score claims.

Backward compatibility: prior October16/18 native ensembles validate in systemPython3.13. The local3.10 interpreter can reject calibrated saved votes due to the pre-existing exact-float equality check; the original committed code behaves identically. This trial uses no prior checkpoint reuse. Fresh3.10 tiny fit/reload/export and same-runtime native contracts pass; cross-interpreter byte-identical vote replay is not claimed.
