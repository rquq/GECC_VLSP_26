# Focused Sailor and correction recovery implementation plan

User approved all recommendations in THREE_DATASET_COMPARISON_20261018.md and requests execution/preparation for the next Kaggle run. This is a bounded extension of existing split bundles, two-phase training, decoding sweeps and notebook launcher. The approved comparison report is the design brief. Implementation is inline to limit token cost.

**Goal:** A self-contained private Kaggle upload and committed default that trains one Sailor classifier and one Qwen corrector, compares them with preserved quality-Sailor/old-Qwen baselines, evaluates bounded greedy/beam decoding, and exports the development-selected pipeline.

**Architecture:** Deterministic organizer-anchored resampling of existing verified quality/diverse rows; no authored examples or inferred gold repairs. A focused coordinator reuses existing two_phase and decoding_sweep workers, verifies bundled baseline hashes and exact shared development rows before fitting, and selects classifier/corrector independently without requiring an old Kaggle output attachment. Baseline and challenger remain separately portable; root W&B monitoring lasts through export.

**Constraints:** native <=8B per component; standard FP16 LoRA; seed/split42; unchanged78 development rows;2034 private inputs inference-only; no folds/hints/notebook toggles; all fresh fits execute at leastfive full epochs; retain best earlier checkpoint; no original row/label/target deletion or repair; corpus/checkpoints remain ignored/private. Unqualified teachers remain unqualified. Do not force equal class totals or claim unseen gains. Preserve pre-existing user edits.

## Tasks and checks

- [x] Add versioned organizer-anchored mixture and consumer support. Tests first: exact originals and validation retained; evidence-limited synthetic rule/parent caps; protected annotations once; tamper detection even after resealing; class counts and deterministic provenance; both-stage grouped leakage check.
- [x] Build one private upload with original/split bundles/private input, exact baseline quality Sailor classifier and old Qwen corrector, metadata and SHA256 proof. Validate old artifacts rather than retrain them; no unsafe archive extraction or unknown paths. Review training-only flags and preserve a private evidence queue; unsupported repairs stay pending.
- [x] Add focused coordinator and committed configs. Train two challengers with existing isolation/reload checks; compare fixed argmax/binary0.5 diagnostics without a threshold search; evaluate deterministic greedy, beam2, beam4 and beam4 edit reranking using existing decoder workers. Choose model/decoder on development only, export through strict typed-ID contracts, retain baseline fallback if challengers lose. Budget stages/whole run and surface failures.
- [x] Run focused regression tests, appropriate existing suite, real native data/model-count preflight, and actual tiny CPU fit/reload/decoder/export smoke. Native7B quality and T4 memory remain untested until Kaggle.
- [ ] Update agents rules/current/results and upload instructions; commit/push only own aggregate/code/config changes under prior standing authorization, preserving unrelated user edits and private artifacts. Verify remote commit and package hashes. No Kaggle launch from this session unless a working authorized interface already exists.

## Review focus

1. Duplicate or changed baseline weights/metadata must fail before GPU work; a weird Kaggle mount prefix must still resolve the one self-contained package.
2. No exact/family/template/development or private-test training overlap; no validation labels in generation prompts; no evaluation-reference rewrites.
3. Single-classifier exports and independently selected correctors must align exact typed IDs; copying classifier-clean inputs is excluded after its measured regression.
4. Optional decoder failure stays explicit; a failed strategy is never a winner. Mandatory training/reload failures stop. Native greedy fallback must remain eligible.
5. Root tracking records numeric metrics/resources only; no raw corpus upload, and no hidden standard-LoRA to QLoRA fallback.

## Rulings and ledger

- Existing user approval applies to the preceding concrete report/design; no repeated approval request is necessary.
- Preserve prior three-model experiments; this next experiment deliberately measures Sailor alone because the deployed vote lost to Sailor. The probability-based ensemble implementations remain available in history/configs.
- The brief's "review repairs" does not authorize fabricating targets: only exact supported evidence may produce repairs. Zero accepted repairs is valid and must be disclosed.
- Source quality/Sailor fitting settings stay fixed for the classifier data challenger. Corrector is a separate data/format challenger; compare it descriptively with the old selected checkpoint, not as an isolated single-factor causal claim.

- Ruling: target-dependent instruction separation is allowed only while encoding existing fit targets. No original correction is repaired; inference keeps the complete-sentence prompt. Native tokenizer lengths183/222 and84 conditioned exposures are checked.
- Ruling: include selected dependencies and a hash-bound exact-decoder replay entrypoint in the output after review identified absolute input-mount paths as a portability problem. Actual relocated greedy and beam/reranking fixture inference succeeds.
- Ruling: root decoder outcomes need separate source AND phase IDs; identical screen/full names silently hid full metrics. Added numeric progress allowlists and event enablement; actual root fixture records8 screen and4 full outcomes.
- Measured: two fresh five-epoch Bami fits on identical78dev, quality Macro.4795333976/Micro.6794871795 vs anchored.5892063492/.7179487179, both reloaded. Different updates705/410; sentence and false-clean outcomes worsen. This is positive exploratory data evidence, not a7B gain.
- Verified:257 regression tests, native input/model-count and actual tokenizer checks; actual tiny four-fit/reload/decoding/export; selected-model relocation/replay. Preliminary invalid toy retained-prefix fixture failed before fitting and is preserved. No native7B fitting or new teacher generation occurred.

- Ruling: the final2034-row correction worker receives the remaining whole-run budget, rather than the shorter5hour fit-stage cap; beam decoding may dominate training. Whole-run41,400seconds stays enforced and native duration is unmeasured.
