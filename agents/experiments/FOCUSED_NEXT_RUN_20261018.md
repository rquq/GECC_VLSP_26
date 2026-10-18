# Organizer-anchored Sailor/Qwen next run

Status: implemented and locally verified; native7B Kaggle fitting/quality remains pending. User approved all recommendations in THREE_DATASET_COMPARISON_20261018.md and requested complete preparation. No official/private score is claimed. Native training was not attempted on the local8GB GPU. All old datasets, runs, adapters and original annotations are preserved.

## What the next notebook does

The unchanged notebook launcher reads committed configs/experiment.json → focused_plan.json. Attach only PRIVATE dataset/kaggle-focused-private-20261018/VGEC26-focused-private-20261018.zip. The package contains data AND hash-bound baseline adapters; no separate October16 output is required, and the Kaggle dataset name/extraction prefix may vary. Internet and GPU T4x2 remain required for pinned pretrained bases and native standard-LoRA execution. No notebook toggles, folds, gold hints, quantized training or repeated three-model fit.

1. Verify all package members/weights against the committed manifest hash and check exact shared development rows before GPU work. Check native data/model placement. Reload the saved quality Sailor classifier in a fresh process and match its selected score.
2. Fit one fresh Sailor classifier and one fresh Qwen corrector, FIVE FULL epochs each. Keep the best earlier epoch, not automatically epoch5. Fresh process isolation and memory-release gates remain active; no implicit QLoRA fallback or costly automatic retry.
3. Compare quality Sailor and the new Sailor using Macro-F1 under their existing joint-probability argmax. Report fixed binary-first0.5 diagnostics, but do not search thresholds or silently deploy a hard gate. This deliberately measures Sailor alone; prior three-model soft-vote recipes/implementations remain intact as history.
4. Evaluate old Qwen and the new corrector: native saved greedy, beam2, beam4 and beam4 edit reranking. Each gets the same10-row class-stratified screen; native greedy and the best alternative then run on all78 rows. This bounds GPU time; it is not four full-development decoding passes per adapter. Failed alternatives remain explicit and ineligible. Native greedy must finish, and the old baseline's replayed GLEU must match its saved score.
5. Select classifier by development Macro-F1; select corrector/decoder by all-row GLEU, then error GLEU and clean preservation. Keep the baseline when the challenger loses/ties under these rules. Export the2,034 private inputs once using the selected components. This is development selection, not a guarantee of official improvement.
6. Retain results.zip plus selected.json and selected_models/ inside the downloadable run. Selected paths are relative, checkpoint/decoder hashes are checked, and tools.replay_focused replays the exact recorded beam/reranking rule after relocation. Ordinary greedy pipeline_predict is not the replay entrypoint for a selected beam/reranking strategy. Base weights remain pinned downloads rather than bundled7B copies.

The outer W&B root is named run and stays alive through decoding/export. Existing GPU/VRAM/CPU/RAM/process-tree/disk/network collection remains. Numeric decoder progress is forwarded; baseline/challenger AND screen/full result names are distinct so screening metrics cannot hide full78-row outcomes. No corpus text is sent to W&B.

## Training mixture and annotation review

The new versioned vgec-organizer-anchored-v1 bundle binds and revalidates its complete historical diverse/quality source, exact organizer snapshot, deterministic policy, parent/rule provenance and unchanged78-row development set. It does not create new grammar examples, rewrite targets or infer labels. All321 eligible classifier originals and309 corrector originals remain. Existing11 note-target exclusions and the separate ID11 corrector exclusion remain; requested originals retained by the classifier's grouped role stay there. Abbreviated targets remain unchanged once in corrector training.

| Training exposure | Clean | Sentence | Phrase | Function word | Punctuation | Total |
|---|---:|---:|---:|---:|---:|---:|
| Classifier |184|165|87|100|115|651|
| Corrector |184|136|43|63|63|489|

Canonical rows:428 classifier/412 corrector, including24 selected licensed human UD clean parents per stage. Classifier has83 selected existing synthetic errors; corrector79. Synthetic errors occur ONCE, with at most6 examples per rule,2 per parent and1 per parent/subtype. Quotation/identity shortcut patterns are withheld from fitting but retained in embedded historical snapshots. Eligible organizer error examples receive at most2 sentence/3 rare-subtype exposures; notes are protected, as are abbreviated/parenthetical correction targets. Organizer rows account for544/651 classifier exposures and386/489 corrector exposures. These are exposure counts, not independent examples or human-adjudicated quality claims. Class totals are intentionally not forced equal using filler/repetition.

Cross-stage family/template/text/near-duplicate audits pass under the existing scoped policy; exact private-source versus training-source/target overlap is0. Do not call this an exhaustive guarantee of semantic independence or erase historical development-set exposure. All78 validation rows/references are identical to old/quality/diverse. The2,034 private inputs remain unlabeled, inference-only, and are never generation parents.

An automatic format/conflict audit examines all321 organizer training originals:95 flagged rows and10 mechanical reconstruction proposals. The private training_review_not_applied.json preserves original rows/evidence. Accepted repairs: ZERO. It does not replace the earlier162-row/114-flag queue or claim linguistic adjudication. Previous local teachers failed qualification; no new unqualified teacher output, larger-assistant grammar wording, inferred replacement label or public/private-test supervision is admitted.

## Correction reference-format experiment

84 abbreviated/parenthetical training exposures use a separate short generic instruction requesting the existing abbreviated/note-style answer. All target bytes remain unchanged. Complete targets keep the existing minimal-edit/full-sentence instruction. This conditioning runs ONLY in encode_training_record; saved inference, validation generation and private generation always use the ordinary complete-sentence prompt, without inspecting a target or gold label. This addresses the training-instruction/reference mismatch without fabricating complete corrections. Its native language-quality effect is unmeasured and the old corrector remains a candidate.

Actual pinned Qwen tokenizer audit: longest training sequence183 tokens, validation222, both below the unchanged512 cap. No source/target truncation or context enlargement is introduced. Unit checks verify target preservation, no target text in the prompt, immutable export settings, complete-target behavior and invalid policy rejection.

## Exact fresh-fit settings

| Component | LR | Epochs | Warmup | LoRA r/alpha/dropout |
|---|---:|---:|---:|---|
| Sailor-7B-Chat classifier |5e-5|5 full|30 steps|16/32/.05|
| Qwen2.5-7B-Instruct corrector |2e-5|5 full|10%|16/32/.05|

Both seed42/split42, native FP16 frozen bases/FP32 trainable LoRA/head, microbatch1/effective8, max512, one process spanning two T4s, gradient checkpointing. Classifier weighted binary +2conditional subtype CE and square-root class weighting remain unchanged. Classifier settings match the quality Sailor baseline for the data challenger. Corrector restores the old measured optimization settings, but changes mixture/format conditioning/max epochs and checkpoint selection; its comparison is descriptive, not a dataset-only causal ablation. Corrector selects all-row GLEU, error GLEU, clean preservation. Model+adapter counts: Sailor7,139,016,704; Qwen corrector7,655,986,688, each below8B. Source revisions and exact hashes are in the pairedJSON/configs.

Whole-run budget41,400 seconds (11h30); fit/validation child stages at most18,000 seconds and remaining whole-run time; final private correction receives the entire remaining whole-run budget. Runtime is not guaranteed; interruption/failure preserves completed evidence. Final inference can take substantial time, particularly if beam/reranking wins. A new native run is required to measure actual7B results.

## Measured local data screen

Two fresh BamiBERT full-encoder fits, same seed42, native102,955,013-parameter model/revision, five epochs, LR2e-5, FP16 autocast, no LoRA/quantization, microbatch2/effective8, max512, warmup10%, same hierarchical objective and same78 development rows. Both selected checkpoints reloaded with matching labels/probabilities. These are new matched-setting five-epoch screens; do not compare them as identical experiments with the earlier420-update BF16/max256 screens.

| Metric | Quality | Organizer anchored |
|---|---:|---:|
| Macro-F1 |47.95%|58.92%|
| Micro-F1 |67.95%|71.79%|
| Sentence F1 |48.28%|35.71%|
| Phrase F1 |18.18%|50.00%|
| Function-word F1 |37.50%|40.00%|
| Punctuation F1 |46.15%|80.00%|
| Error sentences predicted clean |8/38|10/38|
| Optimizer updates |705|410|
| Fit/reload seconds |285.27|175.68|

Macro improves10.97 percentage points, with real sentence/false-clean tradeoffs. This supports a native Sailor experiment; it does not establish a7B or unseen gain. Training examples/updates differ with mixture size despite matching five epochs. Preserve the common78-row set for requested comparisons, and treat repeated checkpoint/data selection as development work. The native baseline fallback does not guarantee official/private non-regression.

## Verification and limits

Native real-data/count preflight passes without attempting7B placement on the8GB local GPU. All package data/weights/metadata hashes and ZIP CRCs are verified. The actual tiny CPU focused pipeline completes fourfive-epoch fits/reloads (two fixture baselines plus two challengers), allfour decoder screens per model, four full finalist evaluations, isolated private inference and strict integer/string ID export. Executed epoch histories, not only configs, confirm five full epochs. Relocated selected checkpoints replay the exact submission using the recorded decoder. A separate actual toy beam/reranking replay checks the portable decoding contract. These are plumbing tests, not Vietnamese quality evidence. Full native T4 execution/OOM behavior/score remain pending.

The final review identified and fixed missing monitoring event enablement, numeric progress allowlists, source/phase event collisions, and selected dependency portability. Hash-bound selected models and exact decoder strategy reject changed dependencies/decoder settings before reload. Full regression suite, source checks and clean-clone verification are recorded in the pairedJSON. A preliminary artificial fixture had an invalid retained-prefix contract and failed before fitting; it was corrected without changing real datasets, with the failed log retained. Native training/data artifacts were not reused from that failed toy invocation.

Prior active experiment is preserved byte-equivalently in configs/expanded_tuned_experiment.json. Quality and older tuning recipes remain available. No notebook cells or training dependency pins changed. Prior user modifications to CURRENT/PRIVATE_RETRAIN records remain separate; no unrelated files or old experiment artifacts are removed. Temporary development formatter installation is outside the training environment/repository dependencies.

Private artifacts: dataset/kaggle-focused-private-20261018/; runs/focused-preparation-20261018/; runs/focused-native-preflight-20261018/; runs/smoke-focused-20261018-final/. The initial package and failed fixture remain private audit history. Aggregate-only companion FOCUSED_NEXT_RUN_20261018.json records hashes/results. Follow NEXT_RUN_PLAN_20261018.md for the implementation ledger and DATA_GUIDELINES.md for corpus restrictions.
