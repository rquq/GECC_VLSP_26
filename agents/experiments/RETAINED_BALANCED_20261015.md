# Retained and balanced dataset — 2026-10-15

Status: built and locally verified as a DATASET CANDIDATE. No corpus model fit, pretrained model download, teacher generation, target repair, independent accuracy result or Kaggle launch. A tiny randomly initialized toy fixture did run the real two-stage mechanics locally. The unchanged notebook/default recovery recipe is not a launcher for this package. The previous 585/495 control and failed model generations remain preserved.

The user authorized building near-equal per-error-class counts, superseding the earlier 45–50% clean-exposure proposal and the local dataset-build hold. Their follow-up clarifies that “ignore the eight rows” means KEEP them unchanged. All401 original rows are now present in the classifier and master, including IDs104,105,116,121,141,143,147,168. No original source, target or classifier label was rewritten. Those eight remain part of the11 explanatory-note targets excluded from corrector training; they are never amplified through resampling.

## Actual populations

Canonical means before resampling, not independent source families. Existing organizer duplicates are retained. All320 newly generated sources are deduplicated against the classifier base and one another.

| Class | Classifier canonical | Corrector canonical | Classifier training | Corrector training |
|---|---:|---:|---:|---:|
|0 clean|272|272|272|272|
|1 sentence structure|180|143|272|272|
|2 phrase structure|101|70|272|272|
|3 function word|130|108|272|272|
|4 punctuation|262|228|272|272|
|Total|945|821|1360|1360|

Classifier canonical composition:401 originals +192 unchanged legacy pairs +32 external clean controls +320 new deterministic errors. Corrector:277 original pairs +192 unchanged legacy pairs +32 controls +320 new errors. Distinct normalized sources:938 classifier /818 corrector. The seven/three existing duplicate-source rows are not called new independent examples. Explicit additional exposures:415 classifier /539 corrector. Clean share is20% in the balanced arrays, deliberately following the latest instruction. This changes training priors; it is not evidence that a 20% clean test prior is appropriate.

## Source and operators

External training source: UD Vietnamese VTB, revision3c971fd1c5d03561e00b267d3d0057b81a9bea84, CC BY-SA4.0. Its native manual annotations were converted to UD. Downloaded only the train split plus upstream README/license. Data bytes, revision and SHA256 are pinned in the private manifest.32 complete filtered parents supply clean controls and annotated edits; external dev/test and competition test data were not used. Attribution and the full upstream license are in the archive. The external component retains its license; it does not relicense VGEC. Keep the mixed archive private under the VGEC agreement.

New deterministic error pairs:4 sentence /44 phrase /74 function /198 punctuation. They copy corpus wording and apply a recorded reversible span edit; no assistant-authored sentence wording, cloud teacher or failed local-model output enters the corpus. The method is code augmentation, not successful model synthesis or human adjudication.

Phrase edits move an existing plural marker/classifier relative to an adjacent complete nominal head. Degree/adjective reversal is rejected because it can remain grammatical Vietnamese. Sentence-front swaps are rejected to avoid incidental capitalization errors. Sentence structure deletes a complete supported nominal predicate, rather than deleting just a verb and possibly leaving valid zero-copula predication. Function edits duplicate applicable copulas/aspect markers or annotated adpositions/conjunctions. Punctuation edits create one unmatched quote/parenthesis/bracket, without changing words. New terminal-period/comma examples are not generated. Three uncertain external parents are excluded; exact reasons are in private metadata and the builder.

New examples are capped at4 per parent and32 per exact rule. That is a rule-level bound, NOT17 distinct linguistic mechanisms: many rules are variants of the same duplication/reordering/delimiter family. Phrase variety and naturalistic function-word coverage remain bottlenecks. The192 legacy pairs are retained exactly, including their previously documented narrow templates, without upgrading their evidence status.

Balance uses explicit aliases containing source_id/draw/is_new_unique_example=false, with at most4 total exposures per canonical row. Repeats prioritize organizer rows, then legacy pairs, then new code pairs. The11 note rows are never repeated. No repeated row is called novel synthetic data. Both canonical and resampled views are exported, enabling future comparisons without rebuilding evidence. Apply ordinary shuffling; do not add another inverse-frequency sampler blindly. Current class weights computed from these balanced training frequencies are equal.

## Corrector supervision still unresolved

The112 abbreviated original targets remain in the master/classifier and private unresolved queue. They are excluded from the active corrector array until supported recovery is validated; no missing context was invented. ID11 also remains unresolved because its target has inline optional-word annotation and was already quarantined by the control. Thus113 review targets plus11 note targets are excluded, retaining277 original corrector pairs. ID48 is retained with its full ORIGINAL organizer target despite an abbreviated source; no new wording was generated. This is a conservative active training view, not a claim that all original targets were repaired. The earlier local7B recovery proposal is still unexecuted.

## Verification and scope

Source base commit0854a2b; exact dirty-source builder/validator/loader hashes and all final data hashes are in the JSON record. The original organizer file remains byte-identical. Real package checks loaded both1360-row arrays through public loaders, verified all401 classifier originals and restored8 exact, all192 legacy pairs unchanged in each stage, note exclusions, no note repeats, zero independent validation, source-code hashes, archive CRC and every archived file SHA256. New candidate edits are regenerated from their recorded original/annotated parents during validation. Resealed target/label/source changes and changed repeats are covered by tests.

The local suite passed213 tests in the preserved Transformers environment. This validates mechanics, not grammar accuracy or score improvement. No classifier/corrector was trained on the VGEC package; no grouped comparison or tokenization/full-size Kaggle run occurred. A separate tiny native Qwen2 fixture completed real CPU classifier/corrector LoRA, save/reload and a valid two-row submission using the new format. It exposed and fixed missing classifier reporting keys and corrector final-fit recognition. Failed/successful logs and the private reproduction helper are bound in the JSON record. This is mechanical evidence, not Vietnamese quality evidence. Former81 are eligible final-training parents and cannot serve as unseen evaluation for this version. Positive local review-quality/data-comparison evidence remains required before recommending another Kaggle run. Do not treat the ZIP as a submission or as automatically selected by the older notebook recipe.

Reproduce from the preserved pinned baseline and private external source snapshot:

```bash
USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 ../.venv/bin/python -m tools.build_retained_balanced --output-dir dataset/<new-private-directory>
USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 ../.venv/bin/python -m unittest discover -s agents/tests
```

Builder refuses existing output directories, verifies baseline/external hashes, writes a versioned vgec-retained-balanced-v1 bundle (seed42, fraction0), raw stage arrays, canonical arrays, master, pending queue, generated pairs, rejection summary, external parent evidence, licenses, file hashes and a private ZIP. Consumption requires the new loader and deliberate training recipe/hash selection. The training branch and default notebook were not changed.

Final artifact: dataset/retained-balanced-20261015/VGEC26-retained-balanced-20261015.zip. Manifest: training_manifest.json; per-file integrity: file_manifest.json. The separate earlier candidate folders are unpromoted private build iterations, not additional experiments or recommended training packages. Historical training was not rerun.

Sources: https://aihub.ml/competitions/1038#learn_the_details-overview ; https://github.com/UniversalDependencies/UD_Vietnamese-VTB ; local ../test/VLSP2026-ViGEC-UserAgreement.docx. External resources and pipelines must be disclosed to organizers; the≤8B limit applies to any model used for generation. This build used no generative model.
