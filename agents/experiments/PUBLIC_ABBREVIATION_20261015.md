# Public ellipsis audit, restored references and complete inventory — 2026-10-15

Status: private v2 dataset built and verified; no real-model training, generation, score claim or Kaggle launch. The user requested original abbreviated references be retained if past public data supports this format. Public inputs/outputs were read for diagnosis only; zero public predictions or public inputs were added as training samples.

## What the saved public runs contain

Public input SHA256: 5a0c7ca466b639a933e8f472223127ac78e418089584d6b754dd0b4b4eb836a7.443 rows,139 with ellipsis markers. Exact normalized source overlap with organizer training:0. Across24 saved JSON/ZIP materializations there are4 distinct correction payloads; classifier variants and copied archives are not separate corrector experiments. All24 decoded and matched the public input IDs; read errors0.

| Corrector output family | Rows with ellipses | Input already had ellipses | Introduced on input without ellipses |
|---|---:|---:|---:|
| Older Sailor |100|100|0|
| Qwen / Qwen-corrector hybrids |134|134|0|
| ViT5 / small-corrector variants |80|80|0|
| Latest saved14_10_2026 LoRA run |142|138|4|

These are predictions, not hidden gold references. An ellipsis marker can denote an excerpt or real punctuation; its presence alone does not prove a bad target or truncation. The official submission-format example also includes an abbreviated correction. The four introduced markers in the latest output are review flags, not automatically correct new gold. Their IDs and private examples are available in the saved audit; do not feed them back as supervised targets.

## Change from v1 to v2

The blanket ellipsis-target exclusion is superseded. Restore112 EXACT original organizer error/reference rows:36 sentence,25 phrase,19 function,32 punctuation.80 sources already have ellipses;32 sources do not. No target completion, source rewrite, replacement label or model repair was invented. Retention is not a claim that every organizer reference is a fully adjudicated complete sentence. These112 appear once per correction epoch and are exempt from all balancing repeats to avoid amplifying shortened-target behavior. The32 full-source/abbreviated-target pairs remain explicitly flagged by this count.

All401 originals remain unchanged for classification, including the8 previously excluded notes. The11 explanatory note targets remain excluded from correction, and ID11 stays queued for its inline optional-word annotation. Correction now retains389 original rows. Its canonical population rises821→933, while classification stays945. Both balanced training arrays remain1360,272 per class. Legacy192, new code320, external clean32 and all generated example identities are unchanged. No new LLM-generated examples were accepted.

| Class | Classifier canonical | Corrector v1 canonical | Corrector v2 canonical | Balanced per stage |
|---|---:|---:|---:|---:|
|0 clean|272|272|272|272|
|1 sentence|180|143|179|272|
|2 phrase|101|70|95|272|
|3 function|130|108|127|272|
|4 punctuation|262|228|260|272|

Canonical counts retain existing original duplicates and are not equivalent to independent families. Explicit repeats:415 classifier /427 corrector. The v1 archive is preserved byte-for-byte and remains accepted by its original v1 loader/exclusion policy. V2 uses a new format and ZIP; old bundles are not silently reinterpreted.

## Why phrase data is sparse

Original phrase class:34 versus200 clean. Six phrase rows have explanatory-note targets. The v1 corrector also excluded25 abbreviated phrase references, leaving only3 original phrase pairs. V2 restores those25 and retains28 original phrase pairs. Add23 unchanged legacy quantifier-reordering examples and44 new phrase edits: classifier34+23+44=101; corrector28+23+44=95.272 is the training-exposure count after explicit repetition.

The44 new phrase edits are12 dictionary-bounded nominal quantifier reorderings,29 annotated noun/quantifier reorderings and3 annotated classifier/noun reorderings. This is narrow nominal-order coverage, not a comprehensive phrase grammar dataset. Only32 external clean parent families are selected, new parent/rule caps apply, and prior LLM augmentation accepted0. Degree/adjective reorderings can remain grammatical Vietnamese and are not promoted to the current package. Broader comparative, attachment and other phrase patterns found in older experimental bundles are historical inventory, not a new quality endorsement. Repeating the current examples increases exposure but cannot add missing linguistic mechanisms.

## All extra-data types in the active classifier and corrector

The canonical extras below are identical in both active stage views:192 legacy +320 new code +32 external clean =544. Original restored references are original data, not augmentation. The private CSV additionally lists actual repeated exposures separately for each stage. Rule variants are not independent linguistic mechanisms.

| Class | Origin | Recorded type | Canonical count per stage |
|---|---|---|---:|
| 0 | legacy | complete_training_gold_identity | 10 |
| 0 | legacy | quoted_complete_training_gold_identity_ascii | 10 |
| 0 | legacy | quoted_complete_training_gold_identity_curly | 10 |
| 0 | legacy | quoted_complete_training_gold_identity_single | 10 |
| 0 | external clean | external_clean_identity | 32 |
| 1 | legacy | missing_main_copular_predicate | 20 |
| 1 | legacy | quoted_original_clause_edit_ascii | 19 |
| 1 | legacy | quoted_original_clause_edit_curly | 19 |
| 1 | legacy | quoted_original_clause_edit_single | 20 |
| 1 | new code | missing_annotated_main_predicate | 1 |
| 1 | new code | missing_main_copular_predicate | 3 |
| 2 | legacy | quantifier_after_nominal_head | 23 |
| 2 | new code | classifier_after_annotated_noun | 3 |
| 2 | new code | quantifier_after_annotated_noun | 29 |
| 2 | new code | quantifier_after_nominal_head | 12 |
| 3 | legacy | adjacent_copula_or_aspect_repetition | 23 |
| 3 | new code | adjacent_copula_or_aspect_repetition | 32 |
| 3 | new code | duplicated_annotated_conjunction | 10 |
| 3 | new code | duplicated_annotated_preposition | 17 |
| 3 | new code | duplicated_aspect_marker | 11 |
| 3 | new code | duplicated_copula | 4 |
| 4 | legacy | comma_after_terminal_period | 28 |
| 4 | new code | missing_parenthesis_closing | 6 |
| 4 | new code | unmatched_bracket_closing | 32 |
| 4 | new code | unmatched_bracket_opening | 32 |
| 4 | new code | unmatched_parenthesis_closing | 32 |
| 4 | new code | unmatched_parenthesis_opening | 32 |
| 4 | new code | unmatched_quotation_closing | 32 |
| 4 | new code | unmatched_quotation_opening | 32 |

## Every saved dataset

`tools.audit_dataset_inventory` inventories26 packaged stage bundles under dataset/, plus2 full-training raw exports and1 standalone validation export:29 materializations,1554 aggregate inventory rows and99 recorded origin/type keys for private examples. These include unpromoted build iterations; copies are not independent datasets. For every dataset/stage/view/class it records original or extra origin, exact mechanism, canonical/repeat exposure, count and distinct-source count. Validation and training views are separate; do not sum them together with canonical views. The standalone validation export remains historical evaluation material, not a synthetic source.

Private artifacts: runs/public-abbreviation-inventory-final-20261015/ALL_DATASET_TYPES.md, all_dataset_types.csv, all_dataset_types.json, private_examples_by_type.json and public_ellipsis_audit.json. The examples JSON contains one actual source/target pair for every recorded extra-data type across all versions. It remains private and ignored. Some older rule names correspond to questionable/ambiguous transformations; the inventory records their existence without accepting them into current training.

## Verification and artifacts

219 local tests passed, including v1 backward compatibility, exact v2 abbreviated-reference retention, no completion/relabel tampering, protected-repeat behavior, typed public IDs and inherited/introduced ellipsis distinctions. The real v1/v2 packages load through public loaders, have identical classifier populations,112 exact restored corrector originals, unchanged synthetic/external rows, original bytes unchanged and272 exposures per class. All112 restored references are kept exactly once. Archive CRC and all payload hashes passed. A tiny CPU Qwen2 protocol fixture ran the actual v2 classifier/corrector LoRA pipeline, saved/reloaded and exported2 valid test submissions without folds/hints/independent metrics. This is mechanics evidence, not model quality.

New package: dataset/retained-balanced-v2-20261015/VGEC26-retained-balanced-v2-20261015.zip. Exact hashes,24 public artifact records, inventory hashes and the verification report are bound in PUBLIC_ABBREVIATION_20261015.json. Default training/augmentation recipes and notebook cells remain unchanged; selecting the new package for a real fit requires a deliberate recipe/hash update. No run is launched by this change.

Reproduce inventory: `python -m tools.audit_dataset_inventory --output-dir <new-private-audit-directory>`. Reproduce v2: `python -m tools.build_retained_balanced --output-dir <new-private-dataset-directory>`. Both refuse existing output directories. Original/derivative corpus data remains private.

Source: https://aihub.ml/competitions/1038#learn_the_details-overview (submission-format excerpt example and external-data/model-size rules). Saved public outputs provide no hidden per-row target labels; organizer training targets are the only supervision restored here.
