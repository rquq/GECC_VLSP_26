# Validation-only export and augmentation rules — 2026-10-14

**Superseded interpretation:** the user clarified they wanted these rows included in a new training dataset. FULL_TRAINING_20261014.md/.json records the corrected585/495-row training export. This earlier validation-only artifact remains preserved for history.

Completed the user-requested separate export of the existing **local** validation split from the organizer training corpus. This is not an official separately supplied development set or the443-row public test.

Private folder: `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/dataset/validation-only-20261014`. `validation.json` contains all81 original raw rows in the exact existing order, preserving IDs, Input/Output, extra fields and null targets. Counts: clean40, sentence20, phrase7, function7, punctuation7. No filtering, repairs, relabeling or synthetic rows. JSON SHA256: `a025c9323e6167689c12924d9ba08b837d77dc59fea6e467e7a4eb6436b76402`; bytes: 24565. Private ZIP: `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/dataset/validation-only-20261014/VGEC26-validation-only-20261014.zip`, 7824 bytes; SHA256 `08cc278e25cda39f262a31fc2d0a84c08c5727e2a6e16eb6921d9e6f1ef13a94`. Members: validation.json, validation_manifest.json, README.md.

The export exactly matches both stage bundles and the grouped seed42 original split. No validation ID is present in original/classifier/corrector training. Original file hashes are unchanged. ZIP CRC/members and full JSON roundtrip pass. `validation_manifest.json` intentionally avoids the production Kaggle manifest filename. This is an evaluation/review export, not a replacement training package. All corpus-bearing files remain Git-ignored. Exporter source commit: `7049040d36d938efedd5a3fe00708e0c8e1c3e8d`; exact code digest is in the JSON. Source and data hashes are recorded; no full training or unit-suite rerun was needed for the actual-copy checks.

## Round-trip translation

Competition1038 permits synthetic generation pipelines with models no larger than8B, external-resource documentation and organizer access upon request. Interpreting this to include round-trip translation is an inference: the page does not name back-translation explicitly. The corpus agreement separately permits research modifications, requires noncommercial use and prohibits redistribution of edited corpus resources. Keep exports/translated derivatives private.

Each translator in both directions and every model creating error examples or new labels must satisfy the cap; using a small final translator does not make an unverified first translator compliant. Pin revisions, count native parameters plus adapters and check licenses before generation. These are eligibility requirements, not evidence that translated examples will improve classification.

NLLB-200-distilled-600M was audited with meta tensors: **615,073,792 parameters**, revision `f8d333a098d19b4fd9a8b18f94170487ad3f821d`, CC-BY-NC-4.0. No weights were downloaded, and no translation ran. Qwen2.5-7B-Instruct is another candidate under its existing7,615,616,512-parameter native audit; its card lists Vietnamese support. Output quality remains unmeasured for this proposed translation use.

GPT-6.1 Sol's official page lists reasoning efforts but does not disclose a parameter count establishing≤8B. Its augmentation eligibility is therefore unverified; exclude it from translation, paraphrasing, example authoring and inferred replacement labels for1038. Extra-high is a reasoning setting, not a parameter-size certification. This does not assert Sol's exact architecture. Codex can implement tools and inspect existing evidence without supplying uncounted training wording/labels.

## Pilot design, not executed

Use only complete reviewed training references/clean seeds for Vietnamese→pivot→Vietnamese translation. Verify meaning, names, numbers and fluency. Then use eligible local generation or documented human adjudication to construct one grammar mechanism with its correct full target and matched clean control. Never assume an original error label survives translation; errors can disappear or change class. Keep parent families together and record model/count/license, prompt/settings, hashes and reviewer evidence. Quarantine ambiguous examples. The81-row validation export is evaluation-only under the current protocol and must not seed training augmentation while serving as validation. These rows originate in organizer training data; this separation is our evaluation policy, not an extra organizer rule. A final fit using all organizer training rows would retire this81-row holdout as independent validation and require a documented protocol change.

No synthesis, model training or production recipe changes occurred for this request. DATA_GUIDELINES.md now records the method, eligibility conclusion and split/privacy constraints.

Sources: [competition rules](https://aihub.ml/competitions/1038#learn_the_details-overview), local `/home/quq/machineLearning/competitions/VLSP26_4/test/VLSP2026-ViGEC-UserAgreement.docx`, [GPT6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol), [reasoning effort](https://developers.openai.com/api/docs/guides/reasoning), [NLLB model card](https://huggingface.co/facebook/nllb-200-distilled-600M), [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
