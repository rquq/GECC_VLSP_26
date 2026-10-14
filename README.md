# Private Kaggle augmentation recovery

Branch `kaggle-augment` now **reviews the saved generation run**. It reuses 148 translated references and 249 Qwen proposals, rejects 198 mechanically invalid proposals before inference, and reviews the remaining 51. It does not repeat translation, authoring, classifier training, or corrector training.

Upload **`VGEC26-augmentation-recovery-20261015.zip`** as a private Kaggle dataset and attach it to the existing notebook. The upload's enclosing folder name can vary. Use Internet and GPU T4 ×2, retain the repository-read `GITHUB_TOKEN` secret, and choose Save Version → Save & Run All. The launcher fetches `kaggle-augment` and runs the committed recipe. There are no notebook experiment toggles. The previous zero-yield augmented ZIP is not the recovery input.

The recovery ZIP contains the unchanged original/585-row classifier/495-row corrector baselines and a pinned private packet of saved generation evidence. It is **an input package, not newly augmented training data**. Both stages retain all 81 former local validation rows, which no longer provide independent validation. Official public/private test inputs are not needed or read.

**Current evidence:** the bounded local NF4 reviewer check failed a known sentence-structure correction. No new training rows are approved. The next Kaggle job is a native FP16 reviewer diagnostic first; it reviews candidates only if that check passes. Do not start retraining from the recovery input or the older baseline-only output.

## What runs

| Role | Checkpoint | Native parameters | Recovery work |
|---|---|---:|---|
| Saved translations | tencent/Hy-MT2-7B | 7,504,568,320 | Reuse recorded outputs; no model load |
| Saved proposals | Qwen/Qwen2.5-7B-Instruct | 7,615,616,512 | Reuse recorded outputs; no model load |
| New reviewer | Qwen/Qwen3-4B-Instruct-2507 | 4,022,468,096 | Review in native FP16 across both GPUs |

Revisions, baseline/payload/archive hashes, model identities and budgets are committed in `configs/augmentation.json`. The new reviewer uses a different checkpoint from the author; correlated model errors are still possible. This is automated screening, not human adjudication or proof of score improvement.

The pipeline validates saved journal hashes, source models, translation prompts, author outputs and retained original parents before loading weights. It discards old Sailor reviews. Mechanical filtering preserves the original minimal-span, vocabulary, punctuation, protected-content and type restrictions.

Before reviewing the pool, Qwen3 checks eight recorded-original controls: unchanged clean sentences, clean sentences that must stay unchanged, unrelated facts that must be rejected, and two recorded sentence-structure errors. A failed check stops the job and links its diagnostic instead of publishing a training dataset. This is a small operational check, not a balanced taxonomy benchmark. Local development used a bounded 4-bit laptop diagnostic; Kaggle uses FP16 and repeats the check at that precision.

Clean-pair reviews use explicit numeric labels and meaning/entity checks. Blind error reviews see only the error sentence, taxonomy and complete mechanism inventory. Hard JSON decoding limits source spans to unique substrings of at most six tokens/35% and bounds replacement length. They return a short source span and replacement; code applies exactly that teacher-selected edit. Whole-sentence rewrites, missing/overbroad spans and unsupported replies are rejected. Acceptance still requires the exact intended subtype, exact mechanism, reference correction, one required error, preserved meaning, confidence ≥0.9, explanatory evidence, all mechanical gates and parent/reference/diversity caps. No assistant-authored corpus wording or inferred replacement labels enter training.

The hard budget is **256 reviewer attempts / 90 active minutes**, including failed attempts and model loading. No call is made to the translator or author. One bounded retry is allowed for malformed review output. A worker owns the model and exits before cleanup; there is no Kaggle quantization or CPU offload.

## Outputs

When review succeeds, the notebook links `VGEC26-augmented-full-training.zip` under `/kaggle/working/augmentation-runs/<run>/dataset/`. It includes canonical augmented bundles, raw stage arrays, accepted/rejected evidence and a manifest. Original rows are immutable prefixes. Preserve the private run's prompts, outputs, hashes and model audit for organizer access.

Check `accepted_error_pairs` and `new_rows_per_stage`. A valid ZIP with zero additions is baseline-only and must not be interpreted as an augmentation success. If the reviewer control check fails, only its diagnostic is linked. Before training on an accepted output, update the training branch consumer and committed data hashes deliberately; the existing `kaggle-linux` training recipe remains preserved.

## Reproduction and checks

`python -m tools.package_augmentation_recovery --source-run <completed-run> --baseline-dir <canonical-baselines> --output-dir <new-private-directory>` reproduces and verifies the input package. It preserves source files and never overwrites an existing package.

`python -m tools.run_augmentation --preflight-only` checks actual inputs and native reviewer metadata/placement without loading weights. `--resume-run <saved-recovery-run>` resumes only an unchanged source/recipe/input identity. It never silently resumes the old generation journal under the new code.

Use the preserved environment for `python -m unittest discover -s agents/tests`. Focused augmentation tests and `python -m agents.tests.smoke_augmentation --device cpu` run in the Transformers 5.6 inference environment. Toy smokes establish mechanics only; the 8GB laptop cannot prove full-size dual-T4 FP16 behavior.

Read `AGENTS.md`, `agents/README.md`, `agents/docs/DATA_GUIDELINES.md` and `agents/experiments/AUGMENTATION_RECOVERY_20261015.md` before changing the experiment. Keep corpus-bearing inputs, derivatives and outputs private. Every generator/reviewer remains within 8B, with pinned revisions, licenses and disclosure records. Both earlier zero-yield runs and the memory failure remain documented; no historical training was rerun.
