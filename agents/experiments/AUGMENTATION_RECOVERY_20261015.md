# Saved-output augmentation recovery — 2026-10-15

## Intent and preserved evidence

The user delegated augmentation recovery after two expensive runs produced zero usable additions. This change does not repeat training, round-trip translation or error authoring. It reuses the private completed run `runs/augment/augmentation-runs/hy-qwen-sailor-reviewed-roundtrip-20261014T132131880892Z`, sourced from clean commit `018a9fbe9ea4ed9ad278cabaf643c069e3167ae4`: 148 translated references, 249 proposals and 1,462 calls in 12,703 seconds. The prior Sailor reviews are discarded as acceptance evidence; their original journal is preserved. Implementation starts from `c6db7949e06b088f35fca1d7055a86fdf1a65173`.

Mechanical filtering accepts 51 proposals for further review: type1=2, type2=22, type3=21, type4=6 across47 references. It rejects198 before inference. Overlapping reasons include111 overbroad edits,47 non-punctuation-only proposals,47 function edits changing content,38 phrase edits changing vocabulary,32 structure edits adding vocabulary,18 protected-fact/name changes,18 mixed punctuation errors and1 identity edit. Mechanical eligibility does not establish a correct label or target.

## Input package and recipe

Private input artifact: `dataset/augmentation-recovery-20261015/VGEC26-augmentation-recovery-20261015.zip`, 1,144,646 bytes. SHA256 `d14b572998bb7dc94ccac5242c98ba747d5a05b8bc65c6fb24867cc38d408f82`. Payload SHA256 `d9cfd503b7674a7ba77de51a02b71760646d06c792716f1357272c926fad1e95`. This is a recovery input, not augmented training data. It contains the unchanged canonical original/585-row classifier/495-row corrector bundles and a saved-generation packet. All81 former validation rows remain in both training stages. Original annotations, including22 review flags, are preserved. Official public/private tests are neither included nor read.

`tools/package_augmentation_recovery.py` validates canonical hashes, original source metadata, journal entry/output/request digests, translator/author identities, exact translation prompts and applicable author patches against retained training parents. The committed payload/archive hashes prevent swapping saved evidence. Extracted and raw-ZIP uploads are supported; an arbitrary enclosing Kaggle dataset folder name is allowed. Verified recovery inputs are scoped separately from older attached datasets with colliding filenames.

`configs/augmentation.json` loads only Qwen/Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554`, Apache2.0, audited4,022,468,096 native parameters. Hy-MT2-7B and Qwen2.5-7B retain their historical identities but perform zero new calls. Qwen3 is a different checkpoint from the author, with possible same-family correlated errors. The model card is https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 . No new model-size waiver is inferred.

Kaggle stays native FP16 across T4×2 with no quantization or CPU offload. Budget:256 reviewer attempts/90 active minutes; seed42. The notebook remains a launcher with no experiment toggles. Changes are committed choices. A small eight-case operational check precedes the pool: two identical clean pairs, two blind clean sentences, two mismatched-fact pairs, and two existing sentence-structure source/target anchors. It stops on the first failure and publishes a diagnostic instead of a training ZIP. These exposed training controls are not a balanced or independent accuracy benchmark.

## Review contract

Clean review requires explicit integer labels, preserved meaning/entities and explanatory evidence. Blind error review receives the source, taxonomy and entire mechanism inventory, without the proposed type or reference. It selects a unique short source span and replacement. Code only copies and replaces the teacher-selected span; it does not generate a new sentence or infer a training label. Hard JSON constraints restrict the span to actual unique source substrings of at most six whitespace tokens and at most35% of the source; punctuation characters are eligible spans too. Replacement length is bounded to35%. The instruction prefers the shortest necessary edit, including deleting a redundant word before repeating an existing subject.

The exact type, exact mechanism, one required error, preserved meaning, reference correction, confidence≥.9, explanatory evidence and every existing mechanical/diversity gate remain required. Valid JSON or model self-confidence alone never establishes acceptance. A clean-control no-edit verdict may omit explanation; actual accepted error examples and clean-pair reviews still need evidence. A changed-fact negative control only needs to reject meaning equivalence; it need not independently certify the sentences' grammar. These control fixes do not loosen training acceptance.

Schema constraints are bound into each journal request identity, so changes cannot silently reuse unconstrained output. `augmentation/constraints.py` uses LMFE0.11.3's parser core and caches tokenizer data per worker. The library's stock Transformers adapter imports a removed module in Transformers5.6; the local adapter avoids that version-specific import. Test smokes exercise actual constrained native generation rather than mocking this interface.

## Bounded laptop diagnostics

Local GPU is RTX4060/8GB with shared Windows use. Local NF4 storage/FP16 compute is diagnostic only. These outputs are not promoted into the native FP16 Kaggle export. Exact prompts/settings/raw text remain private in `runs/augmentation-recovery-probe*-20261015/`. Each run records new calls and explicit replayed controls separately. Compatible replay is diagnostic evidence, not a new inference count or a fresh eight-case evaluation.

| Probe | New calls | Outcome |
|---|---:|---|
|1|5|Initial clean control selection incorrectly treated arbitrary recorded targets as unquestionably clean. Stopped.|
|2|5|Original-label0 controls substituted. Correct changed-fact rejection also found a grammar issue; the control wrongly demanded label0. Stopped.|
|3|2|Negative-control semantics fixed. Known type1 identified correctly but whole correction was rewritten. Failed exact target.|
|4|2|Span-output prompt added. Correct clean no-edit verdict failed an unnecessary explanation requirement in the control.|
|5|2|Control explanation rule corrected. Model still attempted whole-sentence span/replacement; rejected.|
|6|1|Hard span/JSON decoding prevented a full rewrite, but shorter rewrite still failed the recorded target.|
|7|3|Explicit shortest-edit/deletion preference still failed the first known type1 correction. Four clean-pair controls were explicitly replayed; two clean no-edit controls were newly generated. Stopped before candidate review.|

The prior failures are retained, not presented as successful review. Development smokes also exposed and fixed the LMFE adapter import and tokenizer-data constructor arguments. The toy tokenizer cannot spell spaces outside its registered whole-answer tokens; the final actual constrained test uses a representable string enum and sufficient decoding length. Its native19k-parameter model must obey that schema. Source-copy explanations, including punctuation-only variants, are explicitly rejected; minimum string length alone did not ensure explanatory evidence. A focused test command initially named a nonexistent worker test module; the corrected focused command excludes it. The47 focused tests pass under Transformers5.6, and the actual CPU tiny-model smoke verifies constrained generation, journal reuse, all three workers exiting and recovery stopping without a dataset. Actual pinned-package discovery passes for extracted/raw uploads alongside an older conflicting dataset. The full preserved repository suite passes202 tests in22.678 seconds. Both notebook code sources parse; no experiment toggle was added. Read the JSON record for the committed CLI preflight result.

## Validation and current limits

**Current status: recovery implementation prepared; local reviewer quality check failed; native FP16 pilot and accepted yield pending.** This package is suitable for a bounded native reviewer diagnostic, not a claim that Qwen3 has qualified to label the51 candidates. Do not rerun full translation/authoring or start retraining from a baseline-only ZIP.

The local reviewer check **failed**, after20 fresh calls across seven bounded development probes; replayed controls are counted separately. No real augmentation has been accepted, no classifier/corrector training has run, and no downstream score gain is established. An operational reviewer pilot, even when it succeeds, does not demonstrate balanced taxonomy accuracy or generalization. Native full-size dual-T4 execution remains a separate Kaggle result. Do not interpret an input ZIP or a valid baseline-only export as augmentation success.

Successful recovery exports unchanged baseline prefixes plus strictly accepted error/clean pairs. The existing training branch consumer and hashes must be updated deliberately before training on that export. Preserve all private prompts, outputs, lineage, model audits and rejected evidence for organizer access. No assistant-generated corpus wording, automatic repair or unsupported relabeling is allowed.
