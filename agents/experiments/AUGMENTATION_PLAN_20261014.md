# Kaggle augmentation implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline for this user-delegated implementation; perform a fresh whole-branch review at completion.

**Goal:** Ship a separate Kaggle augmentation branch producing audited full-training data while preserving the original585/495 rows.

**Architecture:** Pure corpus/contracts code selects seeds, validates cross-model evidence and exports a strict augmented bundle. A bounded inference engine/journal runs translation, authoring and independent review sequentially. The existing notebook helper launches the augmentation entrypoint and links its private dataset ZIP.

**Tech Stack:** Python, Transformers5.6.0, Accelerate, CUDA Torch retained, native FP16 inference, unittest.

**Spec:** agents/experiments/AUGMENTATION_DESIGN_20261014.md

## Global constraints

- Branch kaggle-augment; no default model training, toggles, folds or hints.
- Preserve585/495 baseline prefixes/all81 added former validation rows; official tests unread.
- Native per-model<=8,000,000,000; Hy tying must survive loading.
- Seed42;96 parents; English/Chinese pivots;200 errors maximum;1344 calls/360 minutes.
- Label caps24/64/64/48,6 mechanisms per class,4 errors per parent/2 per reference.
- Corpus-bearing artifacts private/ignored; automated evidence is not human adjudication.

## Review focus

- Malformed JSON/booleans/nonfinite confidence must be rejected, never coerced.
- Error subtype and correction inferred blindly; author declarations cannot satisfy review.
- Duplicate/conflicting text, edited baselines or wrong parent IDs must fail bundle verification.
- Cached journal calls must match exact prompt/settings and budget across resumes.
- Tied head placement, short-input token limits, model cleanup and runtime failures must be explicit.

### Task1: Data contracts and selection

Files: augmentation/data.py, augmentation/review.py, model/data.py, agents/tests/test_augmentation.py.
Interfaces: select_parents(inputs,recipe), validate_pair(pair,policy), select_pairs(pairs,baselines,policy), export_dataset(inputs,pairs,output,recipe,run_identity), load_augmented(raw,fraction,seed).
- [x] Write rejection/baseline/cap/source-contract tests and observe failure.
- [x] Implement seed selection, independent evidence gates and strict append-only bundle export/load.
- [x] Verify targeted tests and ZIP/data roundtrips.

### Task2: Bounded inference and journal

Files: augmentation/inference.py, augmentation/journal.py, augmentation/pipeline.py, tools/run_augmentation.py, configs/augmentation.json, agents/tests/test_augmentation_runtime.py.
Interfaces: NativeEngine(settings,device), Journal(output,recipe,identity).call(role,key,prompt,sampling), run_pipeline(inputs,recipe,caller,output,identity).
- [x] Write placement/caching/tamper/budget/role-switch tests and observe failure.
- [x] Implement native audits/explicit maps, role-aware chat prompts, durable journaling and sequential stages.
- [x] Verify real tiny CPU/CUDA inference and a complete scripted no-download corpus pipeline.

### Task3: Launcher, documentation and delivery

Files: tools/notebook.py, notebooks/kaggle_two_phase.ipynb, notebooks/local_two_phase.ipynb, requirements-kaggle.txt, README.md, AGENTS.md, agents/experiments/CURRENT.md/history records.
- [x] Test augmentation ZIP presentation and branch/default entrypoint; implement simple launcher reuse.
- [x] Record exact recipes/hashes/budget/tests/status and production training branch preservation.
- [x] Run regression suite, focused runtime checks, fresh review/fixes; commit and push kaggle-augment.
