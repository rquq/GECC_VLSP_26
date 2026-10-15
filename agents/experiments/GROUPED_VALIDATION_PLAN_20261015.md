# Grouped validation dataset implementation plan

**Goal:** Produce an extra private retained-v2-derived dataset with one shared approximately80/20 original-row holdout and explicit cross-stage leakage checks, preserving the full-fit package/default.

**Architecture:** A deterministic seed42 graph groups typed IDs, parent/family/template lineage, normalized source/target equality, equal token bags, shared10-token spans and token near duplicates. Select whole eligible components for validation, export original rows only, withhold their synthetic descendants, then rebalance training only. Embed source packets and validate all derived populations on load. The graph's declared rules bound the no-leakage guarantee; historical model/human exposure is not erased.

**Spec:** User asks for an extra dataset with validation and no leakage; ratio delegated. Existing note exclusions, source/label/target preservation, private-data rules, no notebook toggles, no models trained. The previous request to skip verification concerned the previous training-branch preparation; this request requires dataset leakage audits.

- [x] Add toy regression tests for parent/counterpart/template/duplicate/near-duplicate grouping, typed IDs, held-out originals, training-only repeats and contamination rejection.
- [x] Implement pure grouping, component selection, derivation and loader in model/grouped_validation.py; compatible default fractions remain unchanged.
- [x] Build a separate private ZIP with both stage populations, shared validation, lineage/quarantine metadata, runtime hashes, licenses, audit results and inference-only public input.
- [x] Add loader/classifier validation dispatch and separate committed validation configs; leave active full-fit defaults unchanged.
- [x] Run focused data tests, real-data leakage/ZIP checks and the applicable focused regression suite; record exact boundaries/history caveat in agents/. No full-size/tiny-model training.
- [x] Commit/push source and aggregate experiment records; keep corpus-bearing artifacts private.

Completed:96 focused tests passed, real-data and archive checks passed; no training. See GROUPED_VALIDATION_20261015.md/.json. Implementation and records are committed/pushed together.
