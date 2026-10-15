# Extra grouped validation dataset — 2026-10-15

Status: **completed dataset build and consumer checks; no model training or Kaggle launch**. Source commit: `c33d834` on `kaggle-linux`. This commit contains the implementation. The user requested an extra dataset with validation and delegated the ratio. The previous skipped-verification request concerned the preceding full-fit branch update; this request specifically requires dataset separation checks.

The private package is `dataset/kaggle-grouped-validation-20261015/VGEC26-grouped-validation-20261015.zip`. The separate committed recipe is `configs/validation_experiment.json`. The default `configs/experiment.json`, existing full-fit stage configs, notebook and source datasets remain unchanged. Uploading only the validation ZIP does not switch the default notebook recipe; experiment activation remains a chat/commit choice with no notebook toggles. The runner already accepts `--recipe configs/validation_experiment.json` for an explicit command-line launch; no launch was performed.

## Populations

Choose approximately **80/20 of eligible organizer originals**, seed42. The 389 correction-eligible originals define per-class targets; whole text/lineage groups, annotation exclusions and validation duplicate removal determine the exact split. Validation has78 unmodified original representatives across76 components; classifier training has321 originals and corrector training309. Two duplicate organizer counterparts are withheld with validation and not scored twice. All8 user-retained note originals stay unchanged in classifier training; the existing11 note-target and ID11 correction exclusions remain.

| Class | Classifier canonical train | Corrector canonical train | Exposures per stage | Shared validation |
|---|---:|---:|---:|---:|
| 0 clean |229|229|229|40|
| 1 sentence structure |141|140|229|19|
| 2 phrase structure |91|85|229|6|
| 3 function words |110|107|229|6|
| 4 punctuation |207|205|229|7|
| Total |778|766|1145|78|

Canonical means stored examples before resampling, not guaranteed distinct source strings. Both stages retain158 legacy pairs,267 newer code examples and32 external clean controls. Relative to each source canonical population,78 representatives become validation and89 rows are withheld:87 synthetic descendants/related examples plus2 original duplicate counterparts. Nothing is silently relabeled or repaired. Balancing occurs only after partitioning; validation has no synthetic examples or repeated exposures. Training repeats have distinct alias IDs and at most4 total exposures per canonical example. Corrector84 abbreviated training references occur once each; validation retains26 abbreviated organizer references, while2 abbreviated duplicate originals are withheld. The original112 remain present across these three roles. Abbreviated references remain an annotation/metric limitation, not reconstructed full corrections.

## Separation protocol and limits

Build a joint graph from all original rows and both source canonical stage populations. Connect typed IDs, explicit parents (including resampling sources), specific family/template IDs, normalized source OR target equality, equal token multisets, any shared10-token sequence, and token near duplicates. Normalization is Unicode NFKC/casefold plus word tokens. Near matches require at least8 tokens, length ratio>=0.8, token-set Jaccard>=0.8 and token SequenceMatcher ratio>=0.9. Generic grammar operator names alone do not imply shared content ancestry. Components with unsupported correction annotations or exact-source conflicting annotations cannot become validation. Select whole eligible components toward original class proportions, without examining model predictions. Withhold all descendants and duplicate counterparts of the selected components; then rebalance only the remaining training canonical rows.

The union of BOTH stages' actual training exposures is audited against the shared original validation. All six counters are zero: typed ID intersection, normalized source/target intersection, token-bag intersection, shared10-token spans, direct held-parent hits, and mixed lineage/text components (including transitive and near matches). All544 source extras per stage have explicit parent IDs. Unresolved parents fail closed. The runtime loader revalidates source retained-v2 packets, content hashes, deterministic selected IDs, the derived train arrays and exact organizer validation; edited/resealed contamination is rejected. The classifier now consumes this package's holdout rather than recreating the older random split. The no-hint corrector handoff enforces identical normalized validation records.

Only top-level `train` enters fitting. Embedded `source_packet` (including prior full-fit arrays), `validation`, split manifests and withheld-family records are private audit/evaluation data and must never be concatenated into training. Both stages' source packets are embedded to make runtime recomputation possible. Public_test.json is copied unchanged for inference only; no public labels/predictions enter training. UD-VTB train-only origin, revision and license notices are preserved. Keep the ZIP private under the corpus agreement.

**Bounded guarantee:** these exported fitting/evaluation arrays have zero overlap under the recorded rules. Arbitrary semantic paraphrases and historical project/human/pretraining exposure cannot be ruled out. Originals were previously used in full fits and reviews; this does not create a historically blind test. Use fresh pretrained Sailor/Qwen bases, no prior project adapters/classifier reuse. The extra recipe has `classifier_run=null`, `initial_adapter=null`, no folds/hints, standard LoRA and maximum5/6 epochs; validation selection is enabled. Rare subtype supports6/6/7 imply noisy metrics; treat this as development validation and avoid repeatedly selecting on it as if it were an untouched final test.

## Evidence

96 focused/regression tests passed, including14 new toy-only grouped partition/package tests. Tests include genuine near matching without a shared10-token span, missing-parent failure, cross-stage target leakage, whole-family withholding, typed IDs, training-only repeats, deterministic selection, contamination resealed with a valid outer digest, original-file binding and ZIP corruption. Existing data/builders, no-fold handoff, run reuse and experiment contracts also passed. This was not the entire ML training test suite and no tiny/full model training ran.

Actual source data checks passed: retained-v2 provenance validation, cross-stage leakage audit, both public loaders, classifier organizer partition, no-hint classifier→corrector handoff1145/78, and command preparation using mocked model sizing (no model load/download/launch). The ZIP reopened with CRC, exact member set and every payload hash checked. Existing source/full-fit hashes remain unchanged and their original1360-row/no-validation loaders still work. No quality improvement is claimed. Exact artifact/data hashes, populations and check outcomes are in the paired JSON and private manifests.
