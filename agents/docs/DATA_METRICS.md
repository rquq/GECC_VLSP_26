# Dataset evidence and local evaluation

The original corpus is isolated at `dataset/original/VGEC 2026_v1.json`;
`Metrics_GEC.pdf` specifies evaluation. The accompanying XLSX supplies names and
descriptions only. Original gold is never silently rewritten. The inventory
below describes the original 401 records, not the derived balanced training data.
The correction recipe uses `dataset/newdataset/VGEC_balanced.json`: original-only
splits, unchanged abbreviated original training corrections, explanatory/invalid
target exclusion, and locally authored augmentation capped at 2× retained originals
per error class (also capped at the majority-original count). Class 0 is unchanged.
Seed 42: 310 originals + 150 synthetic = 460 training rows.
Validation stays at 81 original rows; all training source/target exact normalized
matches to validation source/target are excluded. See [../README.md](../README.md)
for provenance, caveats and the builder command.

Both bundles use `vgec-balanced-v3`: `seed: 42` and top-level `train`,
`validation`, and `provenance`. `VGEC_classifier.json` retains all 320 original
training labels and adds 160 synthetic rows; correction-reference quarantine
does not discard classifier labels. The `vgec-manifest-v3` manifest has
`seed: 42`, a correction `split` summary and a separate `classifier_split`.
The builder takes no seed CLI option; the loader rejects other seeds and older
bundle formats.

## Dataset inventory

JSON SHA-256:
`7968ea77cbcf69a26c9f98f1946c4b4dc19237a02c67fb483d54265f109dc728`.

| Class ID | Rows | XLSX `Error_Type_Name` | Raw ID range |
| --- | ---: | --- | --- |
| 0 | 200 | Không có lỗi | 202–401 |
| 1 | 98 | Lỗi cấu trúc câu | 1–98 |
| 2 | 34 | Lỗi cấu trúc ngữ đoạn | 99–132 |
| 3 | 33 | Lỗi về sử dụng từ công cụ | 133–165 |
| 4 | 36 | Lỗi cách chấm câu | 166–201 |
| Total | 401 | | 1–401 |

All 401 IDs are unique and non-null. Inputs and labels have no nulls. Exactly
200 corrections are null, all in class 0; all 201 nonzero-class corrections are
strings and differ from their sources. The class-0 identity correction is an
**inference** from the workbook's “Không có lỗi” name and null-gold pattern, not an
explicit evaluation rule stated in the PDF. The loader substitutes the exact,
untrimmed input for these null corrections. It rejects nonzero-class null gold
corrections when labels are required; it does not convert an empty correction
string to the source.

The workbook's `xl/worksheets/sheet1.xml` has 1,000 row elements but only 401
populated data IDs after its header. ZIP/XML inspection of shared strings and
worksheet cells confirms that all 401 populated IDs, source texts, class IDs, and
corrections agree with the JSON. Class IDs appear as numeric worksheet values
such as `1.0`; their names above are actually present in the workbook, not guessed.
Qwen3-4B QLoRA predicts class IDs 0–4; the corrector receives Vietnamese class descriptions as advisory hints.

### Classification imbalance and workbook explanations

The classifier uses `VGEC_classifier.json`, not the correction augmentation
bundle. Seed-42 original training counts for classes 0–4 are **160, 78, 27, 26,
29**; synthetic counts are **0, 78, 27, 26, 29**, giving **480** mixed rows with
counts **160, 156, 54, 52, 58**. The unchanged holdout has **40, 20, 7, 7, 7**.
The 460-row correction bundle's counts (**160, 156, 42, 46, 56**) are separate.
`configs/classification.json` sets `training.class_weight_power` to 0.5:
weights are proportional to mixed `count ** -0.5`, normalized so their
training-row mean is one. Main-fit weights are approximately **0.802024,
0.812241, 1.380545, 1.406843, 1.332090**. Each OOF fit excludes connected
source/parent groups from both bundles, caps synthetic rows again at its own
original count per class, and recomputes weights from that fit's mixed rows.
Neither outer holdout labels nor OOF prediction rows enter its loss.
Square-root weighting is a conservative choice relative to full inverse
frequency because minority labels include annotation ambiguity. It is not
evidence of improved Macro-F1; compare full runs with power 0 and 0.5 while
holding other settings fixed, and inspect per-class recall and confusion.

`Error Description` is a **per-example annotation explaining the diagnosis**,
not a fifth model input supplied at deployment. XML inspection found nonempty
descriptions on all 201 nonzero-class rows (98/34/33/36), none on 200 class-0
rows. For ID 1 it explains the incompatible pairing of “Với” and “cho nên”;
ID 133 explains a redundant “Với” that prevents the noun phrase from acting as
subject. Such wording can reveal the class and intended edit. Training a
classifier on `[source, gold description]` while inference has only `source`
would introduce unavailable answer-bearing information and a train/inference
mismatch. No workbook descriptions are added to classifier inputs.

Descriptions are useful for **training-only label audits and error analysis**,
or as targets for a separately requested source-to-explanation model, not as
gold-selected inference hints. Fixed, generic definitions of all five classes
are different: those are deployment-available task information. The corrector
already selects advisory generic descriptions using predicted labels and
train-only OOF predictions, rather than the workbook's gold explanation.

Annotation ambiguity is not solved by weighting: IDs 104/105/116/121 have
class 2 and targets saying the source is correct; IDs 141/143/147 similarly
have class 3. All seven lie in the original training split. ID 102 has class 2
but its description mentions a missing comma, illustrating that surface
punctuation alone does not determine class 4. Original labels are preserved;
correction-reference quarantine is not automatic classifier relabeling.


### Duplicate sources and leakage

There are 394 exact source strings and 394 whitespace-collapsed, casefolded
source groups. Six duplicate groups contain 13 rows (seven extra repeated rows):

- IDs 3 and 35;
- IDs 22 and 68;
- IDs 23 and 69;
- IDs 24, 64, and 70;
- IDs 25 and 91;
- IDs 66 and 71.

Every duplicate group has consistent labels and exact corrections. No conflicting
normalized-source group was found. These groups must stay in one partition;
random row splitting can leak their corrections into validation. Rows and IDs
are retained rather than deduplicated, so evaluation weights repeat rows just as
the supplied dataset does.

### Lengths and annotation caveats

Lengths use Python character counts or `str.split()` whitespace-token counts.
Canonical target statistics include inferred class-0 identity targets. P95 is
the sorted value at index `floor(0.95 * (N - 1))`, not an interpolated percentile.

| Field | Minimum | Median | Mean | P95 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source characters | 27 | 94 | 98.5511 | 166 | 240 |
| Source whitespace tokens | 7 | 21 | 22.0973 | 37 | 54 |
| Target characters | 20 | 86 | 89.6010 | 149 | 212 |
| Target whitespace tokens | 4 | 19 | 20.0349 | 34 | 48 |

These are **not model-tokenizer lengths**. Vietnamese whitespace tokens are not
necessarily linguistic words. Choose model sequence limits from the actual model
tokenizer rather than assuming a one-to-one conversion.

Eleven correction strings begin with `(`: IDs 103, 104, 105, 108, 116, 121, 141,
143, 147, 168, and 200. Several are annotation instructions or statements that
the source is already grammatical rather than standalone rewritten sentences.
For example, ID 143's correction is `(Câu đúng ngữ pháp).` despite a nonzero
class label. There are 112 non-null corrections containing the literal `...`,
often abbreviating the sentence. The baseline preserves these references because
inventing gold corrections or relabeling rows changes the task. Consequently,
reference-based scores can reward copying annotation prose or truncation and can
penalize an otherwise sensible full-sentence correction. Treat score changes as
local reference agreement, not proof of grammatical quality.

### Ordered-label pitfall and split policy

The raw array is five contiguous class blocks in the order 1, 2, 3, 4, 0. A
first-32-row “smoke sample” from the raw file has only class 1; an unshuffled
last-20% validation set has only class 0. Neither is a representative holdout.

`split_records(records, validation_fraction=0.2, seed=42)`:

1. Validates canonical gold rows and unique typed IDs.
2. Groups sources by whitespace collapse followed by casefold, without changing
   the actual texts and without Unicode/punctuation normalization.
3. Rejects any group whose exact targets or labels conflict.
4. Stratifies observed classes. Within each class, a seeded group permutation
   and deterministic subset-sum select the nearest feasible validation row count
   to `round(class_rows * validation_fraction)`, with at least one independent
   group in each partition. Ties choose the smaller row count.
5. Shuffles both output row lists using the same seeded RNG. Prefix caps should
   be applied only after this split, never before.

A class with fewer than two independent source groups raises an explicit error:
representing it in both partitions without leakage is impossible. Whole-group
constraints can make the requested validation fraction approximate. For this
file, the per-class rounded targets are 40, 20, 7, 7, and 7, all achievable:
81 validation rows and 320 training rows are the expected uncapped counts.
The generated analysis report is the authoritative record of the actual split.
For very small smoke caps, use stratified selection within the partition if all
five labels must appear; arbitrary shuffled prefixes do not guarantee that.

## Data API and artifacts

```python
from model.data import analyze, load_records, split_records, load_training_split

records = load_records("dataset/original/VGEC 2026_v1.json")
train, validation = split_records(records, validation_fraction=0.2, seed=42)
report = analyze("dataset/original/VGEC 2026_v1.json", "outputs/data-analysis")
balanced_train, original_validation, provenance = load_training_split(
    "dataset/newdataset/VGEC_balanced.json", validation_fraction=0.2, seed=42)
```

A canonical row is:

```json
{"id": 1, "source": "exact Input", "label": 1, "target": "exact correction"}
```

`load_records` accepts only a raw array with `ID`, `Input`, and labeled `Output`;
it does not accept its own canonical split exports as raw input. Labels must be
actual JSON integers 0–4, not booleans, strings, floats, or nulls when required.
IDs must be integers or strings; integer `1` and string `"1"` remain distinct.
`load_records(path, require_labels=False)` additionally permits missing/null
`Output`, null labels, and absent corrections for unlabeled inference input.
Any label present must still be valid, and every input must still be a string.
Non-null targets must be strings. Malformed rows or duplicate IDs fail.

```bash
python -m model.data \
  --data "dataset/original/VGEC 2026_v1.json" \
  --output-dir outputs/data-analysis \
  --validation-fraction 0.2
```

This writes UTF-8 `analysis.json`, canonical `train.json`, and canonical
`validation.json`. `analyze` returns the same report it writes: SHA-256, original
class counts/nulls, source groups/duplicates, length summaries, annotation flags,
label runs, split parameters, partition counts/class counts/source-group counts,
and zero shared groups. A conflicting dataset is rejected rather than exporting
a report claiming a safe split.

## Metric definitions and explicit assumptions

`Metrics_GEC.pdf` pages 1–3 specifies per-class precision/recall/F1, macro-F1,
micro-F1, confusion matrix, exact match, source-aware GLEU+, and ROUGE-L. It
provides a single reference per sentence. Our scores are fractions in [0, 1], not
percentages. This is a transparent **local implementation**, not a claim that
an unpublished official scorer uses identical preprocessing or aggregation.

### Classification

All five numeric classes are included in macro-F1, even if a tiny evaluated
subset lacks some classes. A zero denominator produces zero precision, recall,
or F1. This fixed class set is a local convention because the PDF does not
specify absent-class handling.

Invalid predicted labels (including null, out-of-range integers, booleans,
strings, or missing labels) are wrong predictions, never coerced to a valid
class. Each row remains in the denominator. Micro-F1 is the correct-decision
fraction (single-label accuracy), with every rejected decision contributing one
global false positive and one false negative. Invalid labels are excluded as
named classes from macro-F1 but contribute false negatives to their gold class.

`classification.confusion_matrix` has five gold rows and six prediction columns
`[0, 1, 2, 3, 4, "invalid"]`, making rejected outputs visible instead of dropping
them. `classification.labels`, `prediction_columns`, and `per_class` make these
axes and support counts explicit. `invalid_prediction_rate` is the fraction of
rows with an invalid label; structural prediction errors fail scoring rather
than being folded into this rate.

### Correction

The scorer independently reports correction metrics over **all** rows and over
**error_only** rows (gold class != 0). The PDF does not say whether class 0 should
participate in correction evaluation. Reporting both avoids hiding identity
performance on the 200 no-error rows. Empty subsets have count 0 and null metrics,
not fabricated zero performance.

- **Exact match:** exact Python string equality against canonical gold, with no
  trimming, case folding, Unicode normalization, or punctuation normalization.
- **Tokenization:** GLEU+ and ROUGE-L use case-sensitive `str.split()` for all
  three strings. No Vietnamese segmenter, stemming, or punctuation splitting.
  Thus repeated whitespace is ignored for token metrics but not exact match.
- **ROUGE-L:** per-sentence LCS F1 with beta=1, then arithmetic mean across rows.
  The dynamic program uses one-row storage. One empty token sequence scores 0;
  two empty sequences score 1. This empty-case convention is not specified by
  the PDF.
- **GLEU+:** uses PDF Eq. (7) for each distinct hypothesis n-gram `g`:
  `min(H[g], R[g]) - max(0, min(H[g], S[g]) - min(H[g], R[g]))`.
  Signed numerators and hypothesis n-gram denominators are summed over sentences
  for each order 1–4, **before** checking/clamping nonpositive precision. Source
  counts never cross sentence boundaries. All four orders have weight 1/4.
  The corpus brevity penalty is 1 when total hypothesis tokens exceed total
  reference tokens; otherwise it is `exp(1 - r/c)`. Empty total hypotheses score
  0. Any order with zero denominator or nonpositive numerator makes the score 0.
  There is no smoothing, effective-order reduction, per-sentence averaging, or
  multi-reference sampling.

The PDF prose describes the retention penalty in terms of source frequency,
but its explicit equation clips against both hypothesis and source. We implement
the equation literally. Corpus aggregation is a local convention consistent with
its total-length brevity penalty; the PDF does not fully specify whether
precisions are first pooled or sentence-averaged. It also leaves log(0), negative
modified precision, empty strings, and short sentences unspecified. Our explicit
zero rule resolves those cases reproducibly. A corpus consisting entirely of
sentences shorter than four hypothesis tokens scores GLEU+ 0 even for perfect
predictions, while ROUGE-L and exact match can still be 1. Short sentences within
a mixed corpus contribute available orders normally; missing orders contribute
no counts, not automatically zeroing a mixed corpus.

**Do not replace this with `nltk.gleu`:** a reference/hypothesis-only metric cannot
penalize retention of erroneous source n-grams as required by the PDF.

### Alignment, prediction schema, and identity comparator

```python
from model.metrics import identity_predictions, score

predictions = [{"id": row["id"], "label": 0, "target": row["source"]}
               for row in validation]
metrics = score(validation, predictions)
identity_metrics = score(validation, identity_predictions(validation, label=0))
```

The scorer aligns by exact typed IDs, not list order. Duplicate, missing, or
extra prediction IDs fail. A target must be a string but may be empty. Gold rows
must have valid labels and string targets. It returns:

- `count`;
- `classification.{macro_f1,micro_f1,confusion_matrix,labels,prediction_columns,per_class}`;
- `corrections.{all,error_only}.{count,exact_match,gleu_plus,rouge_l}`;
- `invalid_prediction_rate`.

`identity_predictions(records, label="majority")` is also available. Its majority
is estimated from the records supplied to that function and resolves ties to the
smallest class ID. Using validation labels to compute that majority is a
**descriptive comparator only**; for a deployable majority baseline estimate the
class on training data and pass its fixed integer ID. On the full supplied
file the majority is class 0. Independently of running the scorer, inspection
shows that an identity correction matches 200/401 full-data gold strings and
0/201 error-only gold strings. These are raw-data observations, not held-out
model results.

```bash
python -m model.metrics --data "dataset/original/VGEC 2026_v1.json" \
  --identity-label 0 --output outputs/identity-metrics.json

python -m model.metrics --data "dataset/original/VGEC 2026_v1.json" \
  --predictions outputs/predictions.json --output outputs/metrics.json
```

The metrics CLI accepts the joined comparison exports (`predicted_class` and
`predicted_correction`), historical canonical predictions (`label` and `target`),
or historical dataset-shaped wire arrays (`ID` and `Output`). Only prediction
fields are scored; `target_class` and `target_correction` in comparison files
are gold references for inspection, not hypotheses. Classifier-only exports
have no correction hypothesis and are not correction-scoring inputs.

## What the supplied materials do not establish

The PDF's “Overall Evaluation” section lists complementary classification and
correction metrics but supplies **no combined-score formula, ranking priority,
weights, official submission filename/schema, tokenization script, or submission
checker**. It mentions optional offline LLM assessment without specifying a
judge model, prompt, or aggregation protocol. This baseline does not invent a
combined score, claim its JSON prediction format is official, or fabricate LLM
judgments. Current comparison exports explicitly separate gold and predictions;
historical `ID`/`Output` exports mirrored the dataset for local interoperability.
