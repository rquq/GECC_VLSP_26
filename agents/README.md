# Vietnamese GEC: classification → correction

The user-selected classification baseline is **audited Sailor2-8B NF4 QLoRA,
hierarchical subtype coefficient two, five-epoch horizon, selected epoch 4**.
Its checkpoint is
`runs/sailor2-8b-hierarchical-qlora-audited-seed42/classifier`: Macro-F1
**0.614473**, Micro-F1 **0.790123**, on the repeatedly selected 81-row holdout.
Its actual training provenance includes R-Drop; do not relabel these weights
as a single-forward result. `configs/classification_sailor2_8b.json` now starts
fresh single-forward fits with shared weights and ordinary LoRA dropout, no KL.

The historical 480-row five-epoch Sailor checkpoint and higher-scoring eight-epoch
comparison remain preserved; neither replaces the audited baseline. Equal-weight
ensembles are not selected. The Qwen classification + Sailor correction pipeline
and OOF support remain available, but the standalone Sailor baseline has no OOF
hints and is not an end-to-end correction result.

## Project layout

| Path | Purpose |
| --- | --- |
| `configs/classification.json` | Qwen3-4B QLoRA classification and OOF settings |
| `configs/correction.json` | Sailor2 QLoRA, prompt, decoding and checkpoint selection |
| `model/classification.py` | Classifier training and leakage-safe correction hints |
| `model/encoder_classification.py` | Full-finetuned BamiBERT/XLM-R classifiers and saved serving |
| `model/ensemble.py` | Three fixed equal-probability combinations and sequential serving |
| `configs/encoder_bamibert.json` | Retained standalone encoder recipe |
| `configs/classification_sailor2_8b.json` | Retained five-epoch Sailor classification experiment |
| `model/correction.py` | Correction-only QLoRA training and saved-adapter reload |
| `model/two_phase.py` | Connected stages, preflight and artifact packaging |
| `model/pipeline_predict.py` | Saved classifier + corrector inference |
| `notebooks/kaggle_two_phase.ipynb` | Kaggle source bootstrap, live logs and downloads |
| `tools/build_dataset.py` | Private audited training-bundle preparation |
| `tools/author_synthetic.py` | Explicit locally authored clean/error contrasts, without reading gold |
| `dataset/original/` | Untouched original JSON/XLSX |
| `dataset/newdataset/` | Separate classifier/correction bundles, pool, audit and manifest |
| `runs/` | Active pipeline artifacts; never committed |
| `agents/tests/`, `agents/docs/` | Local checks and scoring notes |

Shared runtime helpers live in `model/runtime.py` and `tools/runtime.py`.
`model/modeling.py` contains only the response-loss collator. Data, metrics and
prediction export modules remain shared between the two tasks.

## Training contract

- Seed **42**; independent JSON configurations for the two stages.
- Classification: `Qwen/Qwen3-4B-Instruct-2507`, pinned to revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`, **504 training rows** (312
  unchanged original rows + 192 capped synthetic examples), five epochs, raw
  Vietnamese text, max length 512. NF4 double quantization;
  LoRA rank 16 / alpha 32 / dropout 0.05 on attention and MLP projections,
  Shared LoRA/head LR **2e-5**, microbatch 1 / effective batch 8. The existing
  linear `score` head still has width five, but its outputs are now one binary
  error logit `z0` and four conditional subtype logits `z1..z4`. With
  `q = sigmoid(z0)` and `r = softmax(z1..z4)`, the joint probabilities are
  `[1-q, q*r1, q*r2, q*r3, q*r4]` for clean, sentence, phrase, function-word,
  punctuation. Training loss and inference composition use FP32 math.
  Prediction is joint argmax, **not a hard binary gate**.
  Five-class sample weights retain baseline inverse-square-root frequencies
  (`training.class_weight_power: 0.5`) and mean sample weight one. A single
  training forward uses
  `L = sum_i w[y_i] * (BCE(z0, y_i>0) + 2 * I(y_i>0) * CE(z1..z4, y_i-1)) / N`.
  `N` includes every row in the full or partial final accumulation window;
  Trainer must not divide again. Clean rows have zero subtype gradient.
  Ordinary LoRA dropout remains 0.05; **R-Drop and its KL penalty are removed**.
  Evaluation/inference uses dropout-disabled joint argmax. No pair-consistency
  loss, oversampling, separate head LR, threshold tuning or calibration is active.
  Best checkpoint maximizes holdout Macro-F1. Counts/weights, architecture and
  single-forward `training_algorithm` provenance are saved with the adapter.
  Architecture version 2 is unchanged; existing lambda-two adapters still load.
  Earlier flat and lambda-one checkpoints remain incompatible.
- Correction: `sail/Sailor2-8B-Chat`, **409 training rows** (217 unchanged originals
  + 192 synthetic), four configured epochs, rank 16 / alpha 32 QLoRA, LR 2e-5;
  NF4 double quantization, microbatch 1 / effective batch 8.
- Both stages use the **same 81 original validation rows**.
- Five fresh train-only OOF classifiers produce correction training hints.
  Normalized duplicate sources, synthetic parents, clean/error families and
  authored template groups across both bundles stay together; folds cap their
  own mixed training partitions again.
  Outer validation never trains or selects OOF models.
- Correction receives raw sentences and advisory Vietnamese class descriptions.
  **Predicted class 0 still reaches the corrector**. Final labels come from Qwen3;
  the corrector returns only `corrected_sentence`.
- For correction, prompt tokens are masked in the loss; completion and EOS are supervised.
  Targets are never silently truncated. Invalid correction JSON remains visible;
  no source-copy fallback hides generation failures.
- Correction checkpoints maximize error-only GLEU+, then error-only exact match,
  then exact clean-source preservation. Ties keep the earlier epoch.

The Qwen3 classifier uses one CUDA GPU; BF16 where supported, FP16 otherwise
(including T4), with nonreentrant gradient checkpointing. FP16 dynamic loss scaling
starts at 1: the local NF4 diagnostic overflowed at 4096 despite finite forward loss;
nonfinite training statistics still fail the run. Resolved dtype and exact base
settings are saved for reload. `bitsandbytes==0.49.2` is required for QLoRA.
Alpha inference uses CUDA BF16/FP16 autocast at the resolved model precision,
matching Trainer, including FP32-prepared QLoRA norms/embeddings. Composition
remains FP32 and CPU inference never enables autocast. Historical saved baseline
scores used a different precision context; reevaluate the baseline under matching
autocast before comparing this experiment. This change is not a measured gain.
Sailor2 now uses NF4 double quantization on one CUDA GPU by default, BF16 where
supported and FP16 otherwise, with nonreentrant gradient checkpointing and FP16
initial loss scale 1. `model.device_map: "balanced"` enables explicit two-GPU
layer placement; neither mode uses DDP or CPU/disk offload. Preflight conservatively
budgets frozen weights plus training reserve. The actual Sailor2 configuration
required an estimated 13.24 GiB in the local preflight, which correctly rejected
the 8GB GPU. This is a budget estimate, not measured Sailor2 training memory.
Use Kaggle **T4 x2** or sufficient hardware; actual T4/Sailor2 training is unverified.

## Prompts and error descriptions

`configs/correction.json` exposes `inference.prompt` (system instruction) and
`inference.hint_descriptions` (five Vietnamese descriptions indexed by label 0–4).
Despite the section name, both are used during training and inference. The user
message contains the predicted label, its description, and the original sentence;
`model/correction.py::_prompt_ids` constructs it. The adapter saves `inference.json`
and `loading.json`, including resolved dtype and quantization, for exact reload.
Editing the training config does not change an already-saved adapter's prompt.

`configs/classification.json` has **no prompt**: a Qwen sequence-classification
model reads raw sentence tokens and trains the hierarchical score head with
weighted binary/subtype losses, not chat generation. A JSON prompt key has no effect.

Generic class definitions are legitimate task information. Correction already
receives the definition selected by a predicted label, using train-only OOF
predictions for training and the main classifier for validation/deployment.
Do not substitute the gold label/description or a gold-derived explanation for
each sentence: it is unavailable at inference and creates an oracle train/test
mismatch. Appending the gold-selected description to classifier input would leak
the answer directly. A source-only predicted explanation avoids gold leakage but
can still propagate incorrect diagnoses; its quality benefit is unmeasured.

Without official dev/test data, the existing 81-row original holdout is local dev,
not an independent test. Compare a small number of otherwise-identical correction
runs with versus without descriptions, retaining predicted labels and the same OOF
hints in both. Measure error-only GLEU+/exact match, clean-source preservation and
invalid JSON; inspect performance when the hint label is wrong. Include all five
definitions, never the gold-selected one, if experimenting with classifier prompt
conditioning; that would require an explicit input/preprocessing code change.
Repeated holdout tuning makes selected scores optimistic; group-safe/nested CV
can assess stability but cannot establish performance on an unseen official test.


## Run

Use the pinned Python stack in `requirements.txt`, retaining CUDA-enabled
PyTorch 2.7.1. The notebook refuses an incompatible torch image rather than
replacing its CUDA build.

```bash
# Active alpha classification-only experiment; no OOF or correction training.
python -m model.classification --config configs/classification.json --skip-hints

# Fresh single-forward local run; use an empty output directory.
python -m model.classification --config configs/classification_sailor2_8b.json \
  --output-dir runs/sailor2-8b-hierarchical-single-forward-seed42 --skip-hints

# Optional connected pipeline support, outside the current ablation.
python -m model.classification --config configs/classification.json

# On suitable correction hardware: compatibility and memory preflight.
python -m model.two_phase --preflight-only

# Both stages, using the two default configs.
python -m model.two_phase

# Or reuse a compatible completed classifier with its OOF hints.
python -m model.two_phase --classifier-run RUN/classification

# Correction stage alone, using a completed classifier run.
python -m model.correction --config configs/correction.json \
  --classifier-run RUN/classification
```

Configured relative paths resolve from the project root; explicit output paths
resolve from the working directory. Nonempty output directories are rejected.
CLI data overrides: `--classification-data` / `--correction-data` on the pipeline,
`--data` on individual stages. Hint artifacts must match the correction bundle's
SHA-256 and connected folds; never reuse hints merely because row counts agree.

### Kaggle

Publish the selected code branch; open `notebooks/kaggle_two_phase.ipynb`, enable
Internet and **T4 x2**, and attach a **private** dataset containing
`VGEC 2026_v1.json`, `VGEC_classifier.json` and `VGEC_balanced.json`.
Run cells in order; regenerate the bundles with `python tools/build_dataset.py`
before uploading.

Separate `CLASSIFICATION_OVERRIDES` and `CORRECTION_OVERRIDES` set seed 42 and
clear `data.path`; classification also clears `data.augmentation_path` so
private mounted files are discovered instead of relying on local paths.
The alpha notebook defaults to classification only with `--skip-hints`; it does
not train OOF classifiers or Sailor2. Pipeline reuse additionally requires a
hierarchical classifier's complete, compatible OOF artifacts.
Training output streams into the notebook and a timestamped `.log`; interruption
terminates the launched process group. Download the log and artifact ZIP and save
the notebook version to retain outputs. Actual Kaggle execution has not been verified.

## Saved inference and artifacts

```bash
python -m model.pipeline_predict --classifier RUN/classification/classifier \
  --adapter RUN/correction/adapter --data INPUT.json --output predictions.json
```

Use `[ {"ID": 1, "Input": "..."} ]` for unlabeled input; add `--evaluate` only
when gold outputs exist. Joined exports contain source, `target_class`,
`target_correction`, `predicted_class` and `predicted_correction`. Gold is joined
only for export; it is not supplied to inference. Classifier-only exports have no
correction hypothesis. This is a local analysis format, not an organizer-verified
submission schema.

Pipeline output contains `classification/classifier/`, `classification/hints.json`,
`correction/adapter/`, stage configs/manifests/metrics/predictions and
`pipeline_manifest.json`. Correction saves unadapted comparisons, epoch metrics
and the best adapter, then reloads it for final evaluation. Optional ZIP packaging
follows `correction.run.archive`.
The classifier stores `adapter_config.json`, adapter/head weights, tokenizer,
`preprocessing.json` and `classifier_config.json` with the exact base revision,
quantization/dtype, five-class mapping and `architecture` metadata. Architecture
version 2 is `soft_hierarchical_error_type`; it records score-output order,
conditional labels `[1,2,3,4]`, joint log-probability composition, supervised
objective and joint-argmax decision. `training_algorithm` records the actual
single-forward objective without changing reload architecture. Alpha rejects missing
or incompatible architecture metadata: old flat checkpoints must be evaluated
from the baseline branch, not reinterpreted here.
Inference keeps `{id, label, probabilities}` with five normalized joint scores.
Pipeline reuse requires a compatible hierarchical mixed-data classifier and its
complete OOF hints matching the current bundle hashes. Rebuild bundles and
retrain rather than reusing historical hints with the audited data.
Correction's base is reloaded with the saved NF4 settings, then prepared with the
same frozen nonquantized-weight dtypes as training before loading its adapter.


## Private data and score caveats

```bash
python tools/author_synthetic.py
python tools/build_dataset.py
python -m unittest discover -s agents/tests -v
```

The organizer JSON is byte-for-byte unchanged. Split before auditing or
augmentation: seed 42, 320 original training rows and the same 81 raw holdout
rows. Do not repair/relabel organizer gold, reconstruct ellipses, or remove
holdout annotations to improve reported scores.

`VGEC_classifier.json` excludes eight nonzero-label training rows whose
parenthesized correction explicitly declares the source correct. Other
annotation/incomplete-reference rows retain classification supervision.
`VGEC_balanced.json` excludes 103 training rows with annotation targets, ellipses
in either source or target, or punctuation-only repairs under a nonpunctuation
label. Exclusions overlap; they are quarantine, not invented replacement gold.
Retained raw originals preserve null class-0 targets; only the training loader
interprets those as exact identity corrections.

| Class | Classifier originals | Correction originals | Added synthetic to each | Classifier total | Correction total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 — clean | 160 | 160 | 40 | 200 | 200 |
| 1 — sentence | 78 | 48 | 78 | 156 | 126 |
| 2 — phrase | 23 | 2 | 23 | 46 | 25 |
| 3 — function word | 23 | 6 | 23 | 46 | 29 |
| 4 — punctuation | 28 | 1 | 28 | 56 | 29 |
| Total | 312 | 217 | 192 | 504 | 409 |

The pool contains 696 explicitly authored rows: 324 local errors, 324 exact
clean counterparts and 48 legal fronted/prepositional/subordinate sentences.
Error coverage is six clause rules and eight rules each for phrases, function
words and punctuation. Every row has its edit rationale, stable family/template
IDs and `assistant_reviewed_not_human_adjudicated` review status. No lexical
Cartesian expansion, external teacher, gold-source reading, or wholesale
preposition deletion is used. Numbers are preserved; punctuation corruptions
preserve all lexical tokens. Missing-predicate targets use the known authored
counterpart, not a uniquely recoverable reference. Clause/type boundaries,
naturalness and synthetic-domain mismatch still require linguistic review.

Policy `quality_capped_2x`: add at most 40 clean rows, bounded by retained clean
count. For error class `c`, add
`min(classifier_eligible_original_count[c], largest_retained_count - retained_count[c])`.
Correction uses eligible classification counts as its augmentation reference,
not its tiny complete-reference counts. Selection rotates across rule families;
unused pool rows are not automatically included. Both bundles reject normalized
training/holdout source-target collisions and quarantine whole leaking authored
families/templates. OOF fits additionally group clean/error counterparts and
templates, so a related clean example cannot train a model predicting its error.

`audit.json` records flags/exclusion reasons; `manifest.json` records hashes,
class counts, selected IDs/rules and privacy policy. Generated files stay under
ignored `dataset/newdataset/`; the author writes the pool atomically with mode
0600 and refuses unrelated output. The corpus agreement prohibits redistribution
of original or edited resources: keep bundles private; no public upload or
external teacher receives the corpus. No new data is published by these commands.

Verification: the builder ran on the private corpus; raw original/holdout
equality and the original SHA-256 were checked. Real Qwen/Sailor tokenizers
encoded all 504 classifier and 409 correction training rows; maximum lengths
were 74 and 275 tokens (correction checked under all five possible hints), below
512 without target truncation. Five real fold plans predicted 81/82/82/82/82
correction rows using 397/399/395/401/397 classifier-fit rows. Declaration checks
accepted the audited quarantine partition and rejected injected family leakage;
dummy probabilities exercised declarations only, not trained-model accuracy.
Notebook schema and six code cells validated. Full Sailor correction placement
preflight rejected the local GPU: 6.93 GiB free versus a conservative 13.24 GiB
requirement. The builder verification itself ran no training; the later audited
Sailor classifier rerun is recorded below. No new correction training ran.
All 95 regression tests passed after deleting an obsolete old-cap assertion;
focused builder regressions also passed (10 tests).

Holdout still includes one annotation target, 17 elliptical sources and 21
elliptical targets. Those can penalize valid full-sentence repairs; report
complete-reference diagnostics alongside raw official-style scores. Historical
holdout-selected scores below used the older 480/460-row bundles; the explicitly
marked audited rerun used 504 classifier rows. Neither is an independent-test
estimate. More synthetic rows are not evidence of better accuracy. No official combined ranking formula
has been supplied.

Full dataset/scorer evidence: [docs/DATA_METRICS.md](docs/DATA_METRICS.md).

## Cleanup and history

Removed obsolete joint-task configs, entrypoints, model-loading helpers and
notebooks from the active project. Historical runs, adapters, ZIPs, diagnostics,
configs and old reports were moved without deleting trained models to:

`../VLSP26_4-experiments/legacy-joint-task-20261008/`

The completed `runs/phobert-two-phase-normalized-seed42/` classifier and saved
OOF models are preserved as historical artifacts, not migrated weights. They are
incompatible with the Qwen3 adapter loader and rejected by pipeline reuse.
Train a fresh Qwen3 classifier and fresh OOF models; old PhoBERT hints are not
the new experiment's classification output.

Verification: **54 behavior regressions passed**. Actual Qwen3-4B NF4 CUDA
training completed one BF16 optimizer step (loss 1.251953125), saved/reloaded
the classification adapter/head, and produced valid correction hints for all
81 holdout rows. Peak allocated memory was 3.960 GiB. Same-batch predictions
were repeatable and unchanged by altered gold labels/corrections. Changing the
inference batch size from 1 to 2 produced maximum probability drift 0.007686;
quantized mixed-precision predictions are not bitwise batch-invariant.

The connected classification CLI also ran with a tiny constructed Qwen3 CPU
diagnostic: 320 original training rows, 81 holdout rows, two fresh OOF fits,
and validated hints for all 460 correction training rows. Notebook configuration
and preflight ran locally with a cached Qwen2.5-0.5B correction-base override;
historical PhoBERT reuse was rejected. These checks verify runtime and artifact
contracts, not full-training quality, Sailor2 execution or actual T4 hardware.

After lowering the FP16 initial scale, the actual Qwen3-4B CUDA diagnostic passed
two optimizer steps with the configured effective batch 8 (loss 2.90093994140625),
finite gradient norms, nonzero saved LoRA-B updates, adapter/head reload, and all
81 correction holdout hints. Peak allocated memory was 3.960 GiB. This exercised
FP16 on the local RTX 4060, not a T4; full five-epoch training remains unmeasured.

Correction QLoRA verification: **54 behavior regressions passed**. The actual
correction CLI ran on CUDA with a cached Qwen2.5-0.5B diagnostic base (not Sailor2),
FP16 NF4 double quantization, 16 training rows, five validation rows, two optimizer
steps and effective batch 8. Loss was 0.4818883836; gradient norms were finite and
saved LoRA-B weights changed. Saved quantized reload and repeated same-input
logits passed, as did configured prompt/description use and notebook JSON/code
validation. Peak allocated memory was 1.409 GiB for this smaller diagnostic only.
Fresh tiny CPU classifier/OOF fits supplied valid hints for all 460 training and
81 holdout rows; connected quantized preflight also passed with the smaller base.
Eight-token smoke decoding exercises generation, not correction quality.
Full five-epoch Sailor2 training, two-GPU quantized placement, actual T4 execution
and any quality improvement remain unverified.

Classification weighting verification: **57 behavior regressions passed**.
Actual Qwen3-4B NF4/FP16 CUDA training ran on 14 imbalanced original training
rows, microbatch 1 / effective batch 8, including a six-row final accumulation
window: two optimizer steps, finite loss 3.2321492434, nonzero saved LoRA-B
updates, and adapter/head reload with valid five-class probabilities. This
verifies weighted training and saved inference, not a Macro-F1 improvement.
Full five-epoch weighted training and actual T4 execution remain unmeasured.
An actual Trainer/SGD diagnostic also matched batch 8 against microbatch 1 /
accumulation 8 exactly, including a three-row final window (11 total rows).

Mixed classifier cutover verification: **60 behavior regressions passed**.
Actual Qwen3-4B NF4/FP16 CUDA training used 24 mixed diagnostic rows
(15 original + 9 synthetic), microbatch 1 / effective batch 8, two optimizer
steps and finite loss 2.9688080549. Saved adapter/head reload returned valid
five-class probabilities. A connected tiny-Qwen3 CPU CLI run consumed all
480 classifier rows, trained two fresh capped mixed OOF fits, and validated
hints for all 460 correction training rows and 81 holdout rows. Deliberately
inserting a predicted row into its OOF training declaration was rejected;
fit-local class-weight counts matched each fold's actual training rows.
Notebook JSON and all code cells validated. These checks establish runtime
and leakage contracts, not Macro-F1 improvement or full Qwen3/Sailor2 training.


### Historical rejected alpha ablations

Matched seed-42 five-epoch Qwen3 NF4/BF16 runs used 480 mixed training rows and
81 original holdout rows. Shared-LR baseline versus 5× head-LR saved-inference
Macro-F1 was **0.45977767 / 0.45260493**. Minimal-pair clean-counterpart mixing
scored **0.40740642**, with errors called clean rising from two to eight.
Neither candidate was promoted. Their runs/configs were deleted; the unchanged
flat baseline remains at `runs/lr-head-vs-lora-seed42/baseline/` as a useful control.

Historical saved inference lacked Trainer's BF16 autocast: baseline Trainer
selection Macro-F1 was **0.43460223**, versus pairs **0.43080141**. The hierarchy
comparison below uses the precision-matched baseline, not the historical
0.45977767 saved-inference score. These are holdout-selected estimates.

Cleansed artifacts: rejected R-Drop and localization runs/configs/tokenizer audit,
minimal-pair run/config, differential-LR candidate/config, XLM-R checkpoint/recipe,
all three failed encoder ensembles, the failed Qwen/Sailor ensemble, and the
mistaken R-Drop separate-weight run/recipe. Selected Sailor baselines, audited
rerun, eight-epoch comparison, Qwen controls, BamiBERT standalone and historical
PhoBERT/OOF artifacts remain. General encoder/ensemble utilities remain available;
no failed ensemble checkpoint is a selected serving path.

### Completed soft hierarchy ablation

Actual Qwen3-4B NF4/BF16 CUDA training completed five epochs / 300 updates on
the unchanged 480-row mixed training population. Baseline was **not retrained**:
its saved epoch-3 adapter was reevaluated with CUDA BF16 autocast and batch one,
matching checkpoint selection and the hierarchy's saved-adapter inference.
The hierarchy selected epoch 5; reloaded Macro-F1 exactly matched selection.
Comparison assertions checked identical holdout/order, ordered training IDs,
data hashes, model revision, package versions, class weights and training args
except output/log paths. Neither correction nor OOF fits ran.

| Measure | Flat baseline | Soft hierarchy |
| --- | ---: | ---: |
| Macro-F1 | 0.43460223 | 0.49679076 |
| Micro-F1 / accuracy | 0.64197531 | 0.67901235 |
| Clean F1 (40 gold rows) | 0.93670886 | 0.94736842 |
| Sentence-structure F1 (20 rows) | 0.50000000 | 0.53658537 |
| Phrase-structure F1 (7 rows) | 0.31578947 | 0.40000000 |
| Function-word F1 (7 rows) | 0.15384615 | 0.20000000 |
| Punctuation F1 (7 rows) | 0.26666667 | 0.40000000 |
| Clean sentences called errors | 3 | 4 |
| Error sentences called clean | 2 | 0 |

Macro-F1 delta is **+0.06218853** (6.22 percentage points); 16 predictions
changed, with six fixes and three regressions. Function-word recall remains
1/7; no general solution to subtype confusion is established. A descriptive
10,000-sample paired class-stratified row bootstrap gives a 95% percentile
interval of **[-0.07346, +0.19536]** for the delta, including zero. It assumes
independent rows and omits duplicate-source dependence and checkpoint/algorithm
selection. Both recipes selected on this repeatedly used 81-row holdout; the
result is promising for this split, **not a statistically established or
independent-test gain**. Keep alpha as the experiment; do not replace the flat
baseline on this evidence alone.

Runtime verification: **64 behavior regressions passed**, including probability
composition, extreme-logit finite gradients, zero subtype gradient on clean
rows, weighted full/partial accumulation equivalence, actual tiny-Qwen LoRA/head
updates and saved reload, plus rejection of historical flat checkpoint metadata.
The actual five-epoch GPU run had finite parameters/statistics and training loss
0.71507604; saved reload returned normalized five-class probabilities for all
81 holdout rows. Notebook schema/code validation also passed before training.
Actual T4 hardware, full hierarchical OOF training and connected Sailor2
correction remain unverified; the measured gain concerns classification only.

Artifacts (relative to this worktree):
- `../runs/hierarchical-seed42/`: adapter/head, predictions, metrics, epoch/trainer
  histories, run manifest and paired `comparison.json`.
- `../runs/hierarchical-seed42-config.json`: exact experiment controls.
- `../runs/hierarchical-baseline-bf16.json`: matched baseline predictions/score.
- `../runs/lr-head-vs-lora-seed42/baseline/`: unchanged trained flat baseline.

### Completed subtype loss coefficient-two ablation

The active alpha recipe doubles conditional subtype CE while leaving binary BCE,
sample weights and joint-argmax inference unchanged. Actual Qwen3-4B NF4/BF16
CUDA training completed five epochs / 300 updates; epoch 5 was selected.
Comparison assertions verified identical ordered 480 training rows, 81 holdout
rows, model/package/precision controls and training arguments except output
paths. The lambda-one adapter was reused, not retrained. No correction or OOF
fits ran.

| Measure | Lambda 1 | Lambda 2 |
| --- | ---: | ---: |
| Five-way Macro-F1 | 0.49679076 | 0.50845162 |
| Micro-F1 / accuracy | 0.67901235 | 0.69135802 |
| Gold-error subtype Macro-F1 | 0.40184435 | 0.41654457 |
| Correct subtype on 41 gold errors | 19/41 | 20/41 |
| Clean F1 | 0.94736842 | 0.94736842 |
| Sentence-structure F1 | 0.53658537 | 0.58536585 |
| Phrase-structure F1 | 0.40000000 | 0.42857143 |
| Function-word F1 | 0.20000000 | 0.20000000 |
| Punctuation F1 | 0.40000000 | 0.38095238 |
| Clean sentences called errors | 4 | 4 |
| Error sentences called clean | 0 | 0 |

Five-way Macro-F1 increased **0.01166086** (1.17 percentage points). Only three
predictions changed: ID 43 was fixed (punctuation -> sentence structure), while
IDs 162 and 120 moved between incorrect subtypes. There were no correct-to-wrong
regressions. Binary decisions were unchanged; function-word recall remains 1/7.

A descriptive 10,000-sample paired class-stratified row bootstrap gives a 95%
percentile delta interval of **[-0.00598, +0.03784]**, including zero. It assumes
independent rows and omits duplicate dependence and checkpoint/algorithm
selection. This repeatedly used holdout does not establish generalization.
Keep coefficient two as the active alpha experiment, not a proven improvement
or a reason to replace the production baseline. Prefer fresh group-safe
evaluation to another coefficient sweep on these same rows.

Verification: **64 behavior regressions passed**, including doubled subtype
gradients without doubling the direct binary-logit gradient, zero subtype
gradient on clean rows, and full/partial accumulation equivalence. Actual GPU
training and saved-adapter evaluation completed; the reloaded classifier scored
0.50845162 Macro-F1. Training loss was 1.23814050, which is not directly comparable
with lambda-one loss because its objective scale changed.

Artifacts (relative to this worktree):
- `../runs/hierarchical-lambda2-seed42/`: saved adapter/head, predictions, metrics,
  training histories, manifest and paired `comparison.json`.
- `../runs/hierarchical-lambda2-seed42-config.json`: matched training controls.
- `../runs/hierarchical-seed42/`: retained lambda-one control artifacts.

### Single-forward cutover verification

The affected regression suite passed **94 tests** after removing R-Drop.
A real tiny-Qwen CPU fit exercised both shared and separate weighting: five
microbatches, exactly five training forwards and one optimizer update per fit,
including a partial effective-batch-eight window. Finite loss/parameters,
saved-adapter reload and source-only normalized joint-argmax inference passed.
This is runtime proof, not a full Sailor fit or a no-R-Drop quality measurement.

### Rejected matched R-Drop ablation

The historical beta-one experiment and coefficient-two control both scored
Macro-F1 **0.50845162**, accuracy **0.69135802**, with identical labels on all
81 rows. Probabilities changed by up to 0.06777167, but there were no fixes or
regressions. R-Drop required two training forwards without observed benefit.
Its run and standalone config were deleted; active training uses one forward
and no KL penalty. Retained successful Sailor checkpoints still record their
actual historical R-Drop provenance; they are not single-forward fit results.

### Rejected alpha/beta comparison

On the same ordered 480-row train / 81-row holdout, coefficient-two hierarchy,
R-Drop and localization scored Macro-F1 **0.50845162 / 0.50845162 / 0.48774208**.
Localization introduced two regressions with no fixes. Neither candidate
demonstrated improvement. Both failed outputs and their experiment configs
were deleted. These historical scores remain negative evidence, not available
serving checkpoints or independent-test results.

## Standalone encoder comparison

BamiBERT and XLM-R were full-finetuned on the Qwen coefficient-two control's
ordered 480 training rows and 81 held-out originals. Both used seed 42, five
epochs, LR 2e-5, microbatch 1 / effective batch 8, raw text up to 512 tokens,
inverse-square-root class weights and flat weighted five-class CE. BamiBERT
selected epoch 2; XLM-R selected epoch 4. Saved reload preserved their decisions.

| Model or equal-weight ensemble | Macro-F1 | Micro-F1 |
| --- | ---: | ---: |
| Qwen hierarchy control | 0.508452 | 0.691358 |
| BamiBERT standalone | 0.509337 | 0.666667 |
| XLM-R standalone | 0.398770 | 0.666667 |
| Qwen + BamiBERT | 0.502429 | 0.691358 |
| Qwen + XLM-R | 0.458957 | 0.703704 |
| All three | 0.446174 | 0.703704 |

No ensemble improved Macro-F1. XLM-R and all three failed ensemble outputs
were deleted, along with the XLM-R recipe. BamiBERT's standalone checkpoint and
recipe remain as a useful small-model comparator, not a replacement for Sailor.
Its advantage over Qwen was only 0.000885 with lower accuracy. These scores
use the repeatedly selected holdout, not an independent test.

Fresh standalone BamiBERT fit (use an empty output directory):

```bash
python -m model.encoder_classification --config configs/encoder_bamibert.json \
  --control-run runs/hierarchical-lambda2-seed42 \
  --output-dir runs/bamibert-repeat-seed42
```

The retained checkpoint is `runs/heterogeneous-ensemble-seed42/bamibert/`.
Encoder loading uses the published `tokenizer.json` through
`PreTrainedTokenizerFast`, without external Vietnamese word segmentation.
BamiBERT metadata names an XLM-R tokenizer despite its byte-level BPE backend;
direct backend loading preserves segmentation, special tokens and punctuation.
[BamiBERT](https://huggingface.co/Qualcomm-AI-Research/BamiBERT) specifies
research/educational use and Qualcomm responsible-AI terms; competition
eligibility is not verified. General ensemble utilities remain available for
explicit future experiments, not as a selected or precomputed serving path.

## Larger Vietnamese QLoRA classifier: Sailor2-8B

Completed on alpha using
[sail/Sailor2-8B-Chat](https://huggingface.co/sail/Sailor2-8B-Chat), a
Vietnamese-capable Southeast Asian checkpoint with Apache-2.0 licensing,
pinned at `577d5b2745863cec11e1b2f1c92586a91cea446e`. This is a standalone
sentence classifier, not a corrected-sentence generator or connected two-stage
run. Default classification/correction configs remain unchanged.

From alpha, using the existing pinned training environment:

```bash
python -m model.classification --config configs/classification_sailor2_8b.json \
  --output-dir ../runs/sailor2-8b-hierarchical-qlora-seed42 --skip-hints
```

Use an empty output directory when rerunning. Recipe: NF4/double quantization,
BF16 compute, rank 16 / alpha 32, attention and MLP adapters plus the trained
five-output hierarchical head, R-Drop beta 1, LR 5e-5, cosine decay with 30
warmup updates, paged 8-bit AdamW, gradient checkpointing, microbatch 1 and
effective batch 8. The ordered 480 training rows and 81 held-out originals
match both saved Qwen controls exactly. Maximum source length was 74 tokens;
no training sources exceeded the 512-token limit.

Five epochs / 300 optimizer updates completed on the local RTX 4060 Laptop
8 GB GPU. Trainer runtime was 3411.14 s (56.85 minutes), excluding initial
download/model loading and final saved-checkpoint reload. There were
46,155,264 trainable parameters. Epoch 4 was selected; epoch 5 tied and
retained the earlier checkpoint. Trainable parameters and training statistics
were finite. Saved-adapter evaluation covered all 81 held-out rows.

| Classifier | Macro-F1 | Micro-F1 | Fixes / regressions vs Qwen |
| --- | ---: | ---: | ---: |
| Qwen3-4B hierarchy control | 0.508452 | 0.691358 | — |
| Qwen3-4B hierarchy + R-Drop | 0.508452 | 0.691358 | — |
| Sailor2-8B hierarchy + R-Drop | 0.607858 | 0.790123 | 11 / 3 |

Sailor changed 19 decisions and gained eight correct rows overall. Macro-F1
increased by 0.099406; class-3 F1 increased from 0.200000 to 0.571429, but
class-2 F1 fell from 0.428571 to 0.181818 (only one of seven class-2 rows
correct). Keep Sailor as the stronger candidate on this holdout, not a proven
generalization winner. The holdout has been repeatedly used for selection;
model, LR, scheduler and optimizer changed together, so this does not isolate
the effect of model size or Vietnamese pretraining. No ensemble was added.

Serving reload matched saved probabilities within atol 1e-6 / rtol 1e-5 and
labels exactly on one held-out row per class. A new source-only sentence had
a finite normalized probability vector, but the clean sentence “Tôi đang đọc
một cuốn sách rất thú vị.” was incorrectly classified as class 2 with about
0.990758 probability. That is an observed quality failure, not a positive
correctness example.

```python
from model.classification import classify
predictions = classify(
    "../runs/sailor2-8b-hierarchical-qlora-seed42/classifier",
    [{"id": "new-1", "source": "Tôi đang đọc sách."}],
    device="cuda", batch_size=1,
)
```

Artifacts: `runs/sailor2-8b-hierarchical-qlora-seed42/` contains the adapter/head,
resolved config, exact splits, metrics, predictions, epoch/trainer histories,
manifest, matched `comparison.json` (paired changed IDs and per-class deltas)
and `serving_smoke.json`. OOF hints were disabled; this run cannot be reused as
a connected correction pipeline without producing its required hints. Only
the new alpha recipe and this documentation were changed; beta and
`rquq/test` were not changed by this experiment.

### Audited private-bundle five-epoch rerun

Completed a fresh fit on 312 retained originals + 192 locally authored synthetic
rows (504 total; class counts 200/156/46/46/56), excluding eight contradictory
original annotations. The resolved config and ordered 81-row raw holdout matched
the historical five-epoch Sailor run exactly. Original JSON SHA-256 was unchanged.
No historical checkpoint was overwritten; no correction or OOF fit ran.

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python -m model.classification \
  --config configs/classification_sailor2_8b.json \
  --data 'dataset/original/VGEC 2026_v1.json' \
  --output-dir runs/sailor2-8b-hierarchical-qlora-audited-seed42 --skip-hints
```

Five epochs / 315 optimizer updates completed on the local RTX 4060 Laptop 8 GB
GPU. Trainer runtime was 3613.241 s (60.22 minutes), excluding initial model
loading and final saved-adapter reload. Epoch 4 was selected; epoch 5 declined
to Macro-F1 0.547676. All 46,155,264 trainable parameters and training statistics
were finite. Maximum training source length was 74 tokens, below the 512 limit.

| Measure | Historical 480-row fit | Audited 504-row fit |
| --- | ---: | ---: |
| Macro-F1 | 0.607858 | 0.614473 |
| Micro-F1 | 0.790123 | 0.790123 |
| Correct holdout rows | 64/81 | 64/81 |
| Class 0 F1 | 0.987342 | 0.963855 |
| Class 1 F1 | 0.727273 | 0.750000 |
| Class 2 F1 | 0.181818 | 0.363636 |
| Class 3 F1 | 0.571429 | 0.461538 |
| Class 4 F1 | 0.571429 | 0.533333 |

Eight decisions changed: three fixes (IDs 117, 287, 81), three regressions
(162, 61, 21), and two wrong-to-wrong changes (142, 17). Clean recall improved
from 39/40 to 40/40, but three errors were classified as clean versus zero
historically. Phrase recall improved from 1/7 to 2/7; function-word recall fell
from 4/7 to 3/7. The small Macro-F1 gain is mixed evidence, not grounds for
replacing the retained user-selected baseline automatically.

Saved-adapter inference rescored all 81 rows. A separate CUDA serving reload
on one held-out row per gold class matched all saved probabilities exactly
(maximum absolute difference 0; tolerance atol 1e-6 / rtol 1e-5) and labels.
The previously misclassified source-only clean sentence “Tôi đang đọc một cuốn
sách rất thú vị.” now predicted class 0 with probability 0.999632. This single
probe does not establish generalization.

Artifacts: `runs/sailor2-8b-hierarchical-qlora-audited-seed42/` contains the
adapter/head, exact splits, resolved config, epoch/trainer histories, manifest,
predictions, metrics, `comparison.json` and `serving_smoke.json`. The repeatedly
selected holdout is not independent test evidence. Filtering, synthetic coverage,
class weights and the cosine update horizon changed together (300 to 315
updates), so the result does not isolate augmentation quality or quarantine.

### Separate detection/subtype weighting ablation

**Rejected experiment:** this run mistakenly retained R-Drop after the user
had rejected it. These numbers describe separate weights **with R-Drop**, not
the intended no-R-Drop recipe. Its artifacts and standalone config were deleted
at the user's request. No full no-R-Drop separate-weight fit has been measured.

The historical candidate used the deleted separate-weight Sailor recipe. The
audited 504-row training population, ordered 81-row holdout, seed 42, Sailor model/revision,
QLoRA settings, LR 5e-5, five-epoch schedule, subtype coefficient two and
R-Drop beta one remained unchanged. The resolved recipe differed only in
`training.hierarchical_weighting: "separate"` and run name. The audited control
was neither retrained nor reloaded; its saved predictions were rescored, and
all baseline artifact content hashes remained unchanged.

The historical supervised term was
`sum_i (d[y_i>0] * BCE_i + 2 * s[y_i] * I(y_i>0) * CE_i) / N`.
Detection/subtype weights were normalized over all rows/error rows respectively;
the R-Drop KL term still used the original five-class weights. Separate weighting
remains supported in the single-forward trainer, but has no measured full fit.

Five epochs / 315 updates completed with finite parameters and loss; Trainer
runtime was 4683.178 s (78.05 minutes), excluding initial loading and final
saved-adapter evaluation. Epoch 4 was selected by holdout Macro-F1.

| Measure | Audited shared-weight control | Separate-weight candidate |
| --- | ---: | ---: |
| Macro-F1 | 0.614473 | 0.596330 |
| Micro-F1 | 0.790123 | 0.802469 |
| Correct holdout rows | 64/81 | 65/81 |
| Class 0 F1 | 0.963855 | 1.000000 |
| Class 1 F1 | 0.750000 | 0.829268 |
| Class 2 F1 | 0.363636 | 0.285714 |
| Class 3 F1 | 0.461538 | 0.333333 |
| Class 4 F1 | 0.533333 | 0.533333 |
| Clean false alarms | 0/40 | 0/40 |
| Errors classified as clean | 3/41 | 0/41 |
| Detected errors with wrong subtype | 14 | 16 |
| Conditional subtype accuracy on gold-error rows | 26/41 | 25/41 |

Six decisions changed: fixes on IDs 61 and 21; regression on ID 136;
wrong-to-wrong changes on IDs 142, 181 and 146. Detection improved, but
minority subtype performance worsened: Macro-F1 fell by 0.018143 despite one
net correct row. Retain the baseline; this is not a successful Macro-F1
replacement and does not establish generalization on an independent test.

Before deletion, source-only CUDA reload matched all saved labels and
probabilities on five held-out inputs exactly. Its former output was
`runs/sailor2-8b-hierarchical-separate-weights-audited-seed42/`.
The four deferred options remain subject to user selection; none was launched.

### Eight-epoch fit and equal-weight Qwen ensemble

Completed a fresh fit with `configs/classification_sailor2_8b_8epochs.json`.
Only the epoch horizon (5 to 8) and run name changed from the Sailor recipe
above; ordered canonical training and validation rows remained identical.
This was not a continuation of the five-epoch checkpoint: cosine decay was
recomputed for 480 optimizer updates, changing the learning-rate trajectory
before epoch 4 as well as extending training.

```bash
python -m model.classification --config configs/classification_sailor2_8b_8epochs.json \
  --output-dir ../runs/sailor2-8b-hierarchical-qlora-8epochs-seed42 --skip-hints
```

Eight epochs / 480 updates completed on the same local 8 GB GPU. Trainer
runtime was 5463.03 s (91.05 minutes), excluding model loading and final
reload. Training loss was 0.720204; trainable parameters and training
statistics were finite. Epoch 4 was selected; epochs 5–8 tied its holdout
Macro-F1, retaining the earlier checkpoint. Extra late epochs did not improve
that selection score.

| Classifier | Macro-F1 | Micro-F1 | Correct / 81 |
| --- | ---: | ---: | ---: |
| Original Qwen3-4B hierarchy control | 0.508452 | 0.691358 | 56 |
| Sailor2-8B, five-epoch horizon, selected epoch 4 | 0.607858 | 0.790123 | 64 |
| Sailor2-8B, eight-epoch horizon, selected epoch 4 | 0.630288 | 0.802469 | 65 |
| Equal-probability Qwen + eight-epoch Sailor | 0.577120 | 0.765432 | 62 |

The eight-epoch recipe fixed ID 181 versus the five-epoch recipe, with no
regressions and one changed decision. Versus Qwen it fixed 12 rows and
regressed three. Class-2 F1 remained 0.181818 (one of seven correct).

The ensemble averaged the two original five-class probability vectors with
weight 0.5 each; no weight tuning or nested ensemble was used. It fixed no
Sailor errors and introduced three regressions (IDs 140, 96 and 20), reducing
Macro-F1 by 0.053168 versus Sailor. The five-epoch Sailor run is the retained
best experiment by user selection; the eight-epoch result remains comparison
evidence, and this equal-weight ensemble is not the default serving choice.

Saved ensemble GPU serving loaded one member at a time and matched offline
probabilities within atol 1e-6 / rtol 1e-5 and labels exactly on held-out IDs
377, 60, 130, 140 and 197. The source-only clean sentence “Tôi đang đọc một
cuốn sách rất thú vị.” produced a finite normalized vector but was wrongly
classified as class 4 (probability 0.484187). Serving consistency is not proof
of classification quality.

The failed Qwen/Sailor ensemble output was deleted. The higher-scoring standalone
eight-epoch checkpoint remains at
`runs/sailor2-8b-hierarchical-qlora-8epochs-seed42/`, with its exact splits,
metrics, histories and comparisons. It is retained comparison evidence, not the
user-selected five-epoch serving baseline.

All results use the repeatedly selected 81-row holdout; no independent test
was evaluated. The small gain does not establish generalization or isolate
the benefit of extra epochs from the changed cosine schedule. Classification
only: correction generation and OOF hints were not evaluated. Only the new
alpha recipe and this documentation were changed for this extension; default
configs, beta and `rquq/test` were left unchanged.

### Retained experiment integration into `rquq/test`

The alpha recipes, encoder/ensemble implementations, regression checks and
experiment evidence are integrated into `rquq/test`. Existing test-branch
alpha/beta comparison notes and its Kaggle source branch are preserved; beta
localization is not merged. Private data links and trained artifacts are not
committed. Sailor recipes discover private data in either worktree or Kaggle;
use explicit `--data` if discovery is ambiguous.

From the integrated `rquq/test` checkout, rerun into a new empty directory:

```bash
python -m model.classification --config configs/classification_sailor2_8b.json \
  --output-dir runs/sailor2-8b-hierarchical-qlora-repeat --skip-hints
```

Serve the retained checkpoint without retraining:

```python
from model.classification import classify
predictions = classify(
    "runs/sailor2-8b-hierarchical-qlora-audited-seed42/classifier",
    [{"id": "new-1", "source": "Tôi đang đọc sách."}],
    device="cuda", batch_size=1,
)
```

Current audited selection is classification-only: Macro-F1 0.614473, Micro-F1
0.790123, 64/81 correct. Do not substitute the historical 480-row or eight-epoch
adapter when reporting this baseline. Connecting it to correction requires
fresh train-only OOF hints and end-to-end correction evaluation first.

Historical integration verification: all 80 regression tests passed. Data discovery
from `rquq/test` reproduced the historical run's ordered 480 training and 81
validation rows. Actual CUDA serving of its five-epoch recipe's selected
adapter matched saved probabilities within atol 1e-6 / rtol 1e-5 and labels
exactly on IDs 377, 60, 130, 140 and 197. No retraining or correction run was
performed for this merge.

## Paper evidence and improvement priorities

The exact alpha architecture—binary clean/error probability multiplied by a
conditional four-way sentence error distribution—is a **local hypothesis**.
No experimentally matched paper was found. It uses existing sentence labels,
not new span annotations. Joint argmax avoids an irreversible hard binary gate,
but the error probability still constrains all subtype probabilities. Historical
lambda-one joint NLL was a reparameterized five-class likelihood; the active
lambda-two objective explicitly increases subtype supervision. Neither establishes
that an additional independent detection objective improves classification.

Published results below are on their authors' datasets and metrics. They are
not predictions of Vietnamese VGEC Macro-F1, and correction F0.5/BLEU cannot be
compared directly with our sentence-classification score.

### Most relevant primary studies

- **[Rei & Yannakoudakis (2017), Auxiliary Objectives for Neural Error Detection
  Models](https://aclanthology.org/W17-5004.pdf)**: joint token correctness and
  fine error labels improve English FCE F0.5 from 43.4 to 45.5; adding POS and
  grammatical relations reaches 47.7 (Table 2). Alternating separate auxiliary
  datasets instead worsens results to 30.3–31.9 (Table 5). Supports shared
  coarse/fine supervision on the same examples, **not** our conditional
  sentence-level hierarchy. Vietnamese parser quality and annotation coverage
  must be checked before adding auxiliary labels.
- **[Li & Wang (2024), Detection-Correction Structure via General Language
  Model for Grammatical Error Correction](https://arxiv.org/html/2405.17804v1)**:
  KEEP/ERROR/INSERT detection localizes spans for controlled infilling. Removing
  detection control reduces English CoNLL14 F0.5 from 68.00 to 66.70 and BEA19
  from 74.43 to 72.03 (Table 5). Gains are not universal: adding a detector to
  GPT-4 worsens Chinese MuCGEC from 34.22 to 32.31 (Table 1). Supports localization
  and conservative correction, not sentence-type hierarchy or guaranteed gains.
- **[Kiyono et al. (2019), An Empirical Study of Incorporating Pseudo Data into
  Grammatical Error Correction](https://arxiv.org/html/1909.00520v1)**: English
  BEA-valid F0.5 is 38.8 with genuine-only data; noisy backtranslation pretraining
  at 1.4M/14M pairs gives 41.1/44.5, while joint mixing gives 40.4/40.3
  (Table 4; results averaged over five seeds). Supports synthetic pretraining
  followed by genuine-data fine-tuning rather than assuming more mixing helps.
  Does not validate minimal-pair sentence classification augmentation.
- **[Flachs, Stahlberg & Kumar (2021), Data Strategies for Low-Resource
  Grammatical Error Correction](https://aclanthology.org/2021.bea-1.12.pdf)**:
  language-aware corruption and artificial→Wikipedia→Lang8→gold curricula improve
  Czech/German/Russian/Spanish correction (Table 2). Gold training sets contain
  4,980–42,210 examples and artificial pretraining uses 10M pairs; this is not
  evidence that 401 rows suffice. Morphology noise can hurt German, and English
  inflection edits should not be copied indiscriminately into Vietnamese.
- **[Bout et al. (2023), Efficient Grammatical Error Correction Via Multi-Task
  Training and Optimized Training Schedule](https://arxiv.org/html/2311.11813)**:
  with optimized three-stage training, English CoNLL14/BEA-test F0.5 is
  69.00/74.74; multitask optimized training gives 68.91/75.28 (Table 4).
  Multitasking is not uniformly better at matched schedules. Correct+Apply
  outperforms Edit+Apply on BEA-dev, 62.67 versus 43.22 (Table 7). Supports
  selected alignment-derived auxiliary tasks, not generic generated explanations.
- **[Omelianchuk et al. (2020), GECToR—Grammatical Error Correction: Tag,
  Not Rewrite](https://aclanthology.org/2020.bea-1.16.pdf)**: token detection and
  constrained edit tags achieve English single-XLNet CoNLL14/BEA19 F0.5 of
  65.3/72.4 (Table 7). Staged training and inference calibration contribute
  materially (Table 5). A Vietnamese adaptation needs enough aligned pairs,
  suitable segmentation and tags for nonlocal word order; English lexical and
  inflection tags are not directly transferable.
- **[Qorib, Aji & Ng (2024), Efficient and Interpretable Grammatical Error
  Correction with Mixture of Experts](https://aclanthology.org/2024.findings-emnlp.997.pdf)**:
  error-specialized experts improve T5-base CoNLL14/BEA19 F0.5 from 65.88/70.39
  to 66.31/71.71 using 2,372,119 cLang8 training sentences (Table 3). Removing
  explicit error-type loss gives 66.34/71.80 (Table 5), so type supervision itself
  does not establish the gain. Too much complexity for the current gold data.

### Vietnamese evidence and limits

**[Pham, Nguyen & Nguyen, Grammatical error correction for Vietnamese using
Machine Translation](https://repository.vnu.edu.vn/server/api/core/bitstreams/156e283e-b3ef-4405-836b-6eefa73d9ae7/content)**
is directly about Vietnamese sentence structure and punctuation, rather than
only spelling. The accessible PDF does not reliably establish publication
year/venue. Its synthetic news-derived corpus has 271,822 training, 29,895
validation and 15,879 test pairs (Table 1). Reported SMT/NMT BLEU is 83.73/87.51,
or 84.32/88.71 with spelling preprocessing (Figure 2). Synthetic test data and
BLEU do not establish natural VGEC classification or correction performance.
Vietnamese spelling-only studies must not be treated as evidence for our four
grammar classes.

### Decisions for this repository

1. The matched hierarchy and subtype-coefficient-two ablations are complete
   above. Keep coefficient two in alpha for fresh group-safe evaluation; the
   small gain does not justify replacing the baseline. Do not tune thresholds
   or sweep more coefficients on the repeatedly used holdout.
2. **Rejected classification candidate:** auxiliary edit localization/type was
   tested above and regressed on this split. It is not active; revisiting it
   requires a different, explicitly selected experiment on audited complete
   source-target pairs. Classification remains source-only.
3. Next **data/schedule** candidate: realistic syntax, phrase, function-word and
   punctuation corruptions with verified labels, then an original-gold-only
   finishing stage. Keep synthetic parents/duplicate sources inside training
   groups. The failed local minimal-pair recipe is negative evidence against
   simply repeating clean-counterpart mixing.
4. For **correction**, investigate detection-guided minimal edits after auditing
   complete targets; evaluate error-only correction and clean preservation
   separately. Detection mistakes can suppress needed edits.
5. Defer a Vietnamese edit tagger and error-specialized MoE until sufficiently
   large, verified aligned data exists. Neither is a paper-proven shortcut for
   the current 401 gold sentences.

### Deferred classifier algorithms

The separate detection/subtype weighting experiment is complete above; the user
has not yet selected among these four algorithms. None ran in that experiment:

1. [LDAM](https://arxiv.org/abs/1906.07413) on conditional subtype logits:
   training-only class margins; preserve binary detection and joint inference.
2. [Supervised contrastive learning](https://arxiv.org/abs/2004.11362) on gold-error
   sentence representations: cross-family positives, actual contrastive batches,
   and a training-only projection head.
3. [Adversarial consistency](https://aclanthology.org/2020.acl-main.197/) under
   constrained embedding perturbations: separate binary/subtype consistency.
   This term alone is not the complete SMART algorithm.
4. [Focal loss](https://arxiv.org/abs/1708.02002) on conditional subtype probabilities:
   a replacement for subtype CE, not initially combined with LDAM.

All four are untested proposals for this Sailor hierarchy. Compare them one at a
time against the retained shared-weight baseline; do not silently stack or launch them.

### Next controlled experiment

First fit a fresh **single-forward shared-weight Sailor lambda-two control** on
the exact audited 504 training rows and ordered 81-row holdout. Retain the audited
R-Drop checkpoint unchanged as historical evidence. This prerequisite measures
the requested R-Drop removal; it must not be combined with a new objective.

Next recommended ablation: conditional subtype focal loss with fixed gamma two,
replacing only subtype CE and retaining coefficient two, shared sample weights,
binary BCE, model/revision, data/order, seed 42, LoRA, LR/schedule, five epochs,
checkpoint selection and joint-argmax serving. Do not add separate weights or
another class-alpha factor. This is an untested hypothesis targeting subtype
confusion, not a promised improvement; ambiguous labels can receive excessive
weight. It works with microbatch one without contrastive-batch machinery.

Compare saved-reload five-way Macro-F1, each subtype F1, gold-error subtype
accuracy, clean false alarms, missed errors and paired fixes/regressions against
the new no-R-Drop control. No gamma sweep on this holdout. Require a genuinely
untouched, independently annotated evaluation set before claiming generalization.
Neither the full control rerun nor focal-loss implementation/training was launched
by this recommendation.


The auxiliary-supervision, data/schedule and correction priorities remain
proposals, not measured improvements. This holdout has already selected
checkpoints and experiments; future quality claims need fresh group-safe
evaluation or repeated splits, not continued optimization of these same 81 rows.

