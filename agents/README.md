# Vietnamese GEC: classification → correction

The active alpha experiment is **soft hierarchical Qwen3-4B NF4 QLoRA
classification with subtype loss coefficient 2 and R-Drop beta 1**. Sailor2
correction and train-only OOF hint support remain available, but are not run in this ablation.

## Project layout

| Path | Purpose |
| --- | --- |
| `configs/classification.json` | Qwen3-4B QLoRA classification and OOF settings |
| `configs/correction.json` | Sailor2 QLoRA, prompt, decoding and checkpoint selection |
| `model/classification.py` | Classifier training and leakage-safe correction hints |
| `model/correction.py` | Correction-only QLoRA training and saved-adapter reload |
| `model/two_phase.py` | Connected stages, preflight and artifact packaging |
| `model/pipeline_predict.py` | Saved classifier + corrector inference |
| `notebooks/kaggle_two_phase.ipynb` | Kaggle source bootstrap, live logs and downloads |
| `tools/build_dataset.py` | Private audited training-bundle preparation |
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
  `cdbee75f17c01a7cc42f958dc650907174af0554`, **480 training rows** (all 320
  original labels + 160 capped synthetic examples), five epochs, raw Vietnamese
  text, max length 512. NF4 double quantization;
  LoRA rank 16 / alpha 32 / dropout 0.05 on attention and MLP projections,
  Shared LoRA/head LR **2e-5**, microbatch 1 / effective batch 8. The existing
  linear `score` head still has width five, but its outputs are now one binary
  error logit `z0` and four conditional subtype logits `z1..z4`. With
  `q = sigmoid(z0)` and `r = softmax(z1..z4)`, the joint probabilities are
  `[1-q, q*r1, q*r2, q*r3, q*r4]` for clean, sentence, phrase, function-word,
  punctuation. Training loss and inference composition use FP32 math.
  Prediction is joint argmax, **not a hard binary gate**.
  Five-class sample weights retain baseline inverse-square-root frequencies
  (`training.class_weight_power: 0.5`) and mean sample weight one. For each of
  two independent dropout forwards, supervised loss is
  `L_j = sum_i w[y_i] * (BCE(z0, y_i>0) + 2 * I(y_i>0) * CE(z1..z4, y_i-1)) / N`.
  The complete training objective is `(L_1 + L_2)/2 + beta * sum_i w[y_i] *
  (KL(p_1||p_2) + KL(p_2||p_1))/ (2*N)`, with **beta fixed to 1** and `p_j`
  the composed five-class joint distribution. `N` includes every row in the
  full or partial final accumulation window; Trainer must not divide again.
  Symmetric KL is computed stably in FP32, with both distributions attached
  to autograd. RNG is not reset between passes; LoRA dropout remains 0.05.
  Clean rows have zero supervised subtype gradient, but their joint subtype
  disagreement is regularized by KL. Evaluation/inference uses one forward
  with dropout disabled and joint argmax. No pair augmentation, oversampling,
  separate head LR, threshold
  tuning or calibration is active. Synthetic class caps and the 480-row baseline
  training population are unchanged. Best checkpoint maximizes holdout Macro-F1.
  Counts/weights, architecture metadata and separate `training_algorithm`
  provenance (coefficient, KL convention, dropout and forward counts) are saved
  with the adapter and manifest. R-Drop does not redefine the width-five
  architecture version 2: existing lambda-two hierarchical adapters still load.
  Earlier flat and lambda-one checkpoints remain incompatible; their artifacts
  and measured predictions remain unchanged. No R-Drop result is claimed here.
- Correction: `sail/Sailor2-8B-Chat`, **460 training rows** (310 retained originals
  + 150 synthetic), five epochs, rank 16 / alpha 32 QLoRA, LR 2e-5;
  NF4 double quantization, microbatch 1 / effective batch 8.
- Both stages use the **same 81 original validation rows**.
- Five fresh train-only OOF classifiers produce correction training hints.
  Normalized duplicate sources and synthetic parent components across both
  bundles stay together; each fold trains on its own capped mixed partition.
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

# Matched local run (from the alpha worktree; resolved data paths in this config).
python -m model.classification --config ../runs/hierarchical-rdrop-seed42-config.json \
  --output-dir ../runs/hierarchical-rdrop-seed42 --skip-hints

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
objective and joint-argmax decision. `training_algorithm` separately records
R-Drop provenance without changing reload architecture. Alpha rejects missing
or incompatible architecture metadata: old flat checkpoints must be evaluated
from the baseline branch, not reinterpreted here.
Inference keeps `{id, label, probabilities}` with five normalized joint scores.
Pipeline reuse requires a compatible hierarchical mixed-data classifier and its
complete OOF hints. Existing correction bundles and holdout are unchanged.
Correction's base is reloaded with the saved NF4 settings, then prepared with the
same frozen nonquantized-weight dtypes as training before loading its adapter.


## Private data and score caveats

```bash
python tools/build_dataset.py
python -m unittest discover -s agents/tests -v
```

The builder writes two bundles. `VGEC_classifier.json` retains every original
training label, including rows whose correction references are explanatory,
and adds capped synthetic examples. `VGEC_balanced.json` preserves original
gold, excludes explanatory/invalid training references, retains abbreviated
corrections, and caps synthetic augmentation separately. Both keep validation
unchanged and reject synthetic source/target or parent leakage into holdout.
Abbreviated references can penalize valid full-sentence corrections and teach
shortened outputs; ROUGE-L's LCS does not
remove that mismatch. Inspect annotation diagnostics and standalone-reference
scores. Holdout-selected scores are not independent test results. No official
combined ranking formula has been supplied.

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


### Historical alpha ablations (retained artifacts, inactive code)

Matched seed-42 five-epoch Qwen3 NF4/BF16 runs used 480 mixed training rows and
81 original holdout rows. Shared-LR baseline versus 5× head-LR saved-inference
Macro-F1 was **0.45977767 / 0.45260493**; both selected epoch 3 and used 300
updates. No stable differential-LR gain was established. Artifacts remain at
`../runs/lr-head-vs-lora-seed42/`.

Minimal-pair augmentation added 215 clean counterparts (695 total rows), used
435 updates and selected epoch 4. Saved-inference Macro-F1 was **0.40740642**
versus baseline **0.45977767**. Clean false alarms fell 3 → 0 but errors classified
clean rose 2 → 8; the recipe was not promoted. Artifacts remain at
`../runs/minimal-pairs-seed42/`. Both abandoned experiments' active code and
alpha-only regression tests have been removed; trained weights/data are retained.

Historical saved inference lacked Trainer's BF16 autocast: baseline Trainer
selection Macro-F1 was **0.43460223**, versus pairs **0.43080141**. Scores are
precision-sensitive and holdout-selected, not independent-test estimates. Alpha's
hierarchy now matches forward precision between fit and reload. The matched
hierarchy experiment below uses the reevaluated baseline, not the historical
0.45977767 saved-inference score.

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

### R-Drop implementation verification

The fixed beta-one R-Drop recipe passed **69 behavior regressions**, including
independent dropout gradients, stable extreme-logit KL and real Trainer updates
for full and partial accumulation windows. A separate CPU tiny-Qwen smoke ran
the actual `_fit` path, updated the score head by a maximum absolute **0.05**,
updated LoRA weights, saved the adapter and reloaded normalized joint-argmax
predictions. Notebook schema and code-cell compilation passed. The matched
five-epoch Qwen3-4B GPU experiment is separate; no quality gain is established
by this implementation smoke.

### Completed matched R-Drop ablation

The beta-one R-Drop experiment completed five epochs / **300 optimizer updates**
on Qwen3-4B NF4/BF16; epoch 5 was selected. Comparison assertions verified the
same ordered 480 training rows, 81 holdout rows, model revision, LoRA settings,
class weights, package versions, preprocessing and training arguments except
artifact paths. The coefficient-two hierarchy control was reused, not retrained.
Saved-adapter inference retained one forward, BF16 autocast and joint argmax.

| Measure | Hierarchy coefficient 2 | R-Drop beta 1 |
| --- | ---: | ---: |
| Five-way Macro-F1 | 0.50845162 | 0.50845162 |
| Micro-F1 / accuracy | 0.69135802 | 0.69135802 |
| Gold-error subtype Macro-F1 | 0.41654457 | 0.41654457 |
| Correct subtype on 41 errors | 20 | 20 |
| Clean false alarms | 4 | 4 |
| Errors called clean | 0 | 0 |

All 81 predicted labels are identical: **zero fixes, zero regressions, zero
Macro-F1 delta**. Joint probabilities changed by up to **0.06777167**, so this
is not a reused prediction file. The paired row-bootstrap delta is [0, 0] only
because these labels are identical; it does not prove generalization equivalence.
The repeatedly selected holdout is not an independent test. Training loss was
1.23480150 with finite parameters and statistics. R-Drop used two training
forwards and took about 50.5 minutes; **no classification benefit was observed**.
Prefer the simpler hierarchy control for efficiency on current evidence.
R-Drop remains active in alpha and its requested integration into `rquq/test`;
there is no silent algorithm rollback or localization merge.

Shared project artifacts:
- `runs/hierarchical-rdrop-seed42/`: adapter/head, saved predictions, metrics,
  epoch/trainer histories, manifest and matched `comparison.json`.
- `runs/hierarchical-rdrop-seed42-config.json`: exact experiment controls.

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
2. Next **classification** candidate: auxiliary edit localization/type only on
   audited complete source-target pairs. Incomplete corrections cannot define
   trustworthy KEEP/error spans; omit their auxiliary loss rather than invent
   targets. Classification still receives source text only—never gold-selected
   descriptions, corrections or explanations.
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

The auxiliary-supervision, data/schedule and correction priorities remain
proposals, not measured improvements. This holdout has already selected
checkpoints and experiments; future quality claims need fresh group-safe
evaluation or repeated splits, not continued optimization of these same 81 rows.

