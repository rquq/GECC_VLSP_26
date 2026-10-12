# Sailor-7B-Chat Task 1 → Task 2: Kaggle and local

Both stages use `sail/Sailor-7B-Chat`, retaining the existing Task 1 loss and
Task 2 correction recipe. Task 2 prompts contain the source sentence and Task 1's
predicted label plus error-type description. Training hints come from train-only
out-of-fold (OOF) classifiers; validation hints come from the selected main
classifier. Gold labels are not substituted into prompts. These are independently
initialized QLoRA adapters, not classifier-to-corrector weight transfer. Every
input reaches correction, including predicted class 0.

## Run locally

Local and Kaggle runs share the same training code, configs and both notebooks. Use
`notebooks/local_two_phase.ipynb` on `main`, `beta`, `rquq/test` and `rquq/alpha`;
use `notebooks/kaggle_two_phase.ipynb` on Kaggle. The local notebook operates on
the current checkout, never clones it, discovers private data locally, defaults
to one CUDA GPU, and writes resolved configs, logs and outputs to
`runs/`. It supports the same Task 1/Task 2, OOF, saved-run continuation and
diagnostic controls as the Kaggle notebook. Training runs in subprocesses, not
inside the notebook kernel. Dependencies use `requirements-kaggle.txt` while
retaining the installed CUDA PyTorch.

## Weights & Biases

W&B is enabled in both configs and notebooks. Local authentication uses
`WANDB_API_KEY` in the environment or an existing `wandb login`. For Kaggle,
add `WANDB_API_KEY` under **Add-ons → Secrets** and enable notebook access.
The notebook loads the key into the environment inherited by training subprocesses;
no interactive `wandb login` is needed. The isolated training import check includes
W&B. Never put credentials into source, configs or outputs.
Rotate the key previously pasted into chat before using it in future runs.
`WANDB_MODE=offline` retains runs for later `wandb sync`; set
`tracking.enabled=false` (or notebook `TRACKING_ENABLED=False`) to disable it.

Main classification, each OOF fit, correction and paired diagnostics have
distinct W&B runs in one `WANDB_RUN_GROUP`. Tracking records optimizer-step
loss/LR/gradient metrics, epoch validation, final selected/reloaded metrics,
per-class tables, confusion matrices, data hashes/counts, Git provenance,
selected adapter artifacts and reused-adapter lineage. `wandb_run.json` records
the run ID, URL, group, status and artifact references next to local outputs.
Only rank zero logs in distributed jobs; a failed fit finishes with failed status.
No model watching or source-code upload is enabled.

`tracking.log_text=false` is required: corpus text and hint uploads are rejected
under the supplied agreement. `tracking.log_models=true` uploads adapters/tokenizers,
not datasets; disable it with `WANDB_LOG_MODELS=False`. Keep projects private and
confirm authorization for external model hosting; private hosting alone does not
grant redistribution rights. Local reports retain corpus-bearing diagnostics.

## Run on Kaggle

Use Kaggle only with authorization for third-party hosting/access to the restricted
resources: the supplied agreement prohibits redistribution, including edited data.
A private notebook or Dataset does not itself establish that authorization.
Once authorized, import `notebooks/kaggle_two_phase.ipynb` into a **private** notebook,
enable Internet, select **GPU T4 ×2**, and attach the following private files from
the same rebuilt dataset. Use **Save Version → Save & Run All** for an unattended
saved run; interactive Run All uses the same execution path:

The GitHub repository is **private**. Before Run All, create a fine-grained
GitHub token restricted to `rquq/GECC_VLSP_26` with **Contents: Read-only**.
In Kaggle **Add-ons → Secrets**, add it as `GITHUB_TOKEN` and enable notebook
access. The Git bootstrap uses that secret to clone without storing it in the
remote URL or Git config. Each run creates a fresh `gecc-two-phase-source-*`
directory under `/kaggle/working`; old source edits and outputs are untouched,
so a dirty checkout never blocks Run All. This authenticates code access only;
datasets still come from the private Kaggle Dataset attached through Add Input.

If Git access fails, the bootstrap reports sanitized Git stderr and the exit code:
authentication failures require checking the token's repository access and expiry;
DNS/connection failures require checking Kaggle Internet. Never paste tokens into
notebook source, output or chat. Revoke any exposed token and replace the Secret.

- `VGEC 2026_v1.json`: original corpus.
- `VGEC_classifier.json`: capped mixed classifier dataset.
- `VGEC_balanced.json`: correction dataset with split/provenance metadata.

The rebuilt local files are `dataset/original/VGEC 2026_v1.json` and
`dataset/newdataset/{VGEC_classifier.json,VGEC_balanced.json}`. No upload package
is produced. Only after obtaining the required authorization, attach those three
JSON files and confirm they appear under `/kaggle/input`. No XLSX, audit, manifest
or synthetic-pool upload is needed. Restricted corpus and generated datasets are
intentionally Git-ignored; a fresh Git clone does not contain them.


The notebook clones `kaggle-linux` and supports `RUN_MODE="task1"`, `"task2"`,
or `"both"`. Fresh Task 1 keeps the baseline recipe; default `GENERATE_OOF=True`
also produces the connected train-only hints needed by later Task 2 runs.
Task 2 can reuse a prior classifier without another Task 1 fit. Explicit attachment
paths resolve duplicate filenames. Outputs use `/kaggle/working/runs/RUN`,
retained by Save Version without a symlink. Keep notebook versions, datasets,
checkpoints and text-containing outputs private and within authorized access.

For a Kaggle terminal, first configure Git authentication for the private
repository; Kaggle Secrets are read automatically only by the notebook bootstrap:

```bash
git clone --single-branch --branch kaggle-linux https://github.com/rquq/GECC_VLSP_26.git /kaggle/working/gecc-two-phase-source
cd /kaggle/working/gecc-two-phase-source
python -m venv --without-pip --system-site-packages /tmp/gecc-training
python -m pip --python /tmp/gecc-training/bin/python install -r requirements-kaggle.txt
mkdir -p /kaggle/working/runs
/tmp/gecc-training/bin/python -m model.two_phase --classification-config configs/baseline.json \
  --correction-config configs/correction.json --output-dir /kaggle/working/runs/RUN \
  --submission-data /kaggle/input/DATA/input.json
```

`requirements-kaggle.txt` pins non-torch dependencies and preserves Kaggle's
CUDA-enabled PyTorch. The notebook records the cloned commit and resolved configs.

The dependency cell creates a separate training virtual environment with
`--without-pip --system-site-packages`, then uses Kaggle's existing pip with
`--python` to install into it. This avoids the system Python's failing `ensurepip`
bootstrap and requires pip 22.3 or newer. Kaggle's CUDA-enabled PyTorch is reused;
the pinned training packages are installed locally. It never replaces the notebook kernel's
NumPy or other packages, and **does not require a kernel restart**. Imports,
pipeline preflight, DDP workers and training all use the fresh training
interpreter. The notebook kernel only prepares paths/configs and streams output.
Pip may still report unrelated preinstalled-app conflicts because the environment
can see Kaggle's system packages; these do not imply installation failed. The
training import check and subprocess exit codes remain required gates.

## Two-T4 training

Both configs default to `run.world_size=2`. The pipeline launches each stage
with `torchrun`, one NF4 model replica per GPU and synchronized adapter gradients.
The main classifier, every OOF fit, and the corrector use both GPUs for training.
Per-GPU microbatch 1 × two GPUs × four accumulation steps = global effective
batch 8. T4 uses FP16; supported newer GPUs may use BF16.

Epoch validation, hint prediction and correction generation run on rank 0 while
the other rank waits. This is data parallelism for training throughput, not model
splitting, and does not guarantee a 2× end-to-end speedup. Each replica must fit
one T4. The default launch fails before downloads if two CUDA GPUs are unavailable.
For deliberate single-GPU runs, set `run.world_size=1` in **both** copied configs.
Direct stage entrypoints require a matching distributed launch:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m model.classification --config configs/baseline.json
```

## Retained recipes

Both stages use pinned `sail/Sailor-7B-Chat` revision
`19066fae0a8a3ba029c190d8e3dacccf4d1234b8`, NF4 double quantization,
LoRA rank 16 / alpha 32 / dropout 0.05, sequence length 512, effective batch 8,
and seed 42. Correction uses the model's `question`/`answer` chat roles and
pretrained generation stop tokens, saved with the adapter contract.

- **Task 1:** single-forward soft hierarchical BCE + twice conditional CE,
  square-root class weights, five epochs, LR `5e-5`. The raw corpus is split
  before augmentation: 312 retained original training rows + 192 synthetic rows
  = 504 training rows; 81 raw validation rows. Selection: validation macro F1.
- **OOF hints:** five connected folds, three epochs per fold, no outer-holdout
  selection. Duplicate sources, synthetic parent relationships and contrast/template
  families stay connected. Hints cover the correction training population,
  including correction rows not retained in the main classifier's training set.
- **Task 2:** response-only JSON supervision, eight epochs, LR `2e-5`.
  Prompt tokens are masked from loss; target completions are never truncated.
  Selection: error-only GLEU+, then error-only exact match, then clean preservation;
  ties retain the earlier epoch. Invalid generated JSON is reported with raw
  generation and parse error, not replaced with a fake successful correction.

Validation selection reuses the same holdout; reported scores are exploratory,
not independent test performance. Fresh training need not reproduce historical
scores or identical numerics across hardware/library versions.

## Saved pipeline inference and submissions

Local output is `runs/RUN/`; Kaggle output is `/kaggle/working/runs/RUN/`.
After Task 2, the pipeline reloads both saved stages and infers `SUBMISSION_DATA`
(CLI `--submission-data`) into `RUN/submission.json` before archiving the run.
With no explicit input, it exports the untouched holdout and marks the scope
`validation`; this is not an unseen-test submission. Task 1 alone cannot produce
a full GEC submission. Every input reaches Task 2, including predicted class 0.

The organizer JSON array contains only this shape, preserving typed IDs and order:

```json
[{"ID": 1, "Output": {"Error_Type_Id": 0, "Fixed Sentence": "A corrected sentence."}}]
```

Gold/source fields, raw generation and parse errors stay in
`submission.comparisons.json`. `submission.evaluation.json` records scope/status;
`submission.metrics.json` is written only for labeled evaluation. Invalid labels,
blank corrections or parse errors fail export without a source/gold fallback.
The standalone CLI exits nonzero. A completed training run retains its adapters,
failed-export diagnostics and archive without fabricating `submission.json`.

The run also contains `classification/classifier`, `classification/hints.json`,
`correction/adapter`, selected-adapter evaluations, epoch histories, predictions,
metrics, resolved configs, `run_handoff.json` and `pipeline_manifest.json`.
Notebook logs and the artifact ZIP sit alongside the run.

For saved-run inference, set `RUN_TRAINING=False`, `PRIOR_RUN` and `SUBMISSION_DATA`,
or run:

```bash
python -m model.pipeline_predict --run runs/RUN \
  --data /absolute/private/input.json --output runs/new-submission/submission.json
```

Use `/kaggle/working/runs/RUN` and a mounted input on Kaggle. Omitting `--output`
creates a fresh repository `runs/submission-*/submission.json`. Output artifacts
must not already exist; inference never overwrites a completed submission.
Use `--evaluate` for labeled records. Direct `--classifier` plus `--adapter`
paths are also supported.

### Combined score

Correction epoch/final evaluation, labeled submission inference, pipeline logs,
JSON reports and W&B scalar metrics include `combined_score`:

`0.5 * classification.macro_f1 + 0.5 * corrections.error_only.gleu_plus`

This is a **local, non-official equal-weight composite**: the supplied metrics
PDF specifies separate metrics, not a combined formula. Metadata records that
distinction. No gold error rows means `combined_score=null`; unlabeled input is
not scored. Checkpoint selection remains error-only GLEU+, exact match, then
clean preservation, rather than this composite.

## Run modes and next-version inputs

| Notebook setting | Behavior | Required inputs |
|---|---|---|
| `RUN_MODE="task1"`, `GENERATE_OOF=True` | Train baseline classifier and OOF fits; export reusable classifier | Three audited data files |
| `RUN_MODE="task1"`, `GENERATE_OOF=False` | Classifier only; no future correction hints | Original + classifier datasets |
| `RUN_MODE="task2"`, `PRIOR_RUN=...` | Reuse classifier/OOF hints; train a fresh corrector | Prior run + identical correction dataset |
| `RUN_MODE="both"`, no prior | Train both stages | Three audited data files |
| `RUN_MODE="both"`, `PRIOR_RUN=...` | Reuse classifier; train corrector | Prior run + identical correction dataset |
| `RUN_TRAINING=False` | Saved-run submission inference and/or correction diagnostics | `EXPERIMENT_RUN` or `PRIOR_RUN`; `SUBMISSION_DATA` or an experiment flag |

Within the agreement's authorized access, save the notebook version privately
and use **Add Input** in the next notebook to attach its output. Set `PRIOR_RUN`
to its actual mounted run folder, e.g.
`/kaggle/input/my-previous-version/runs/sailor-7b-task1-RUN_ID`. Kaggle chooses
the mount name; this is not the previous `/kaggle/working/runs` path. Run All
cannot automatically attach outputs to a future notebook version.

`run_handoff.json` contains relative stage paths. A new run includes the reused
classifier weights/tokenizer, OOF hints, configs and provenance, so another
handoff does not depend on an older mount. Direct classifier-stage paths,
legacy enclosing pipeline roots, and unique run folders inside mounts are
supported; multiple candidates require an explicit run folder. Inputs are
read-only. Reuse requires a compatible five-class QLoRA classifier and valid
OOF hints for the exact correction data/split. Changing the dataset requires
new compatible OOF fits, not applying stale hints to new examples.
The rebuilt corpus invalidates historical OOF coverage. Retrain Task 1 and OOF
for this build; old Sailor2 adapters are not compatible with a Sailor-7B warmstart.

`CLASSIFIER_RUN` explicitly selects a classifier. `RESUME_CORRECTION=True`
selects the prior corrector for continued fine-tuning; alternatively set
`INITIAL_ADAPTER` to a saved corrector. This starts a **new optimizer/scheduler**,
not an exact trainer-state resume. Base model/revision, LoRA layout, hint mode
and sentence/edit output representation must match. Arbitrary incompatible
adapters cannot be converted automatically; a classifier is not a causal-LM
corrector, and a sentence adapter is not an edit adapter.

```bash
python -m model.two_phase --mode task1 --classification-config configs/baseline.json \
  --correction-config configs/correction.json --output-dir /kaggle/working/runs/task1-run
python -m model.two_phase --mode task2 --prior-run /kaggle/input/prev/runs/task1-run \
  --correction-config configs/correction.json --output-dir /kaggle/working/runs/task2-run
```

## Correction experiments

All are opt-in. Defaults retain the original sentence-generation recipe and
classifier hints; no performance improvement is claimed.

- **`HINT_ABLATION=True`:** paired evaluation of the same saved corrector and
  validation records with predicted hints versus no label/type hint. Gold labels
  never enter prompts. This removes a conditioning input without retraining, so
  the no-hint result is a diagnostic distribution shift.
- **`EDIT_CORRECTION=True`:** fresh training predicts
  `{"edits":[{"start":0,"end":3,"replacement":"..."}]}` rather than a
  whole sentence. Offsets are Python Unicode character indices into the original
  source; spans must be ordered, nonoverlapping and in bounds. Insertions and
  deletions are supported. Deterministic application preserves untouched spans;
  invalid edit JSON remains a parse error, never a fake unchanged success.
- **`CANDIDATE_RERANKING=True`:** deterministic beam candidates plus unchanged
  source. A frozen base model (correction adapter disabled) scores a Vietnamese
  grammaticality judgment; a CPU multilingual sentence encoder scores meaning
  similarity. Weighted quality is ranked before edit-size tie-breaking. Invalid
  generated candidates remain in diagnostics but cannot win. The source is an
  ordinary candidate, not an automatic fallback. This adds inference cost and
  a semantic-model download; both learned scoring signals can be wrong.

Set `RUN_TRAINING=False`, `PRIOR_RUN` and either diagnostic flag to evaluate
an existing corrector without retraining. After training, the same flags evaluate
the newly saved corrector. Predictions retain candidate text, parse errors,
component scores and selection metadata. Reports use the unchanged holdout,
including correction quality and clean preservation separately.

```bash
python -m model.correction_experiments --run /kaggle/input/prev/output/RUN \
  --output-dir /kaggle/output/diagnostics --device cuda --hint-ablation --rerank
```

For training/production inference configure `inference.hint_mode="none"` or
`inference.output_format="edits"` in `CORRECTION_OVERRIDES`. Enable persistent
reranking with `inference.reranking={"enabled":true,"num_candidates":4}`.
Saved inference settings travel with the adapter; the diagnostic rerank flag
does not mutate the saved adapter. Sentence-mode legacy adapters keep their
original saved prompt.


### Recover a run that failed after training

If training finished but the final `load_corrector` call failed its GPU-memory
preflight, the selected checkpoint is already in `correction/adapter`. Preserve
the entire run directory before restarting the Kaggle session. Do not rerun
training merely to evaluate that checkpoint. In `epoch_history.json`, the highest
`selection_values` tuple wins (earliest epoch on ties); the saved adapter is not
necessarily from the last epoch.

Run this notebook cell after the dependency bootstrap, with `RUN` set to the
preserved run directory. It evaluates the saved corrector in a fresh subprocess
using the original validation hints, without loading a second classifier or
changing the configured memory reserve:

```python
RUN = Path("/kaggle/working/output/YOUR_COMPLETED_RUN")
recovery = r'''
import json, sys
from pathlib import Path
from model.correction import load_corrector, correct
from model.data import _id_key
from model.metrics import experiment_report
from model.prediction_export import comparison_rows
from model.runtime import write_json
run = Path(sys.argv[1])
stage = run / "correction"
records = json.loads((stage / "validation.json").read_text())
ids = {_id_key(row["id"]) for row in records}
hints = json.loads((run / "classification" / "hints.json").read_text())["validation"]
hints = [row for row in hints if _id_key(row["id"]) in ids]
model, tokenizer, inference = load_corrector(stage / "adapter", device=sys.argv[2])
predictions = correct(model, tokenizer, records, hints, inference)
metrics = experiment_report(records, predictions)
write_json(stage / "predictions.json", comparison_rows(records, predictions))
write_json(stage / "metrics.json", metrics)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
'''
subprocess.run([str(TRAINING_PYTHON), "-c", recovery, str(RUN), "cuda"],
               cwd=PROJECT_DIR, check=True)
```

This recovers predictions and metrics only; it does not fabricate the missing
run/pipeline completion manifests or reproduce lost trainer optimizer state.


## Recommended local correction experiments

Keep the audited split, seed 42 and Task 1/OOF artifacts fixed when changing
only Task 2. Do not use gold labels in correction prompts or skip predicted
class 0. Compare error-only GLEU+, error-only exact match, clean preservation,
invalid response rate and per-error-type results, not classifier accuracy alone.
The current 81-row selection holdout is exploratory; repeated tuning needs a
separate untouched evaluation set before claiming a quality improvement.

1. Run `HINT_ABLATION=True` on the saved corrector with `RUN_TRAINING=False`.
   This paired inference comparison costs no training. A no-hint prompt changes
   the training distribution: if it helps, train a fresh `hint_mode="none"`
   corrector before drawing a final conclusion.
2. Train fresh sentence-output correctors at LR `1e-5`, `2e-5` and `5e-5`,
   leaving the eight-epoch budget and effective batch eight unchanged. Compare
   selected, reloaded adapters; do not continue one adapter across this ablation.
3. Compare a fresh `EDIT_CORRECTION=True` run against sentence output using the
   same hints and optimizer budget. Localized edits preserve untouched spans,
   but invalid JSON/offsets remain failures; they are not silently repaired.
4. Evaluate `CANDIDATE_RERANKING=True` on a saved adapter after measuring the
   greedy baseline. It adds beam generation, frozen-base grammaticality scoring
   and a multilingual encoder; accept its added inference cost only if paired
   metrics improve without damaging clean preservation.
5. After isolating a useful Task 2 recipe, consider reviewed training-only
   corrections and realistic minimal-edit/clean examples. Changing dataset
   contents requires rebuilding and retraining Task 1 OOF coverage; old hint
   artifacts are not valid for a modified corpus.

These are experiment priorities, not measured correction improvements.

## Private dataset rebuild and terms

The previous pool's provenance was `locally_authored_synthetic` and
`assistant_reviewed_not_human_adjudicated`: it was assistant-authored, not original
organizer data. It has been replaced completely, not merely relabeled.
`tools/author_synthetic.py` now derives deterministic minimal contrasts exclusively
from complete original training corrections/clean sentences after the grouped
seed-42, fraction-0.2 split and existing quality audit. Holdout source/target text
is only an exclusion boundary, never generation evidence.

The rules delete a sole nominal copular predicate, replay original class-1 edits
inside identical balanced quotations, move `các`/`những` after bounded existing
nominal heads, repeat adjacent copula/aspect markers, or append a dangling comma
after a terminal full stop. Clean additions are exact complete training-gold
identities, optionally quoted. No external generator, invented sentence product,
semantic substitution, original relabeling or gold repair is used.

Rows record `parent_ids`, parent-level `template_id`, target `family_id`, rule
evidence and original SHA-256; OOF keeps those connected. Current provenance is
`corpus_derived_deterministic_augmentation` /
`deterministic_rule_checked_not_human_adjudicated`. Quoted replays preserve the
organizer's label ambiguities; no human adjudication or quality improvement is
claimed. The rules assume ordinary edited sentence prose, not headings or
creative typography.

| Private artifact | Rows | Use |
|---|---:|---|
| `synthetic_pool.json` | 730 | Deterministic training-only candidates |
| Classifier training | 504 | 312 retained originals + 192 selected additions |
| Correction training | 409 | 217 retained originals + 192 selected additions |
| Validation in either bundle | 81 | Identical raw original holdout |

Existing caps remain: 40 clean additions; error additions do not exceed the
retained classifier-original count per label. Original JSON SHA-256 remains
`7968ea77cbcf69a26c9f98f1946c4b4dc19237a02c67fb483d54265f109dc728`;
the XLSX is also unchanged. Regenerated pool, both bundles, audit and manifest
are mode `0600` under a mode-`0700` directory. They stay local and Git-ignored.

Reproduce from the repository root:

```bash
python tools/author_synthetic.py --original 'dataset/original/VGEC 2026_v1.json' \
  --output dataset/newdataset/synthetic_pool.json
python tools/build_dataset.py --original 'dataset/original/VGEC 2026_v1.json' \
  --synthetic dataset/newdataset/synthetic_pool.json \
  --output-dir dataset/newdataset --validation-fraction 0.2
```

The supplied **VLSP2026-ViGEC-UserAgreement** §§3.1–3.6 permits research
modification, forbids commercial use and redistribution (including edited
resources), and requires published research to acknowledge VLSP via its website
or publications. Keep inputs, derivatives, hints, logs, submissions and archives
within authorized access. No Kaggle upload package was created. Obtain organizer
authorization before third-party hosting; a private Dataset is not permission.
Corpus/hint W&B uploads are rejected even if `tracking.log_text=true` is requested.


## Local checks and provenance

The local shared virtual environment is in sibling `test`:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false ../test/.venv/bin/python -m unittest discover -s agents/tests -q
```

Local runs need copied configs with explicit paths for all three datasets and
an appropriate `run.world_size`. Dataset authoring/building utilities remain in
`tools/` for provenance; training never authors or rebuilds attached inputs.

The Sailor-7B-Chat/submission/dataset cutover passed **108 regression tests**.
Actual local notebook execution trained a tiny CPU Qwen2 classifier, two OOF
folds and a corrector on the rebuilt corpus, then reloaded both adapters and
exported a labeled two-row submission, metrics, log and ZIP. Five offline W&B
runs finished. The Kaggle notebook's config/training/streaming/display cells
also executed with temporary local input/working mounts and an existing
interpreter: they exported all 81 validation rows, archived the submission,
then reloaded an attached saved-run copy to infer two unlabeled rows without
inventing evaluation metrics. Both exports preserved integer/string ID types.
Kaggle clone/dependency bootstrap was not rerun in this cutover smoke.
Standalone saved inference with no `--output` also created a fresh repository
`runs/submission-*/submission.json` and exported both unlabeled typed IDs.
An independent rebuild reproduced the pool, both bundles and audit byte-for-byte;
all parents were training-only, no generated source/target matched heldout text,
and the 81 validation objects were identical to their raw organizer counterparts.
Both organizer files were unchanged; generated files/directories retained
`0600`/`0700` permissions.

Before committing, the same 108-test suite passed again. Actual Kaggle config,
streaming and display cells ran with temporary local mounts: a compatible tiny
CPU Task 2 continuation exported all 81 validation rows, logged the non-official
combined score, finished grouped offline pipeline/correction W&B runs and
archived the submission. Saved-run inference exported two unlabeled typed IDs.
A simulated Kaggle Secrets client exercised online key loading, environment-key
precedence, missing/empty-secret errors and offline/disabled opt-outs. A real
child process inherited the key and finished an offline W&B run; emitted key
text was redacted from displayed output and the persisted log. No online W&B
authentication or hosted Kaggle execution was performed in this commit check.
The local smoke interpreter imports W&B 0.27.0 but lacks `sentence_transformers`;
the full Kaggle dependency-import command was not verified in that interpreter.

The real pinned Sailor tokenizer passed response-only JSON supervision,
`question`/`answer` prefix and generation-stop checks, plus a finite forward
with explicitly reduced random Qwen2 weights. These checks prove the executable
contracts, not full 7B training, hosted Kaggle/T4 memory headroom, or useful
correction quality. Original and derived corpus files were not uploaded.


Local verification passed 83 regression tests and real connected notebook-runtime
smokes with tiny unquantized Qwen2 models at world sizes 1 and 2. The smokes used
the attached audited 504/81 classifier split, trained two OOF folds with complete
correction hint coverage, updated both adapters, reloaded saved weights, generated
new-input predictions, and retained the output link, ZIP and log. A complete
correction epoch consumed its final partial accumulation window in both modes.

Save Version bootstrap verification used real Papermill execution on Python 3.13
with NumPy 2.3.3 preloaded in the notebook kernel. The kernel kept that version
and its original module path; the separate training interpreter imported the
pinned stack with NumPy 2.2.6 and ran a finite tiny Qwen2 forward. A second
unattended run exercised the notebook's config and log-streaming cells through
real CPU pipeline preflight on the audited 504/81 split. Seven existing Kaggle
bootstrap/data regression tests also passed. These checks do not claim full
Sailor2 GPU training or execution inside Kaggle's hosted image.

The pipless bootstrap also passed real Papermill execution with `ensurepip`
deliberately blocked in every child interpreter: training imports and a finite
tiny Qwen2 forward passed while the kernel retained NumPy 2.3.3. A separate smoke
used `/usr/bin/python3` 3.13 to create a pipless environment and install/import
NumPy via host pip's `--python`. The seven Kaggle regression tests still passed.

Correction reload cleanup was exercised with a real tiny NF4 Qwen2 CUDA fit
and two-process CPU DDP. Each fit completed two epochs including partial
accumulation windows; saved selected-adapter tensors matched the reloaded
adapter exactly. A retained callback-handler smoke failed before cleanup was
fixed and passed afterward with zero live training-model/parameter references
before reload. The ordinary old-code tiny runs did not reproduce the hosted
T4 memory failure; this proves the ownership cleanup, not its complete cause
in that image or full Sailor2 T4 memory headroom.
The current full local regression suite passed 99 tests, including strict edit
validation, hint isolation, reranking precedence, compatible adapter continuation,
and portable run reuse. A real tiny unquantized Qwen2 CPU smoke trained Task 1
with two OOF folds on the audited 504/81 population, reused it for edit-mode
Task 2, removed the original Task 1 folder, and continued the saved corrector
from the second portable output with a new optimizer. It exported all five
saved/hint/rerank diagnostic variants using the real pretrained multilingual
semantic encoder. The saved-run notebook cells also executed end to end with
a temporary input/output mount and CPU device override, retaining diagnostics
and logs. Tiny generated edit beams were invalid and remained visible errors;
these runs prove execution and handoff, not useful correction quality.


Full Sailor2 training on Kaggle T4 ×2 has not been exercised locally; the local
machine exposes only one CUDA GPU. Distributed CPU smoke evidence does not prove
T4 memory headroom or a measured throughput gain.

## Branch changes

Restored alpha correction, response-only collation, connected OOF handoff,
saved pipeline inference, and the full two-phase notebook. Kept the existing
Task 1 loss and dataset recipe; did not restore alpha experimental losses,
encoder/ensemble paths, alternate experiment configs or duplicate dependencies.
Both stages now default to two-process data-parallel training.
Fixed Transformers 4.51 epoch accounting and rank-local tail consumption so
OOF and correction fits finish every configured epoch rather than silently
dropping the last accumulation window.
Save Version now isolates training dependencies from the notebook kernel and
runs model preflight inside the training subprocess; unattended runs no longer
stop at a loaded-NumPy restart guard.
Environment creation skips `ensurepip`; the existing host pip installs into the
training interpreter using `--python`.
Correction cleanup now detaches callback-handler model/optimizer/scheduler
references and clears the training scaler before collecting GPU allocations
and reloading the selected saved adapter. The conservative memory check remains
unchanged; no offload or warning suppression was added.
Run modes now support independent Task 1/Task 2 execution and self-contained
saved-output reuse. Optional correction experiments add paired hint ablation,
localized edit supervision and frozen-base/multilingual candidate reranking.
Compatible prior correctors can continue fine-tuning with a new optimizer;
incompatible prompt/output/LoRA contracts fail before model loading. Saved
classifier validation accepts PEFT's automatically added classification-head
metadata while still requiring configured modules and saved score weights.
Both stages now use the pinned Sailor-7B-Chat base and its native chat roles.
Training automatically exports organizer-format submissions after Task 2;
standalone inference exports into a fresh runs folder by default. Validation
exports are explicitly marked, invalid generations retain failure diagnostics,
and local combined-score metadata is logged without changing model selection.
The assistant-authored pool was replaced with private deterministic training-only
corpus derivatives, with parent-linked OOF grouping and unchanged organizer files.


W&B integration adds explicit grouped runs for pipeline orchestration, main/OOF
fits, correction and saved diagnostics; metrics, safe reports, selected adapters
and input lineage are tracked without automatic corpus/code uploads. The local
notebook uses an existing interpreter and checkout, with no package mutation.
The unchanged 99-test suite passed with tracking disabled. A real local notebook
execution used tiny CPU Qwen2 adapters and the audited data, completed Task 1
plus two OOF fits and Task 2, and ran paired hint diagnostics: six distinct
finished offline W&B runs shared one group. Safe report payloads excluded corpus
fields. Authenticated online verification checked server-side metrics,
successful/failed lifecycle status, model/text-upload opt-outs, adapter artifact
lineage, sanitized artifact JSON and rank-one suppression in a private project.
The Kaggle saved-run cells also completed offline paired hint diagnostics using
a real temporary input-mount copy, with no W&B secret access; the finished run,
metrics, log and persisted-output symlink were retained.
This does not claim full Sailor2 training or a correction-quality gain.
