> The current five-fold Sailor-7B pipeline is reserved in [preserved/kaggle-fold-hints-47f3a23](preserved/kaggle-fold-hints-47f3a23/README.md). Its Kaggle notebook pins tag alpha-fold-hints-20261013. Existing alpha experiments below remain historical evidence. The next no-fold/no-hint run is in ../kaggle-linux; see agents/experiments/CURRENT.md and HISTORY.md. Historical Sailor2 components exceed competition 1038's strict 8B limit.

# Sailor2 Task 1 → Task 2: Kaggle and local

This branch keeps the existing Sailor2 Task 1 baseline and restores alpha's
Task 2 correction stage. Task 2 prompts contain the source sentence and Task 1's
predicted label plus error-type description. Training hints come from train-only
out-of-fold (OOF) classifiers; validation hints come from the selected main
classifier. Gold labels are not substituted into prompts. These are independently
initialized QLoRA adapters, not classifier-to-corrector weight transfer. Every
input reaches correction, including predicted class 0.

## Run locally

All branches share the same training code, configs and both notebooks. Use
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
`WANDB_API_KEY` in the environment or an existing `wandb login`; Kaggle reads
the `WANDB_API_KEY` Secret. Never put credentials into source, configs or outputs.
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

The default `tracking.log_text=false` excludes corpus text and hint tables;
`tracking.log_models=true` uploads adapters/tokenizers, not datasets. Disable
model uploads with `WANDB_LOG_MODELS=False` in the notebook. Enable
`WANDB_LOG_TEXT=True` only with permission to upload licensed examples. Keep
the W&B project private; derived weights and diagnostics may be sensitive too.

## Run on Kaggle

Import `notebooks/kaggle_two_phase.ipynb` into a **private** Kaggle notebook.
Enable Internet, select **GPU T4 ×2**, attach the following private files from
the same audited dataset build. Use **Save Version → Save & Run All** for an
unattended saved run; interactive Run All uses the same execution path:

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

The local Kaggle checkout contains the ready-to-upload package:
`kaggle-input.zip`. Its three JSON files are also in `dataset/kaggle-input/`.
In Kaggle, create a **private Dataset**, upload that ZIP (or the three loose JSON
files), then add the Dataset to the notebook using **Add Input**. Confirm the
three JSON files appear under `/kaggle/input`, then Run All. Do not attach the
ZIP as an unextracted file. No XLSX, audit, manifest or synthetic-pool upload is
needed. The ZIP and corpus files are local-only and intentionally Git-ignored;
a fresh Git clone does not contain restricted corpus data.


The notebook clones `kaggle-linux` and supports `RUN_MODE="task1"`, `"task2"`,
or `"both"`. Fresh Task 1 keeps the baseline recipe; default `GENERATE_OOF=True`
also produces the connected train-only hints needed by later Task 2 runs.
Task 2 can reuse a prior classifier without another Task 1 fit. Explicit attachment
paths resolve duplicate filenames. Outputs use `/kaggle/output`, linked to
`/kaggle/working/output` so Save Version retains artifacts. Keep notebook versions,
datasets, checkpoints and text-containing outputs private.

For a Kaggle terminal, first configure Git authentication for the private
repository; Kaggle Secrets are read automatically only by the notebook bootstrap:

```bash
git clone --single-branch --branch kaggle-linux https://github.com/rquq/GECC_VLSP_26.git /kaggle/working/gecc-two-phase-source
cd /kaggle/working/gecc-two-phase-source
python -m venv --without-pip --system-site-packages /tmp/gecc-training
python -m pip --python /tmp/gecc-training/bin/python install -r requirements-kaggle.txt
mkdir -p /kaggle/working/output
ln -s /kaggle/working/output /kaggle/output  # Only if this path does not already exist.
/tmp/gecc-training/bin/python -m model.two_phase --classification-config configs/baseline.json \
  --correction-config configs/correction.json
```

If `/kaggle/output` exists, ensure it resolves to `/kaggle/working/output`.
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

Both stages use pinned `sail/Sailor2-8B-Chat` revision
`577d5b2745863cec11e1b2f1c92586a91cea446e`, NF4 double quantization,
LoRA rank 16 / alpha 32 / dropout 0.05, sequence length 512, effective batch 8,
and seed 42.

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

## Saved pipeline inference

The run contains `classification/classifier`, `classification/hints.json`,
`correction/adapter`, selected-adapter evaluations, epoch histories, predictions,
metrics, resolved configs and `pipeline_manifest.json`. The notebook retains a
ZIP and training log alongside the run.

```bash
python -m model.pipeline_predict \
  --classifier /kaggle/output/RUN/classification/classifier \
  --adapter /kaggle/output/RUN/correction/adapter \
  --data /kaggle/input/DATA/input.json \
  --output /kaggle/output/predictions.json
```

Use `--evaluate` for labeled records. To reuse a completed compatible Task 1 run,
pass `--classifier-run /kaggle/input/DATA/completed-classification` to
`model.two_phase`; its saved classifier, OOF hints and provenance are validated.

## Run modes and next-version inputs

| Notebook setting | Behavior | Required inputs |
|---|---|---|
| `RUN_MODE="task1"`, `GENERATE_OOF=True` | Train baseline classifier and OOF fits; export reusable classifier | Three audited data files |
| `RUN_MODE="task1"`, `GENERATE_OOF=False` | Classifier only; no future correction hints | Original + classifier datasets |
| `RUN_MODE="task2"`, `PRIOR_RUN=...` | Reuse classifier/OOF hints; train a fresh corrector | Prior run + identical correction dataset |
| `RUN_MODE="both"`, no prior | Train both stages | Three audited data files |
| `RUN_MODE="both"`, `PRIOR_RUN=...` | Reuse classifier; train corrector | Prior run + identical correction dataset |
| `RUN_TRAINING=False` | Saved-corrector diagnostics only | `EXPERIMENT_RUN` or `PRIOR_RUN`; enable an experiment flag |

Save the notebook version privately. In the next notebook, use **Add Input** to
attach that saved notebook output, or upload the extracted run as a private
Kaggle Dataset. Set `PRIOR_RUN` to its actual mounted run folder, e.g.
`/kaggle/input/my-previous-version/output/sailor2-task1-RUN_ID`. Kaggle chooses
the mount name; this is not the previous `/kaggle/output` path. Run All cannot
automatically attach outputs to a future notebook version.

`run_handoff.json` contains relative stage paths. A new run includes the reused
classifier weights/tokenizer, OOF hints, configs and provenance, so another
handoff does not depend on an older mount. Direct classifier-stage paths,
legacy enclosing pipeline roots, and unique run folders inside mounts are
supported; multiple candidates require an explicit run folder. Inputs are
read-only. Reuse requires a compatible five-class QLoRA classifier and valid
OOF hints for the exact correction data/split. Changing the dataset requires
new compatible OOF fits, not applying stale hints to new examples.

`CLASSIFIER_RUN` explicitly selects a classifier. `RESUME_CORRECTION=True`
selects the prior corrector for continued fine-tuning; alternatively set
`INITIAL_ADAPTER` to a saved corrector. This starts a **new optimizer/scheduler**,
not an exact trainer-state resume. Base model/revision, LoRA layout, hint mode
and sentence/edit output representation must match. Arbitrary incompatible
adapters cannot be converted automatically; a classifier is not a causal-LM
corrector, and a sentence adapter is not an edit adapter.

```bash
python -m model.two_phase --mode task1 --classification-config configs/baseline.json \
  --correction-config configs/correction.json --output-dir /kaggle/output/task1-run
python -m model.two_phase --mode task2 --prior-run /kaggle/input/prev/output/task1-run \
  --correction-config configs/correction.json --output-dir /kaggle/output/task2-run
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

## Local checks and provenance

The local shared virtual environment is in sibling `test`:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false ../test/.venv/bin/python -m unittest discover -s agents/tests -q
```

Local runs need copied configs with explicit paths for all three datasets and
an appropriate `run.world_size`. Dataset authoring/building utilities remain in
`tools/` for provenance; training never authors or rebuilds attached inputs.

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
