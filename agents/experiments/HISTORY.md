**2026-10-16 local retuning completed:**11 successful10-row decoder screens after a reproduced/fixed candidate-likelihood chunk error;3 complete78-row NF4 inference passes. Original greedy wins0.6149706110 all/0.2278072605 error GLEU+, clean40/40; no measured decoder improvement or local OOM. Saved classifier soft-vote search reaches development Macro-F1=0.6377559133, partition sensitivity0.5789647045 versus0.5615999132 baseline. Native four-model standard-LoRA challenger workflow prepared, not run;214 tests passed. See RETUNING_20261016.md/.json.

# Recovered experiment history

**Retuning request — 2026-10-16:** the default now selects configs/tuning_plan.json through the unchanged notebook launcher; prior recovery is preserved as configs/recovery_experiment.json. It binds the completed October16 four-adapter output and unchanged grouped-validation data, compares eleven decoders, fits one standard-LoRA challenger per classifier plus a Qwen corrector, compares/selects and exports. Classifier LR2.5e-5/dropout0.1/max4 epochs; corrector LR1.5e-5/max5 epochs; ranks16/alpha32/lambda2/seeds42/data unchanged. Native FP16 7B cannot fit the local8GB GPU/available WSL RAM; a separate NF4 inference proxy loaded at5.33GiB and is being evaluated, with no quantized training. Saved-probability tuning reaches development Macro-F1=0.6377559133; partition sensitivity0.5789647045 versus baseline0.5615999132 shows optimism. Do not claim unseen gains or deploy quantized proxy winners as native. Read RETUNING_20261016.md/.json; source-file/data checks and tiny full-controller tests passed; full-size Kaggle fits remain pending. Older default descriptions below are historical.


**Completed Kaggle results — 2026-10-16:** source bb7cdf5 finished all recovery/corrector/export stages. Equal-weight Qwen/Sailor/Mistral soft voting:78-row Macro-F1=0.5615999132, Micro-F1=0.7948717949; phrase recall0/6 remains the main subtype gap. Selected Qwen corrector epoch4: all-row GLEU+=0.6313346045, error-only=0.2377164465, combined-v2=0.5964672588, clean preservation40/40. Old81-row and new78-row holdouts share12 rows;69 old rows enter current training and66 new rows entered old training, so cross-set score differences are not controlled gains. Saved-probability binary-first threshold0.5 lowers ensemble Macro-F1 to0.5488461538; no retraining. Read VALIDATION_RESULTS_20261016.md/.json and private runs/analysis-validation-20261016/ for side-by-side rows. Beam2/4 and edit-aware reranking are proposals, not executed gains. Existing minimal-edit/clean-preservation prompting is already active; current correction has no classifier hints. Earlier pending-full-Kaggle statements below are historical.


**Classifier reload recovery — 2026-10-15:** updated Kaggle output confirms Mistral finished 720 steps; saved epoch1 Macro-F1=0.5137657474/Micro-F1=0.7051282051. It failed afterward at same-process inference placement (7.46 GiB free versus8.43 required). The default now binds all three saved classifiers from this latest full output and the unchanged78-row validation dataset: skip all classifier fits, verify Mistral in a fresh child, then train the Qwen corrector and export. New no-hint/single-process fits also separate fitting from verification with a saved training receipt and GPU release gate.202 tests, native51-file/data checks, two real tiny CPU continuations and a standard-LoRA FP16 CUDA pipeline passed; full-size/T4 verification remains pending. No notebook toggles, quantization or dataset changes. Read CLASSIFIER_RELOAD_RECOVERY_20261015.md/.json. Older two-member recovery instructions below are historical for the current default.


**Cold-cache Mistral tokenizer fix — 2026-10-15:** the new pasted preflight log shows LlamaTokenizerFast still entering slow conversion. Pinned Transformers4.51.3 forces from_slow=True when add_prefix_space is supplied; Mistral metadata sets itTrue, and the worker had no resolved SentencePiece vocabulary. Native assets are now explicitly downloaded to one pinned snapshot and loaded locally; add_prefix_space=None preserves the serialized backend's existing prefix behavior and prevents forced conversion.198 tests passed plus an actual empty-cache Python3.13.7 test matching all401 original input encodings and offline export/reload. Existing inputs and standard-LoRA recovery remain unchanged. See MISTRAL_COLD_CACHE_20261015.md/.json.


**Recursive Kaggle output discovery — 2026-10-15:** the continuation now accepts the original failed notebook's full output directly, together with the original78-row validation dataset. It recursively discovers the classification directory beneath arbitrary input mount prefixes and checks every required file against committed original-output hashes before reuse. Duplicate candidates, missing/changed checkpoints or wrong data fail before training. The combined private resume ZIP remains supported. No notebook switches or model/data changes. See KAGGLE_OUTPUT_DISCOVERY_20261015.md/.json.


**Mistral recovery — 2026-10-15:** the supplied Kaggle run completed Qwen and Sailor, then failed before Mistral training because production selected its slow tokenizer. Native fast-backend loading and early tokenizer preflight are fixed. The user explicitly chose standard-LoRA Kaggle continuation; no local full-size/QLoRA training. Default recipe now consumes PRIVATE VGEC26-soft-vote-resume-20261015.zip, preserving exact78-row validation data and both saved adapters. It skips validated Qwen/Sailor, trains Mistral then Qwen corrector and exports the soft-vote submission.196 tests plus native data/checkpoint and tiny end-to-end recovery checks passed. Read KAGGLE_MISTRAL_RECOVERY_20261015.md/.json. Older statements that no full-size stage has run are historical; continuation/new score remain pending.


**System monitoring — 2026-10-15:** W&B native resource collection is enabled, sampling every15seconds in root and model runs. Process-tree CPU/RAM tracking follows the root run's child trainers; all visible GPUs are monitored, plus root/output filesystems, disk I/O and network activity. Actual offline SDK0.27.0 recording verified GPU utilization/VRAM/temperature/power, CPU/RAM, disk/network metrics.7 focused tests passed. Read `WANDB_SYSTEM_MONITORING_20261015.md`/.json; no model/data/notebook changes or training run. Applies to newly started processes, not retroactively to an existing Kaggle session.


**Launcher fix — 2026-10-15:** the Kaggle log failed before training because its filename-loaded launcher could not import `model.reproducibility`. `launch()` now resolves/adds the cloned project root before package imports and uses an absolute cleanup-helper import. Both reproduced failures now pass;21 focused launcher/bootstrap/final-training tests passed. The earlier189 tests missed this outside-checkout filename-loading boundary. Pip dependency conflict warnings were not the reported abort: the provided log passed training imports before the launcher exception. Recipe/model/data/requirements/notebook cells remain unchanged. Read `KAGGLE_LAUNCHER_IMPORT_FIX_20261015.md`/.json.


**Active request — 2026-10-15:** the user superseded three seeds with three distinct classifiers and explicitly required **equal-weight soft voting**. Ordered members are pinned Qwen2.5-7B-Instruct, Sailor-7B-Chat and Mistral-7B-Instruct-v0.3; each trains once with seed42. The default recipe now selects the user-approved78-row grouped validation package (1145 training exposures per stage). Corrector remains Qwen, maximum6 epochs, standard LoRA; no folds, hints or notebook switches. Read `MODEL_SOFT_VOTING_20261015.md`/.json for exact revisions, counts, verification and limits. Earlier full-fit/default/three-seed descriptions below are historical. No full7B training or Kaggle launch has been performed for this change.


**Latest extra dataset request — 2026-10-15:** `GROUPED_VALIDATION_20261015.md`/.json records the completed PRIVATE extra package `dataset/kaggle-grouped-validation-20261015/VGEC26-grouped-validation-20261015.zip`: approximately80/20 organizer grouping,78 shared original validation rows,778/766 canonical training rows and1145 exposures per stage/229 per class. Whole source/target/parent/family/template/near-duplicate components are separated across BOTH stages;87 related synthetic and2 duplicate originals per stage are withheld. No relabeling/new generation/training/Kaggle launch.96 focused data/consumer regression tests and actual partition/archive/handoff checks passed. This new request authorizes leakage/data verification; the preceding skip-verification request applied to the preceding full-fit update. Default recipe/notebook/full-fit artifacts stay unchanged; separate `configs/validation_experiment.json` requires explicit recipe selection. Use fresh pretrained bases: prior full fits/reviews exposed these originals, so this is a new internal partition, not a historically blind test. Only top-level train is fit input; embedded source snapshots/validation/withheld records are audit material. Read the paired record before the historical snapshots below.

**Current training request — 2026-10-15:** user explicitly requests updating `kaggle-linux` for retained v2 and skipping verification. The active recipe is Sailor7B classification5 epochs /Qwen7B correction6 epochs, standard LoRA,1360 exposures per stage/272 per class; stored945/933. Use the complete PRIVATE `VGEC26-sailor-qwen-retained-v2-20261015.zip`, including the existing public inference input. All401 classifier originals (including8 notes) remain; corrector389 originals include112 abbreviated references once each;11 note targets and ID11 remain excluded. The v2 loader/rules are ported without augmentation-pipeline imports. No notebook toggles or model/loss/precision changes. No tests, smoke runs or model preflight are to be run for this update, per the user. Runtime input/size/memory checks remain active. Read KAGGLE_RETAINED_V2_20261015.md/.json. Earlier holds/585-495 training defaults below are historical; this instruction authorizes preparation of the selected v2 training recipe, with no score claim or automatic Kaggle launch.

**Latest dataset decision — 2026-10-15:** PUBLIC_ABBREVIATION_20261015.md/.json records139/443 public inputs with ellipses and saved output counts100 Sailor /134 Qwen /80 ViT5 /142 latest run. V2 restores112 exact ORGANIZER training references with ellipses, each once per correction epoch, without inferred completions or public pseudo-labels. This supersedes blanket ellipsis exclusion. Corrector canonical933 (389 originals), classifier945 (all401 originals,8 notes unchanged); balanced1360 per stage/272 each class.11 note targets remain corrector-excluded; ID11 is the sole remaining review item. Phrase canonical is101 classifier/95 corrector, not272 independent examples. All29 saved stage/export materializations and every extra type are inventoried privately. V1 remains preserved; no real-model training, new synthesis, recipe/notebook change or Kaggle launch. Read this record before the historical v1 snapshots below.

2026-10-15 latest: RETAINED_BALANCED_20261015.md/.json — private retained/balanced candidate built and verified;320 new code pairs,32 external controls,8 originals restored unchanged, each training view1360/272 per class,213 tests. No new model/score/Kaggle run.113 unresolved corrector targets remain pending; prior model failures are not promoted.

Recovery uses surviving run artifacts, archived agent notes, Git history, and maintenance records. No historical training was rerun. Private recovered notes can contain corpus text and remain ignored by Git.

## Surviving measured artifacts

| Run | Macro-F1 | Micro-F1 | Evidence |
|---|---:|---:|---|
| Kaggle 13_10_2026: fresh Qwen both-stage | 0.595176 | 0.753086 | `runs/13_10_2026/runs/qwen25-7b-reviewed-both-20261013T161655075757Z/run/classification/metrics.json`; full stages completed; failed public export recovered without retraining; QWEN_RUN_20261013.json |
| test:runs/sailor2-8b-lambda2-audited-no-rdrop-seed42 | 0.632750 | 0.802469 | `/home/quq/machineLearning/competitions/VLSP26_4/test/runs/sailor2-8b-lambda2-audited-no-rdrop-seed42/metrics.json` |
| alpha:runs/balanced-decision-seed42 | 0.643934 | 0.802469 | `/home/quq/machineLearning/competitions/VLSP26_4/alpha/runs/balanced-decision-seed42/metrics.json` |
| alpha:runs/domain-reviewed-seed42 | 0.551511 | 0.802469 | `/home/quq/machineLearning/competitions/VLSP26_4/alpha/runs/domain-reviewed-seed42/metrics.json` |
| alpha:runs/real-adaptation-seed42 | 0.632750 | 0.802469 | `/home/quq/machineLearning/competitions/VLSP26_4/alpha/runs/real-adaptation-seed42/metrics.json` |
| kaggle-linux:runs/11_10_2026/output/sailor2-two-phase-20261011-191339-434011/classification | 0.619418 | 0.777778 | `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/runs/11_10_2026/output/sailor2-two-phase-20261011-191339-434011/classification/metrics.json` |
| kaggle-linux:runs/12_10_2026/runs/sailor-7b-both-20261012-145038-393535/classification | 0.434133 | 0.679012 | `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/runs/12_10_2026/runs/sailor-7b-both-20261012-145038-393535/classification/metrics.json` |

## Recovered notes

These entries index historical claims; consult their private source lines before using a metric. Duplicate archive versions are retained explicitly.

- Historical rejected alpha ablations — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-038-README.md` lines 389–410.
- Completed soft hierarchy ablation — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-038-README.md` lines 411–461.
- Completed subtype loss coefficient-two ablation — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-038-README.md` lines 462–511.
- Single-forward cutover verification — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-038-README.md` lines 512–520.
- Rejected matched R-Drop ablation — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-038-README.md` lines 521–530.
- Rejected alpha/beta comparison — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-038-README.md` lines 531–539.
- Standalone encoder comparison — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-038-README.md` lines 540–580.
- Larger Vietnamese QLoRA classifier: Sailor2-8B — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-038-README.md` lines 581–649.
- Audited private-bundle five-epoch rerun — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-038-README.md` lines 650–702.
- Separate detection/subtype weighting ablation — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-038-README.md` lines 703–753.
- Eight-epoch fit and equal-weight Qwen ensemble — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-038-README.md` lines 754–812.
- Retained experiment integration into `rquq/test` — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-038-README.md` lines 813–851.
- Paper evidence and improvement priorities — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-038-README.md` lines 852–866.
- Deferred classifier algorithms — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-038-README.md` lines 954–972.
- Next controlled experiment — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-038-README.md` lines 973–1000.
- Historical alpha ablations (retained artifacts, inactive code) — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-073-README.md` lines 327–348.
- Completed soft hierarchy ablation — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-073-README.md` lines 349–399.
- Completed subtype loss coefficient-two ablation — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-073-README.md` lines 400–449.
- R-Drop implementation verification — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-073-README.md` lines 450–460.
- Completed matched R-Drop ablation — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-073-README.md` lines 461–494.
- Heterogeneous encoder ensemble — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-073-README.md` lines 495–556.
- Completed fixed-split experiment — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-073-README.md` lines 557–599.
- Larger Vietnamese QLoRA classifier: Sailor2-8B — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-073-README.md` lines 600–668.
- Eight-epoch fit and equal-weight Qwen ensemble — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-073-README.md` lines 669–738.
- Retained experiment integration into `rquq/test` — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-073-README.md` lines 739–770.
- Paper evidence and improvement priorities — `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/archive-073-README.md` lines 771–785.

## Code and recipe chronology

Implementation and recipe commits are evidence of code changes, not proof that their experiments completed.

Recovered draft recipes from stashes (no training outcome inferred):

- `stash:b43e34c6ae804ddb7ce264b9803a33be46529553:configs/classification.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-b43e34c-False-classification.json`
- `stash:b43e34c6ae804ddb7ce264b9803a33be46529553:configs/classification_sailor2_8b.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-b43e34c-False-classification_sailor2_8b.json`
- `stash:b43e34c6ae804ddb7ce264b9803a33be46529553:configs/classification_sailor2_8b_8epochs.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-b43e34c-False-classification_sailor2_8b_8epochs.json`
- `stash:b43e34c6ae804ddb7ce264b9803a33be46529553:configs/correction.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-b43e34c-False-correction.json`
- `stash:b43e34c6ae804ddb7ce264b9803a33be46529553:configs/encoder_bamibert.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-b43e34c-False-encoder_bamibert.json`
- `stash:b43e34c6ae804ddb7ce264b9803a33be46529553^3:configs/baseline.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-b43e34c-True-baseline.json`
- `stash:b43e34c6ae804ddb7ce264b9803a33be46529553^3:configs/classification_sailor2_8b_focal.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-b43e34c-True-classification_sailor2_8b_focal.json`
- `stash:b43e34c6ae804ddb7ce264b9803a33be46529553^3:configs/classification_sailor2_8b_smoothing.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-b43e34c-True-classification_sailor2_8b_smoothing.json`
- `stash:b43e34c6ae804ddb7ce264b9803a33be46529553^3:configs/classification_sailor2_8b_supcon.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-b43e34c-True-classification_sailor2_8b_supcon.json`
- `stash:b43e34c6ae804ddb7ce264b9803a33be46529553^3:configs/domain_reviewed.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-b43e34c-True-domain_reviewed.json`
- `stash:b43e34c6ae804ddb7ce264b9803a33be46529553^3:configs/domain_v2_reviewed.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-b43e34c-True-domain_v2_reviewed.json`
- `stash:b43e34c6ae804ddb7ce264b9803a33be46529553^3:configs/real_adaptation.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-b43e34c-True-real_adaptation.json`
- `stash:fc8ab8fc07e5c27700a5d49ce0eeec40884da00f:configs/baseline.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-fc8ab8f-False-baseline.json`
- `stash:fc8ab8fc07e5c27700a5d49ce0eeec40884da00f:configs/classification.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-fc8ab8f-False-classification.json`
- `stash:fc8ab8fc07e5c27700a5d49ce0eeec40884da00f:configs/classification_sailor2_8b.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-fc8ab8f-False-classification_sailor2_8b.json`
- `stash:fc8ab8fc07e5c27700a5d49ce0eeec40884da00f:configs/classification_sailor2_8b_8epochs.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-fc8ab8f-False-classification_sailor2_8b_8epochs.json`
- `stash:fc8ab8fc07e5c27700a5d49ce0eeec40884da00f:configs/correction.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-fc8ab8f-False-correction.json`
- `stash:fc8ab8fc07e5c27700a5d49ce0eeec40884da00f:configs/encoder_bamibert.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-fc8ab8f-False-encoder_bamibert.json`
- `stash:fc8ab8fc07e5c27700a5d49ce0eeec40884da00f^3:configs/classification_sailor2_8b_focal.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-fc8ab8f-True-classification_sailor2_8b_focal.json`
- `stash:fc8ab8fc07e5c27700a5d49ce0eeec40884da00f^3:configs/classification_sailor2_8b_smoothing.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-fc8ab8f-True-classification_sailor2_8b_smoothing.json`
- `stash:fc8ab8fc07e5c27700a5d49ce0eeec40884da00f^3:configs/classification_sailor2_8b_supcon.json` → `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/agents/private/recovered-notes/stash-fc8ab8f-True-classification_sailor2_8b_supcon.json`

- 2026-10-07 `045f441`: Initial commit
- 2026-10-07 `0f7e8f7`: Add Vietnamese GEC LoRA baseline and verified Kaggle branch workflow
- 2026-10-07 `a4818d9`: Record GPU and distributed smoke verification
- 2026-10-07 `1715e1e`: Publish Kaggle-ready Qwen3-4B QLoRA baseline
- 2026-10-07 `a7565b3`: Centralize experiment configuration and save adapters under runs
- 2026-10-07 `519fad1`: Add balanced training pipeline and HiGANplus-compatible local notebook
- 2026-10-09 `a553229`: Freeze mixed-data Qwen3 classification and Sailor2 correction baseline
- 2026-10-09 `49d4dcc`: Split classifier head and LoRA learning rates for alpha ablation
- 2026-10-09 `c62f5f0`: Record matched head LR ablation and precision sensitivity
- 2026-10-09 `44be38a`: Record lambda-two soft hierarchy and matched ablation evidence
- 2026-10-09 `0bafd90`: Add fixed R-Drop to soft hierarchical QLoRA classification
- 2026-10-09 `b48652f`: Add reference-edit localization auxiliary to hierarchical QLoRA classifier
- 2026-10-09 `c510663`: Record matched R-Drop GPU result with no label-level gain
- 2026-10-09 `3dfeeb2`: Point integrated Kaggle notebook at rquq/test
- 2026-10-09 `e246781`: Merge branch 'rquq/alpha' into rquq/test
- 2026-10-09 `232aa5a`: Record completed alpha and beta matched experiment comparison
- 2026-10-09 `3d90b5f`: Record matched localization GPU result and serving verification
- 2026-10-09 `f7f5a4e`: Retain five-epoch Sailor classification experiment and record ensemble results
- 2026-10-09 `5abb3de`: Merge alpha experiments into test; retain five-epoch Sailor baseline
- 2026-10-09 `92b100b`: Record integrated five-epoch Sailor GPU verification
- 2026-10-09 `55f2cac`: Audit training bundles and remove rejected R-Drop experiments
- 2026-10-11 `c10586e`: Prepare Sailor2 baseline for Kaggle Linux experiments
- 2026-10-11 `67ac63c`: Fix baseline-only tokenizer and epoch-budget placement
- 2026-10-11 `3d86741`: Record clean-clone Kaggle baseline smoke verification
- 2026-10-11 `d501948`: Trim Kaggle branch to Sailor2 classification baseline
- 2026-10-11 `29b063f`: Restore connected Kaggle two-phase Sailor2 training with dual-GPU DDP
- 2026-10-11 `a666667`: Update correction recipe to eight epochs and revised prompt
- 2026-10-11 `bae464d`: Clarify local Kaggle dataset upload package and Add Input steps
- 2026-10-11 `24dcc31`: Authenticate Kaggle notebook private GitHub clone and fetch
- 2026-10-11 `78e99a2`: Expose sanitized Git bootstrap failure diagnostics
- 2026-10-11 `a8f9c47`: Require kernel restart for replaced scientific stack before preflight
- 2026-10-11 `0d9ee4a`: Simplify Kaggle reruns with fresh non-destructive source clones
- 2026-10-12 `e5e34b3`: Isolate Kaggle training dependencies for unattended Save Version
- 2026-10-12 `c014bfe`: Skip ensurepip in Kaggle training environment bootstrap
- 2026-10-12 `42c0720`: Release correction callback GPU owners before saved adapter reload
- 2026-10-12 `3918ee3`: Add portable Kaggle task modes and saved correction experiments
- 2026-10-12 `0b9a9b9`: Integrate private W&B tracking and local two-phase notebook
- 2026-10-12 `4824ec1`: Align all branches to canonical Kaggle/local implementation while preserving history
- 2026-10-12 `972387e`: Clarify local notebook single-GPU default
- 2026-10-12 `981d5a8`: Document successful offline Kaggle saved-run tracking smoke
- 2026-10-12 `9ab2d1d`: Use Sailor-7B-Chat with run submissions and Kaggle W&B secrets
- 2026-10-12 `be14981`: Integrate public test inference in local and Kaggle pipelines
- 2026-10-13 `c87ad98`: Clean W&B logs and stage results for local and Kaggle
- 2026-10-13 `4b3077a`: Reuse correction base for selected-adapter evaluation
- 2026-10-13 `6d2f6b9`: Run committed Kaggle recipes with validated submissions and T4 recovery
- 2026-10-13 `47f3a23`: Keep notebook cells minimal and move environment setup into helper

## Approved new local review on 2026-10-13

LOCAL_REVIEW_RESULTS.md / .json record the Qwen 64-attempt pilot, native Sailor second opinion, two Bami hierarchical fits and the reviewed corrector handoff. No historical training was rerun. Both teacher protocols and Bami promotion were negative; only five original clean annotations enter the prepared corrector data. Bami internal Macro-F1 was 0.408220 control versus 0.247619 candidate; selected control holdout was 0.325577 versus saved Sailor 0.434133. Failures include two interrupted quality passes, FP16/FP32 reload mismatch recovered without retraining, native single-file mmap failure recovered by unchanged streaming shards, and corrected hardware/mock preflight probes. Real tiny reviewed-v4/classifier reuse passed. Full-size corrector training remains pending on Kaggle.

The subsequent Qwen/SeaLLMs substitution request received a native size/tokenizer/license audit, with no weight downloads or fits. Both are size eligible; SeaLLMs drops the existing `question` role and its linked license restricts improving other LLMs with its outputs. The current recipe retains Sailor. Evidence and source links: REPLACEMENT_REVIEW.md and private `runs/audit-seallms-20261013/audit.json`.

The user then explicitly requested Qwen for both classifier and corrector, with Kaggle edits, commit and push. The recipe now trains two fresh Qwen2.5-7B-Instruct adapters without folds/hints/toggles; it does not reuse Sailor. QWEN_BOTH.md/.json records native parameter audits, unchanged data/config hashes, actual tokenizer/data checks, 141 passing tests, a real tiny pipeline and T4x2 placement simulation. Full-size Kaggle training and scores remain pending.

Following the historical epoch review, the user explicitly requested pushing six corrector epochs, a readiness recap and a Kaggle upload ZIP. KAGGLE_READY.md/.json records this final5/6-epoch recipe and the396,081-byte data-only private archive, with original data hashes retained. New archive/provenance/tokenizer/T4-simulation checks and141 tests passed; no full-size training or additional teacher inference ran.


## Authorized local smaller-model screening — 2026-10-14

LOCAL_SMALL_MODELS_20261014.md/.json records nine unique architectures,13 screening configurations,12 completed fits, one aborted divergence and two primary selected refits and three exploratory secondary refits. Source commits, immutable model revisions, data hashes, budgets, measured metrics, failures/recoveries and saved-checkpoint verification are retained. Prior training was not rerun to reconstruct history. Combined reporting now matches all-row official scope while retaining error-only diagnostics. LEADERBOARD_20261014.json records the user-supplied third KANO score/drop. HYBRID_20261014.json records the eligible component-preserving Sailor7B/Qwen7B archive and conditional expected0.5423; it and five native candidate archives are unsubmitted. No corpus/synthesis promotion or notebook toggles.


## User public-result PNG follow-up — 2026-10-14

PUBLIC_RESULTS_20261014.md/.json imports the six PNGs placed in candidate folders, preserving screenshot/ZIP hashes and all historical results. Sailor/Qwen0.5423 matches the hybrid prediction and is best supported among these results; native classifier Macro scores decline. ViT5 trades lower GLEU for higher exact match. Five candidate associations are consistent, while Bami/ViT5 attribution is unresolved because the PNG repeats Bami/Qwen correction scores despite174 differing verified outputs. No old/new training was rerun. Active records now link this evidence; prior unsubmitted statements above describe the earlier preparation snapshot.


## Validation export and back-translation eligibility — 2026-10-14

VALIDATION_EXPORT_20261014.md/.json records the requested private81-row validation-only JSON/ZIP, exact original/bundle/order/ID/hash checks and rule review. Broad synthetic-generation permission supports round-trip translation in principle under per-generator8B/license/disclosure constraints; method permission is inferred, not explicitly named. Sol6.1 size eligibility is unverified, so it is excluded as augmentation teacher. NLLB metadata audit counts615073792 parameters; no weights, generation or training. Validation stays evaluation-only; future augmentation uses training families exclusively.


## Corrected training-dataset interpretation — 2026-10-14

The user clarified they wanted all81 validation rows included in the new training dataset. FULL_TRAINING_20261014.md/.json records the corrected classifier585/corrector495 raw arrays and ZIP, with exact81-row membership in both. The validation-only export was the wrong interpretation and remains historical. Baseline data/runs are preserved; old81 are not independent validation for the new dataset.22 added corrector targets are retained and flagged. No new augmentation/training or Kaggle recipe change occurred.


## Standard LoRA full-training preparation — 2026-10-14

SAILOR_QWEN_LORA_20261014.md/.json records the user-requested Sailor7B classification / Qwen7B correction LoRA recipe, FP16 frozen bases and FP32 trainable adapters/head, fixed5/6 epochs and full585/495 populations including all81 former holdout rows. New source-validated final-training schema/ZIP, one-process dual-T4 layer splitting, AMP head recovery and null-score notebook handling are implemented. Real CPU/CUDA tiny fits/reloads/ZIPs and native architecture/memory simulations pass. Full-size Kaggle training and public scores remain pending;22 added target flags are retained, not inferred repairs. No historical training was rerun.


## Translation model recommendation and native size audit — 2026-10-14

TRANSLATION_MODELS_20261014.md/.json recommends Apache-2.0 Hy-MT2-7B and its1.8B sibling for a pilot. Native meta audits verify7,504,568,320 and1,791,080,448 unique parameters with tied embeddings. The initial stored-tensor8.029B exclusion was corrected. No model weights, generation, training or active recipe change occurred; corpus quality and downstream improvement remain unmeasured.


## Separate Kaggle augmentation preparation — 2026-10-14

AUGMENTATION_20261014.md/.json records the user-requested kaggle-augment branch, preserving the585/495/all81 baseline and training branch030cb13. The initial recipe used96 parents,200 accepted-error caps plus matched clean controls,1344 calls/360 active minutes, Hy-MT2-7B round trips, Qwen7B minimal proposals and independent blind Sailor7B review. The actual dual-GPU FP16 run then completed1071 calls, translated148 references and authored249 candidates, but Sailor returned no parseable JSON in391 review responses. All249 failed strict review gates, leaving zero accepted pairs and zero added rows. See AUGMENTATION_RUN_20261014.md/.json. Follow-up source commit969f22c adds one format-only retry and raises the call ceiling to1536; it preserves precision and every quality gate. Local tests and a toy CPU smoke pass, but a fresh full-size yield/quality run and downstream score remain pending.


## Augmentation phase-memory failure and worker isolation — 2026-10-14

AUGMENTATION_MEMORY_FIX_20261014.md/.json records the first user Kaggle log:96 parents/148 translated references then Qwen startup memory precheck fails9.30GiB needed/7.36GiB free. Exact retained owner was unknown. Worker isolation was followed by a successful full-size dual-GPU FP16 run with all three models exiting cleanly and no OOM. Sailor's review output format—not memory—then caused zero accepted augmentation. Follow-up code commit969f22c asks for strict JSON and allows one schema retry; the repeat full-size quality/yield run is pending. No score gain is claimed.

## Full-size augmentation run and reviewer output-contract diagnosis — 2026-10-14

The completed private run at `runs/augment/augmentation-runs/hy-qwen-sailor-reviewed-roundtrip-20261014T094859674639Z` was sourced from clean commit15d2694093e9b6e758bf8cd36d11a009c3c8ca6d. Dual GPUs each exposed15,636,037,632 bytes; Hy-MT2-7B, Qwen2.5-7B-Instruct and Sailor-7B-Chat loaded sequentially in unquantized FP16. The job completed1071 calls in8272.5 seconds without OOM. It made148 round-trip references and249 authored candidates. All142 clean and249 blind error review outputs were non-JSON prose; every candidate failed both independent review gates. The ZIP is valid but contains zero accepted pairs and zero additional classifier/corrector rows. No training or score evaluation ran.

The code follow-up is commit969f22c99c7a3f6f51d90b2d1434217c85ad586e. It preserves strict parsing and data-quality gates, explicitly asks Sailor for a single JSON object, retries a malformed/schema-invalid response once, records both call IDs and sets the maximum to1536 calls.11 focused augmentation tests,178 repository tests and the CPU tiny-model smoke passed; those are protocol/mechanics checks, not proof of improved real-data output. New run record: AUGMENTATION_RUN_20261014.md/.json.

## Full-size augmentation retry — 2026-10-14

AUGMENTATION_RETRY_20261014.md/.json records the 1,462-call run from clean commit018a9fbe9ea4ed9ad278cabaf643c069e3167ae4. Dual-GPU native FP16 completed without OOM in12,703 seconds. It translated148 references and authored249 candidates. The format-only retry still failed: all142 final clean reviews were schema-invalid; only187 of249 final error reviews had type-valid fields, and just10/5/13 agreed with the candidate label/exact mechanism/reference correction respectively. All249 candidates failed independent review; mechanical edits included111 overbroad examples and other taxonomy/protected-content failures. The verified ZIP is baseline-only:585 classifier/495 corrector rows, zero accepted pairs or added rows. No training or score was measured. Keep all strict gates and pause another full run until a small saved-output pilot demonstrates reliable review behavior.


## Saved-output recovery — 2026-10-15

AUGMENTATION_RECOVERY_20261015.md/.json records recovery of the completed148-reference/249-proposal run. Mechanical checks retain51 candidates across47 references and reject198 before inference. The pinned private ZIP carries unchanged585/495 baselines and saved journal proof. Only Qwen3-4B-Instruct-2507 is loaded in Kaggle native FP16; translation and authoring perform zero calls. Hard JSON/bounded source-span decoding preserves exact correction and every original review gate. Eight exposed-original operational controls stop the job early on failure. Budget256 calls/90 minutes; laptop NF4 diagnostics are not promoted. Every failed local probe and verification limit is retained in the new record; no training/public score is claimed.

Recovery implementation8cd7de0 passes202 repository tests,47 augmentation tests under Transformers5.6 and an actual CPU native tiny-model constrained-decoding/recovery smoke. The committed CLI passes reviewer-only meta preflight and fails cleanly/resumes without extra calls using the actual private payload plus a tiny fixture. Twenty fresh local NF4 reviewer calls across seven probes still fail the known type1 correction; no candidate inference or training promotion followed. Native FP16 pilot and accepted real yield remain pending.


## Augmentation held after local cross-check — 2026-10-15

The user rejected further Kaggle risk without local review results. LOCAL_AUGMENTATION_SCREEN_20261015.md/.json records actual inference with four saved native checkpoints in122.9 seconds,179 rows/model. On51 candidates, Bami/Qwen0.5 subtype agreements are4/7; ViT5/BARTpho exact proposed references are10/2. Both classifier agreement and both corrector exact reference hold for0/51. Their exposed-original error diagnostics are weak too, so disagreement does not adjudicate labels. No new rows or training/Kaggle run occurred. The next-run recommendation was withdrawn; require positive local quality evidence and a small grouped data comparison before another Kaggle recommendation. Preserve the best-supported hybrid and all original artifacts.


## Retention-first recovery proposal —2026-10-15

The user requested review and a better written plan, not execution. RETENTION_RECOVERY_PLAN_20261015.md/.json records a proposed immutable401-row master,393-original classifier view after8 user exclusions,11 note-target corrector exclusions,32 complete-source/80 source-abbreviated recovery cases,192 retained code-synthetic baseline rows, qualified≤8B local teacher candidates and explicit class0 exposure/grouped local comparison gates. It distinguishes preserving a row from inventing a target, and grammatical test errors from unverified hidden-label errors. No model download, generation, dataset promotion, training or Kaggle run occurred.
