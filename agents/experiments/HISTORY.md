# Recovered experiment history

Recovery uses surviving run artifacts, archived agent notes, Git history, and maintenance records. No historical training was rerun. Private recovered notes can contain corpus text and remain ignored by Git.

## Surviving measured artifacts

| Run | Macro-F1 | Micro-F1 | Evidence |
|---|---:|---:|---|
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
