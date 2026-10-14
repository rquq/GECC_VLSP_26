# Private Kaggle dataset augmentation

Branch `kaggle-augment` creates a new dataset from the approved full-training population: **585 classifier rows and495 corrector rows**, including all81 former local validation rows. It preserves every baseline row and appends accepted error/clean pairs. It runs inference and exports data; classifier and corrector training stay on `kaggle-linux`.

Import `notebooks/kaggle_two_phase.ipynb`, enable Internet, select **GPU T4 ×2**, attach the private `VGEC26-sailor-qwen-lora-full-training-20261014.zip`, and enable the repository-read `GITHUB_TOKEN` secret. `HF_TOKEN` is optional. Run All. The upload name/enclosing folder can vary: input discovery checks canonical filenames and committed hashes. No public/private test input is needed. The input ZIP may contain public_test.json; this pipeline does not read its contents.

The existing notebook launcher clones this branch, installs its isolated inference environment while retaining Kaggle CUDA PyTorch, then runs `python -m tools.run_augmentation`. All experiment choices are committed in `configs/augmentation.json`; there are no notebook experiment toggles.

## Committed pipeline

| Phase | Model | Native parameters | Purpose |
|---|---|---:|---|
| Translation | tencent/Hy-MT2-7B | 7,504,568,320 | Vietnamese → English/Chinese → Vietnamese |
| Authoring | Qwen/Qwen2.5-7B-Instruct | 7,615,616,512 | Propose a single minimal taxonomy error |
| Review | sail/Sailor-7B-Chat | 7,721,324,544 | Independently assess reference equivalence, infer subtype and correction blindly |

All three revisions are pinned, use native FP16 weights, and run sequentially, each in its own spawned worker across two GPUs. Hy's embedding/output weights must remain tied. Native counts and placement are audited before the first weight download and checked after each load. Each model's temporary cache is removed on normal phase completion. There is no fine-tuning, quantization, CPU offload, fold training or hint pipeline in this launcher.

The recipe selects at most96 eligible original parent references from the two retained training populations. Incomplete/annotation-style targets and legacy synthetic parents are excluded from generation. The22 flagged correction rows remain in the baseline. Two pivots and two proposals per new reference provide varied contexts rather than repeating one rare-class pattern.

Accepted error caps are24 sentence-structure,64 phrase-structure,64 function-word and48 punctuation examples. Caps also limit each of six mechanisms per class, four errors per parent and two per reference. Each unique accepted reference contributes one matched clean control. The maximum addition is200 errors plus200 controls, giving at most985/895 rows; actual output can be much smaller, including zero accepted additions. There is no quota filler.

Every accepted pair needs finite confidence≥0.9, agreement with a different reviewer model on the exact mechanism, one required error, a minimally corrected reference, preserved meaning/names/numbers, valid provenance, and mechanical span/taxonomy gates. The reviewer receives the error source without the author's intended label, mechanism or target. This is automated review, not human adjudication or proof of correctness. No Codex-authored corpus examples or inferred replacement labels enter the dataset.

The budget is **1,344 inference attempts and360 active minutes** across resumes. After a hard interruption, elapsed time since the last checkpoint is conservatively charged, including downtime. Model loading counts toward active elapsed time; environment setup and metadata preflight do not. A single generation step or download can overrun a time boundary, so this is a bounded inference budget rather than a strict wall-clock cancellation deadline. Failed/interrupted inference attempts consume the call allowance. Only fully reviewed candidates are exported when the budget expires.

## Outputs and reuse

The final notebook link is `VGEC26-augmented-full-training.zip` under `/kaggle/working/augmentation-runs/<run>/dataset/`. It contains the canonical original corpus, augmented classifier/corrector bundles, raw training arrays, accepted/rejected pair evidence and a dataset manifest. The run separately retains exact prompts, generated outputs, hashes, generation settings, model audits, selected parents and phase diagnostics. Keep both the Kaggle notebook and its outputs private.

`vgec-augmented-final-v1` embeds the unchanged baseline and validates its pinned recipe hash/count/original, exact baseline prefix, parent origins and accepted evidence on load. Export validates every bundle and ZIP before atomic publication. Resume recovers incomplete staging and can reuse a published export if interruption occurred before the outer run manifest was written.

The output is **a training dataset ZIP**, not a competition submission ZIP. Before training with it, deliberately update the training branch's dataset consumer and committed input hashes; the current `kaggle-linux` recipe still consumes its previous package. Preserve family groups for any fresh internal validation. The former81 rows are training rows and cannot provide independent validation for this dataset. Augmentation quality and a downstream score gain remain unmeasured until real generation and controlled training results exist.

## Checks and local use

`python -m tools.run_augmentation --preflight-only` checks the committed inputs and native architecture/device placement without loading full weights. `python -m tools.run_augmentation --resume-run <saved-run>` resumes an unchanged source/recipe/input identity. Resume is a recovery CLI, not a notebook experiment switch. Full-size inference requires two visible GPUs; the8GB laptop is used for audits and tiny-model mechanics.

`python -m unittest discover -s agents/tests` runs the historical regression suite in the preserved training environment. The focused augmentation tests and `python -m agents.tests.smoke_augmentation --device cpu` / `--device cuda` run in the new Transformers5.6 inference environment. CUDA smoke uses an actual tiny model on the available single laptop GPU; only its placement is adapted. Native7B counts and dual-T4 memory maps use real pinned metadata with simulated hardware, not a completed full-size Kaggle run.

Every experiment and failure belongs in `agents/experiments/`. Read `AGENTS.md`, `agents/README.md`, `agents/experiments/CURRENT.md` and `agents/docs/DATA_GUIDELINES.md` before changing this pipeline. Code and aggregate records can enter Git; original/derived corpus, prompts, generated text and model artifacts remain private and ignored. The competition1038≤8B per-component/generator rule, model licenses and corpus noncommercial/no-redistribution agreement continue to apply.

The Kaggle translation→authoring transition reported insufficient free memory after148 translated references. Read `agents/experiments/AUGMENTATION_MEMORY_FIX_20261014.md`/.json. Model weights now live in separate spawned workers, with exit verified before the next model and memory/PID diagnostics retained. Native FP16, revisions, prompts, generation settings and baseline hashes remain unchanged; full-size rerun remains pending.
