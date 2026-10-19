# Track1039: larger models on two T4s

Current recipe: `configs/experiment.json` selects `unconstrained_plan.json`. Experiments are controlled by chat/config commits; notebooks contain no experiment toggles.

1. Upload `dataset/kaggle-track1039-private-20261019/VGEC26-track1039-private-20261019.zip` as a **PRIVATE** Kaggle dataset. Attach that version alone; the extraction folder name can vary. No older notebook output is needed.
2. Import `notebooks/kaggle_two_phase.ipynb`, enable Internet and **GPU T4 x2**, and provide repository-read `GITHUB_TOKEN`. `HF_TOKEN` is optional for these ungated bases; `WANDB_API_KEY` enables monitoring.
3. Save Version / Run All. The launcher reads the committed recipe and verifies all dataset/baseline hashes. Download `run/results.zip` for the2,034 private IDs; retain the full run output and selected model bundle.

| Challenger | Model | Precision | Epochs | LR |
|---|---|---|---:|---:|
| Classifier | sail/Sailor2-20B-Chat | NF4 QLoRA / FP16 compute |5 full|3e-5|
| Corrector | Qwen/Qwen2.5-14B-Instruct | NF4 QLoRA / FP16 compute |5 full|2e-5|

Both use seed42, split42, rank16/alpha32/dropout.05, maximum512 tokens, microbatch1/effective batch8, nonreentrant gradient checkpointing, FP32 trainables and AdamW. One process explicitly splits each model across both GPUs; fits/reloads run sequentially. Classifier loss remains weighted binary detection plus lambda2 subtype supervision. Retain the best earlier checkpoint after all5 epochs. Correction uses minimal-edit / preserve-clean prompting, no hints, greedy192-token decoding. No folds, voting requirement or beam sweep.

The package also contains the verified October19 Sailor7B classifier and old Qwen7B greedy corrector. Each new component must beat its corresponding baseline on the exact same78 development rows; selection can retain either baseline. The new challengers are not assumed to win. Training remains the unchanged organizer-anchored651/489 exposures, with321/309 original training rows. The78 development rows and2,034 private inputs remain excluded from fitting. No new model-authored examples, target rewrites or private-test supervision.

The pipeline keeps root W&B run `run` alive through training/decoding/export, reporting numeric quality/progress and CPU/RAM/GPU/VRAM/temperature/power/disk/network metrics. Stage processes and release gates prevent retaining two live models. Track1039 alone uses an owned temporary Hugging Face weight cache, outside saved outputs, retaining one base at a time; base changes may redownload weights. Frozen checkpoints need roughly36GiB of download space at peak; keep at least40GiB free on the temporary-cache filesystem. User caches, original runs, adapters and quality backups are never cleanup targets. Native memory estimates are about11.2/11.4GiB per T4 including training reserves; these are estimates, not measured20B/14B training peaks. A full native two-T4 run is still required.

## Results and preserved track1038 experiment

October19 selected development scores: Sailor7B Macro68.9874%, Micro83.3333%; old Qwen7B greedy all-GLEU63.1335%, error-GLEU23.7716%; combined0.660604. Four native fits/decoding alternatives from earlier comparisons are historical; the fresh focused run trained two models. Full greedy/beam2 comparisons favored old greedy. See `agents/experiments/OCTOBER19_RESULTS_20261019.md` and `TRACK1039_20261019.md`. Repeated-development selection is not an official/private score.

The quality-data track1038 experiment reported about0.58 combined by the user is **frozen**, at source tag `archive/track1038-quality-20261019` (338293f). Private source/docs/runtime/submission backups: `backups/track1038-quality-20261019/`; original quality data/run/adapters remain in place and hash-inventoried. See `agents/experiments/PRESERVED_QUALITY_TRACK1038_20261019.md`. Extract the source ZIP into a separate directory for historical reporting; do not substitute current configs. Its measured development combined score is0.58926286, a different scope from the reported leaderboard score.

The preceding focused7B recipe remains `configs/constrained_focused_experiment.json`; the earlier quality recipe is retained separately and exactly recoverable from the archive tag. Configs without `competition_id` retain track1038 and its8B native-parameter ceiling. Larger-model configs explicitly use1039; never submit those models as1038 entries.

## Verification

Run `python -m unittest discover -s agents/tests -p 'test_*.py' -q`. A tiny CPU controller check is `python -m agents.tests.smoke_focused --plan configs/unconstrained_plan.json --output <new-private-directory>`; it performs four actual five-epoch fits/reloads and export using artificial data and small, unquantized CPU models. It does not validate native multi-GPU QLoRA performance. Exact model revisions, data/proof hashes and native meta/tokenizer checks are documented in the paired experiment records. Corpus resources, backups, checkpoints, submissions and generated comparisons remain private/ignored.
