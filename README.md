# Track1039: fresh20B/14B quality-data retrain

The default `configs/experiment.json` now selects `configs/fast_training_plan.json`: fresh Sailor2-20B classifier and Qwen2.5-14B corrector, five full epochs each, one final private submission. The earlier four-way trial remains in `configs/controlled_quality_experiment.json`.

1. Keep the existing PRIVATE quality-track1039 Kaggle dataset attached. Its extraction folder name may vary; old expanded-control uploads remain supported. For a new upload, use `dataset/kaggle-quality-track1039-private-20261019-json/VGEC26-quality-track1039-private-20261019.zip`. It contains plain data/model files, no nested ZIP/TAR archives.
2. Import `notebooks/kaggle_two_phase.ipynb`, enable Internet and GPU T4 x2. Repository-read GITHUB_TOKEN is required; WANDB_API_KEY enables monitoring; HF_TOKEN is optional.
3. Save Version / Run All. Download `run/results.zip`; it now means BOTH newly trained20B/14B models. Keep the entire output. Nothing is automatically submitted. No notebook toggles or previous notebook output required.

|Setting|Sailor2-20B classifier|Qwen2.5-14B corrector|
|---|---|---|
|Precision|NF4 double-quant QLoRA, FP16 compute|NF4 double-quant QLoRA, FP16 compute|
|Epochs|5 full, best development Macro-F1|5 full, best development all-GLEU then error-GLEU and clean preservation|
|Learning rate|2e-5, cosine|2e-5, linear|
|LoRA rank / alpha / dropout|16 /32 /.10|16 /32 /.075|
|Weight decay|.05|.02|
|Warmup|20 steps|.05 ratio|
|Training microbatch / accumulation / effective|4 /2 /8|2 /4 /8|
|Evaluation|Every epoch, batch8|Epochs2/4/5, generation batch4|

Both use pinned revisions, seed/split42, max512, AdamW and nonreentrant gradient checkpointing. Classifier remains weighted binary + lambda2 conditional subtype loss, argmax joint probabilities. Corrector remains minimal-edit/no-hint, deterministic greedy192, with constrained retry only for malformed JSON. Private correction uses left-padded batches up to4 with smaller-batch fallback on OOM. Batched generation and larger microbatches have not been tested on T4s for this change.

Quality data is unchanged:1128 classifier and1011 corrector exposures;78 development rows stay excluded from fitting. The2034 private sentences are inference-only, not synthetic generation. There is no new annotation repair, data expansion or private supervision.

The old run completed both five-epoch fits, then our41,400-second controller cap killed private correction at856/2034. On the SAME reused78 development rows, quality control combined.5893 versus new20B/14B.6290; this is not an official/private score. Native training-loop runtimes alone were3h27m43s and4h11m6s. This next run removes controller wall/stage caps; Kaggle's own session limit still applies.

For speed, no model preflight/test fit, untrained-baseline generation, postfit saved-checkpoint score replay, decoder sweep, four-way baseline comparison or duplicate selected-model copies. Selected-epoch predictions supply development reports; reload-verification flags remain false. Classify private inputs immediately after Sailor training, before switching the owned base cache to Qwen. Append-only private prediction journals preserve progress; final output order follows exact typed IDs. Root W&B run `run` tracks numeric metrics and CPU/RAM/GPU/VRAM/temperature/power/disk/network through export. Its Logs tab also receives the combined child console stream: progress bars, losses, warnings, errors, stage boundaries and final output path. Credentials are redacted; corpus-bearing lines are withheld. `run/wandb_console.log` saves the sanitized mirror locally; the existing `console.log` remains the full credential-redacted notebook transcript. Bootstrap/pip output before the root run starts is outside this mirror. Only controller-owned temporary base caches are evicted.

Standard LoRA does not safely fit these current models on twoT4s: Sailor classifier FP16 base alone is about34.26GiB; Qwen14B is about27.51GiB before training overhead. Keep QLoRA. No full-model training, tests, smoke checks or new score verification were run for this update, as requested. Runtime savings, higher microbatch fit and quality gains remain unmeasured.

The original0.5806 track1038 quality experiment remains frozen at `archive/track1038-quality-20261019`, source338293f, with protected code/docs/data/adapter backups. All old outputs and partial inference remain intact. See `agents/experiments/FAST_RETRAIN_20261020.md` and `.json` for loss/metric evidence and exact scope.
