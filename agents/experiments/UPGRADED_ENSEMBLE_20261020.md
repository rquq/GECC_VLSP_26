# Fresh upgraded ensemble gamble — October20

Prepared, NOT run. User explicitly requests a new training gamble with the best quality-system mechanism and better models, superseding the earlier recommendation to rely only on already-scored component outputs. No training or inference is executed locally. The previously prepared final component ZIP remains intact as a separate option; no submission slot is consumed by preparation/training and no automatic submission is implemented.

Default configs/experiment.json → upgraded_training_plan.json → upgraded_training_recipe.json. Previous one-classifier default is preserved as configs/fast_single_experiment.json. Same compact VGEC26-fast-quality-private-20261020.zip (2.17MB) remains attached; no new dataset upload required. Same1128/1011 exposures,78 shared development rows fit-excluded,2034 private inputs inference-only. No new synthetic examples or annotation edits. Preserve all source weights, outputs, quality archive/tag/docs and three preexisting edited experiment files.

## Four fresh fits

|Task|Base|Epochs|LR|LoRA dropout|Decay|Micro/accum/effective|
|---|---|---:|---:|---:|---:|---|
|Classifier Qwen|Qwen/Qwen2.5-14B-Instruct|5 full|3e-5|.05|.01|4/2/8|
|Classifier Sailor|sail/Sailor2-20B-Chat|5 full|3e-5|.05|.01|4/2/8|
|Classifier Mistral|mistralai/Mistral-Nemo-Instruct-2407|5 full|5e-5|.05|.01|4/2/8|
|Corrector|Qwen/Qwen2.5-14B-Instruct|6 full|2e-5|.075|.02|2/4/8|

All fresh pinned pretrained bases, NF4 double-quant QLoRA/FP16 compute/FP32 trainables, rank16/alpha32, seven projection targets, seed/split42, max512, AdamW and nonreentrant gradient checkpointing. No prior adapter initialization/reuse. Model-specific member names are qwen14b/sailor20b/mistral12b, not misleading7B aliases. Native model counts remain recorded by normal preparation;1038 component cap remains unchanged. QLoRA is required for the large bases on twoT4s; no standard-LoRA claim.

Classifier mechanism is the preserved quality system's soft hierarchy: binary clean/error probability times conditional subtype probability, weighted binary plus lambda2 conditional CE, sqrt class weighting. Three independently trained model families SOFT vote joint probabilities. Fixed weights1:1:2 in Qwen/Sailor/Mistral order, temperature1, phrase_scale1.25; no calibration sweep, threshold tuning, hard vote or new gate. Each member selects its own best development Macro-F1 epoch among five full epochs. Classifier baseline generation and saved-checkpoint development replay remain disabled; selected-epoch predictions form the ensemble report and saved adapter reload-verification flags remain false.

Classifier dropout/decay restore the quality recipe instead of the failed .10/.05 regularization trial. Sailor3e-5 uses the earlier20B fit's stronger development outcome; Qwen3e-5 and Nemo5e-5 are model-specific hypotheses, not verified improvements. Warmup30steps/cosine. This is not a controlled one-factor experiment: bases, precision versus1038 and optimization differ. No claim that higher parameter count guarantees higher classification F1.

Corrector retains the completed14B recipe, with5→6 epochs as the new gamble. LR2e-5/linear/warmup.05, same no-hint minimal-edit prompt, greedy192, no sampling/repetition penalty1.0, batch4. It evaluates epochs2/4/5/6 and selects all-row GLEU then error-GLEU then clean preservation. Epoch5 remains evaluated because prior development continued improving through5; epoch6 need not be deployed if worse. No baseline/saved-model score-replay/decoder sweep. All private rows are generated, including classifier-clean rows; schema retry remains for malformed JSON and generation OOM reduces batches. Existing tokenizer/target length guards and finite-stat checks remain.

## Schedule and resource choices

Sailor fit → Sailor private classification → Nemo fit → Nemo private classification → Qwen fit → Qwen private classification → cached weighted vote → fresh Qwen corrector fit → private correction → one results.zip. Vote weights are tied to canonical names, not execution order. Each model worker runs sequentially across bothT4s, one model in VRAM at a time. Private predictions are inference-only and never used in corrector fitting. Qwen classifier/corrector share only downloaded pretrained base files, not fitted adapters, and run in separate fresh processes. Keeping Qwen last avoids downloading its14B base again. Only marked controller-owned temporary HF caches may be evicted.

Nemo official metadata: Apache-2.0, ungated, revision04d8a90549d23fc6bd7f642064003592df51e9b3, native causal12,247,782,400 parameters; Mistral architecture40layers/hidden5120/intermediate14336/32attention+8KVheads/head_dim128/vocab131072. Tokenizer is native PreTrainedTokenizerFast, so existing generic-fast loading applies. Its metadata leaves pad_token null but backend reserves <pad> ID10; classification now selects an existing distinct padding token when present instead of resizing vocabulary and saving a huge embedding matrix. No local tokenizer/model execution verified this new checkpoint. Model-card multilingual benchmarks do not establish Vietnamese VGEC superiority. Nemo12B is a third-family candidate selected for additional capacity while limiting session/cache cost versus24B alternatives.

Prior actual T4 evidence: Sailor20B classifier micro4 and Qwen14B corrector micro2/generation4 completed one run, root~7h35m. The extra two classifier fits and extra corrector epoch add time; new full runtime, Nemo/Qwen classifier memory peaks and ensemble score are UNMEASURED. No artificial controller deadline; Kaggle's own session cap still applies. Do not claim allfour fits are guaranteed to finish or that a memory allocation chart proves standard-LoRA feasibility. Native placement/data integrity and GPU release checks remain, but there is no separate model preflight/smoke/testing stage.

W&B root remains run through export with member-specific metrics, console mirror and resource monitoring. Training logs map to classifier-qwen14b/classifier-sailor20b/classifier-mistral12b; private stage names identify the active member. Member completion events are explicitly published, correcting missing fit-only completion counts. Corpus/targets/raw generations stay out of W&B. log_models=false retains local adapters without uploading them. Notebook stays a simple launcher with choices committed in config/code, no toggles.

The exported upgraded ensemble descriptor binds allthree adapters with actual member names; model.classification.classify dispatches to its explicit portable protocol. Historical7B ensemble paths/rules remain separate. Strict saved-model verification gates are not weakened and skipped development replay is not called verified. Future full-model replay still requires all saved member weights and bases.

No tests, smoke runs, native model preflight, training or inference performed for this update, following standing user instructions. Read official model metadata/cards only. The new ensemble is a speculative challenger, not a validated replacement for0.5806. After Kaggle, retain the full output and review actual completed metrics/submission before using the one remaining private slot.

Sources: [Qwen14B](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct), [Sailor2-20B](https://huggingface.co/sail/Sailor2-20B-Chat), [Mistral-Nemo12B](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407). Model-card capacity/benchmarks inform candidate choice; task-specific improvement is untested.
