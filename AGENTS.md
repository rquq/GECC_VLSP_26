# Experiment workflow

Read `agents/README.md`, `agents/experiments/CURRENT.md`, and the relevant history before changing training, data, or inference. The user explicitly requested fresh Qwen2.5-7B-Instruct training for both classification and correction on Kaggle. Keep no folds or hints, experiment choices in chat and commits, and no notebook toggles.

Record every user-requested experiment in `agents/experiments/`: intent, source commit, exact config/data hashes, budget, status, results, validation scope, and artifact location. Distinguish planned, implemented, started, completed, failed, and recovered historical claims. Update the record when the outcome arrives, including failures and negative results. Do not invent scores or rerun old training to fill documentation gaps. Runtime launcher events are stored in ignored `agents/experiments/local.jsonl` and copied into each run.

Original corpus/holdout and saved runs must be preserved. Corpus-bearing notes, repair overlays, synthetic data, and generated text stay private and ignored. Follow `agents/docs/DATA_GUIDELINES.md` before preparing a dataset or using a teacher model. Historical models exceeding 8B are evidence, not eligible components for competition 1038.

The approved local teacher and Bami screens are recorded in agents/experiments/LOCAL_REVIEW_RESULTS.md and .json. Do not silently promote their failed synthetic candidates or the weaker Bami checkpoint. Keep notebook choices committed and the saved-classifier handoff private.

Read agents/experiments/REPLACEMENT_REVIEW.md before a Qwen/SeaLLMs substitution. Size eligibility does not override a generator's license or tokenizer interface. SeaLLMs outputs must not enter Sailor/Qwen training under its linked license. The user-authorized Qwen run completed both full-size stages; read agents/experiments/QWEN_RUN_20261013.md/.json for measured local classification improvement, mixed correction results and post-training export recovery. A new official score remains pending. Preserve the earlier Sailor artifacts.
