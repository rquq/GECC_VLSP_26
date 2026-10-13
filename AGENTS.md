# Experiment workflow

Read `agents/README.md`, `agents/experiments/CURRENT.md`, and the relevant history before changing training, data, or inference. The user's current preference is a corrector-only Kaggle run reusing the saved Sailor classifier, without folds or hints, with experiment choices made in chat and commits and no notebook toggles.

Record every user-requested experiment in `agents/experiments/`: intent, source commit, exact config/data hashes, budget, status, results, validation scope, and artifact location. Distinguish planned, implemented, started, completed, failed, and recovered historical claims. Update the record when the outcome arrives, including failures and negative results. Do not invent scores or rerun old training to fill documentation gaps. Runtime launcher events are stored in ignored `agents/experiments/local.jsonl` and copied into each run.

Original corpus/holdout and saved runs must be preserved. Corpus-bearing notes, repair overlays, synthetic data, and generated text stay private and ignored. Follow `agents/docs/DATA_GUIDELINES.md` before preparing a dataset or using a teacher model. Historical models exceeding 8B are evidence, not eligible components for competition 1038.

The approved local teacher and Bami screens are recorded in agents/experiments/LOCAL_REVIEW_RESULTS.md and .json. Do not silently promote their failed synthetic candidates or the weaker Bami checkpoint. Keep notebook choices committed and the saved-classifier handoff private.

Read agents/experiments/REPLACEMENT_REVIEW.md before a Qwen/SeaLLMs substitution. Size eligibility does not override a generator's license or tokenizer interface. SeaLLMs outputs must not enter Sailor/Qwen training under its linked license. Keep the current Sailor recipe until a measured replacement is selected.
