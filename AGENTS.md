# Alpha preservation workspace

The active Kaggle experiment lives in ../kaggle-linux. Preserve this checkout's Git identity, historical runs and original data. The five-fold Sailor-7B pipeline is frozen under preserved/kaggle-fold-hints-47f3a23 and tag alpha-fold-hints-20261013; its notebook pins that source. Do not change its recipe when editing the next no-fold experiment.

Document every user experiment, negative result and failure under agents/experiments. Read agents/docs/DATA_GUIDELINES.md before data/teacher work. Archived corpus-bearing notes and private data remain ignored. Recover history from evidence without rerunning old training. Historical Sailor2 scores are retained as evidence and exceed the strict 8B limit for competition 1038.

Completed active-branch outcomes are mirrored in agents/experiments/LOCAL_REVIEW_RESULTS.md/.json; REPLACEMENT_REVIEW.md records Qwen/SeaLLMs size, tokenizer and license limits. These records do not change alpha runtime or its frozen snapshot.
