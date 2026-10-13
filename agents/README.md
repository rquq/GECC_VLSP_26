# Preserved experiment workspace

Use preserved/kaggle-fold-hints-47f3a23 for the reserved five-fold Sailor-7B pipeline. Its source/tag and private bundle hashes are in PRESERVATION.json; selected weights are retained with verified hard links and selected metadata is copied without duplicating tensor storage. Existing exploratory alpha runs remain intact.

The next Kaggle run in ../kaggle-linux is independent correction with no folds or label hints. Follow that checkout's AGENTS.md and agents/experiments/CURRENT.md. Record all future experiments under agents/experiments, with exact configs/data hashes, budget, status, evidence and outcome. HISTORY.md and history.json recover old records without retraining. DATA_PLAN.md specifies training-only repairs and eligible local-teacher candidates.

Corpus-bearing recovered source notes are private under agents/private. Apply agents/docs/DATA_GUIDELINES.md and the corpus agreement.

The active checkout’s latest Vietnamese-model/data proposal is mirrored in agents/experiments/MODEL_SELECTION.md and MODEL_SELECTION.json. New training and data promotion await the user’s requested approval; frozen source under preserved remains unchanged. Private review artifacts referenced by the proposal live in ../kaggle-linux.
