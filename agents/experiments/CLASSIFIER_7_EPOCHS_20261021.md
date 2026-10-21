# Seven-epoch classifier trial — October21

Prepared, not run. The user explicitly requests seven training epochs for all three classifiers and authorizes commit/push on `kaggle-linux`. This supersedes the five-epoch classifier budget in UPGRADED_ENSEMBLE_20261020.md/.json; those preparation records remain historical.

The active default remains configs/experiment.json → configs/upgraded_training_plan.json → configs/upgraded_training_recipe.json. In configs/upgraded_classification.json, change training.args.num_train_epochs and each ensemble member's training_args.num_train_epochs from5 to7. Explicit member overrides must change along with the shared default or the members would still train for five epochs.

|Classifier|Full epochs|Learning rate|
|---|---:|---:|
|Qwen/Qwen2.5-14B-Instruct|7|3e-5|
|sail/Sailor2-20B-Chat|7|3e-5|
|mistralai/Mistral-Nemo-Instruct-2407|7|5e-5|

Each member still selects its own best development Macro-F1 checkpoint after completing the full training budget; using the seventh checkpoint is not mandatory. The cosine learning-rate schedule spans the longer fit, so extending epochs also changes the schedule's progress per update. Keep all other configured hyperparameters, pinned bases, seed42, QLoRA, lambda2 hierarchy and weighted SOFT vote1:1:2 unchanged. Qwen14B correction remains six full epochs with its existing evaluation schedule and decoding settings.

Keep the same quality training payloads,78 development rows and2034 private inference rows, with no data edits or new uploads. The existing Kaggle launcher clones the branch afresh, so Save Version / Save & Run All picks up the change; notebook code cells are unchanged. Only its explanatory markdown and current README are updated. The original1038 quality archive, saved candidates and prior runs remain preserved.

Training/inference and native GPU checks are not run during this preparation. Static checks cover JSON parsing, effective member epochs, unchanged corrector and other classification settings, unchanged notebook code cells and a clean patch. Seven rather than five epochs increases the classifier epoch budget by40%; total session runtime and task quality remain unmeasured, and the previous10–12h whole-session estimate for five-epoch classifiers does not establish that this longer recipe fits Kaggle's external limit. No controller timeout or automatic submission is added.
