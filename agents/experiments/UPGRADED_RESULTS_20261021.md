# Completed upgraded ensemble — October21

Reviewed the user-downloaded runs/21_10_2026/runs/track1039-upgraded-soft-vote-20261020-20261020T144909923321Z. The source manifest binds clean commit95b6697: all three classifiers trained five full epochs and Qwen14B correction trained six. This is the predecessor of the newly pushed seven-epoch classifier configc7fee39, which remains unchanged by this review. No official private score has been supplied for this run.

## Completion and resources

All eight model stages completed. Root W&B runtime32,742seconds (9h05m42s), before notebook dependency setup; private correction generated all2034 rows with batch8 throughout. Console contains no traceback, CUDA OOM or timeout markers and ends with Submission ready. All eight GPU transition records restore both GPUs to their original14.55957GiB free baseline. This establishes native fit/cache cleanup for this run only.

Stage durations: Sailor classifier1h23m02s/private10m14s; Nemo classifier51m39s/private6m10s; Qwen classifier1h01m45s/private7m36s; corrector training3h34m36s/private1h50m33s. Individual stages include setup/loading/saving overhead; root duration also includes transitions/reporting. The private decoding stage is shorter than the previous batch4 run, but the selected adapter/output lengths also differ, so batch size is not an isolated causal comparison.

Measured classifier training-loop runtimes4606.0352/2851.8677/3430.342seconds imply an extra4355.30seconds for two more epochs each at the same rates. Adding this to the measured root gives roughly10h18m for the seven-epoch recipe, before notebook setup and variable download/evaluation/generation costs. This is an extrapolation, not a completion guarantee; the older10–12h estimate described an unmeasured five-epoch recipe.

## Same development population

Compared complete canonical validation arrays, including typed IDs, source strings, labels and references: the new classifier/corrector and all three members exactly match the quality7B, previous20B/14B and fast20B/14B78-row development population. No validation inference or training was rerun. Numbers below read already saved metrics; the Sailor component join uses the existing classifier metric and classifier-independent corrector GLEU.

|System|Macro-F1|Micro-F1|All GLEU+|Combined|
|---|---:|---:|---:|---:|
|Protected quality7B ensemble/Qwen7B|.572774|.743590|.605752|.589263|
|Earlier20B/14B selected pair|.594122|.782051|.663852|.628987|
|October20 fast20B/14B|.545092|.756410|.656091|.600591|
|New14B/20B/12B ensemble +14B corrector|.534541|.717949|.685001|.609771|
|New Sailor20B alone +same14B corrections|.622771|.794872|.685001|.653886|

These are repeatedly reused DEVELOPMENT results, not hidden private scores. The historical0.5806 quality number is user-reported official1038 feedback and is a different scope from its.589263 development metric. Do not compare.653886 development directly against0.5806 official feedback as a measured gain.

## Classification

|Member|Selected epoch|Macro-F1 epochs1→5|
|---|---:|---|
|Sailor20B|5|.434148 / .613472 / .597329 / .590458 / .622771|
|Nemo12B|4|.285901 / .377392 / .527643 / .586570 / .582110|
|Qwen14B|4|.463939 / .460806 / .448158 / .526185 / .526185|

The new vote underperforms Sailor by8.82 Macro-F1 percentage points. They differ on18 development rows: voting breaks10 Sailor-correct answers and recovers4 Sailor mistakes, leaving56/78 correct versus62/78 for Sailor. Sailor/ensemble function-word F1 is.600/.222 and sentence F1.800/.629; the ensemble improves phrase F1.250→.375 but the tradeoff loses overall Macro-F1. Clean recall is40/40 Sailor versus37/40 ensemble.

Manually recomputed the committed weighted SOFT vote from the saved probabilities on all78 development and2034 private rows, with Qwen/Sailor/Nemo weights1:1:2, temperature1 and phrase multiplier1.25. All saved probability vectors agree within tolerance; zero label mismatches. The loss is a measured combination/weight-transfer failure, not evidence of an ID/order/weight implementation defect. These fixed weights came from the smaller quality ensemble and are not proven calibrated for the new bases. No weight search, default change or private-target fitting was performed.

Logged mean training loss in epoch5 is Sailor.00526/Qwen.04888/Nemo.05465. These are averages of logged loss windows, not sample-weighted evaluation loss or training accuracy. Sailor's best checkpoint is still epoch5; Qwen/Nemo plateau around4–5. Extra epochs may help or hurt and cannot by themselves guarantee a better vote. Keep the user-requested seven-epoch experiment distinct from these results.

## Correction

|Epoch|All GLEU+|Error-only GLEU+|Exact error rows|Clean preserved|
|---|---:|---:|---:|---:|
|2|.596151|.199736|5/38|36/40|
|4, selected|.685001|.340415|7/38|37/40|
|5|.670672|.312891|6/38|38/40|
|6|.631449|.250131|6/38|38/40|

This corrector's development GLEU improves over both earlier14B checkpoints, but it declines after4 while mean logged train loss falls.01279→.00497→.00172 across4/5/6. This is evidence consistent with late-training overfitting on this development set. Best-checkpoint selection correctly retains4 rather than deploying the final6 checkpoint. Extending the scheduler from five to six epochs changes the learning-rate trajectory, so this epoch4 is not a controlled repeat of the older epoch4. Corrector hint_mode remains none and its predictions are independent of classifier labels.

## Private artifacts

The original run/results.zip is preserved,91,659bytes, SHA2564b2f2178fc7260c79293be568b3130bc42cff1c6bc3ba15d0a2be777217313c9. It contains one root results.json with2034 exact typed private IDs/order and valid organizer output fields. Saved source comparisons match the pinned private input hash0762703e83e59feac963b2e63f2986a351340cf32da07ce2db7d151a8abe6045; all final corrected strings are present and parse-error-free. Ensemble private label counts0..4 are810/1008/95/73/48. Counts are diagnostics without private gold.

Prepared a separate whole-component alternative at runs/analysis-upgraded-20261021/sailor20b-qwen14b/results.zip,91,778bytes, SHA256dcbdc8283ae67d6e6dfa005f1ea2cd06265926465d7d854c4891e8b40914609c. It uses all saved selected Sailor20B labels and the exact same selected Qwen14B corrected strings as the original ZIP.482 private labels change; corrected strings do not. Sailor private label counts0..4 are878/883/124/104/45. Exact typed IDs/order, unchanged strings, strict schema and ZIP CRC/root checked. No row-specific mixing, generation, training or calibration was needed. Provenance is alongside the candidate; full local aggregate review is runs/analysis-upgraded-20261021/review.json.

The alternative is supported by stronger reused-development classification evidence, but private score is UNKNOWN. The protected quality classifier still has stronger prior official-score evidence; no candidate is automatically promoted or submitted. Preserve the original run, the0.5806 archive and the previous cached final candidate. Current seven-epoch model/optimization configuration remains exactlyc7fee39.
