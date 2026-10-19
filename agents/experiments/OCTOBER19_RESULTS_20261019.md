# Completed October19 focused run

Status: completed native Kaggle run; this analysis rescored existing predictions only. Source4876c4093d02d9a1437b73940cbfdb1546f783dc, exact downloaded run: runs/19_10_2026/runs/focused-sailor-qwen-20261018-20261018T124500964626Z. Artifacts and original annotations are unchanged.

The selected pair is the NEW organizer-anchored Sailor7B classifier (epoch4 selected after ALL5 epochs/410 updates) and OLD October16 Qwen7B greedy corrector. The new anchored Qwen corrector completed ALL5 epochs/310 updates and verified reload, but did not beat the baseline. Both fits used standardLoRA, FP16 frozen bases, rank16/alpha32/dropout.05, seed42, no folds/hints. The saved selection contains portable classifier/corrector files and a hash-bound exact decoder contract; all bound files rehashed successfully.

| Development component | Macro-F1 | Micro-F1 | All-row GLEU+ | Error GLEU+ |
|---|---:|---:|---:|---:|
| Quality Sailor7B baseline |59.0196%|76.9231%|—|—|
| Anchored Sailor7B challenger, argmax |68.9874%|83.3333%|—|—|
| Old Qwen7B, greedy |—|—|63.1335%|23.7716%|
| Old Qwen7B, beam2 |—|—|61.7434%|21.7486%|
| New anchored Qwen7B, greedy |—|—|61.6951%|22.3050%|
| New anchored Qwen7B, beam2 |—|—|61.9359%|23.0698%|

Selected local combined-v2:0.6606040759843331. Independent rescoring agrees within1e-12. Clean preservation40/40; error exact3/38. Per-class F1: clean97.50%, sentence85.71%, phrase54.55%, function52.63%, punctuation54.55%. Phrase recall3/6; function precision5/13 reveals overprediction. Only1/38 errors becomes false-clean. Fixed binary-first0.5 lowers challenger Macro to67.8188%; keep argmax. Three-way soft voting was weaker on preceding runs and is not forced into the new focused selection.

Each corrector screened greedy/beam2/beam4/beam4-edit-reranking on10rows, then evaluated greedy and beam2 on ALL78. Beam4/reranking were NOT fully evaluated. All-row GLEU includes40 clean rows; annotation-like26 error references score0 GLEU for the selected corrector, while12 standalone error targets score about74.22%. A larger model alone does not remove this annotation-format mismatch; no held-out reference has been repaired or used as fitting data.

The78 typed development records/references are EXACTLY identical to the quality and old baseline artifacts; repeated use for tuning/checkpoint selection means these are development results, not independent generalization estimates. No official private labels/score are supplied for this run. The existing results.zip passes CRC, strict organizer schema,2034 typed private IDs/order and valid labels0..4. It is ready as an inference artifact, not proof of a private score.

Detailed private analysis: runs/analysis-track1039-20261019/october19_results.json. Aggregate JSON accompanies this document. No training or generation rerun was used to create this analysis.
