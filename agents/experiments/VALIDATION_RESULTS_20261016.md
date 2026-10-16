# Validation comparison — 2026-10-16

The full October16 output completed classifier recovery, Qwen correction training, saved-adapter reload, inference and results.zip export. Source commit bb7cdf5b28e8f4f161087e819827aaae129e3e51. These are internal validation results; no new official leaderboard result is supplied.

## Historical side-by-side

| Metric | Old Qwen, Oct13 (81 rows) | Current soft vote/Qwen (78 rows) |
|---|---:|---:|
| Macro-F1 | 59.52% | 56.16% |
| Micro-F1 | 75.31% | 79.49% |
| GLEU+ all | 62.92% | 63.13% |
| GLEU+ error only | 21.86% | 23.77% |
| Exact match all | 53.09% | 55.13% |
| Exact match error only | 7.32% | 7.89% |
| Combined with v2 formula | 61.22% | 59.65% |

Only 12 exact validation rows overlap. Current classifiers trained on 69 old validation rows; the old Qwen classifier trained on 66 current validation rows. These are descriptive historical comparisons, not a controlled model-quality difference. Never re-evaluate either checkpoint over the full opposite holdout and call it unseen. The old saved combined score used error-only GLEU; the table recomputes it using the current all-row formula. The old Sailor classifier scored Macro-F1 43.41% / Micro-F1 67.90% on the 81-row partition.

## Same 78 rows

| Classifier | Macro-F1 | Micro-F1 |
|---|---:|---:|
| Qwen | 50.03% | 74.36% |
| Sailor | 49.89% | 71.79% |
| Mistral | 51.38% | 70.51% |
| Soft vote | 56.16% | 79.49% |

The ensemble recognizes all40 clean rows, detects35/38 errors, and identifies22/38 error subtypes. Per-class recalls: clean40/40, sentence15/19, phrase0/6, function-word3/6, punctuation4/7. Six phrase errors split as one sentence, two function-word and three punctuation predictions; detection is not their bottleneck.

## Binary-first counterfactual (no fit)

From exactly the saved joint probabilities, use q=sum(P1..P4), threshold q>=0.5, then argmax over P1..P4. This changes only the decision, not training. Ensemble Macro-F1 drops from56.16% to54.88%; Micro-F1 drops79.49% to78.21%; phrase recall remains0/6. It does not prove separately trained binary/subtype specialists could never improve; no such extra training was run. Keep the current soft hierarchy as the reference.

## Correction diagnostics

Selected epoch4: all-row GLEU+63.13%, error-only23.77%, clean preservation40/40, error exact matches3/38. Copying every input already gets all-row GLEU+55.76%. Of38 errors,26 have annotation-like references (slice GLEU+0) and12 have standalone references (74.22%). Flags are not proof of erroneous gold data, and the unsmoothed corpus metric can be zero despite some matches. Nine error inputs remain unchanged. Prompting already explicitly requires minimal edits and preserving correct sentences; hint_mode=none means the corrector receives no classifier labels. Primary decoding is greedy; malformed-JSON recovery alone uses four-beam schema-constrained decoding. Existing optional beam/frozen-base-grammar/semantic reranking code is disabled; its edit cost is only a tie-breaker, not a general edit penalty.

Epoch1..6 error GLEU+:13.78,16.86,21.83,23.77,23.12,22.80 percent. Six epochs ran; the retained checkpoint is epoch4. More epochs are not supported by this curve.

## Recommendations, not executed changes

Keep soft hierarchy and lambda2 (binary BCE + twice error-only subtype CE). Prioritize diverse reviewed phrase-type training examples and train/reference convention audits. Evaluate greedy versus beam2 versus beam4 on this fixed78-row development set with the saved epoch4 adapter; no retraining required. Consider edit-aware candidate ranking including the unchanged source, but check against excessive copying and avoid treating an uncalibrated frozen model as authoritative grammar supervision. Do not rewrite held-out references to flatter a model. Any eligible teacher/reranker must follow the recorded agreement/license rules. None of these suggestions is a measured beam or augmentation gain. Model-based/hierarchical RL does not describe this independent-sentence supervised task or the three-classifier ensemble.

Detailed private files in runs/analysis-validation-20261016/: comparison.json, side_by_side_rows.json and side_by_side_rows.md (all78 rows, including references/current outputs and old outputs for the12 shared IDs). No new fitting or beam generation was run.
