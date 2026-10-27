# Completed quality-run ablations — 2026-10-27

Status: completed on Kaggle, independently checked from saved predictions, data and histories. Source d17828a6d135354aea14a954d8c328a0bc59149f; protected recipe source338293f. Three standard-FP16-LoRA Sailor7B fits and three selected-adapter reloads completed. No new corrector training, private inference or official submission.

## Controlled fresh fits

| Dataset / objective | Training entries | Best step | Macro-F1 | Micro-F1 | Phrase F1 | Function-word F1 | Final Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| quality_hierarchical | 1128 | 423 | 59.02 | 76.92 | 25.00 | 53.33 | 55.08 |
| organizer_hierarchical | 321 | 141 | 57.27 | 73.08 | 60.00 | 38.10 | 56.55 |
| quality_flat | 1128 | 423 | 46.92 | 71.79 | 0.00 | 40.00 | 43.82 |

Scores are percentages on the same repeatedly used78 development rows. All arms use seed42,705 updates and five selection steps141/282/423/564/705. Quality arms complete five epochs/5640 sample exposures; organizer-only cycles321 originals for17.199 passes/5521 exposures. Same inverse-square-root weighting formula, with weights recalculated per training mixture.

Quality/hierarchy improves selected Macro-F1 by1.75 points over organizer-only, and12.10 points over quality/flat. This is one fixed-recipe comparison: the data contrast includes mixture/resampling/weight changes; the objective contrast includes the lambda-two subtype emphasis. It does not isolate augmentation volume, hierarchy factorization or lambda individually.

The quality/hierarchical reproduction has exactly the same78 predictions AND probabilities as the original selected Sailor quality output (identical prediction-file SHA256 a888868e32ae03855b169557a359a3e6e956b60faa5f4ace8924c75fb9ffebdb).

## Per-class F1

| Arm | Clean | Sentence | Phrase | Function word | Punctuation |
|---|---:|---:|---:|---:|---:|
| quality_hierarchical | 91.76 | 75.00 | 25.00 | 53.33 | 50.00 |
| organizer_hierarchical | 92.86 | 62.07 | 60.00 | 38.10 | 33.33 |
| quality_flat | 90.70 | 70.59 | 0.00 | 40.00 | 33.33 |

Quality gains three net correct predictions over organizer-only (seven gains, four losses), but phrase recall falls3/6→1/6 and phrase F1 falls60→25. Clean correctness remains39/40 in both. Quantity balancing has not solved subtype coverage. Prefer diverse reviewed phrase mechanisms and explicit subtype boundaries over more repetitions.

## Learning curves

| Optimizer step | Quality/hierarchy | Organizer/hierarchy | Quality/flat |
|---|---:|---:|---:|
| 141 | 48.64 | 57.27 | 43.79 |
| 282 | 51.96 | 56.55 | 45.04 |
| 423 | 59.02 | 56.55 | 46.92 |
| 564 | 56.45 | 56.55 | 44.76 |
| 705 | 55.08 | 56.55 | 43.82 |

Final logged training losses round to0.0 in all three arms, while final development Macro-F1 is below the selected peak. This is consistent with overfitting/saturation; it does not establish that a particular regularizer would improve generalization. Retain best-checkpoint selection. Final-step ranking differs: organizer-only56.55 exceeds quality/hierarchy55.08.

## Frozen quality voting

| Rule | Macro-F1 | Micro-F1 | Phrase F1 | Function-word F1 |
|---|---:|---:|---:|---:|
| qwen7b | 47.22 | 64.10 | 13.33 | 21.05 |
| sailor7b | 59.02 | 76.92 | 25.00 | 53.33 |
| mistral7b | 55.07 | 71.79 | 40.00 | 33.33 |
| equal_plain | 50.95 | 74.36 | 0.00 | 40.00 |
| equal_phrase | 56.52 | 75.64 | 25.00 | 42.86 |
| quality_112_plain | 57.28 | 74.36 | 40.00 | 33.33 |
| quality_112_phrase | 57.28 | 74.36 | 40.00 | 33.33 |
| sailor_121_phrase | 60.11 | 78.21 | 22.22 | 53.33 |

Weight order is Qwen:Sailor:Mistral. Sailor-heavy1:2:1/phrase1.25 gives60.11 Macro-F1 versus57.28 for submitted1:1:2/phrase1.25. No new private score exists. Phrase multiplier1.25 changes zero labels at1:1:2, although it helps equal voting. The protected official0.5806 remains attributed solely to the deployed1:1:2 quality submission.

## Runtime and verification

All six child stages exited and returned both GPUs exactly to the recorded14.55957GiB free idle baseline. No traceback, CUDA OOM, OutOfMemoryError or TimeoutError appears in the console. Summed fit-plus-reload arm time is13,462.44s (3h44m22s), excluding initial setup and root export. Root W&B named run finished online and includes the forwarded console.

Independent checks: complete package file hashes, clean captured source commit, exact train/dev arrays and original subset identity, no normalized source/target or typed-ID overlap, probability validity/argmax, independently calculated five-class F1/confusion matrices, all eight cached comparisons, selected/final step histories, finite model receipts, exact historical reproduction and output ZIP CRC. No model execution was repeated for this review.

The manuscript replaces the historical mixture table with the controlled ablation table, expands frozen-quality voting, and explains the unequal epoch counts. Prior historical results and all protected artifacts remain available in their original records. These results are sufficient to strengthen the competition report; additional model-size training is optional and not needed to support these findings.

Run: /home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/runs/27_10_2026/runs/quality-sailor-controlled-ablations-20261027-20261026T192117776519Z/run
Independent evidence: /home/quq/machineLearning/competitions/VLSP26_4/paper/work/quality-ablation-results-20261027/evidence.json
W&B: https://wandb.ai/ddq007-uit/vlsp26-gec/runs/e12b7265c5974eb9b570a69c330dd173
