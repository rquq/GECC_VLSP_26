# Historical corrector epoch and repair-teacher review — 2026-10-13

After this review the user explicitly requested pushing six epochs. That change is implemented in configs/correction.json and documented with the upload package in KAGGLE_READY.md/.json. The review below records the earlier eight-epoch state and its actual historical evidence.

The user asked whether an eligible QLoRA fine-tuned model could repair the noisy dataset and whether the current eight corrector epochs are appropriate. This was a read-only artifact review; no training, teacher inference, dataset edits or training config changes occurred. Source inspected: `86d275580ae3a8124a1965e2e586555b84329b1f`. The active Qwen config still specifies eight epochs; a shorter budget is a recommendation, not a measured Qwen optimum.

Latest full-size run: `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/runs/12_10_2026/runs/sailor-7b-both-20261012-145038-393535/correction/`. It used Sailor-7B-Chat, 409 training rows, predicted hints, LR2e-5, seed42 and 81 validation rows (41 error/40 clean). The current Qwen run differs in base, data (414 rows), chat role and no hints, so its optimal epoch cannot be inferred exactly.

| Epoch | Error-only GLEU+ | All-row GLEU+ | Error exact match | Clean preservation |
|---|---:|---:|---:|---:|
| 1 | 0.155030 | 0.576098 | 0.048780 | 0.775 |
| 2 | 0.202460 | 0.576575 | 0.097561 | 0.800 |
| 3 | 0.201870 | 0.609166 | 0.073171 | 0.975 |
| 4 | 0.215836 | 0.616234 | 0.073171 | 0.950 |
| 5 | 0.234896 | 0.625571 | 0.073171 | 0.950 |
| 6 | 0.233426 | 0.622779 | 0.073171 | 0.950 |
| 7 | 0.229509 | 0.621078 | 0.073171 | 0.950 |
| 8 | 0.229956 | 0.624653 | 0.073171 | 0.975 |

Epoch5 is best under the recorded selection rule (error-only GLEU+, then error exact, then clean preservation). Error exact match at epochs3–8 remains3/41. Mean logged training loss falls from0.003444 at epoch5 to0.000440 at epoch8, without improved error correction. This is consistent with memorization/plateau; it is not proof of a universal overfitting threshold. Training plus validation runtime was12,065.3sec (about3h21m). The final fresh-base reload failed after all eight epochs because CUDA0 had9.92GiB free against a13.25GiB reserve. Epoch histories and the selected adapter survived; the current implementation releases trainer/DDP owners and reuses the frozen base for selected-adapter reload.

The preceding historical Sailor2 run selected epoch1 (error GLEU0.252938), versus0.229504 at epoch8. It is an ineligible historical architecture for the constrained track and a different model/protocol, not a deployment candidate or direct Qwen comparison. Its curve also provides no evidence that eight epochs are inherently better.

Recommendation: six maximum epochs with existing per-epoch best-adapter selection is a reasonable runtime-oriented first Qwen budget. Five is the best observed Sailor epoch; eight remains a defensible upper ceiling for learning Qwen's first curve. Changing eight to six also shortens the linear LR schedule, so it does not reproduce the first six epochs of the previous run exactly. Future tuning should select on a training-only grouped internal split, with the frozen81-row holdout used sparingly for reporting; it has already been repeatedly inspected and has uncertain references. These local GLEU+ values are not the official public score.

An eligible GEC-tuned QLoRA model can propose repairs; QLoRA itself is not the data-quality problem. Audit original base plus active adapter counts (the current Qwen corrector upper bound is7,655,986,688), immutable base/adapter revisions, fine-tuning/generation data provenance, license and decoding settings. Preserve private originals and record before/after evidence. A model fitted to the same noisy rows can reproduce their annotation mistakes or memorize targets; its agreement is not independent validation. Prefer trusted training-only curator supervision excluding the rows/families being audited, then blind independent review. Do not feed the frozen holdout or public/private test to repair generation, invent missing context, apply optional style edits, or automatically overwrite original labels/targets. Admit supported minimal grammatical changes, quarantine uncertain cases and validate a candidate against a fixed grouped internal control before promotion.

Sources: [constrained rules](https://aihub.ml/competitions/1038), [PEFT quantization guidance](https://huggingface.co/docs/peft/developer_guides/quantization), local DATA_GUIDELINES.md and the actual epoch/trainer/config artifacts. CORRECTOR_EPOCH_REVIEW.json contains exact metrics, hashes and paths; no corpus text is published in this summary.
