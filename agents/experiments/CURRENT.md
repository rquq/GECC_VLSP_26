# Current experiment — 2026-10-14

The authorized local smaller-model screen is **completed**: nine architectures,13 configurations (12 complete, one divergence), two internally selected primary refits and three exploratory secondary refits, verified reloads and443-row exports. Details and exact recipes/provenance are in LOCAL_SMALL_MODELS_20261014.md/.json, `tools/screen_small_models.py` and `configs/local_screen.json`. The primary internally selected native classifier is bami-flat (5 epochs); selected native corrector is vit5-base-edit5 (5 epochs, raw). Frozen81 classifier Macro/Micro-F1: **0.414722/0.641975**. All-small correction all/error GLEU+: **0.607623/0.139081**; local combined **0.511173**. Same classifier with saved Qwen corrections: **0.521946** local combined. This exposed holdout is exploratory; subsequent user-screenshot public results are recorded below. No automatic Kaggle recipe replacement.

**Public screenshots now show the Sailor7B/Qwen7B hybrid at0.5423 combined**, matching the prior prediction and beating the prior Qwen/Qwen0.5282. It is the best supported candidate among the six supplied folder PNGs. Public Macro-F1: Sailor0.3473; PhoBERT0.3065; Qwen0.5 0.3010; Bami0.2593. Native Micro-F1 improvements do not offset their Macro losses. Qwen correction GLEU0.7373 exceeds ViT5 0.7279, but ViT5 EM0.5440 exceeds Qwen0.5214. Read PUBLIC_RESULTS_20261014.md/.json for exact scores, hashes and scopes.

The candidate-small-both Bami/ViT5 PNG repeats Bami/Qwen scores despite different verified correction bytes; attribution remains unresolved. Five folder associations are internally consistent; do not claim a confirmed Bami/ViT5 score. All six ZIP hashes remain unchanged. Original datasets/holdout remain unchanged. Best supported submission is the existing Sailor7B/Qwen7B hybrid; production Kaggle Qwen recipe remains committed with no silent replacement, folds, hints or notebook toggles.

**qwen25-7b-reviewed-both completed both full-size Kaggle training stages.** The user supplied the actual run from `runs/13_10_2026/runs/qwen25-7b-reviewed-both-20261013T161655075757Z`, using clean source commit `b2aa2800ca2e294bea70b42b2392de947966fd09`. Fresh Qwen2.5-7B-Instruct classifier: five epochs, LR 5e-5, 504 training rows. Corrector: six epochs, LR 2e-5, 414 reviewed training rows; selected epoch five. Both retain rank16, seed42, effective batch8 and the frozen 81-row holdout. No folds, hints or notebook toggles.

Measured local classification: **Macro-F1 0.595176 / Micro-F1 0.753086**, versus prior Sailor 0.434133 / 0.679012. Rare-class recalls are still 3/7 each for phrase, function-word and punctuation. Selected correction: all-row GLEU+ 0.629170, error-only GLEU+ 0.218588, error exact match 3/41 and clean preservation 40/40. Classification improves locally, but the third official KANO entry fell to combined 0.5282 (Macro-F1 0.3191, GLEU+ 0.7373). The public classification decline outweighed correction improvement. Full evidence and hashes are in QWEN_RUN_20261013.md/.json.

Public export failed after training because inference used the training memory guard (7.77 GiB free versus 12.65 GiB required). Recovery completed all 443 public outputs from saved checkpoints without retraining. The verified private submission is `runs/13_10_2026/results.zip`, also copied into the original run at `run/results.zip`. The fix isolates CUDA classification in a child process, persists its predictions, and uses a separate corrector inference reserve while preserving training precision and the training budget. Final recovery/ZIP status is recorded in QWEN_RUN_20261013.json.

The private data-only upload remains `dataset/kaggle-qwen-ready-20261013/VGEC26-qwen-reviewed-v4.zip`; its pinned hashes and all original/holdout bytes are unchanged. KAGGLE_READY.md/.json retains the original readiness evidence and links the completed run. QWEN_BOTH.md/.json records initial preparation; CORRECTOR_EPOCH_REVIEW.md/.json records the decision to shorten correction to six epochs. This recovery adds no repairs or synthesis.

## Previous prepared Sailor handoff (historical)

**sailor7b-reviewed-corrector-only** is prepared and locally checked. The user will retrain the corrector on Kaggle. The recipe reuses the completed Sailor-7B classifier and trains one Sailor-7B-Chat corrector for 8 epochs: no folds, no hints, no notebook toggles. The production ZIP includes the saved classifier and reviewed 414-row correction dataset. Full-size Kaggle training and new official scores remain pending.

The approved local Qwen pilot and two Bami screens are completed. No synthetic candidate or classifier replacement was promoted. Five supported original clean annotations enter the correction dataset; the legacy augmentation remains unchanged. See LOCAL_REVIEW_RESULTS.md / .json for scores, hashes, failed/recovered attempts, evidence and limits.

Qwen/SeaLLMs replacement review is complete in REPLACEMENT_REVIEW.md: both size eligible, no measured replacement win, SeaLLMs cross-model teacher use excluded by its linked license. The production base remains Sailor; the classification data bottleneck is still unresolved.

## Earlier no-fold preparation (historical)


**sailor7b-no-folds** is implemented and locally verified; the full-size Kaggle run is pending. The user requested removal of expensive fold training and preservation of the existing system in alpha.

The previous source/configs and canonical private bundle are preserved at `../alpha/preserved/kaggle-fold-hints-47f3a23/`, with source commit/tag `47f3a23` / `alpha-fold-hints-20261013`. Selected classifier and corrector weights are also preserved there using verified hard links; metadata and selected outputs are copied. The original run remains intact, without duplicating the 16 GB of fold weights.

The new run trains one main Sailor-7B classifier (5 epochs, LR 5e-5, rank 16/alpha 32, weighted BCE + 2*subtype CE) and one independent corrector (8 epochs, LR 2e-5). OOF is disabled. Corrector prompts contain no predicted or gold error label during training or inference. Saved main-classifier validation predictions supply classification metrics only. No in-sample training hints are created. Effective batch remains eight; hardware uses one/two GPUs with native precision checks.

The same pinned classifier/correction bundles are used (504/409 training rows, 81 unchanged validation rows). This isolates removal of hints/fold overhead. No subtype score gain is claimed for the refactor. Future data/model comparisons follow DATA_PLAN.md rather than mixing unmeasured changes into this run.

Local evidence: 130 tests passed. Real tiny-model CUDA NF4/FP16 and two-process CPU pipelines exported valid typed-ID submissions with OOF absent and hint_mode=none. The production notebook runner also completed both tiny stages and wrote matching start/completion events with commit/data hashes under agents and in the run. Details and artifact locations are in VERIFICATION.md. These are mechanical checks, not a full-size training result.

Known public baseline, supplied by the user on 2026-10-13: first entry combined 0.1737 / GLEU+ 0 / Macro-F1 0.3473 / Micro-F1 0.5485; format-corrected entry combined 0.5349 / GLEU+ 0.7224 / Macro-F1 0.3473 / Micro-F1 0.5485. The second result reflects correction-list formatting, not a classification improvement. The public competition page also displayed KANO at 0.5349 when checked.

Review preparation: 103 flagged original-training rows were checked for safe reconstruction opportunities. Eleven have mechanically anchored target-completion proposals. Further review found seven explicit annotation/label contradictions; the previous eight-row claim included a false positive from a punctuation instruction. The new builder uses explicit sentence-claim policy v2; the pinned baseline still loads its legacy policy and remains unchanged. Five completions and five annotation-supported clean candidates are prepared privately for blind verification/approval. No external teacher or full-size training ran for this review.

Historic evidence and failures: HISTORY.md and history.json index surviving metrics, recovered private archive notes, maintenance records and Git chronology. An artifact-backed metric is distinguished from an archived reported result or an implementation-only commit.

The historical model review proposed PhoGPT, PhoBERT-large, BamiBERT, BARTpho, Sailor and a bounded Qwen teacher pilot. The user subsequently approved execution; the completed negative comparisons and prepared handoff are recorded above and in LOCAL_REVIEW_RESULTS.md. No unmeasured model replacement is active.
