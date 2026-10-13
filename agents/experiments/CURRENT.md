# Current experiment — 2026-10-13

**sailor7b-no-folds** is implemented and locally verified; the full-size Kaggle run is pending. The user requested removal of expensive fold training and preservation of the existing system in alpha.

The previous source/configs and canonical private bundle are preserved at `../alpha/preserved/kaggle-fold-hints-47f3a23/`, with source commit/tag `47f3a23` / `alpha-fold-hints-20261013`. Selected classifier and corrector weights are also preserved there using verified hard links; metadata and selected outputs are copied. The original run remains intact, without duplicating the 16 GB of fold weights.

The new run trains one main Sailor-7B classifier (5 epochs, LR 5e-5, rank 16/alpha 32, weighted BCE + 2*subtype CE) and one independent corrector (8 epochs, LR 2e-5). OOF is disabled. Corrector prompts contain no predicted or gold error label during training or inference. Saved main-classifier validation predictions supply classification metrics only. No in-sample training hints are created. Effective batch remains eight; hardware uses one/two GPUs with native precision checks.

The same pinned classifier/correction bundles are used (504/409 training rows, 81 unchanged validation rows). This isolates removal of hints/fold overhead. No subtype score gain is claimed for the refactor. Future data/model comparisons follow DATA_PLAN.md rather than mixing unmeasured changes into this run.

Local evidence: 130 tests passed. Real tiny-model CUDA NF4/FP16 and two-process CPU pipelines exported valid typed-ID submissions with OOF absent and hint_mode=none. These are mechanical checks, not a full-size training result.

Known public baseline, supplied by the user on 2026-10-13: first entry combined 0.1737 / GLEU+ 0 / Macro-F1 0.3473 / Micro-F1 0.5485; format-corrected entry combined 0.5349 / GLEU+ 0.7224 / Macro-F1 0.3473 / Micro-F1 0.5485. The second result reflects correction-list formatting, not a classification improvement. The public competition page also displayed KANO at 0.5349 when checked.

Review preparation: 103 flagged original-training rows were checked for safe reconstruction opportunities. Eleven have mechanically anchored target-completion proposals; eight have explicit annotation/label contradictions (already quarantined by the classifier builder). These remain private candidates, with no automatic label/data promotion and no holdout changes. No external teacher or full-size training ran for this review.

Historic evidence and failures: HISTORY.md and history.json index surviving metrics, recovered private archive notes, maintenance records and Git chronology. An artifact-backed metric is distinguished from an archived reported result or an implementation-only commit.
