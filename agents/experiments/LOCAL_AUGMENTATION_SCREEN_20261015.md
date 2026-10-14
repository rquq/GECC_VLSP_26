# Local augmentation cross-check and stop decision — 2026-10-15

**Decision: augmentation is ON HOLD. Do not recommend another Kaggle run.** The user explicitly rejected spending Kaggle resources on an unqualified reviewer. The previous recommendation to try native FP16 despite a failed local check was not supported by quality evidence. Preserve all existing datasets/checkpoints and the best-supported Sailor/Qwen submission. The recovery ZIP is an input artifact, not an improved dataset.

A read-only local cross-check used four existing, reload-verified small-model final checkpoints. No training, checkpoint changes, new large-model generation, label repair, promotion or Kaggle execution occurred. In122.9 seconds it produced179 outputs/model:51 mechanically eligible proposed errors,47 proposed clean references and81 exposed original controls. All models loaded locally without downloads and retained FP32 weights with native BF16 autocast. Greedy correction uses the saved max256-token recipe. Checkpoint names, exact native counts, training-summary/weight/script/input/output digests and memory use are in the JSON record. Private script and outputs: `runs/augmentation-local-screen-20261015/`.

| Existing checkpoint | Candidate result | Exposed recorded-error result |
|---|---|---|
| BamiBERT flat | Proposed subtype matched4/51;42/51 predicted clean | Recorded subtype matched14/41 |
| Qwen2.5-0.5B flat | Proposed subtype matched7/51;37/51 predicted clean | Recorded subtype matched17/41 |
| ViT5-base edit-weight5 | Proposed reference reproduced10/51;36/51 sources unchanged | Recorded target reproduced1/41 |
| BARTpho-syllable | Proposed reference reproduced2/51;46/51 sources unchanged | Recorded target reproduced0/41 |

Only1/51 has both classifiers matching the proposed subtype. Only1/51 has both correctors reproducing the reference. **0/51 has both conditions. No candidate is approved.** This is cross-model agreement, not a quality threshold or evidence that all disagreeing candidates are wrong. The classifiers' own rare-class weaknesses and the correctors' poor recorded-error exact match make them unsuitable as automatic adjudicators. Identical clean predictions and high all-row correction metrics do not establish an ability to identify or repair the errors that need augmentation.

The81 original controls have been exposed in prior selection and are not an independent new validation set. Recorded targets may contain annotation-style or questionable corrections; exact-reference matching does not measure every grammatically valid alternative. Models trained on overlapping corpora can share errors. Existing classifier decisions reproduce all81 saved predictions for both classifiers; correction outputs additionally apply the existing tokenizer-identity restoration helper. Do not convert disagreement into fresh labels or promote agreements as human review.

The per-candidate private readable report is `runs/augmentation-local-screen-20261015/reviews.md`; machine-readable evidence is `candidate_reviews.json`. All51 existing source/reference pairs and all four outputs are preserved without rewriting or new labels. The compact tracked record contains only metrics/hashes, not corpus text.

The best-supported user-screenshot submission remains the preserved Sailor7B/Qwen7B hybrid: reported combined0.5423, Macro-F1.3473, GLEU+.7373. Its existing ZIP CRC/hash was reverified; this turn did not obtain a new official score. See PUBLIC_RESULTS_20261014.md/.json for attribution limitations. Keep that model pair as the control rather than replacing it with these weaker judges.

Before spending Kaggle time again, establish a reliable eligible local reviewer on supported known cases and evaluate any proposed additions against the same baseline in a small local experiment with parent-group separation. Preserve uncertain originals for explicit annotation review; do not scale pseudo-labeling or change training epochs/model size as a substitute for reliable subtype supervision. Neither a runtime smoke, model self-confidence, checkpoint consensus nor an untested FP16 hypothesis satisfies this prerequisite. If reliable labels cannot be established, leave augmentation off; the user does not need to upload or run anything now.
