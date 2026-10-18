# Old, quality and diverse: verified shared-development comparison

User requested comparison of the completed October18 diverse run against quality and old, rerunning validation only if needed, and analysis of next improvements. Status: completed saved-prediction reanalysis, with zero model fits, zero model inference, zero synthetic generation, no production recipe change and no deletion. The diverse trial completed successfully but failed to improve the deployed combined score.

All 15 validation artifacts (ensemble, corrector and three members for each run) contain exactly the same ordered 78 rows, typed IDs, source texts, labels and unchanged references. Canonical SHA256: `5cf0e53bdc49e56145d526f63d88585d6067d3b14d8b54230ef01dc125522055`. Supports: clean40, sentence19, phrase6, function-word6, punctuation7. Each stage has zero exact typed-ID or normalized-source overlap with these development rows. Historical grouped-family checks remain in the dataset records; this comparison does not claim exhaustive semantic independence. These rows have repeatedly selected checkpoints and decoding/voting rules, so they are development evidence, not a blind test.

Saved predictions were rescored using the same current `model.metrics.experiment_report`; saved Macro/Micro-F1, all/error GLEU+, exact match, ROUGE-L and combined-v2 match within1e-12 for all three runs. No validation inference rerun is necessary.

| Shared 78-row development metric | Old | Quality | Diverse V2 tuned |
|---|---:|---:|---:|
| Macro-F1 |56.16%|57.28%|55.25%|
| Micro-F1 / accuracy |79.49%|74.36%|73.08%|
| GLEU+, all rows |63.13%|60.58%|61.08%|
| GLEU+, error rows |23.77%|20.09%|21.06%|
| Exact corrections, all |43/78|45/78|45/78|
| Exact corrections, errors |3/38|6/38|5/38|
| Correct sentences preserved |40/40|39/40|40/40|
| Combined v2 = (Macro-F1 + all-row GLEU+)/2 |59.65%|58.93%|58.16%|
| Actual error sentences classified as clean |3/38|8/38|12/38|

| Per-class F1 | Old | Quality | Diverse |
|---|---:|---:|---:|
| Clean |96.39%|90.91%|86.96%|
| Sentence structure |81.08%|60.61%|64.29%|
| Phrase structure |0.00%|40.00%|25.00%|
| Function word |50.00%|33.33%|57.14%|
| Punctuation |53.33%|61.54%|42.86%|

Diverse gets 40/40 clean,9/19 sentence,1/6 phrase,4/6 function-word,3/7 punctuation correct. Against quality it fixes6 classifications and breaks7; against old it fixes3 and breaks8. Function-word gains are real on these rows, but quantity balancing did not solve subtype coverage or generalization.

## Model and decision diagnostics

| Selected individual classifier Macro-F1 | Old | Quality | Diverse |
|---|---:|---:|---:|
| Qwen |50.03% (epoch3)|47.22% (epoch1)|47.06% (epoch5)|
| Sailor |49.89% (epoch2)|59.02% (epoch3)|58.87% (epoch2)|
| Mistral |51.38% (epoch1)|55.07% (epoch4)|52.18% (epoch5)|

Diverse Sailor alone has79.49% accuracy, beating its deployed ensemble's55.25% Macro/73.08% Micro. Diverse Mistral calls18/38 errors clean; mean P(clean) on true errors is.4632, versus Sailor.2473 and Qwen.1581. This supports checking members' probability quality rather than assuming three models must beat one. It does not prove calibrated probabilities, a broken loader or a defective architecture. All nine final member manifests confirm fresh-process classifier reload verification.

Predeclared common soft-voting rules, without a new weight search:

| Rule | Old Macro | Quality Macro | Diverse Macro |
|---|---:|---:|---:|
| Equal probability average |56.16%|50.95%|51.55%|
| Weights1:1:2, T1, phrase1.25 |63.78%|57.28%|52.18%|
| Weights1:2:1, T1, phrase1.25 |51.34%|60.11%|55.25%|

The63.78% old and60.11% quality figures have already been selected using this development set. They are not new unseen improvements. Choice of rule changes the apparent dataset ranking. A truly controlled dataset-only comparison must hold fitting, checkpoint-selection and decision settings fixed.

Fixed binary-first diagnostic uses P(error)=1-P(clean), gate0.5, then the largest error-subtype posterior. It changes diverse Macro55.25%→58.31%, accuracy73.08%→75.64%, false-clean errors12→9, while preserving40/40 clean classifications. It slightly hurts old Macro56.16%→54.88% and quality57.28%→56.31%. This is a useful diverse-specific diagnostic, not a universal binary-first recommendation or a separately trained binary classifier. Final subtype selection is discrete; this probe does not replace the committed soft-voting protocol.

Preserving the source whenever the classifier predicts clean was also tested from saved predictions. It decreases all-row GLEU+ in all three runs: old63.13%→62.74%, quality60.58%→60.04%, diverse61.08%→59.65%. Do not deploy this shortcut: false-clean classifications can suppress useful repairs. No hints/gates or model implementations were changed.

## Correction and reference format

All three use the same unmodified references:26 error targets start with a parenthesis or contain literal `...`, and12 are standalone targets. These are diagnostic format flags, not confirmed invalid labels. Annotation-like slice corpus GLEU+ is0 for all three despite some exact matches; the unchanged corpus GLEU+ calculation can yield0 when any aggregate n-gram numerator is nonpositive. This does not mean every correction is wrong or establish an official-scoring bug.

| Error-reference slice GLEU+ | Old | Quality | Diverse |
|---|---:|---:|---:|
| Annotation-like,26 rows |0.00%|0.00%|0.00%|
| Standalone,12 rows |74.22%|70.50%|73.82%|

Slice scores are separately computed corpus metrics, not values that can be averaged to recover all-row GLEU+. The standalone slice is small and potentially easier; it cannot replace the official-style primary metric. Retain both views and all original golds. The inherited complete-sentence prompt and abbreviated training references remain a format tension to investigate using training-only evidence.

Diverse corrector selected epoch4; all-row GLEU by epoch1..5 is57.65/59.95/60.87/61.08/60.80%. Clean preservation reaches40/40 at epoch3. More epochs have no demonstrated benefit here. Beam search was not rerun by this analysis; earlier decoder experiments are retained and this comparison establishes no new beam-search gain.

## Training and submission scope

Old source `bb7cdf5b28e8f4f161087e819827aaae129e3e51`; quality `338293f90e664803ee3dde12e82dd5d4d5601d43`; diverse `7f2cceed903adb718a84f457868ee89c75451c9b`; all source manifests are clean. Classifier training exposures are1145/1128/1394; exact distinct sources773/612/844. Corrector exposures1145/1011/1289. See JSON for exact per-class exposure counts, saved settings, curves and hashed evidence.

Old uses equal votes and corrector LR2e-5/dropout.05/max6 epochs with error-GLEU checkpoint selection. Quality uses votes1:1:2, corrector2e-5/.05/max4 with all-GLEU selection. Diverse uses1:2:1, Qwen classifier LR2e-5/warmup70 (others5e-5/warmup30), corrector1.5e-5/.10/max5 with all-GLEU selection. These are descriptive pipeline comparisons with optimization/data/voting confounds, not causal dataset ablations. The newer user minimum-five-epoch instruction applies going forward; do not rewrite the historical quality fit as five epochs.

Diverse completed allfour fits forfive epochs and allsix pipeline stages in35653.31 seconds (~9h54m). No fatal traceback, CUDA-error, OOM or RuntimeError marker was found in any final console log. Allthree submission ZIPs have valid CRCs, exactly results.json, unique typed IDs and valid output fields. Old exports443 public inputs; quality/diverse export the exact same2034 private typed IDs and source texts (canonical input SHA a6790f449952567a5971254fd9564395a3509ab9943707d3b9451a19c1833cbb). No private labels or official score are available. Diverse predicts1510/2034 private inputs clean (74.24%), versus quality1029/2034 (50.59%); this is a distribution warning, not a measured private error rate. Do not compare the old443 outputs as the same test population.

## Recommended next work, in order

1. **Exploit and preserve existing checkpoints first.** Use Sailor's quality/diverse checkpoints as baselines and retain the old corrector. Saved-probability comparisons already show the ensemble can hurt; no additional10hour four-model fit is justified just to recover these baselines. Calibrate component probabilities/weights only within an explicitly documented selection protocol; do not sweep more thresholds and present the same78 rows as unseen improvement. Cross-component development scores are diagnostic only: quality Sailor Macro59.02% plus old corrector GLEU63.13% implies combined61.08%; no hybrid inference/export has been performed. Existing three-model soft voting remains the committed default until a deliberate change is requested.
2. **Review training subtype evidence before adding volume.** Use the existing private162-row/114-flag organizer-training review queue, distinguish valid annotation formats from real mistakes, and record evidence for every supported repair. Focus on phrase/sentence/function-word boundaries and natural errors mistakenly learned as clean. Preserve raw originals and ambiguous rows; no large-assistant replacement wording or inferred labels. New training candidates need compliant<=8B authors/reviewers or documented human evidence, licensed train-only parents, matched clean controls and lineage. Model agreement alone is insufficient.
3. **Change sampling from label counts to parent/mechanism coverage.** Retain a strong organizer-data anchor; cap repeated parent families and transformations, inspect each subtype mechanism separately, and downweight uncertain synthetic material. More mechanically generated examples with the same shortcut can hurt even when counts are balanced. The negative Bami screens and this native run justify pausing blind expansion. Do not force every subtype to match clean count unless quality and diversity support it. Keep official test inputs out of mining, generation and supervision.
4. **Run one controlled Sailor data ablation before the next ensemble retrain.** Compare quality against an organizer-anchored/reweighted expanded mixture with the same model, seed, LoRA, optimizer, decision rule and shared78dev. Require at leastfive full epochs per the user, save/select the best earlier checkpoint, and report optimizer updates/tokens because different dataset sizes change exposure. Qwen's new Macro curve rises29.81→35.66→43.41→44.30→47.06 throughfive epochs; Sailor peaks58.87 at epoch2 then falls47.23. Do not apply the prior Qwen-epoch1-overfitting explanation to this new curve or assume longer fitting will improve all models. Native learning-rate/data effects remain confounded.
5. **Fix correction supervision format with evidence.** Keep primary scores on unchanged gold; track standalone and annotation-like slices. Review train-only abbreviated/parenthetical targets and construct complete targets only when compliant evidence supports them, preserving provenance/retained originals. Then test one factor at a time. Start with the old corrector as the measured GLEU baseline; compare greedy with a small beam2/4 candidate set only on that fixed checkpoint when inference budget permits. Check clean preservation, edit counts and both GLEU slices; beam width or extra epochs do not repair incorrect supervision.
6. **Treat validation uncertainty honestly.** Phrase/function have onlysix examples each and punctuationseven. Repeated selection over78 rows can favor accidental wins. Keep this shared set for requested comparisons; predeclare experiments and seek a separate grouped organizer-only audit or trustworthy official score before claiming70–80% Macro-F1. No such score is demonstrated by these artifacts.

Private full results, executable analysis and all78 side-by-side source/reference/prediction rows: `runs/comparison-three-datasets-20261018/`. Aggregate-only companion: `THREE_DATASET_COMPARISON_20261018.json`. No corpus-bearing files are added to tracked documentation. Existing user edits to CURRENT and PRIVATE_RETRAIN records are preserved; no model/data/config/notebook changes or new Kaggle launch were made.
