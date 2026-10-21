# Final seven-epoch run — October21

Reviewed runs/final_run/runs/track1039-upgraded-soft-vote-20261020-20261021T052620184170Z. Clean source commitc7fee39 trains all three classifiers seven full epochs, Qwen14B correction six full epochs. The saved ensemble descriptor binds names Qwen14B/Sailor20B/Nemo12B to weights1:1:2, temperature1 and phrase multiplier1.25. The proposed1:2:1 was discussed but not committed. Original results.zip doubles Nemo, not Sailor.

All eight stages completed in36,546seconds root W&B runtime (10h09m06s), before notebook dependency setup. Console ends Submission ready, without traceback/CUDA OOM/timeout markers. All2034 private rows generated at batch8; all eight GPU transitions returned both GPUs to original idle memory. This is a complete run.

## Models

|Member|Selected epoch|Development Macro-F1|Micro-F1|
|---|---:|---:|---:|
|Qwen14B|3 of7|.589003|.794872|
|Sailor20B|3 of7|.622136|.794872|
|Nemo12B|7 of7|.565299|.730769|

Versus preceding five-epoch fits, Qwen improves.526185→.589003; Sailor is approximately unchanged.622771→.622136; Nemo declines.586570→.565299. Extending the cosine schedule changes optimization throughout training; this is not simply the same first five epochs followed by two more. Qwen/Sailor select epoch3, so later is not automatically better.

Corrector selects epoch4 of6 again. Development GLEU at2/4/5/6 is.596151/.685001/.670672/.631449. Selected all-row GLEU.685001/error GLEU.340415/exact44/78/error exact7/38/ROUGE-L.889261/clean preservation37/40. All78 selected development corrected strings AND all2034 private corrected strings are exactly identical to the preceding run. Hints remain disabled. No correction improvement occurred here.

## Weights

Full canonical78-row validation arrays exactly match the preceding run, including typed IDs/sources/labels/references. Recomputed the one previously proposed1:2:1 alternative from saved member probabilities, holding temperature1/phrase1.25/corrector fixed. Existing1:1:2 recalculation matches saved probabilities/labels on all78 development and2034 private rows. No training, model inference, private-target evaluation or broader calibration search.

|Classifier choice|Macro-F1|Micro-F1|Combined with same corrector|
|---|---:|---:|---:|
|Previous five-epoch1:1:2|.534541|.717949|.609771|
|Previous five-epoch1:2:1 diagnostic|.558238|.756410|.621620|
|Final seven-epoch actual1:1:2|.555508|.743590|.620254|
|Final seven-epoch reweighted1:2:1|.627076|.807692|.656038|
|Final seven-epoch Sailor alone|.622136|.794872|.653568|

Sailor-heavy voting improves final ensemble Macro-F1 by7.16 percentage points/five net correct answers. It exceeds Sailor alone by only0.49 points/one net correct answer. Phrase recall remains1/6, phrase F1.166667. New combined.656038 exceeds the previous Sailor-only candidate.653886 by just.002152. These are reused DEVELOPMENT comparisons, not private-score guarantees or proof of a major generalization gain. Do not compare.656038 directly against the reported official1038 quality.5806 as a measured improvement. Final private score is unknown.

## Artifacts

Original runs/final_run/runs/track1039-upgraded-soft-vote-20261020-20261021T052620184170Z/run/results.zip is preserved:91,717bytes, SHA2561586cc11c24e3e1b64a85b0e9bd8785fe5aa4e534f54eea2d1678222276a1b71. It uses1:1:2 and contains valid root results.json for2034 typed private IDs/order.

Separate1:2:1 candidate: runs/analysis-final-run-20261021/soft-vote-121/results.zip,91,730bytes, SHA256e5992355fae68110bc01c69d544c77e6f1ebbaa68460cb7266e662dd44fd3bf7. All three saved probability vectors contribute;409 private labels change, corrected strings remain identical. Pinned private-input hash, exact typed IDs/sources, strict schema and archive root/CRC checked. Adjacent provenance binds all member prediction hashes. No dataset/notebook/pipeline changes or model run required.

Top-level runs/final_run/results(6).zip is1,232,868,788bytes/536 entries: the complete notebook download, including source/model files, not a competition submission. Submit only a small results.zip with root results.json if the user chooses; nothing is automatically uploaded.

Local analysis: runs/analysis-final-run-20261021/review.py/.json. Preserve the original run, protected0.5806 quality experiment, old candidates and all original weights. Training configuration remains unchanged.
