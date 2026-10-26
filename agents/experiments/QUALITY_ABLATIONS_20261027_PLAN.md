# Quality-run ablations implementation plan — 2026-10-27

Goal: one Save Version on the existing Kaggle launcher plus one private data upload runs three sequential Sailor7B fits and saved-probability voting comparisons.

Approved scope: organizer-only vs quality, flat CE vs lambda-two soft hierarchy, and fixed voting rules. No corrector retrain or private-test inference is necessary for these paper ablations. Keep the protected 338293f quality run and unrelated local edits intact.

Architecture: reuse model.classification._fit and classify; optional objective and step-based selection extend their historical defaults. A small controller verifies an extracted hash-bound package, runs each fit/reload in a fresh process through the existing GPU-release/console/resource machinery, and produces paper-ready results. Kaggle notebook code cells remain launcher-only.

Global constraints: Sailor-7B-Chat revision19066fae0a8a3ba029c190d8e3dacccf4d1234b8, standard FP16 LoRA r16/alpha32/dropout.05, attention+MLP projections, head trained, seed42, LR5e-5/cosine/warmup30/decay.01, micro1/effective8, length512. Three arms each705 optimizer steps, validation at141/282/423/564/705. Quality arm therefore retains its five-epoch budget. Organizer-only321 rows cycle until the same update budget; actual epochs and row exposure differences are reported. Class weights use the original inverse-square-root formula on each arm. Flat changes only joint distribution/objective. Best Macro-F1 with earlier-step tie preference; final-step metrics also reported.

Review focus: flat checkpoint reload must use softmax; epoch callbacks must not create extra selection chances; no mixed/test/development rows in fitting; arbitrary Kaggle mount prefixes and no nested archives; all model transitions release GPU memory; root W&B remains named run and mirrors safe notebook console.

- [x] Tests first: flat weighted accumulation, objective serialization, exact schedule, data discovery/tamper/overlap, cached voting identities.
- [x] Extend classifier with opt-in flat and fixed-step evaluation, preserving hierarchical defaults.
- [x] Package exact quality training records,321 eligible organizer-only records, unchanged78dev, cached selected-member probabilities and correction outputs; no weights/private test/nested archives.
- [x] Add three-arm runner using original training/reload functions; per-arm metrics, curves, sample counts, probabilities, summary JSON/CSV/Markdown and small results ZIP. No model/controller deadline.
- [x] Reuse launcher, default recipe dispatch and notebook code cells; report new mode without expecting a submission ZIP.
- [x] Run focused regression tests plus actual tiny three-fit CPU run and relocated package/controller run. Verify source matches quality except required objective/schedule/runtime changes. No full7B local fitting or new data generation.
- [x] Record exact hashes/configs/status, review whole diff, commit/push only task changes to kaggle-linux, verify remote commit. Deliver one upload ZIP and exact Save Version instructions; native full-size Kaggle completion remains unmeasured.
