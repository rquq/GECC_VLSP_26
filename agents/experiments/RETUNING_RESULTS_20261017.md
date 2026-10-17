# Native retuning outcome and timeout correction — 2026-10-17

The user supplied runs/17_10_2026 and requested score diagnosis plus console-log debugging. Sourcefd83a65 completed11 native decoding screens,3 full baseline decoder passes, all3 classifier fits/reload verifications and the Qwen corrector fit/reload. The contrastive repair is now verified on the actual dual-T4 run. The outer experiment failed during an intermediate public export after25944.48seconds; it did not complete challenger decoding/final selection or write the new final submission ZIP. Preserve every completed adapter; no full7B training was rerun locally.

## Measured outcomes on identical78 development rows

| Macro-F1 | Old | Tuned |
|---|---:|---:|
|Qwen|50.03%|50.60%|
|Sailor|49.89%|44.16%|
|Mistral|51.38%|48.95%|
|Equal soft vote|56.16%|51.24%|
|Best saved-probability calibration|63.78%|55.87%|

Both normalized training arrays and both validation arrays are exactly equal to the baseline. LR/dropout/max epochs changed together, so causal attribution to one knob is not established. Sailor's score continues rising through epoch4, suggesting the slower schedule may need more steps; this remains a hypothesis, not grounds for another blind run. The classifier retune did not improve the ensemble. Equal-vote phrase recall remains0/6;229 exposures per class represent229/136/91/110/207 unique inputs and160/77/28/27/29 original IDs. Exact same-text cross-label conflict count is0. Limited rare-subtype evidence and correlated model errors persist despite resampling.

New corrector: overall GLEU62.84% vs63.13%; error-only24.85% vs23.77%; exact errors7/38 vs3/38; clean40/40 both. Its W&B classification/combined values use the classifier's predictions; the corrector has no label head. Combined falls59.65% to57.04%, mostly from classification.26/38 abbreviated/annotation-like references yield aggregate GLEU0 in both runs; the12 standalone references score74.22% vs73.95%. Flagging is not proof gold is wrong, and no validation targets were repaired.

Native baseline decoder full78: greedy/repetition1.05 all GLEU63.1335%, greedy/repetition1.0=63.4938%, contrastive=62.0745%. This native result supersedes the NF4 proxy's decoding ordering. Beams2/4/8 score32.71% on the ten-row panel and are not full78 results. Old calibrated classification plus old repetition1.0 correction recomputes combined63.6347% on the same78 reused selection rows. This is not an independent or public score; no public ZIP for this selected combination is claimed. Challenger decoding never ran.

## Timeout bug and fix

The outer retuned-models subprocess is a coordinator for four fits, reload verification and its normal public export. Applying the6h ordinary-stage cap to that whole coordinator terminated it during443-row public correction, before actual candidate selection. Use the remaining9h experiment budget for that coordinator and retain the ordinary6h stage cap elsewhere. Defer the intermediate two-phase public export to the tuning parent, so public classification/correction happen after candidate selection. A selected fresh challenger has no cached public greedy output; explicitly generate it instead of assuming results.json exists. Internal CLI control is committed by the parent; no notebook switches. Training/decoder settings and data are unchanged. The fix does not magically finish the already stopped run or extend the9h limit.

220 unit/regression tests passed, including a real short coordinator surviving the ordinary-stage allowance and ordinary-stage timeout cleanup. A real tiny CPU end-to-end controller completed baseline decoding, four new fits/reloads, deferred intermediate export, challenger decoding and a valid final selected ZIP preserving integer/string IDs. One temporary verifier assumed3 rows; the immutable fixture has2, and independent schema/CRC/typed-ID checks passed without rerunning fits. Full-size dual-T4 execution of this updated controller remains pending. Private comparison, hashes and full results are linked in the paired JSON.

Recommendation: retain the stronger old classifier and address supported rare-subtype training supervision before more broad tuning. Do not discard these completed adapters or describe the interrupted challenger dashboard as the final selected pipeline. No new dataset generation, relabeling, model replacement or expensive recovery job was started for this investigation.

Screening-population caveat: each10-row decoder panel has2 clean and8 error rows; full78 has40 clean and38 error rows. The32.71% beam panel score is not comparable directly to the63% full-set corrector score.
