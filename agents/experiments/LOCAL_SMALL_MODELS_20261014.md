# Local smaller-model results — 2026-10-14

Completed: nine distinct architectures, 13 screening configurations (12 completed, one divergent), then two primary refits and three exploratory secondary refits. Every completed checkpoint passed local save/reload verification. All six submission ZIPs contain443 rows; user PNG public results are now recorded in PUBLIC_RESULTS_20261014.md/.json, with one candidate attribution unresolved. Exact revisions, configs, hashes, metrics, budgets and artifact paths are in LOCAL_SMALL_MODELS_20261014.json.

## Leaderboard diagnosis and metric change

The user screenshot shows KANO entry 3: combined 0.5282, GLEU+ 0.7373, exact match 0.5144, ROUGE-L 0.8357, Macro-F1 0.3191, Micro-F1 0.5192. Entry 2 was combined 0.5349, GLEU+ 0.7224, Macro-F1 0.3473, Micro-F1 0.5485. Classification fell by 0.0282 while correction GLEU+ improved by 0.0149: their equal-weight contributions imply a combined change of -0.00665 from rounded scores. Micro-F1 implies 230 versus 243 correct decisions out of 443. Labels changed on 229 public rows and corrections on 155; public gold labels are unavailable. LEADERBOARD_20261014.json preserves this evidence.

Local combined version 2 is `(Macro-F1 + all-row GLEU+) / 2`, including clean rows. Version 1 used error-only GLEU+ and is numerically incomparable without recomputation. Saved Qwen predictions yield local v2 0.612173 versus historical v1 0.406882; no predictions changed. Error-only GLEU+, clean preservation and exact match remain diagnostics. `official=false` identifies the local GLEU+ approximation. Production Qwen corrector checkpoint selection is unchanged.

## Protocol and data limits

Classifier screen: 440 fit / 64 training-only validation rows (32 clean, 16 sentence, 5 phrase, 5 function, 6 punctuation). Correction screen: 373 fit / 41 validation (32 clean, 4 sentence, 1 phrase, 3 function, 1 punctuation). Both use grouped parent/template families, seed 42, immutable reviewed bundles and the unchanged 81-row holdout. Separate task splits avoid the correction quarantine leaving almost no classifier validation errors. A family component of 113 rows and only nine internal correction errors make the correction screen weak evidence of generalization.

Primary models were selected internally, then refitted on all 504 classifier / 414 correction training rows for fixed selected epochs before evaluating the holdout. The disappointing Bami holdout result triggered three explicitly exploratory secondary refits: PhoBERT-large five epochs, Qwen-0.5B three epochs and BARTpho one epoch. These epochs came from internal selection; LR/data stayed fixed. The 81-row holdout has prior exposure, and the extension follows new holdout exposure: neither is independent test evidence.

No corpus bytes, original IDs, holdout references or public inputs were modified. No new synthesis, inferred replacement labels or historical training reruns occurred. Private text, segmented inputs, predictions and weights stay ignored under runs/local-small-models-20261014/. Only code and aggregate records enter Git.

Correction reference audit flags 0 annotation-like error targets internally versus 22/41 on the frozen holdout. Flags are based on parenthetical beginnings or literal ellipses, not adjudicated replacement gold. Whole-split copy baseline GLEU+ is 0.932610 internally and 0.597583 on the holdout; error-only is 0.664076 and 0.0 respectively. Thus a high all-row overlap score does not demonstrate successful error correction. Holdout references are never rewritten.

## Screening results

Classifier Macro-F1 below uses the internal 64-row split. Correction all/error GLEU+ uses the internal 41-row split; these are not official or frozen-holdout scores.

| Configuration | Parameters with adapters | Selected epoch | Internal Macro-F1 | Internal all/error GLEU+ |
|---|---:|---:|---:|---:|
| phobert-base | 135,002,117 | 3 | 0.414323 | — |
| phobert-large | 369,168,389 | 5 | 0.576084 | — |
| bami-flat | 102,955,013 | 5 | 0.600198 | — |
| xlmr-base | 278,047,493 | 4 | 0.295063 | — |
| visobert | 97,569,797 | 3 | 0.441550 | — |
| bartpho-syllable | 395,814,912 | 1 | — | 0.926460 / 0.664076 |
| vit5-base | 225,950,976 | 6 | — | 0.930467 / 0.671082 |
| vit5-large-lora | aborted | — | — | divergent |
| bami-flat-lr5e5 | 102,955,013 | 5 | 0.220559 | — |
| qwen05-flat | 494,037,248 | 3 | 0.527884 | — |
| bami-flat-originals | 102,955,013 | 6 | 0.525782 | — |
| vit5-base-edit5 | 225,950,976 | 5 | — | 0.933825 / 0.686716 |
| vit5-large-lora-stable | 796,003,328 | 5 | — | 0.545562 / 0.458131 |

Primary selection was Bami-flat plus ViT5-base-edit5, raw correction mode. Tokenizer-only identity restoration was assessed on internal predictions: it improves BART/ordinary ViT5 clean copying, but the selected edit-weighted raw checkpoint retained the highest internal all-row GLEU+. No gold labels or targets are used by that helper, and blank generations are never replaced with source copies.

Full classifier fitting uses sqrt inverse-frequency weighted five-class cross-entropy. Mean per-row weighted loss preserves class influence with microbatch one. Effective batch eight, BF16 autocast, max length 256, warmup 10%, decay 0.01; full fine-tuning uses AdamW with 8-bit optimizer states above 200M parameters. Correction is native sentence-to-sentence seq2seq, with no JSON prompts or classifier hints. ViT5 edit5 applies weight five to changed target tokens and a deletion anchor; padding contributes no loss and edit metadata is removed before generation. CompleteEpochMixin retains partial accumulation windows. ViT5-large uses rank16/alpha32/dropout0.05 q/v LoRA. Candidate recipes record immutable revisions and exact learning rates; trial budget is 90 minutes.

## Frozen-holdout outcomes

| Classifier | Macro-F1 | Micro-F1 / accuracy | Full-data epochs | Measured refit and reload time |
|---|---:|---:|---:|---:|
| bami-flat | 0.414722 | 0.641975 | 5 | 1.57 min |
| phobert-large | 0.536678 | 0.765432 | 5 | 4.35 min |
| qwen05-flat | 0.546681 | 0.691358 | 3 | 3.93 min |
| Saved Qwen-7B baseline | 0.595176 | 0.753086 | 5 | Historical Kaggle run |

PhoBERT-large gets 62/81 decisions correct and 40/40 clean, but 0/7 phrase errors. Bami gets 52/81 correct and 0/7 phrase. Qwen-0.5B gets 56/81 correct, 2/7 phrase and 4/7 function errors. None beats saved Qwen-7B Macro-F1; no 70–80% Macro-F1 claim is supported. PhoBERT reaches 76.54% accuracy, which is a different measure.

| Corrector | All-row GLEU+ | Error-only GLEU+ | Error exact matches | Clean preserved |
|---|---:|---:|---:|---:|
| BARTpho, one epoch | 0.602319 | 0.105179 | 0/41 | 40/40 |
| ViT5-base-edit5, five epochs | 0.607623 | 0.139081 | 1/41 | 40/40 |
| Saved Qwen-7B, selected epoch five | 0.629170 | 0.218588 | 3/41 | 40/40 |

On the 19 standalone error references, ViT5-base-edit5 GLEU+ is 0.569497 versus Qwen 0.573910; BARTpho is 0.499640. ViT5 is therefore a useful inexpensive correction candidate, although it has fewer exact fixes and does not establish a full-holdout/public win. Its native full-data fit and reload took 5.72 minutes on the RTX4060; BARTpho took 1.49 minutes.

Combined local v2: Qwen/Qwen 0.612173; Qwen-0.5B/Qwen 0.587925; PhoBERT/Qwen 0.582924; Qwen-0.5B/ViT5 0.577152; Bami/Qwen 0.521946; Bami/ViT5 0.511173. These combinations share fixed component predictions; they do not imply separate training trials or official scores.

## Negative results and recoveries

- Bami LR5e-5 falls to internal Macro-F1 0.220559; increasing LR is rejected.
- Bami originals-only reaches internal Macro-F1 0.525782 versus mixed-data 0.600198. Nine epochs roughly match optimizer updates (279 versus 275), so wholesale removal of augmentation is unsupported.
- Prior scaling powers 0/0.5/1 lowers Bami from 0.600198 to 0.570164/0.516299 and PhoBERT-large from 0.576084 to 0.552204/0.543961. Retain plain argmax. XLM-R improves to 0.461075 at power0.5 but remains weaker.
- ViT5-large BF16/LR2e-4 diverged: logged loss 4.5496 → 7.256 → 24.0215. Stopped before completing epoch two; retained logs/checkpoint, excluded from selection. One FP32/LR5e-5 recovery completed and reload verified, but clean preservation was 0/32 and internal GLEU+ only 0.545562. Precision and LR both changed, so the cause is not isolated. The runner now aborts nonfinite or logged loss above10.
- Initial PhoBERT-base was interrupted to repair the microbatch weighted-mean cancellation before a completed fit; excluded from results.
- A private queue.py shadowed the standard library during prefetch and triggered duplicate startup. Output guards blocked overwrites; launcher renamed and extra parent stopped. Two valid early classifier fits overlapped, so their screen wall times are not fair isolated speed comparisons. Later GPU fits were sequential.
- Adoptium setup HTTP403/read-only repeat extraction recovered through verified public asset metadata/checksum and idempotent extraction; native VnCoreNLP RDR segmentation was used for PhoBERT.
- First three successful classifier source commits were captured at completion with HEAD unchanged throughout. Later trials capture commit/code hash before fitting. Initial roster expanded with bounded variants/recovery; each candidate has its own revision/config/provenance.

## Submission artifacts and verification

Best-known eligible hybrid combines actual Sailor-7B-Chat classifier labels from the 12 October run with latest Qwen corrections. Sailor classifier conservative upper bound: 7,139,016,704 parameters; Qwen corrector including adapter: 7,655,986,688. The historical 11 October Sailor2 component is ineligible and is not used. The originally predicted combined0.5423 now matches the user-supplied public hybrid PNG. Local hybrid combined is 0.531651, illustrating the local/public ranking mismatch. HYBRID_20261014.json identifies sources and archive hashes.

Six private root-only ZIPs are verified: best-known-hybrid; Bami/Qwen; Bami/ViT5; PhoBERT-large/Qwen; Qwen-0.5B/Qwen; Qwen-0.5B/ViT5. All have exactly443 typed IDs, labels0–4, singleton correction lists, valid CRC and only results.json at root. Public tokenizer lengths fit256; native PhoBERT segmentation is preserved. Source predictions are unchanged in the hybrid. Public screenshots for all six folders are now available; Bami/ViT5 attribution is unresolved. Production Kaggle Qwen recipe and simple notebooks remain unchanged; no experiment toggles or automatic model promotion.

All147 unit/contract tests passed in27.473s after the final code changes. All12 completed screens and five final refits passed actual local save/reload verification. Original datasets and public input hashes are unchanged. The original Qwen classifier replay matches80/81 saved Kaggle decisions with unchanged Micro-F1 and max probability difference0.004173; one almost tied wrong clean decision becomes a wrong sentence decision. FP16 and batch1 match; local Torch2.7.1/CUDA11.8 on RTX4060 differs from Kaggle Torch2.11/CUDA12.8/T4. This is recorded as a numerical discrepancy, not exact reproduction. The separately verified original Qwen corrector replay remains81/81 exact.

## Rules and next decisions

Every individual component and synthetic generator must be at most8B; quantization does not reduce parameter count. External resources must be documented and accessible to organizers. Corpus/derivatives remain private under DATA_GUIDELINES.md. Native models are below800M including adapters. PhoBERT/BARTpho/ViT5/XLM-R are MIT; Qwen-0.5B Apache2; Bami BSD-3-Clause-Clear plus Qualcomm Responsible AI license; ViSoBERT model card grants research use only. Source revisions/licenses are in committed configs.

The component-preserving hybrid is now the best supported public result0.5423. The reported native classifier submissions do not beat its Macro-F1. Retain Qwen corrector for the immediate known public score. ViT5 is the fast correction research option. Future curation should target rare subtype consistency, varied minimal error/clean pairs, parent diversity and complete standalone references. Review training-only flags through documented human or eligible local-teacher evidence; preserve original annotations and pending uncertain cases. Build meaningful grouped training-only error coverage rather than changing the existing81 holdout or adding unreviewed repetitions.

Primary sources: [competition rules and evaluation](https://aihub.ml/competitions/1038#learn_the_details-evaluation), [PhoBERT](https://github.com/VinAIResearch/PhoBERT), [BARTpho](https://huggingface.co/vinai/bartpho-syllable), [ViT5](https://huggingface.co/VietAI/vit5-base), [BamiBERT](https://huggingface.co/Qualcomm-AI-Research/BamiBERT), [ViSoBERT](https://huggingface.co/uitnlp/visobert), [Qwen-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).


## Subsequent public screenshots

PUBLIC_RESULTS_20261014.md/.json records all six uploaded folder PNGs and reverified ZIP hashes. Hybrid0.5423, PhoBERT/Qwen0.5219, Qwen0.5/Qwen0.5191, Qwen0.5/ViT50.5145, Bami/Qwen0.4983. Bami/ViT5 PNG also displays0.4983 but repeats the Qwen correction fields, so its candidate attribution is unresolved. Local ranking did not predict public ranking. No training or data change occurred for this follow-up.
