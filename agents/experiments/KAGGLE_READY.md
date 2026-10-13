# Kaggle upload and six-epoch Qwen run — 2026-10-13

**Outcome imported on 2026-10-14:** both full-size Qwen stages completed in the supplied 13_10_2026 run. Public export then failed at the inference memory guard; saved-checkpoint recovery and the fix are documented in QWEN_RUN_20261013.md/.json. The text below records the original preparation, not the latest run status.

The user requested a readiness check, experiment recap, upload ZIP and explicit push of six corrector epochs. The current recipe trains fresh Qwen2.5-7B-Instruct classification for 5 epochs and correction for 6 epochs, selecting the best evaluated corrector adapter. Both use pinned revision `a09a35458c702b33eeacc393d103063234e8bc28`. No folds, hints, notebook toggles or prior adapter reuse. Full-size Qwen training and new official scores remain pending; local readiness is not a completed Kaggle run.

Upload `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/dataset/kaggle-qwen-ready-20261013/VGEC26-qwen-reviewed-v4.zip` as a private Kaggle dataset and attach only this data version. ZIP size: 396,081 bytes (about 387 KiB); SHA256 `3813d5c80e7031db4540698d863bacd3825a0c802097f12f700bc8c5480e28f8`. Members at root: original corpus JSON, VGEC_classifier.json, VGEC_balanced.json, public_test.json and manifest.json. No model weights or private review logs are packaged. The old 2.47 GiB saved-classifier handoff remains preserved separately and is unnecessary for this fresh run. Qwen base weights download through Hugging Face when the notebook loads the model; enable Internet.

The new recipe names this data-only ZIP. Its original/classification/correction bytes and pinned hashes are unchanged: 504 classifier training rows, 414 corrector training rows, 81 frozen holdout rows, 443 public inference inputs. The corrector has 222 usable originals including five supported repairs and 192 exact legacy augmentation rows. No new synthetic pairs were admitted. Public inputs are inference-only.

Readiness checks passed: ZIP CRC and exact entry/hash checks, actual classifier/corrector provenance and matching holdout, native Qwen tokenizer with maximum 222 correction tokens under 512, both-stage fresh launch without reuse, native parameter/LoRA bounds below 8B, 141 unit tests in 17.580 sec, and simulated T4x2 FP16 placement using the actual archive/configs. The earlier real tiny-Qwen two-stage CPU submission smoke also passed. The only new runtime choice is 6 corrector epochs; the linear schedule now ends at 6. No additional full-size training or teacher generation occurred for packaging.

Import notebooks/kaggle_two_phase.ipynb or use the existing launcher that clones the latest kaggle-linux branch. Enable Internet, choose T4x2 and grant GITHUB_TOKEN access in Kaggle Secrets. Attach the private ZIP (it already contains public_test.json). Optional WANDB_API_KEY enables online tracking. Run All/Save and Run All follows the committed recipe and exports run/results.zip. Source parent for this change is d666f26; the commit containing this record/config hashes is the implementation, and the launcher logs the actual runtime commit. KAGGLE_READY.json contains exact config hashes and private audit paths.

| Experiment/review | Outcome |
|---|---|
| Previous Sailor-7B | Local classifier Macro-F1 0.434133 / Micro-F1 0.679012. Corrector best at epoch 5, error-only GLEU+ 0.234896; epoch 8 was 0.229956. Its old final fresh-base reload failed after training; current code reuses the base after releasing trainer owners. |
| Qwen teacher synthesis | 64 attempts, 0 admitted pairs; no weak filler promoted. |
| Native Sailor second opinion | 5 candidate completions reviewed, 0 admitted. |
| Bami hierarchical data screen |Internal Macro-F1 0.408220 control versus 0.247619 candidate; selected control holdout 0.325577, below saved Sailor. No Bami promotion. |
| Annotation repairs |Five supported complete-source clean repairs accepted for correction. Five reference completions remain withheld. |
| Qwen/SeaLLMs audit |Both audited for size/tokenizer. SeaLLMs excluded as teacher for other LLMs under its linked license; no SeaLLMs inference or training. |
| No-fold Qwen pipeline |Two fresh stages configured and mechanically verified; full-size Kaggle run pending, now 5/6 epochs. |
| Historical recovery/preservation |Recovered surviving alpha/test/Kaggle metrics, notes and failures without repeating training; five-fold system preserved in alpha. |

The older Bami flat-CE and Sailor2 historical results use different protocols and are not matched new comparisons. No classifier gain or 70–80% Macro-F1 has been established by this preparation. All corpus-bearing data remains private and ignored; only aggregate readiness and experiment records enter Git.
