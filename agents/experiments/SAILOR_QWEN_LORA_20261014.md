# Sailor classifier / Qwen corrector: standard LoRA preparation

The user requested standard LoRA instead of QLoRA. The active Kaggle recipe trains a fresh Sailor-7B-Chat classifier and Qwen2.5-7B-Instruct corrector, using the approved full-training population including all81 former validation rows in both stages. Status: locally verified preparation; full-size Kaggle training and official results are pending. Source parent commit: `8e5eaae05da842a9d7ec9ee4855fc3aa210924f5`; exact configuration/data/artifact hashes are in the companion JSON. The runtime journals the actual executing commit.

| Setting | Classifier | Corrector |
|---|---|---|
| Base | sail/Sailor-7B-Chat | Qwen/Qwen2.5-7B-Instruct |
| Revision | 19066fae0a8a3ba029c190d8e3dacccf4d1234b8 | a09a35458c702b33eeacc393d103063234e8bc28 |
| Native parameters with adapters and head allowance | 7,139,016,704 | 7,655,986,688 |
| Rows / epochs / optimizer updates | 585 /5 /370 | 495 /6 /372 |
| Learning rate | 5e-5 | 2e-5 |
| Selection | Final epoch | Final epoch |

Both stages use FP16 frozen bases and FP32 trainable LoRA tensors/head, rank16/alpha32/dropout0.05, AdamW, effective batch8/microbatch1, nonreentrant gradient checkpointing, max512 tokens and seed42. There is no bitsandbytes loading or k-bit preparation. Models run sequentially, each with one process (`world_size=1`) splitting contiguous decoder layers across both T4 GPUs; do not replicate complete FP16 bases with two DDP processes. Classifier uses its existing weighted soft hierarchical objective; corrector learns response-only JSON with no predicted/gold labels. No folds, hints or notebook experiment switches.

Native meta audits estimate classifier requirements11.081/11.465GiB and correction11.802/11.802GiB per GPU, including10% weight margin and4GiB training reserve. These are conservative estimates tested against simulated15GiB free memory, not physical T4 measurements. Inference uses a separate1GiB reserve and classifier work runs in a child process that exits before correction.

## Full-training data and upload

Upload the private `dataset/kaggle-sailor-qwen-lora-20261014/VGEC26-sailor-qwen-lora-full-training-20261014.zip` and attach it to the notebook with Internet and T4×2 enabled. The enclosing Kaggle dataset name can vary; canonical member filenames and committed SHA256 determine identity. `tools/package_final_training.py` reproduces the package from preserved source bundles and the requested raw arrays. ZIP SHA256: `a0890a985bfe2809d4366d98975e6238aa55c4ddd7cd56abc86bd989ef55ffa3`,889383bytes.

`vgec-final-training-v1` embeds each exact source bundle and its hash, validates the original v3/v4 contract, and requires the final raw rows to equal source train+validation exactly. Classifier additionally revalidates its full source contract against the original organizer file. Empty validation and fraction0 are deliberate; raw arrays cannot silently resplit at fraction0. No original labels/targets are inferred or rewritten. All22 flagged added correction targets remain included and documented privately. Official public443 inputs are inference-only.

The former81 rows are training data for this experiment. Reports leave local Macro/Micro-F1, correction scores and combined score unavailable. Final checkpoint reload probes use one training input and discard its predictions; these establish mechanics, not unseen performance. The notebook still links the public `results.zip`. Historical files, run weights/logs and baseline ZIPs remain intact. This jointly changes model, precision and training population, so a future public score difference does not isolate a single cause.

## Verification and recovered failures

155 unit/contract tests passed in31.734s. Real tiny native-Qwen CPU and CUDA pipelines completed nonquantized LoRA for both stages, final checkpoint reloads, portable handoffs and a valid two-row ZIP preserving integer/string ID distinction. Their manifests assert no quantization/validation/hints and unavailable combined score. Actual pinned full-size architectures, counts, final population/hash checks and preflight passed with simulated dual-T4 memory availability. Both notebooks' code cells parse and contain no saved output. Full-size7B fitting was not run locally.

The first final-fit smoke caught a null-score CLI print; it now reports unavailable scores safely. Real CUDA training then exposed FP16 gradients in the PEFT saved classifier head; all trainable tensors are now FP32 while frozen base tensors remain FP16. A native-half PEFT regression checks this explicitly. A later toy model generated an unterminated string and strict export rejected it; the fixture now generates a complete JSON token followed by EOS, without weakening production validation. Five historical reuse tests required explicit historical .2 protocol configuration after the active recipe changed to fraction0. All failed artifacts are retained and indexed in the companion JSON.

Detailed local logs and preflight evidence live in `runs/lora-readiness-20261014/`; actual tiny run paths are in the JSON. No new synthetic examples, inferred label repairs or historical training reruns occurred. Await the actual Kaggle version before claiming full-size success or score improvement.
