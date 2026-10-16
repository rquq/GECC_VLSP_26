# Actual contrastive output compatibility fix — 2026-10-17

The user clarified that merely skipping contrastive search was insufficient. The earlier3f2f543 patch contained the abort; this change repairs the reproduced inference path. No failed-run cache or training continuation is added. Same model, data, LoRA/training/decoder parameters, package pins and notebook launcher.

## Reproduced cause

A real tiny Qwen LoRA model reproduces the exact Kaggle TypeError when given Accelerate1.6 AlignDevicesHook(io_same_device=True), the root output-alignment path used by native multi-GPU dispatch. Accelerate send_to_device reconstructs mapping subclasses as type(output)({key:value,...}). For CausalLMOutputWithPast this assigns the dictionary to the first dataclass field, loss. ModelOutput expands the dictionary into its mapping, but the loss attribute remains a dictionary even though inference emitted no loss key. Transformers4.51.3 low-memory contrastive search stacks every dataclass field using getattr, reaches loss=dict and raises Unexpected attribute type. The local one-device NF4 path did not exercise this alignment hook. The tiny reproduction proves this hook interaction without needing a full7B checkpoint or two GPUs.

## Repair and scope

During low-memory contrastive model.generate only, attach a temporary forward hook to the underlying causal base model, including beneath PEFT. Reconstruct ModelOutput with keyword arguments from its actual mapping. An absent loss becomes None; actual losses, logits, hidden states and cache objects are preserved without tensor copies, movement or changes to the contrastive algorithm. Remove the hook in finally on success or exception. Greedy, beam strategies and malformed-JSON beam retry use their existing paths. Failure isolation remains for unexpected errors but is not the compatibility mechanism.

## Verification

Before integration, the real hooked model raises the exact exception, including through production decode. After integration it completes actual contrastive generation and matches tokens from the unhooked control. Tests assert cache/logit/true-loss identity and hook cleanup. Both Python3.10/Torch2.7.1 and an isolated Python3.13.7/Torch2.11.0 CPU environment with pinned Transformers4.51.3/Accelerate1.6/PEFT0.15.2 pass the two real compatibility tests. All217 regression tests passed in42.507seconds. A real tiny FP16 Qwen LoRA model on the RTX4060 reproduces the same pre-fix exception with the alignment hook, completes contrastive search after the fix and matches unhooked tokens. Formatting/diff checks passed. Full-suite and CUDA outcomes are recorded in the paired JSON. A full7B two-T4 run remains pending; this is mechanical compatibility evidence, not a new score claim.
