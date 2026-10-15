# Model soft voting, reproducibility and tracking

Latest user decisions: use Qwen7B, Sailor7B and a comparable third model; use **soft voting**; activate the78-row validation package. Mistral7B was selected as the third pinned, ungated Apache-2.0 candidate. Final record: MODEL_SOFT_VOTING_20261015.md/.json.

Implemented: three separate seed42 fits on the identical split42; equal-weight probability averaging; portable hash-bound member bundles; fresh process loading with GPU release gates; literal W&B root named run, child progress and heartbeat; committed default validation selection. No notebook toggles or full-size training. Focused/whole-suite and tiny CPU/GPU evidence are recorded in the final record.

## Superseded initial plan, retained as history

The original three-seed/majority design below was implemented and mechanically tested, then superseded by the user before any full7B run. Its unchecked list records original planning, not outstanding requirements of the final design. The shared-base parameter-budget assumption applies only to that superseded design.

# Three-seed classifier, reproducibility and pipeline tracking

User request: same Sailor classifier trained three times, seeds42/43/44, voting; explicit RNG controls; prevent model-transition memory leaks/OOM; one W&B root named run through all tasks with live metrics. No notebook toggles. Existing dataset arrays/holdouts and prior runs remain immutable. Work inline on kaggle-linux; no subagent delegation. Implement and check within the user's authorized scope, record measured limits.

Design: keep split_seed42 independent of training_seed. Sequential fresh processes train the same pinned model/LoRA/config; only training RNG varies. Majority votes decide classes; mean probability breaks tied votes, then lower label resolves an exact tie. Persist all members and a portable relative-path ensemble manifest. Normalized vote scores retain the downstream five-score/argmax contract, while mean member probabilities and votes are separately recorded. No OOF/hints. Inference loads one member at a time, then the corrector in its own process. Corrector stays seed42 and maximum6 epochs.

One parent W&B run named run has a unique experiment group, seed-specific child names, live safe numerical events and a30-second wall-clock heartbeat while stdout is quiet. It finishes only after export/archive, or records failure on exceptions. GPU-model processes have their own groups, are waited/reaped, and are terminated as groups on interruption. CPU-only nvidia-smi diagnostics check release relative to pre-stage free memory; every child retains its absolute weight/workspace budget guard. No automatic quantization, CPU offload, silent batch/length changes or expensive training retries.

Python hash seed and CUDA workspace are set before child interpreter startup. Python, NumPy, CPU/CUDA torch, sampler/trainer and inference RNGs are seeded. Strict deterministic kernels, disabled cuDNN benchmarking/TF32 and deterministic attention are recorded; unsupported nondeterministic operations fail explicitly. Exact outputs across hardware/library versions and universal OOM absence cannot be guaranteed.

- [ ] Red tests: voting ties/majority, typed IDs, malformed probabilities, member mismatch/copy/reuse; RNG replay/different seeds/fixed split; root lifecycle/live event/quiet heartbeat/failure; GPU release/recovery/timeouts/process cleanup.
- [ ] Add reproducibility.py and independent split-seed config/runtime integration. Test all RNG streams, startup env and Trainer data seeds.
- [ ] Add classifier_ensemble.py for strict voting, bundle creation/validation and sequential inference. Integrate classify/reuse/copy/two_phase; reject incompatible hints/reuse populations. Count each constituent and shared-base plus three adapters against8B.
- [ ] Add GPU lifecycle/process-stage supervision, per-child seeded environment, no weight loading in coordinator, separate inference stage and exit/release diagnostics.
- [ ] Extend TrackingRun with literal root/child names, unique grouping, safe progress events and durable periodic status; verify actual offline SDK behavior and fake-SDK failure cases.
- [ ] Configure default and separate validation classifier recipes for42/43/44. Activate selected dataset through committed recipe, without notebook switches; dataset selection clarification is pending while implementation proceeds.
- [ ] Run focused tests, applicable suite, fresh tiny actual pipeline and repeatability smoke; no full7B training or Kaggle launch. Inspect saved member voting/submission and offline root history.
- [ ] Update agents operational docs/history/results and README, commit/push only code and aggregate records. Preserve private text/weights/artifacts and original datasets.
