# No-fold implementation checks — 2026-10-13

Full-size Sailor-7B training and official evaluation are pending. No score improvement is established by these checks. All training below used newly created tiny fixtures; no historical full-size experiment was rerun.

- 130 unit/contract tests passed; log: `runs/audit-kano-20261013/no-fold-tests-final.log`.
- Real single-GPU CUDA NF4/FP16 pipeline passed with no OOF artifacts and no prompt hints; summary: `runs/audit-kano-20261013/no-fold-cuda-smoke.json`.
- Real two-process CPU pipeline passed with the same contract; summary: `runs/audit-kano-20261013/no-fold-cpu-smoke.json`.
- The production `tools.run_experiment` launcher ran both tiny stages successfully from clean commit `138fdc3476732952581398eb42713423edcada79`. The root ZIP contains only `results.json`; integer ID `1` and string ID `"1"` remain distinct. The run has no OOF hints, and the corrector's saved inference mode is `none`. Summary: `runs/audit-kano-20261013/no-fold-runner-smoke.json`; artifacts: `runs/runner-smoke/tiny-pipeline-20261013T125822655926Z/`.
- Launcher events `started` and `completed` match in `agents/experiments/local.jsonl` and the run's `experiment_events.jsonl`. They record the actual code commit, recipe hash, input hashes and compact outcomes. Local corpus-bearing artifacts and event files remain ignored.
- Saved no-hint classifier reuse and correction-experiment loading were verified against the actual tiny CUDA artifacts. Changed holdout sources, incompatible typed IDs and invalid probabilities are rejected by targeted tests.
- Alpha snapshot source matches commit `47f3a23`, except its documented notebook source-ref pin to `alpha-fold-hints-20261013`. Private data is ignored. Preserved selected weights share verified inodes with the original run and retain recorded hashes; the original run remains intact.

The recovery index contains 105 evidence records: six surviving run metric records, 26 archived note sections, 46 code-history commits, four maintenance records, 21 recovered draft configurations and two negative lexical-probe results. Historical reported results, implementation commits and draft recipes are not treated as verified completed training runs.
