# Data and score plan

The native-model review was approved and executed. LOCAL_REVIEW_RESULTS.md / .json record the failed synthetic pilot, negative Bami comparison and prepared five-repair corrector dataset. The production recipe is now corrector-only with the saved Sailor classifier. The earlier planning sequence below is historical; new work must use current evidence and receive authorization for any expanded budget.

## First comparison: cheap no-fold system

Run the committed two-fit recipe on Kaggle. Keep existing data/model/loss so the first result measures removal of hints. Classifier F1 is expected to be comparable, subject to runtime variation; correction quality must be measured. Compare local per-class counts and official Macro-F1/GLEU+ with the preserved fold system. A lower correction result does not automatically justify restoring five extra fits; evaluate the quality/time tradeoff explicitly.

## Repair original training supervision

The 103-row queue has 69 elliptical sources, 91 elliptical targets and 10 parenthesized annotations (overlapping counts). Treat flags as uncertainty. Eleven one-tail-ellipsis targets admit unique four-token anchoring followed by copying an existing source suffix, with at most four edited words already supplied by the original target. These proposals generate no new wording, preserve original labels and remain unpromoted. Seven rows explicitly say the source is correct despite a nonzero label. The previous eight-row claim included a procedural punctuation annotation; versioned policy v2 fixes that detection while retaining compatibility with the pinned baseline.

Private evidence: `dataset/review/training-repair-proposals.json`. Reproduce with `python -m tools.propose_training_repairs --original <original-json> --queue <private-queue> --output <private-overlay>`.

Review the 11 reconstructable targets first because they require no missing-context invention. Check the complete sentence and subtype independently. Approve only a supported correction and record the rationale/model provenance. Keep elliptical/ambiguous sources quarantined. Add approved complete corrections to a separately versioned bundle. Do not change the original file or validation. In a classifier comparison, first alter supervision/data only; keep the model/loss and budget fixed.

## Diverse external candidates with an eligible local model

Use `Qwen/Qwen2.5-7B-Instruct` as an optional local teacher candidate. Its official model card lists 7.61B parameters and Vietnamese support: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct. Resolve and pin the exact revision and count the actual unquantized causal architecture before use. The laptop has 8 GB VRAM: prefer a verified 4-bit local inference backend with modest context, one request at a time, and a recorded memory limit. Quantization is a capacity aid, not an eligibility exemption. This teacher has not been run or promoted here.

Initial pilot budget: 64 proposed error/clean pairs (16 per nonzero class), plus matched clean counterparts. Require at least five independent linguistic mechanisms for each rare subtype before expanding. Phrase candidates should cover internal phrase ordering, degree/comparison constructions, and attachment; function-word candidates should cover preposition/conjunction/aspect usage and paired constructions; punctuation should include internal boundaries, balanced delimiters and quotation marks. Avoid a pool dominated by duplicated tokens or terminal `.,`.

Prompt contract: provide the task taxonomy and a training-only context/domain, request JSON containing source, complete correction, one error label, mechanism, minimal edit and explanation. No validation/test text is provided to the teacher. Keep topic/domain variety and preserve factual content. Every teacher output is a candidate, not ground truth. Save the actual prompt, revision, parameter count, seed and raw output privately. Use an independent local verification pass plus structural checks; reject subtype disagreement, ambiguous acceptable sources, changed meaning, malformed targets and duplicate families. No larger model may synthesize the examples.

Replace weak single-pattern examples within the existing synthetic budget before increasing its size. Target at most 20% of a subtype's synthetic examples from one mechanism, enforce parent-family caps, and maintain lexical clean controls. Rebuild with documented policy/version and unchanged holdout; commit new hashes only for a reviewed candidate dataset. Evaluate parent-grouped training-only internal validation, then one frozen-holdout comparison. Do not restore a previously rejected head-LR, R-Drop, localization or equal-weight ensemble experiment without a new hypothesis from recovered evidence.

## Screening and promotion

Screen classifier-only candidates so no corrector or folds are trained for each preliminary trial. The revised proposal starts with two cheap BamiBERT hierarchy fits, using current versus reviewed candidate data. If that screen fails to support replacement, retain Sailor. Lower Sailor learning rates (1e-5, 2e-5 versus current 5e-5) are a later separately authorized experiment. Do not launch the deferred PhoBERT, BARTpho or Qwen classifier comparisons as an automatic sweep.

Promote only when rare-subtype recall/F1 improves without a large clean/error regression. Then run the selected classifier with the independent no-hint corrector and measure the joint score. Record all trials, including failures and no gains, under `agents/experiments/`. No 70–80% score is promised.
