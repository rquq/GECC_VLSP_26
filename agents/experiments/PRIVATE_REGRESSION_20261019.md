# Private results and corrected next-run decision — 2026-10-19

The user identifies0.5307 as the October19 Sailor7B + oldQwen7B submission on1039. It is not a Sailor20B/Qwen14B result. The user supplies the exact0.5806 quality-system result on1038. These are user-reported official metrics; no authenticated leaderboard or hosted reference identity was independently inspected. Paired JSON records source paths, ZIP/input hashes, counts and score arithmetic.

| Metric | Quality system,1038 | October19 focused,1039 | Focused minus quality |
|---|---:|---:|---:|
| Combined |0.5806|0.5307|-0.0499 reported|
| Macro-F1 |0.4149|0.3178|-0.0971|
| Micro-F1 |0.6659|0.4876|-0.1783|
| GLEU+ |0.7463|0.7435|-0.0028|
| Exact match |0.5237|0.5147|-0.0090|
| ROUGE-L |0.8241|0.8108|-0.0133|

Using the displayed component metrics,0.5*(Macro+GLEU) gives0.5806 and0.53065. The unrounded difference is0.04995, of which0.04855/97.2% comes from classification. Rounding explains the last displayed decimal; do not fabricate per-row accuracy counts from the rounded Micro-F1.

Both local submissions have the same2034 typed private IDs in order. Quality class counts0..4 are1029/842/63/41/59; focused counts381/1121/201/211/120.902 labels change.685 quality-clean predictions become focused-error;37 change in the other direction.1560 correction strings are identical. This supports error over-detection as a leading hypothesis. Without private gold, these changed decisions cannot individually be called false positives or mislabeled data.

The quality classifier is the Qwen/Sailor/Mistral calibrated soft ensemble, weights1:1:2, temperature1, phrase multiplier1.25. The focused classifier is one newly trained Sailor7B. Dataset mixture, ensemble versus single model and training choices changed together. Therefore this comparison does not isolate data as the cause. The repeatedly reused78-row development Macro68.99% failed to predict private performance; the previous recommendation to use this focused classifier as the new fallback was too strong and is superseded.

Next default: quality-data20B/14B trial in QUALITY_TRIAL_20261019.md. The exact quality submission is an inference-only control. Its track1039 score is unknown until the human submits that same ZIP there; matching local IDs and task descriptions alone do not establish matching hosted gold/scorer. A verified unchanged copy is at Windows Downloads/VLSP26-quality-control-1039-20261019/results.zip, SHA256d9d14ac425dcc0a3ddf3e1fdd60686bbb09bf59293106845e3683dba6f859f79. No submission, quota use or organizer message was automated.

Preservation: allsix original protected artifacts were rehashed against backups/track1038-quality-20261019/preservation_manifest.json. The archive tag, source/docs backups, original data/configs/adapters/submission remain unchanged. New exact official scores are recorded here, rather than rewriting the frozen historical report.

Separate diagnostic: the current strict full-ensemble reuse validator rejects the quality ensemble's JSON probabilities on exact float equality. Recomputing from saved members produced150 probability-only differences, maximum3.33e-16, zero label changes. This is serialization precision, not evidence explaining the private regression. Original ensemble artifacts were not rewritten and validation checks were not bypassed. The new trial uses the frozen full-system ZIP and its saved development predictions; it does not reload that ensemble through this validator.

The existing three user-edited experiment files remain untouched. No native weights were fitted or generated samples admitted for this analysis.
