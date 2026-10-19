# Preserved quality experiment for track1038

User request2026-10-19: retain the quality-dataset experiment, reported approximately0.58 combined on the leaderboard, including the whole code and documentation for later reporting. Status: preserved; no original data, adapter, run, submission or quality config changed. No training rerun.

Exact run: runs/18_10_2026/quality/runs/quality-private-fresh-20261017-20261017T080116892693Z. Its clean-source manifest identifies338293f90e664803ee3dde12e82dd5d4d5601d43. Immutable Git tag: archive/track1038-quality-20261019. The tag's experiment.json and stage configs preserve the actual historical pipeline; current stage files must not substitute for this historical source. Native Qwen/Sailor/Mistral classifiers and Qwen corrector remain at their original paths.

Private backup: backups/track1038-quality-20261019/. quality-source-338293f.zip contains ALL tracked code, notebook, configs and historical docs from that commit. report-docs-as-of-20261019.zip additionally snapshots current agent documentation, including preexisting report edits. Exact runtime configs/manifests and the quality results.zip are included. preservation_manifest.json inventories hashes/sizes of six protected original adapter/dataset/submission artifacts; large weights are preserved in place, avoiding redundant copies. All backup ZIPs pass CRC. These backups remain ignored/private; only the source tag and aggregate record enter Git.

Restore/read by extracting quality-source-338293f.zip into a NEW directory, or checking out the archive tag separately. Do not overlay the current track1039 checkout. Source restoration requires the existing private quality ZIP and external pinned base downloads. The archive does not embed the huge pretrained base weights.

The user-reported leaderboard combined score is approximately0.58; it has not been re-fetched from the site. Separately verified78-row development metrics are Macro57.2774%, all-GLEU60.5752%, combined0.5892628599865507. Keep leaderboard and reused-development evidence distinct in the report. Full three-dataset analysis remains THREE_DATASET_COMPARISON_20261018.md/.json.

Do not delete this backup, the original quality run, its four adapters, the original quality dataset/upload ZIP, or the archive tag during cleanup. Larger-model trials are explicitly track1039 and do not supersede this frozen1038 report evidence.
