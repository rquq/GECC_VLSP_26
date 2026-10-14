# Public submission screenshots — 2026-10-14

Six user PNGs were visually inspected in the six candidate folders. Scores below transcribe the KANO rows. Folder placement identifies the intended candidate; authenticated submission details and uploaded server ZIPs are unavailable. Entry count is not treated as a verified submission ID. ZIP hashes match the previously verified artifacts.

| Intended folder / models | Entry counter | Combined | GLEU+ | Exact match | ROUGE-L | Macro-F1 | Micro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sailor-7B / Qwen-7B | 5 | 0.5423 | 0.7373 | 0.5214 | 0.8357 | 0.3473 | 0.5485 |
| PhoBERT-large / Qwen-7B | 6 | 0.5219 | 0.7373 | 0.5214 | 0.8357 | 0.3065 | 0.6208 |
| Qwen-0.5B / Qwen-7B | 7 | 0.5191 | 0.7373 | 0.5214 | 0.8357 | 0.3010 | 0.6027 |
| Qwen-0.5B / ViT5-base-edit5 | 8 | 0.5145 | 0.7279 | 0.5440 | 0.8231 | 0.3010 | 0.6027 |
| BamiBERT / Qwen-7B | 9 | 0.4983 | 0.7373 | 0.5214 | 0.8357 | 0.2593 | 0.5508 |
| BamiBERT / ViT5-base-edit5 † | 10 | 0.4983 | 0.7373 | 0.5214 | 0.8357 | 0.2593 | 0.5508 |

Sailor7B/Qwen7B is the best supported result among these screenshots: **0.5423**, matching the predicted hybrid score. Gain is **0.0141** versus the prior Qwen/Qwen entry0.5282 and **0.0074** versus the earlier Sailor/Sailor entry0.5349. The rank column is snapshot-specific; this does not establish the current global leaderboard rank.

The smaller classifiers improve some overall accuracy numbers but reduce Macro-F1: PhoBERT0.6208 Micro versus Sailor0.5485, yet0.3065 Macro versus0.3473. Classification balancing/subtype discrimination remains the ranking bottleneck. All four Qwen-corrector archives have identical corrected sentences and identical displayed GLEU+0.7373; the classification differences explain their combined ranking.

Qwen0.5/Qwen versus Qwen0.5/ViT5 is a component-controlled public comparison: identical classifier labels, different correction fields. ViT5 has lower GLEU+0.7279 versus0.7373 and lower ROUGE0.8231 versus0.8357, but higher exact match0.5440 versus0.5214. Preserve this tradeoff; do not call ViT5 worse on every correction measure. Historical entry3 PNG exact match0.5144 is retained as originally observed rather than silently rewritten to0.5214.

## Unresolved Bami/ViT5 attribution

† The PNG in candidate-small-both shows entry counter10 but repeats the Bami/Qwen counter9 scores exactly, including GLEU+0.7373 / EM0.5214 / ROUGE0.8357. The verified Bami/ViT5 ZIP contains the same Bami labels, but its corrections differ from Qwen on174 rows and are exactly identical to the scored Qwen0.5/ViT5 correction fields. Thus the PNG cannot confidently establish the Bami/ViT5 score.

If the same test/scorer evaluates fields independently, its arithmetic expected score would be0.4936, using Macro0.2593 and the observed ViT5 GLEU0.7279. This is an inference, **not an official observed result**. Pending scoring/cached leaderboard, wrong upload/screenshot or another selected submission are possible; none is established. Inspect the individual submission result/status and uploaded ZIP checksum before resolving this record. Do not retry/upload automatically.

## Artifact and record checks

All six archives preserve their recorded SHA256, contain only root results.json, pass CRC/schema checks and cover443 exact typed IDs. All four Qwen correction fields match; both ViT5 correction fields match; Bami labels match across its two archives. Displayed combined scores agree with half Macro-F1 plus half GLEU+ within rounded-display tolerance. Source PNG paths/hashes and source ZIP hashes are in PUBLIC_RESULTS_20261014.json. Screenshots and corpus-bearing outputs remain private. No training, dataset changes, notebook toggles or production recipe changes occurred.

Best supported submission remains Sailor7B/Qwen7B. Next work should target rare subtype supervision and validation representativeness; neither improved Micro-F1 nor the local81 ranking establishes improved public Macro-F1.
