# Training dataset including the former validation rows — 2026-10-14

The user clarified that the requested new dataset was for training and must include the81 validation rows. The earlier validation-only export was a misunderstanding, not the requested training dataset. It remains preserved; this separate export corrects the role/membership.

Private ZIP: `/home/quq/machineLearning/competitions/VLSP26_4/kaggle-linux/dataset/full-training-including-validation-20261014/VGEC26-full-training-including-validation-20261014.zip`. SHA256 `b06ede54133787c78b3a5475f041849b1dd8f3072382c0789937d22c1ad96b50`; 112210 bytes. All81 former validation rows are now present in **both** training arrays:

| Stage | Existing train | Added former validation | New training total |
|---|---:|---:|---:|
| Classifier | 504 | 81 | 585 |
| Corrector | 414 | 81 | 495 |

Files are classification_train.json, correction_train.json, training_manifest.json, added_correction_review_queue.json and README.md. Exact stage hashes/class counts and source bundle hashes are in FULL_TRAINING_20261014.json. Source commit `cf153ec1abbada9a132a7583b1b90cfeb4e2cb4c`; exporter code digest recorded.

Every baseline training row and every added raw row is preserved, including IDs, labels, extra fields and correction targets. There are no duplicates by typed ID and no generated wording/new labels. Both stage memberships/load roundtrips and ZIP bytes/CRC passed actual checks. Original corpus, historical held-out split and saved runs remain untouched.

For this new training protocol, the former81 are training data, so they cannot also establish independent validation. This user clarification supersedes the project default that kept these rows held out. The rows originate in organizer training data; official public/private test inputs remain inference-only. This does not alter the model-size, licensing, provenance or privacy rules for augmentation.

The corrector export includes all81 requested rows without filtering. Existing audit rules flag22 added targets for review, and the private queue preserves those flags. Raw inclusion does not certify explanation/ellipsis-style targets as complete correction supervision; no inferred repairs or labels are introduced.

This is a **standalone raw training-data export**, not the prior v3/v4 stage-bundle format. The existing Kaggle recipe still points at the earlier package. A final-fit consumer must load all rows and must not reuse these81 as unseen validation. No pipeline promotion, new augmentation or model training ran in this request.
