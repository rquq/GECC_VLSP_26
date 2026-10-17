# Private-input fresh retraining — 2026-10-17

Implemented research candidate; fresh full-size fitting and score improvement remain unmeasured. Supersedes the default evaluation-sweep entrypoint. Existing runs, source bundles and tuning plan are preserved. Initial source commit: 8a4bb01; exact final changes and configuration provenance are bound by Git and runtime experiment manifests.

The previous recipe inferred on public_test.json (443 rows). The supplied private_test.json is a different, unlabeled 2,034-row file; no exact raw input or typed-ID overlap with public input. New default pins its SHA256 0762703e83e59feac963b2e63f2986a351340cf32da07ce2db7d151a8abe6045. Missing/changed input fails before loading weights. Arbitrary Kaggle upload folder names remain supported. Attach only the new private package; no October16 output/checkpoints needed.

## Model decision

Native old/new equal-vote Macro-F1 was .5615999132/.5123875597 on identical78 development rows. Restore the older classifier LR5e-5, dropout.05, maximum5epochs. Three separate pinned bases: Qwen2.5-7B-Instruct, Sailor-7B-Chat and Mistral-7B-Instruct-v0.3. Each uses seed42, standard LoRA r16/alpha32, seven projection targets, FP16 frozen bases/FP32 trainables, microbatch1/effective8, max512, gradient checkpointing, lambda2 subtype loss. Keep checkpoint selection by member development Macro-F1. Existing fresh-process reload/memory release gates remain active.

Weighted SOFT voting uses normalized weights1:1:2, temperature1 and phrase multiplier1.25. All fields are serialized/validated and reused during inference, not just development reporting. These values achieved .6377559133 when selected on OLD development predictions; a sensitivity partition scored .5789647045. These are selection results, not an independent gain, and must not be promised after data changes. No binary-first gating or hard majority vote.

Corrector stays pinned Qwen2.5-7B, LoRA r16/alpha32/dropout.05, LR2e-5, maximum4epochs (older best epoch4), seed42, max512/effective8/micro1. Greedy192 new tokens, repetition penalty1.0, no hints, same minimal-edit/preserve-clean prompt. Old native greedy1.0 scored all-row GLEU+.6349384746; original penalty1.05 scored .6313346045, contrastive .62074457. Beam/contrastive sweeps are preserved as diagnostics rather than rerun by default. New checkpoint selection explicitly prioritizes ALL-row GLEU+, then error-only GLEU+, then clean preservation. Historical configs retain legacy error-only selection. No inference on validation labels or private-input augmentation.

## Dataset decision and limitations

PRIVATE package: dataset/kaggle-quality-private-20261017/VGEC26-quality-private-20261017.zip. Built with tools/package_quality_training.py from the exact prior grouped dataset and supplied private input. It wraps immutable source snapshots, caps deterministic corruption rules at8 canonical rows/rule and2/parent, caps code exposures at2 and original/legacy exposures at4; ellipsis corrector references and annotation-note rows are never oversampled. All321 eligible organizer classifier training rows and309 corrector originals are retained verbatim.78 unchanged organizer development rows remain excluded from fit; two duplicate originals remain withheld under the existing grouping. Full retrain means all four components fit fresh; it does not mean training on this development set.

Class order:0 clean,1 sentence structure,2 phrase structure,3 function word,4 punctuation.

| Stage/view |0|1|2|3|4|
|---|---:|---:|---:|---:|---:|
|Classifier canonical|229|141|67|80|100|
|Classifier unique sources|229|136|67|80|100|
|Classifier exposures|229|229|212|229|229|
|Corrector canonical|229|140|61|77|98|
|Corrector exposures|229|229|149|185|219|

Code rows267→106 per stage. This removes operator overrepresentation, not proven bad annotations: capped rows remain in embedded immutable source/audit material. Existing clean parents already cover selected corruptions, so zero new matched controls were needed. This candidate DOES NOT add independent samples; retained rows are mechanically screened, not human-adjudicated. Do not equate229 exposures with229 independent rare errors. Corrector shortfalls are deliberate: protect incomplete references and avoid filling quotas with a few code errors. Prior dataset/run remains available for controlled comparisons.

Cross-stage source/target/parent/family/template/near-duplicate audit passes with zero reported conflicts; exact normalized source/target intersection with private input is also0. These checks do not prove absence of semantic overlap, latent annotation errors or historical model/human exposure. Validation remains repeatedly consulted development data. No new text, replacement labels or grammar examples were authored by an unverified-size assistant.

## External data

Official1038 rules checked2026-10-17: public corpora, pretrained resources and synthetic datasets allowed; every separate model and model-generated external-data source must be<=8B, documented and supplied to organizers on request. Follow licenses and private VLSP agreement independently. Native architecture-plus-adapter counts remain7.111B/7.139B/7.156B classifiers and7.656B corrector, each below8B; ensemble aggregate is not a shared7B model.

Existing UD-VTB licensed train parents are retained (CC-BY-SA4.0; pinned revision3c971fd1c5d03561e00b267d3d0057b81a9bea84; license/source notice included). No new external corpus admitted. yammdd's correction corpus declares MIT but explicitly uses an unidentified LLM labeler; <=8B eligibility is unresolved. bmd1905 lacks sufficient verified provenance. VSEC is a possible human spelling/correction source after exact license verification; spelling errors cannot automatically become the four grammar subtypes. Metadata decisions and sources are in EXTERNAL_DATA_REVIEW_20261017.json.

Next evidence-based data expansion should add different phrase/function-word mechanisms with eligible independent author/reviewer models, preserve meaning/facts and matched clean vocabulary, abstain on uncertain subtype, and group all derivatives by parent before splitting. Evaluate per mechanism/unique parent as well as sample totals. This preparation does not claim such generation has completed or that70–80% Macro-F1 is achievable from the existing corpus.

## Verification

See paired JSON for completed checks and exact artifact paths. Full-size training is prepared for Kaggle T4×2 and has not been launched locally or remotely by this change. Runtime environment, data/hash and model memory checks remain active; no blanket zero-OOM or unseen-score guarantee.
