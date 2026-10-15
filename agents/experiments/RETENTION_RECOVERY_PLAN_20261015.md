# Retention-first dataset recovery implementation plan

> Planning-only deliverable requested by the user. No dataset promotion, model download, inference, training or Kaggle launch is authorized by this document's existence. For implementation use superpowers:executing-plans inline; do not delegate unless the user or applicable instructions explicitly request delegation.

**Status:** PROPOSED, NOT EXECUTED. The existing augmentation hold and production recipes remain in force.

**Goal:** Retain as much original supervision as possible, recover supported correction targets, retain the existing deterministic pool, and add diverse controlled error examples without collapsing the clean-class distribution.

**Architecture:** An immutable master holds all401 organizer rows and every original field. Separate classifier/corrector views select usable supervision; evidence-bearing overlays supply accepted recovered targets without changing original input or labels. Legacy synthetic data, new deterministic examples and eligible-model examples retain separate provenance, so source weights and acceptance decisions remain auditable.

**Tech stack:** Existing Python/JSON data loaders and grouped evaluation; pinned Hugging Face checkpoints; local single-GPU quantized inference when necessary; existing small-model training recipes for bounded comparisons. Notebook remains a launcher with no experiment switches.

**Spec:** The user's retention/recovery proposal in this chat, refined below. This plan supersedes older planning suggestions only after implementation is explicitly requested; it does not silently change the currently pinned datasets.

## Verified starting point

- Raw source: `dataset/kaggle-sailor-qwen-lora-20261014/VGEC 2026_v1.json`.
- SHA256: `7968ea77cbcf69a26c9f98f1946c4b4dc19237a02c67fb483d54265f109dc728`.
- Raw counts0/1/2/3/4:200/98/34/33/36;401 total,201 error-labeled rows,194 distinct normalized error-source strings.
-112 targets contain ellipsis markers:32 sources have no ellipsis marker;80 sources also contain ellipsis markers. A marker is a triage signal, not proof that the visible sentence is ungrammatical or unrecoverable.
-11 parenthetical targets: IDs103,104,105,108,116,121,141,143,147,168,200. None overlaps the112 abbreviated targets.
-The8 correct-description notes are a subset of those11:104,105,116,121,141,143,147,168. They are NOT another8 corrections to remove on top of the11. Interpret the user's 'ignore' as exclude them from active training while preserving the master; do not silently relabel them as class0. ID168 is a correct-punctuation demonstration note; this exclusion is a user data policy, not a new linguistic adjudication.
-78 non-note error targets have no ellipsis marker;77 also have a source without ellipsis. The remaining source-abbreviated/full-target case is ID48. Formatting alone does not certify target correctness.
-Current classifier585 =393 retained originals +192 deterministic synthetic rows. Class counts0/1/2/3/4 are240/176/53/53/63; clean share41.03%. The classifier already retains all originals except the8 specified exclusions.
-Current corrector495 =303 original-ID rows +192 deterministic synthetic rows; some original rows carry prior accepted repair overlays. This is not303 trustworthy raw error targets.
-All192 current synthetic rows declare `origin=corpus_derived_deterministic_augmentation`, `uses_external_teacher=False`. Preserve the exact baseline pool and provenance; its inclusion is not proof of a measured augmentation gain.

## Review of the proposal

1. Error-containing test inputs are expected by the GEC task. Incorrect hidden labels/references are a different, unverified hypothesis. Do not train to guessed hidden annotation mistakes. Project protocol keeps official test text inference-only.
2. Preserve original error inputs and labels. A bad or missing correction does not automatically invalidate classification supervision, although uncertain labels stay flagged.
3. A reconstructed target must be supported by visible source text, surviving organizer correction fragments and a necessary grammatical edit. 'Recover' does not mean invent missing names, numbers, events, predicates or opinions.
4. Keep a source-fragment row in the master and classifier. It can enter correction training if the visible grammatical edit is supported and the original ellipsis is preserved where necessary. Missing context is not a license to expand it into a fictitious full sentence.
5. The11 note targets are not correction strings. All11 are excluded from the corrector view under this proposal. The3 remaining note rows103/108/200 may remain classifier-only under their original labels. The8 user exclusions are excluded from both active views.
6. A newly generated example supplements the corpus; it does not repair a different original row. Only an overlay tied to that original ID/source can resolve its target status.
7. Existing weak-model disagreement is diagnostic, not an automatic rejection label. Do not require all four previous small checkpoints to reproduce every target; their own error performance was poor. Conversely, self-confidence or consensus alone cannot approve a repair.

## Master data and active views

Create a new private version, never overwrite the raw original, baseline585/495 package or saved runs.

- Master original record: typed ID, exact original input/label/target, normalized duplicate/family group, format flags, classifier/corrector eligibility and exclusion reason.
- Repair overlay: original ID and source digest; original target preserved; proposed target; exact source-relative edits; supporting original target fragments; author/verifier identities, native parameter counts, revisions, precision, prompt/settings/seed/raw-output digests; status and rejection reason.
- Synthetic record: distinct ID, parent IDs/family, clean parent digest, rule/mechanism, source/target/label, exact edit, author/reviewer evidence, origin, private-use status and hashes.
- Classifier:393 original rows after the8 exclusions, plus the192 legacy rows; this reproduces the current585 starting population. Keep all200 original clean rows and the given labels of retained error rows.
- Corrector: all200 original class0 rows as identity pairs, plus supported original error targets and eligible synthetic pairs. Unresolved originals remain in the master/classifier; do not teach error-source identity mappings or annotation prose as their correction.
- Original corrector ceiling under this exclusion policy is390 rows =401 minus11 notes. This is an upper bound, not a promise that all190 error targets can be validated. Keeping192 legacy rows would give582 total only if every remaining original target qualifies.
- Derived clean counterpart: an accepted corrected reference may be added as a distinct class0 row tied to its original parent. Never turn the original erroneous input into class0 to obtain balance. Deduplicate matching clean references.

## Target recovery workflow

**Tier A: retain supported existing complete targets.** Check current accepted overlays and full non-note pairs; preserve valid evidence and avoid regenerating them. Record mixed lexical/structural edits separately from a simple punctuation/function edit.

**Tier B: source without ellipsis, target with ellipsis —32 rows first.** Align the surviving target fragments to the source. Copy unchanged spans deterministically. Ask the eligible model only for the localized correction that the fragments do not determine. Return a source-relative edit or abstention. Reconstruct the full target in code and verify all unchanged content. Never strip ellipses and blindly concatenate fragments.

**Tier C: source and target both contain ellipsis —80 rows.** Recover grammar of the available text, not the absent original passage. Preserve existing markers when they denote missing material. Copy only supported content from the source and organizer target fragments. If subject/reference/meaning requires unavailable context, keep `unresolved_missing_context`, retain the classifier row and exclude only the unresolved corrector target. Do not blanket discard all80 or promise to recover all80.

**Tier D: explanatory notes.** Keep original note text in the master, remove it from the corrector view, apply the8 user exclusions, and retain the3 remaining classifier-only rows.

Acceptance requires a reconstructable teacher-selected edit, supported content, a necessary grammatical correction, no unexplained rewriting, a usable complete target or faithful visible-fragment target, and review evidence. Any name, number or factual clause must be supported by the source or surviving organizer target text; copying a subject already named by the organizer is not inventing a subject. Polarity/event-role changes require explicit annotation support and extra review, not blind acceptance or a blanket ban on repairing the original error. Reject a fluent completion that invents context. Do not apply a universal35%-edit cutoff to every original repair; flag broader necessary changes for stricter review instead. The older generated-pair policy is not automatically the right original-recovery policy.

## Teacher selection and bounded qualification

**Preferred candidate for the requested Vietnamese-focused7B role:** `Viet-Mistral/Vistral-7B-Chat`. Its publisher describes Vietnamese continued pretraining and instruction tuning. It is gated and lists AFL-3.0. Access, full license conditions, immutable revision, native parameter count including expanded embeddings, tokenizer/chat-template integrity and compatibility must be checked before generation. No weights have been downloaded and no GEC competence is established. Do not accept contact-sharing terms on the user's behalf merely to run the plan.

**Fallback:** `vinai/PhoGPT-4B-Chat`, a Vietnamese-native model with BSD-3-Clause. The older PhoGPT-7B5-Instruct publisher recommends newer4B versions for performance/efficiency, so do not choose the old7.5B merely for its size. Fallback is a separately measured candidate, not a claimed winner. Both need actual checkpoint audits before use.

**Verifier candidate:** existing eligible `Qwen/Qwen2.5-7B-Instruct`, run separately on the focused source/proposed-target preservation/necessity task. Its previous general synthesis failure does not qualify this role; it must pass this narrower qualification too. If no verifier qualifies, repairs stay proposals rather than being automatically promoted.

Do not use VinaLLaMA/Llama2-based output to improve unrelated Sailor/Qwen systems without a compatible permission basis; the linked Llama2 license restricts that use. Retain the existing SeaLLMs exclusion. Model size and Vietnamese fluency do not override output-use restrictions. No unverified-size cloud model authors corpus wording or replacement labels.

Local7B native FP16 does not fit the8GB laptop with normal overhead. A pinned4-bit local teacher is allowed in principle; quantization changes memory, not parameter count. Qualify and generate at the same recorded precision. This would be a NEW teacher experiment; never promote the failed earlier Qwen3 diagnostic outputs. No silent quantization of the existing Kaggle FP16 training recipe.

**First budget:**12 cases, maximum36 proposer/verifier attempts including retries, maximum20 active minutes including model loading. Construct controls using only eligible training-side originals:8 supported known-target recovery cases,2 no-edit clean cases and2 missing-information abstention cases. Artificially abbreviate known targets deterministically; do not author benchmark content with an uncounted assistant. Where insufficient supported examples exist for a class, report that limitation rather than invent gold. Conservative existing code pairs can supply separately labeled operator controls; do not misrepresent those as independently adjudicated organizer examples.

**Proposed pilot gate:** at least7/8 known repairs reproduce the supported correction edits,2/2 clean controls unchanged,2/2 missing-information cases do not invent context, and zero critical fact/polarity changes. The verifier must distinguish known supported edits from controlled changed-fact/style-only alternatives. Report every case, not just aggregate pass/fail. This tiny screen only permits a limited real batch; it is not an accuracy guarantee or proof of independent validity.

Then review an8-row Tier-B batch. Inspect accepted/rejected/abstained evidence before continuing to the other24. Tier-C work follows supported Tier-B results. Preserve partial progress and hard budgets; no all112-row blind generation run up front.

## Augmentation and balance

Retain the192 legacy examples for the control and initial retention candidate, with their exact provenance. Keep source-family and template exposure controlled. Quote-style variants count as one underlying family, not independent error mechanisms. Inspect weak template families such as duplicated copulas and terminal punctuation artifacts; source weighting can cap their training exposure without erasing stored examples.

New deterministic candidates use complete, supported clean parents: selected punctuation/balanced-delimiter errors and constrained function-word edits with explicit applicability checks. Phrase reordering needs an identified grammatical constituent; do not shuffle arbitrary word windows. Sentence-structure deletion needs a known clause frame; do not assume every deleted preposition belongs to the same label. Skip ambiguous operators.

New model candidates start from original class0 inputs or accepted clean references. Keep the clean parent, create exactly one required error, record the exact edit, and obtain a separately qualified label/necessity check. Meaning/facts remain fixed. Emphasize phrase/function/punctuation diversity rather than repeating the existing shortcut patterns. No test parent is eligible.

**First generation budget after qualification:** at most64 proposals, distributed as8 sentence /20 phrase /20 function /16 punctuation. This is a proposal budget, not an accepted-row quota. Keep provenance separated between code and model outputs. Do not fill failed quotas with weak candidates. Preserve the prior249 proposals/51 mechanically eligible candidates privately; they are not automatically admitted or silently rewritten.

**Balance policy:** current clean share41.03%, raw-original clean share49.88%. Proposed effective clean exposure45–50%, with initial stored-row clean share kept at least40% unless a measured local comparison justifies a change. Do not force all five labels to20%. Preserve original clean rows, deduplicate new clean counterparts and use ONE fixed source/family-aware weighting scheme where counts alone cannot meet the exposure target. Do not combine strong inverse-class sampling with inverse-class loss weights blindly.

For585 starting rows with240 clean, adding E error rows and C unique clean counterparts gives clean share `(240+C)/(585+C+E)`. With C=0, only15 extra errors fit a40% floor; adding100 errors needs at least57 unique clean counterparts or a documented sampling-exposure policy. This is why 'add errors but leave class0 distribution unchanged' needs an explicit count/sampling rule.

Cap each new template family at20% of its class's effective training exposure and each parent at two new error variants initially. Preserve all stored rows; limit repetitive exposure instead of maximizing duplicates. Prioritize types2/3/4 because type1 already has176 rows in the current classifier.

## Evaluation before any Kaggle recommendation

1. Freeze a seed42 parent/normalized-source/clean-counterpart grouping before teacher prompts and augmentation for the development experiment. Any validation family and its legacy/new descendants stay out of training and teacher contexts for that comparison. Final full fits can again use all eligible originals, including former81, after the recipe is selected.
2. Freeze classification labels for retained originals. For correction metrics, use only pre-existing supported evaluation targets and report denominators; unresolved/annotation targets are not silently turned into gold by the repair teacher. The old81 are exposed development evidence, not an independent test.
3. First compare existing baseline versus retention/recovery only. Use the same small classifier architecture, seeds42/43, optimizer-step budget and original validation rows. Report Macro-F1, type2/3/4 recall, clean false positives and confusion matrices. Data-size differences must not silently increase optimizer updates.
4. Add code augmentation in a separate comparison; add model augmentation only after the previous comparison. This attributes gains and avoids another large uninformative mixture.
5. A small corrector comparison uses matched budgets and reports error-only GLEU+, all-row GLEU+, exact targets and clean preservation. A high all-row correction score alone is insufficient. Neither weak corrector exact disagreement nor a high self-confidence value is a grammatical adjudication.
6. Kaggle stays on hold until local qualification and the data comparison produce positive evidence without a material clean-class regression. Small-model improvements are proxy evidence, not proof that Sailor/Qwen or the public score will improve. No70–80% score is promised.

## Implementation tasks and checks — all pending

- [ ] Task1: add `tools/inventory_original_data.py` and `agents/tests/test_retained_data.py` to reproduce the401/201/112/32/80/11/8 sets, typed IDs, overlaps and source hashes. Include ID48, legitimate ellipses and source duplicates. No grammar labels are inferred.
- [ ] Task2: add `model/retained_data.py`, dispatch the new versioned format in `model/data.py`, and add `tools/build_retained_dataset.py`. Preserve legacy loaders. Validate raw/master hashes, overlay binding,8 exclusions, classifier-only notes, unresolved corrector exclusion, original clean identity and max390 original corrector rows. Test source/label/target-evidence tampering.
- [ ] Task3: add `configs/target_recovery.json` and `tools/recover_original_targets.py`. Reuse journal/worker isolation without inheriting the old full-generation loop. Test bounded calls/time, abstention, exact edits, unknown names/numbers, no free missing-context completion, tokenizer content preservation and same-source resume identity. Run actual tiny mechanics before real local qualification; mechanics are not teacher qualification.
- [ ] Task4: add `tools/augment_retained_dataset.py` and `agents/tests/test_retained_augmentation.py`. Reuse eligible deterministic rules/192-row provenance, require new mechanism/parent eligibility and separate reviewer evidence, preserve label0 parents, enforce clean count/exposure math and family caps. Test that a synthetic example cannot replace an unrelated original target.
- [ ] Task5: add `configs/retention_screen.json` and `tools/screen_retained_dataset.py`. Reuse small-model training helpers, enforce grouped splits and matched update budgets, compare retention before code/model augmentation, and record all negative outcomes. No model sweep or full-size retraining by default.
- [ ] Task6: add `tools/package_retained_dataset.py` after promotion. Produce a new PRIVATE versioned ZIP with original snapshot, classifier/corrector views, overlays, accepted/rejected provenance and hashes. Validate exact row lineage and archive round trip. Update training consumer/recipe hashes in a deliberate later commit. No notebook toggles; keep the existing585/495 artifacts unchanged.

## Rules and sources checked2026-10-15

Competition1038 explicitly allows documented external data/synthetic pipelines and requires each component and synthetic model to be≤8B. The local signed-form template permits research modifications but prohibits redistribution of original/edited corpus resources. Keep samples, overlays and ZIPs private; provide organizer access as required. Local retention/exclusion and review thresholds above are project design choices, not claims that the organizer mandates them.

- https://aihub.ml/competitions/1038#learn_the_details-overview
- Local `../test/VLSP2026-ViGEC-UserAgreement.docx`, paragraphs under Terms and Conditions.
- https://huggingface.co/Viet-Mistral/Vistral-7B-Chat — Vietnamese adaptation, AFL-3.0 and access gate; full license/access audit pending.
- https://huggingface.co/vinai/PhoGPT-4B-Chat — Vietnamese-native model, BSD-3-Clause.
- https://huggingface.co/vinai/PhoGPT-7B5-Instruct — publisher recommends newer4B versions.
- https://huggingface.co/vilm/vinallama-7b-chat and https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/LICENSE.txt — output-use restriction relevant to unrelated Sailor/Qwen training.

No dataset changes or new model results are claimed by this proposed plan.
