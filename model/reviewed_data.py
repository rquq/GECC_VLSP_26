"""Strict reviewed v4 bundles, separate from reproducible legacy v3 policy."""
from collections import Counter
import hashlib
import json

from .data import (_counts, _id_key, _records_from_raw, _source_key,
                   declares_correct_sentence, split_records)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def load_reviewed(raw, fraction, seed):
    if raw.get("seed") != seed or raw.get("validation_fraction") != fraction:
        raise ValueError("Reviewed bundle split configuration differs")
    provenance = raw.get("provenance", {})
    snapshot = provenance.get("original_snapshot")
    if provenance.get("original_snapshot_sha256") != canonical_hash(snapshot):
        raise ValueError("Reviewed original snapshot digest differs")
    originals, validation = split_records(_records_from_raw(snapshot), fraction, seed)
    original_ids = {_id_key(r["id"]) for r in originals}
    raw_by_id = {_id_key(r["ID"]): r for r in snapshot}
    if raw.get("validation") != [raw_by_id[_id_key(r["id"])] for r in validation]:
        raise ValueError("Reviewed bundle must preserve exact raw original holdout")
    before = {key: raw_by_id[key] for key in original_ids}
    expected = dict(before)
    seen_repairs = set()
    for repair in provenance.get("repairs", []):
        key = _id_key(repair["before"]["ID"])
        if key not in before or key in seen_repairs or repair["before"] != before[key]:
            raise ValueError("Reviewed repair must match an original training row exactly")
        seen_repairs.add(key)
        after = repair["after"]
        original_record = _records_from_raw([before[key]])[0]
        record = _records_from_raw([after])[0]
        review = repair.get("verification", {})
        if (after["ID"] != before[key]["ID"] or after["Input"] != before[key]["Input"]
                or repair.get("status") != "accepted" or review.get("source_label") != record["label"]
                or review.get("target_label") != 0 or review.get("meaning_preserved") is not True
                or repair.get("teacher_parameters", 8_000_000_001) > 8_000_000_000
                or not repair.get("teacher_revision") or not repair.get("evidence")):
            raise ValueError("Reviewed repair lacks eligible verification or changes its source")
        if record["label"] != original_record["label"]:
            if not (record["label"] == 0 and record["target"] == record["source"]
                    and declares_correct_sentence(original_record["target"])):
                raise ValueError("Only independently verified explicit clean annotations may be relabeled")
        elif not (review.get("single_error") is True and review.get("required_edit") is True
                  and repair.get("method") == "deterministic_reference_prefix_plus_source_suffix"):
            raise ValueError("Reviewed reference repair must retain subtype and require the edit")
        else:
            from tools.propose_training_repairs import trailing_reference_candidate
            candidate = trailing_reference_candidate(original_record["source"], original_record["target"])
            if not candidate or candidate["target"] != record["target"] or candidate["evidence"] != repair["evidence"]:
                raise ValueError("Reviewed reference repair differs from deterministic source-span reconstruction")
        expected[key] = after
    from tools.build_dataset import audit_record
    stage = provenance.get("stage")
    if stage not in ("classification", "correction"):
        raise ValueError("Reviewed stage must be classification or correction")
    excluded = set()
    for key, value in expected.items():
        r = _records_from_raw([value])[0]
        audit = audit_record(r, value, original=True)
        if (stage == "correction" and audit["quarantine_reasons"]
                or stage == "classification" and r["label"] != 0 and declares_correct_sentence(r["target"])):
            excluded.add(key)
    retained = original_ids - excluded
    for field, keys in (("original_train_ids", original_ids), ("retained_original_train_ids", retained),
                        ("quarantined_ids", excluded)):
        values = provenance.get(field, [])
        if len(values) != len(keys) or {_id_key(v) for v in values} != keys:
            raise ValueError("Reviewed original retention differs from declared audit policy")
    train = _records_from_raw(raw.get("train"))
    grouped = {}
    for r in train:
        key = _source_key(r["source"])
        annotation = (r["label"], r["target"])
        if key in grouped and grouped[key] != annotation:
            raise ValueError("Reviewed duplicate sources have conflicting annotations")
        grouped[key] = annotation
    train_ids = {_id_key(r["id"]) for r in train}
    if { _id_key(r["ID"]): r for r in raw["train"] if _id_key(r["ID"]) in original_ids} != {k: expected[k] for k in retained}:
        raise ValueError("Reviewed original rows differ from supported repair overlay")
    synthetic_ids = {_id_key(v) for v in provenance.get("selected_synthetic_ids", [])}
    if synthetic_ids & original_ids or train_ids != retained | synthetic_ids:
        raise ValueError("Reviewed training IDs differ from retained originals and synthetic IDs")
    validation_ids = {_id_key(r["id"]) for r in validation}
    heldout_text = {_source_key(r[f]) for r in validation for f in ("source", "target")}
    if train_ids & validation_ids or any(_source_key(r[f]) in heldout_text for r in train for f in ("source", "target")):
        raise ValueError("Reviewed train/holdout collision")
    metadata_rows = provenance.get("train_rows", [])
    metadata = {_id_key(r["id"]): r for r in metadata_rows}
    if len(metadata) != len(metadata_rows) or set(metadata) != train_ids:
        raise ValueError("Reviewed train-row provenance must cover IDs exactly once")
    synthetic = [r for r in train if _id_key(r["id"]) in synthetic_ids]
    train_by_id = {_id_key(r["id"]): r for r in train}
    mechanisms = Counter()
    synthetic_counts = Counter(r["label"] for r in synthetic)
    original_counts = Counter(r["label"] for r in train if _id_key(r["id"]) in retained)
    if set(original_counts) != set(range(5)) or synthetic_counts[0] > 40:
        raise ValueError("Reviewed originals must cover all classes; clean augmentation cap is 40")
    if any(synthetic_counts[c] > original_counts[c] for c in range(1, 5)):
        raise ValueError("Reviewed error augmentation exceeds original class count")
    for row in train:
        m = metadata[_id_key(row["id"])]
        if m.get("kind") != ("synthetic" if _id_key(row["id"]) in synthetic_ids else "original"):
            raise ValueError("Reviewed provenance kind differs")
        if _id_key(row["id"]) not in synthetic_ids:
            continue
        if (not m.get("parent_ids") or any(_id_key(p) not in original_ids for p in m["parent_ids"])
                or m.get("teacher_parameters", 8_000_000_001) > 8_000_000_000
                or not m.get("teacher_revision") or not m.get("output_sha256")
                or m.get("review_status") != "accepted" or not m.get("family")
                or not m.get("review_evidence")):
            raise ValueError("Reviewed synthetic lacks eligible teacher/parent/review provenance")
        check = m.get("blind_review", {})
        if row["label"] != 0 and (check.get("source_label") != row["label"] or check.get("target_label") != 0
                or any(check.get(f) is not True for f in ("single_error", "meaning_preserved", "required_edit"))):
            raise ValueError("Reviewed synthetic blind subtype/meaning verification differs")
        if not row["source"].strip() or audit_record(row, {"Output":{"Fixed Sentence":row["target"]}})["quarantine_reasons"]:
            raise ValueError("Reviewed synthetic violates structural curation checks")
        if row["label"] == 0:
            if row["source"] != row["target"]:
                raise ValueError("Reviewed matched clean control must be exact identity")
        else:
            clean = train_by_id.get(_id_key(m.get("clean_id")))
            if not clean or clean["label"] != 0 or clean["source"] != row["target"]:
                raise ValueError("Reviewed synthetic error must have its matched clean counterpart")
            mechanisms[row["label"], m.get("mechanism")] += 1
    if any(count * 5 > synthetic_counts[label] for (label, mechanism), count in mechanisms.items()):
        raise ValueError("Reviewed mechanism exceeds 20 percent of class synthetics")
    for field, value in (("training_class_counts", _counts(train)), ("synthetic_class_counts", _counts(synthetic))):
        if provenance.get(field) != value:
            raise ValueError("Reviewed declared class counts differ")
    return train, validation, {"format": raw["format"], "original_sha256": raw.get("original_sha256"),
                               "provenance": provenance}
