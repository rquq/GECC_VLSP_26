"""Propose conservative target completion from existing source/reference text only."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re

from model.data import _id_key, load_records, split_records


def trailing_reference_candidate(source, target):
    """Copy a known source suffix after one uniquely anchored abbreviated edit."""
    if not isinstance(target, str) or re.search(r"\.\.\.|…", source):
        return None
    parts = re.split(r"\.\.\.|…", target)
    if len(parts) != 2 or parts[1].strip() or not parts[0].strip():
        return None
    prefix = parts[0].strip()
    words = prefix.split()
    if len(words) < 8:
        return None
    anchor = " ".join(words[-4:])
    if source.count(anchor) != 1:
        return None
    end = source.index(anchor) + len(anchor)
    old_prefix = source[:end]
    if old_prefix == prefix or difflib.SequenceMatcher(None, old_prefix, prefix).ratio() < .85:
        return None
    edits = [op for op in difflib.SequenceMatcher(None, old_prefix.split(), words).get_opcodes() if op[0] != "equal"]
    if sum(max(b-a, d-c) for _, a, b, c, d in edits) > 4:
        return None
    candidate = prefix + source[end:]
    return {"target": candidate, "evidence": {"method": "unique_four_token_anchor_then_copy_source_suffix",
            "anchor": anchor, "source_anchor_end": end, "explicit_reference_prefix": prefix,
            "new_wording_generated": False, "grammatical_adjudication": False}}


def propose(original, queue_path, output):
    raw = json.loads(original.read_text())
    records = load_records(original)
    training, validation = split_records(records, .2, 42)
    allowed = {_id_key(row["id"]) for row in training}
    by_id = {_id_key(row["ID"]): row for row in raw}
    queue = json.loads(queue_path.read_text())
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if (queue["original_sha256"] != digest or queue["split_seed"] != 42
            or queue["validation_fraction"] != .2):
        raise ValueError("Repair queue must match the original training partition")
    rows = []
    for item in queue["rows"]:
        key = _id_key(item["id"])
        if key not in allowed or item["original"] != by_id[key]:
            raise ValueError("Repair queue contains a non-training or changed original row")
        original_row = item["original"]
        source, target = original_row["Input"], original_row["Output"]["Fixed Sentence"]
        candidate = trailing_reference_candidate(source, target)
        annotation_correct = "annotation_declares_source_correct" in item["flags"]
        rows.append({**item, "proposed_full_correction": candidate["target"] if candidate else None,
                     "proposed_label": None, "review_status": "candidate_requires_review" if candidate else "quarantined_pending_review",
                     "evidence": candidate["evidence"] if candidate else None,
                     "label_review": "explicit_source_correct_annotation_conflicts_with_label" if annotation_correct else "original_label_preserved",
                     "confidence": "mechanical_reconstruction_only" if candidate else None})
    result = {"schema": "vgec-training-repair-proposals-v1", "original_sha256": digest, "seed": 42,
              "partition": "training-only", "changes_applied": False, "teacher_model": None,
              "holdout_count": len(validation), "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    output.chmod(0o600)
    return {"reviewed_flags": len(rows), "reconstruction_candidates": sum(row["proposed_full_correction"] is not None for row in rows),
            "explicit_label_contradictions": sum(row["label_review"].startswith("explicit") for row in rows),
            "changes_applied": False, "holdout_count": len(validation)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(propose(args.original, args.queue, args.output), indent=2))


if __name__ == "__main__":
    main()
