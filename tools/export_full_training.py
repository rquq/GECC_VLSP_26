"""Export both training populations with every former local validation row added."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from model.data import _id_key, _records_from_raw, load_classifier_split, load_records, load_training_split, split_records
from tools.build_dataset import audit_record

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_full_training(data_dir: Path, output: Path) -> dict:
    original = data_dir / "VGEC 2026_v1.json"
    classifier = data_dir / "VGEC_classifier.json"
    corrector = data_dir / "VGEC_balanced.json"
    _, expected = split_records(load_records(original), .2, 42)
    load_classifier_split(original, classifier, .2, 42)
    _, correction_validation, _ = load_training_split(corrector, .2, 42)
    raw = {"classification": json.loads(classifier.read_text()), "correction": json.loads(corrector.read_text())}
    added = raw["classification"]["validation"]
    if added != raw["correction"]["validation"] or _records_from_raw(added) != expected or correction_validation != expected:
        raise ValueError("The two stages do not share the exact former validation rows")
    output.mkdir(parents=True, exist_ok=False)
    source_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    summary = {"schema": "vgec-private-full-training-export-v1", "status": "raw_training_dataset_export",
               "source_commit": source_commit, "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "former_validation_rows_added_to_each_stage": len(added), "independent_local_validation_rows": 0,
               "old_holdout_retired_for_this_dataset": True, "raw_fields_unchanged": True,
               "new_synthetic_rows": 0, "sources": {p.name: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                                       for p in (original, classifier, corrector)},
               "training_files": {}, "pipeline_compatibility": "Standalone raw training arrays; current Kaggle recipe still uses the prior split bundles. A final-fit consumer must load all rows without reusing the old validation set."}
    added_by_id = {_id_key(r["ID"]): r for r in added}
    for stage, bundle in raw.items():
        baseline = bundle["train"]
        if {_id_key(r["ID"]) for r in baseline} & set(added_by_id):
            raise ValueError("Former validation IDs already occur in baseline training")
        combined = baseline + added
        records = _records_from_raw(combined)  # Strict labels, target types and unique typed IDs.
        path = output / (stage + "_train.json")
        write_json(path, combined)
        actual = json.loads(path.read_text())
        actual_by_id = {_id_key(r["ID"]): r for r in actual}
        if actual[:len(baseline)] != baseline or any(actual_by_id[k] != r for k, r in added_by_id.items()) or load_records(path) != records:
            raise ValueError("Training export changed or omitted source rows")
        summary["training_files"][stage] = {"filename": path.name, "baseline_rows": len(baseline),
                                             "added_rows": len(added), "total_rows": len(combined),
                                             "class_counts": dict(sorted(Counter(r["label"] for r in records).items())),
                                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
                                             "all_former_validation_rows_present": True}
    reviews = [audit_record(row, raw_row, original=True) for row, raw_row in zip(expected, added)]
    reviews = [r for r in reviews if r["quarantine_reasons"]]
    write_json(output / "added_correction_review_queue.json", reviews)
    summary["added_correction_rows_flagged_for_review"] = len(reviews)
    summary["correction_target_policy"] = "All81 requested original rows retained verbatim, including flagged targets; no inferred repairs. Raw inclusion is not a claim that annotation/incomplete targets are curated correction supervision."
    write_json(output / "training_manifest.json", summary)
    (output / "README.md").write_text(
        "# Private full-training dataset export\n\n"
        "All former81 local validation rows are included in BOTH training arrays. "
        "classification_train.json has585 rows (504+81); correction_train.json has495 (414+81). "
        "Original fields and existing training rows are preserved. No new examples or labels were generated.\n\n"
        "These files are raw training arrays. Load all records for a final fit; "
        "the current Kaggle recipe still points to the previous split bundles. "
        "There is no independent local validation set in this export. "
        "The former81 cannot also be used to claim unseen validation performance for these models. "
        "Historical baseline files/runs remain intact.\n\n"
        "All requested correction rows remain present, including annotation/incomplete targets. "
        "added_correction_review_queue.json flags these without inventing repairs; "
        "the raw corrector array is not a claim of curated target quality. "
        "Keep this corpus derivative private and noncommercial.\n",
        encoding="utf-8",
    )
    archive = output / "VGEC26-full-training-including-validation-20261014.zip"
    names = ["classification_train.json", "correction_train.json", "added_correction_review_queue.json", "training_manifest.json", "README.md"]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(output / name, arcname=name)
    with zipfile.ZipFile(archive) as z:
        if z.namelist() != names or z.testzip() is not None:
            raise ValueError("Training ZIP CRC/members failed")
        for stage, file in summary["training_files"].items():
            if z.read(file["filename"]) != (output / file["filename"]).read_bytes():
                raise ValueError("Training ZIP differs from exported rows")
    summary["archive"] = {"path": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size}
    summary["verification"] = "Both81-row memberships/raw fields, baseline preservation, typed IDs, load roundtrip and ZIP CRC verified"
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset/kaggle-qwen-ready-20261013")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset/full-training-including-validation-20261014")
    args = parser.parse_args()
    print(json.dumps(export_full_training(args.data_dir.resolve(), args.output.resolve())))


if __name__ == "__main__":
    main()
