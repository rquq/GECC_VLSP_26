"""Export the existing local holdout from the organizer corpus for private evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from model.data import _id_key, load_classifier_split, load_records, load_training_split, split_records

ROOT = Path(__file__).resolve().parents[1]


def export_validation(data_dir: Path, output: Path) -> dict:
    original = data_dir / "VGEC 2026_v1.json"
    classifier = data_dir / "VGEC_classifier.json"
    correction = data_dir / "VGEC_balanced.json"
    original_train, expected = split_records(load_records(original), .2, 42)
    classifier_train, _ = load_classifier_split(original, classifier, .2, 42)
    correction_train, heldout, _ = load_training_split(correction, .2, 42)
    classifier_raw = json.loads(classifier.read_text(encoding="utf-8"))
    correction_raw = json.loads(correction.read_text(encoding="utf-8"))
    originals = {_id_key(row["ID"]): row for row in json.loads(original.read_text(encoding="utf-8"))}
    rows = [originals[_id_key(row["id"])] for row in expected]
    if rows != classifier_raw["validation"] or rows != correction_raw["validation"] or heldout != expected:
        raise ValueError("Canonical validation rows differ between original/classifier/corrector")
    validation_ids = {_id_key(row["id"]) for row in expected}
    for training in (original_train, classifier_train, correction_train):
        if validation_ids & {_id_key(row["id"]) for row in training}:
            raise ValueError("Validation IDs overlap a training population")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "validation.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if load_records(path) != expected:
        raise ValueError("Export altered normalized validation fields or order")
    manifest = {
        "schema": "vgec-private-validation-export-v1",
        "purpose": "evaluation_only",
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split": {"method": "existing local grouped split of the organizer corpus", "seed": 42, "validation_fraction": .2},
        "row_count": len(rows),
        "class_counts": dict(sorted(Counter(row["label"] for row in expected).items())),
        "original_fields_and_order_unchanged": True,
        "all_rows_including_annotation_references_retained": True,
        "validation_ids": [row["ID"] for row in rows],
        "training_id_overlap": 0,
        "synthetic_rows": 0,
        "sources": {p.name: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in (original, classifier, correction)},
        "file": {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size},
        "privacy": "Private noncommercial corpus derivative; do not redistribute",
        "augmentation": "Do not derive training augmentation from these validation rows or their families",
    }
    (output / "validation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Private validation-only dataset\n\n"
        f"validation.json contains all {len(rows)} rows of the existing local grouped seed-42 holdout. "
        "Original ID, Input, Output, extra fields, null clean targets and row order are preserved. "
        f"Class counts by label: {manifest['class_counts']}. "
        "Incomplete or annotation-style references are retained unchanged.\n\n"
        "Load the entire file with model.data.load_records; do not split it again. "
        "This is an evaluation/review export, not the production Kaggle training bundle. "
        "validation_manifest.json intentionally differs from the production manifest filename. "
        "Use training parents exclusively for synthetic augmentation.\n\n"
        "The corpus agreement limits use to noncommercial research and prohibits redistribution "
        "of original or edited corpus resources. Keep this JSON and ZIP private.\n",
        encoding="utf-8",
    )
    archive = output / "VGEC26-validation-only-20261014.zip"
    members = ["validation.json", "validation_manifest.json", "README.md"]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for name in members:
            zip_file.write(output / name, arcname=name)
    with zipfile.ZipFile(archive) as zip_file:
        if zip_file.namelist() != members or zip_file.testzip() is not None or json.loads(zip_file.read("validation.json")) != rows:
            raise ValueError("ZIP verification failed")
    return {"row_count": len(rows), "class_counts": manifest["class_counts"], "file": manifest["file"],
            "archive": {"path": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                        "bytes": archive.stat().st_size}, "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset/kaggle-qwen-ready-20261013")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset/validation-only-20261014")
    args = parser.parse_args()
    print(json.dumps(export_validation(args.data_dir.resolve(), args.output.resolve())))


if __name__ == "__main__":
    main()
