"""Join frozen classifier and independent corrector outputs for one private submission."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from model.data import _id_key, load_records
from model.prediction_export import read_submission_archive, submission_rows, write_submission
from model.runtime import PROJECT_ROOT, write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(plan, output):
    if plan["format"] != "vgec-final-component-join-v1" or plan["competition_id"] != 1039:
        raise ValueError("This final component join is for unconstrained track1039 only")
    paths = {key: PROJECT_ROOT / plan[key] for key in
             ("private_data", "classification_submission", "correction_submission", "correction_run")}
    for key in ("private_data", "classification_submission", "correction_submission"):
        if digest(paths[key]) != plan[key + "_sha256"]:
            raise ValueError("Frozen input changed: " + key)
    records = load_records(paths["private_data"], require_labels=False)
    if len(records) != plan["expected_rows"] or any(r["label"] is not None or r["target"] is not None for r in records):
        raise ValueError("Expected the complete unlabeled private input")
    inference = json.loads((paths["correction_run"] / "adapter/inference.json").read_text())
    if inference.get("hint_mode") != "none":
        raise ValueError("Correction must be independent of classifier labels")
    # Each source run includes its input text in local comparisons; never join
    # solely on coincidentally matching ID numbers from different inputs.
    expected = {_id_key(row["id"]): row["source"] for row in records}
    for key in ("classification_submission", "correction_submission"):
        comparisons = paths[key].with_suffix(".comparisons.json")
        rows = json.loads(comparisons.read_text())
        actual = {_id_key(row["id"]): row["source"] for row in rows}
        if len(actual) != len(rows) or actual != expected:
            raise ValueError("Source input rows differ: " + key)
    classes = read_submission_archive(paths["classification_submission"], records)
    corrections = read_submission_archive(paths["correction_submission"], records)
    labels = {_id_key(row["id"]): row["label"] for row in classes}
    predictions = [{**row, "label": labels[_id_key(row["id"])]} for row in corrections]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    archive = write_submission(output / "results.json", submission_rows(records, predictions))
    # Check the actual bytes the user will upload, including compressed members.
    with zipfile.ZipFile(archive) as z:
        if z.namelist() != ["results.json"] or z.testzip() is not None:
            raise ValueError("Final ZIP failed root member/CRC checks")
    replay = read_submission_archive(archive, records)
    if (any(a["label"] != b["label"] for a, b in zip(replay, classes))
            or any(a["target"] != b["target"] for a, b in zip(replay, corrections))):
        raise ValueError("Final output changed frozen component predictions")
    metadata = {"format": plan["format"], "competition_id": 1039, "rows": len(replay),
                "private_data_sha256": plan["private_data_sha256"],
                "classification_source": plan["classification_submission"],
                "classification_source_sha256": plan["classification_submission_sha256"],
                "correction_source": plan["correction_submission"],
                "correction_source_sha256": plan["correction_submission_sha256"],
                "submission_sha256": digest(archive), "submission_bytes": archive.stat().st_size,
                "labels_identical_to_quality_submission": True,
                "corrections_identical_to_completed14b_submission": True,
                "scope": "Frozen component join only. Private score unknown. No fitting, generation, calibration or row-specific rules."}
    write_json(output / "provenance.json", metadata)
    print(json.dumps(metadata, indent=2))
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "configs/final_submission_1039.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/final-private-1039-20261020")
    args = parser.parse_args()
    build(json.loads(args.plan.read_text()), args.output)


if __name__ == "__main__":
    main()
