"""Export strict organizer submissions and separate source/gold diagnostics."""
import argparse
import json
from pathlib import Path
import zipfile

from .data import _id_key, _valid_label, _validate_records


class SubmissionExportError(ValueError):
    """Predictions cannot form an organizer submission without fabrication."""


def _indexed_predictions(records, predictions):
    """Join by exact typed ID; missing, unknown, or duplicate IDs fail."""
    _validate_records(records, require_labels=False)
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a JSON array")
    indexed = {}
    for prediction in predictions:
        if not isinstance(prediction, dict) or "id" not in prediction:
            raise ValueError("Every prediction must be an object containing id")
        key = _id_key(prediction["id"])
        if key in indexed:
            raise ValueError(f"Duplicate prediction ID: {prediction['id']!r}")
        indexed[key] = prediction
    expected = {_id_key(record["id"]) for record in records}
    if indexed.keys() != expected:
        raise ValueError("Prediction IDs must match input IDs exactly")
    return indexed


def submission_rows(records, predictions):
    """Return only organizer ID/Output fields, without repairing invalid output."""
    indexed = _indexed_predictions(records, predictions)
    rows = []
    for record in records:
        prediction = indexed[_id_key(record["id"])]
        if not _valid_label(prediction.get("label")):
            raise SubmissionExportError(f"Invalid predicted label for ID {record['id']!r}; expected integer 0..4")
        if prediction.get("parse_error") is not None:
            raise SubmissionExportError(f"Generation parse_error for ID {record['id']!r}: {prediction['parse_error']}")
        target = prediction.get("target")
        if not isinstance(target, str) or not target.strip():
            raise SubmissionExportError(f"Blank or invalid corrected text for ID {record['id']!r}")
        rows.append({"ID": record["id"], "Output": {
            "Error_Type_Id": prediction["label"], "Fixed Sentence": [target],
        }})
    return rows


def write_submission(path, rows):
    """Write the leaderboard-tested schema and a ZIP containing only results.json."""
    from .runtime import write_json
    path = Path(path)
    if path.name != "results.json":
        raise ValueError("Organizer prediction file must be named results.json")
    write_json(path, rows)
    archive = path.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.write(path, arcname="results.json")
    return archive


def read_submission_archive(path, records):
    """Read frozen predictions only for their exact typed inference IDs/order."""
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["results.json"] or archive.testzip() is not None:
            raise SubmissionExportError("Control ZIP must contain only valid root results.json")
        rows = json.loads(archive.read("results.json"))
    return _submission_predictions(rows, records)


def read_submission_json(path, records):
    """Dataset controls are plain JSON; ZIPs are output-only containers."""
    return _submission_predictions(json.loads(Path(path).read_bytes()), records)


def _submission_predictions(rows, records):
    if not isinstance(rows, list):
        raise SubmissionExportError("Control predictions must be a JSON array")
    predictions = []
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"ID", "Output"}
                or not isinstance(row["Output"], dict)
                or set(row["Output"]) != {"Error_Type_Id", "Fixed Sentence"}):
            raise SubmissionExportError("Invalid control prediction fields")
        output = row["Output"]
        if not isinstance(output["Fixed Sentence"], list) or len(output["Fixed Sentence"]) != 1:
            raise SubmissionExportError("Control predictions require one corrected string per ID")
        predictions.append({"id": row["ID"], "label": output["Error_Type_Id"],
                            "target": output["Fixed Sentence"][0]})
    if submission_rows(records, predictions) != rows:
        raise SubmissionExportError("Control predictions must preserve typed input order")
    return predictions


def comparison_rows(records, predictions):
    """Join diagnostics and gold by exact typed ID in input order."""
    indexed = _indexed_predictions(records, predictions)
    rows = []
    for record in records:
        prediction = indexed[_id_key(record["id"])]
        row = {
            "id": record["id"],
            "source": record["source"],
            "target_class": record["label"],
            "target_correction": record["target"],
            "predicted_class": prediction.get("label"),
            "predicted_correction": prediction.get("target"),
        }
        for field in ("raw_generation", "parse_error", "probabilities", "fold",
                      "votes", "mean_probabilities",
                      "candidates", "selected_candidate", "reranking", "regenerated"):
            if field in prediction:
                row[field] = prediction[field]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True, help="Canonical validation.json")
    parser.add_argument("--predictions", required=True, help="Saved prediction JSON")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from .metrics import _load_predictions
    from .runtime import write_json
    records = json.loads(Path(args.validation).read_text(encoding="utf-8"))
    rows = comparison_rows(records, _load_predictions(args.predictions))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, rows)
    print(f"Wrote {len(rows)} joined predictions: {output}")


if __name__ == "__main__":
    main()
