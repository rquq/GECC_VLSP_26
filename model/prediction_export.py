"""Export source, gold and predictions together without changing inference inputs."""
import argparse
import json
from pathlib import Path

from .data import _id_key, _validate_records


def comparison_rows(records, predictions):
    """Join by exact typed ID in input order; unknown or duplicate IDs fail."""
    _validate_records(records, require_labels=False)
    indexed = {}
    for prediction in predictions:
        key = _id_key(prediction["id"])
        if key in indexed:
            raise ValueError(f"Duplicate prediction ID: {prediction['id']!r}")
        indexed[key] = prediction
    expected = {_id_key(record["id"]) for record in records}
    if indexed.keys() != expected:
        raise ValueError("Prediction IDs must match input IDs exactly")
    rows = []
    for record in records:
        prediction = indexed[_id_key(record["id"])]
        row = {
            "id": record["id"],
            "source": record["source"],
            "target_class": record["label"],
            "target_correction": record["target"],
            "predicted_class": prediction["label"],
            "predicted_correction": prediction.get("target"),
        }
        for field in ("raw_generation", "parse_error", "probabilities", "fold"):
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
