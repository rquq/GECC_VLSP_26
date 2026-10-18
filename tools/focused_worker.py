"""Isolated classifier reload/inference for the focused experiment coordinator."""

import argparse
import json
import math
from pathlib import Path

from model.runtime import write_json


def read_records(path):
    from model.data import load_records, _validate_records

    raw = json.loads(Path(path).read_text())
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "id" in raw[0]:
        _validate_records(raw, require_labels=False)
        return raw
    return load_records(path, require_labels=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["classifier", "records", "output"]:
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--expected-metrics", type=Path)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = p.parse_args()
    from model.classification import classify, _classification_metrics
    from model.tracking import publish_stage_result

    records = read_records(a.records)
    inference = [
        {"id": r["id"], "source": r["source"], "label": None, "target": None}
        for r in records
    ]
    predictions = classify(a.classifier, inference, a.device, 1)
    if a.expected_metrics:
        metrics = _classification_metrics(records, predictions)
        expected = json.loads(a.expected_metrics.read_text())["classification"]
        if any(
            not math.isclose(metrics[k], expected[k], abs_tol=1e-8)
            for k in ["macro_f1", "micro_f1"]
        ):
            raise ValueError("Reloaded baseline classifier score differs")
        write_json(
            a.output.with_suffix(".metrics.json"),
            {"classification": metrics, "saved_classifier_reload_verified": True},
        )
        publish_stage_result("baseline-classifier-reload", {"classification": metrics})
    write_json(a.output, predictions)


if __name__ == "__main__":
    main()
