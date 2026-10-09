"""Equal-probability heterogeneous ensembles, saved serving and offline comparison."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from numbers import Real
from pathlib import Path

from .data import LABELS, _id_key, _validate_records
from .prediction_export import comparison_rows
from .runtime import write_json


_AGGREGATION = "equal_probability_mean"
_LIMITATIONS = [
    "The holdout has been repeatedly used for model selection and comparison.",
    "No independent test set was evaluated; holdout improvements are exploratory.",
    "No ensemble weights were tuned; each member has equal probability weight.",
    "These are heterogeneous architectures, not matched-architecture controls; only ordered canonical training and validation data are matched.",
]


def _record_keys(records):
    keys = []
    seen = set()
    for row in records:
        if not isinstance(row, dict) or "id" not in row or not isinstance(row.get("source"), str):
            raise ValueError("Records must contain an ID and string source")
        key = _id_key(row["id"])
        if key in seen:
            raise ValueError(f"Duplicate record ID: {row['id']!r}")
        seen.add(key)
        keys.append(key)
    return keys


def _probabilities(value):
    if not isinstance(value, (list, tuple)) or len(value) != len(LABELS):
        raise ValueError("Probabilities must be a five-class vector in label order 0..4")
    probabilities = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError("Probabilities must contain finite numeric values")
        try:
            number = float(item)
        except (OverflowError, ValueError) as error:
            raise ValueError("Probabilities must contain finite numeric values") from error
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError("Probabilities must be finite and in [0, 1]")
        probabilities.append(number)
    if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Probabilities must be normalized to sum to one")
    return probabilities


def _align_member(keys, predictions):
    indexed = {}
    for prediction in predictions:
        if not isinstance(prediction, dict) or "id" not in prediction:
            raise ValueError("Each member prediction must contain an ID")
        key = _id_key(prediction["id"])
        if key in indexed:
            raise ValueError(f"Duplicate prediction ID: {prediction['id']!r}")
        indexed[key] = _probabilities(prediction.get("probabilities"))
    if indexed.keys() != set(keys):
        raise ValueError("Member prediction IDs must match record IDs exactly")
    return [indexed[key] for key in keys]


def average_predictions(records, members):
    """Average original member vectors equally, aligning exact typed IDs.

    Inputs are source records and an iterable of member prediction lists. Labels
    are derived from the mean probabilities, never votes or supplied decisions.
    Exact ties select the lowest label. Tiny floating-point sum error is accepted
    (absolute tolerance 1e-6), but vectors are never silently renormalized.
    """
    records = list(records)
    keys = _record_keys(records)
    aligned = [_align_member(keys, member) for member in members]
    if not aligned:
        raise ValueError("At least one member is required")
    result = []
    for index, record in enumerate(records):
        probabilities = [math.fsum(member[index][label] for member in aligned) / len(aligned)
                         for label in LABELS]
        result.append({"id": record["id"], "label": max(LABELS, key=probabilities.__getitem__),
                       "probabilities": probabilities})
    return result


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_config(checkpoint):
    config = _read_json(Path(checkpoint) / "ensemble_config.json")
    if not isinstance(config, dict):
        raise ValueError("Ensemble config must be an object")
    labels = config.get("labels")
    if (labels != list(LABELS) or not isinstance(labels, list)
            or any(type(label) is not int for label in labels)):
        raise ValueError("Ensemble labels must be 0..4 in that order")
    if config.get("aggregation") != _AGGREGATION:
        raise ValueError("Only equal-probability averaging is supported")
    members = config.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError("Ensemble config must contain at least two original members")
    seen = set()
    for member in members:
        if not isinstance(member, dict) or member.get("kind") not in ("qwen", "encoder"):
            raise ValueError("Member kind must be qwen or encoder, never a nested ensemble")
        path = member.get("checkpoint")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("Member checkpoints must be absolute paths")
        resolved = Path(path).resolve()
        if resolved in seen:
            raise ValueError("Ensemble members must have distinct checkpoints")
        seen.add(resolved)
        weight = member.get("weight", 1.0 / len(members))
        if (isinstance(weight, bool) or not isinstance(weight, Real)
                or not math.isfinite(weight)
                or not math.isclose(weight, 1.0 / len(members), rel_tol=0.0, abs_tol=1e-12)):
            raise ValueError("Only equal member weights summing to one are supported")
    return config


def classify(checkpoint, records, device="cuda", batch_size=8):
    """Serve a saved ensemble on unseen sources, loading one member at a time."""
    config = _load_config(checkpoint)
    records = list(records)
    _record_keys(records)
    # Canonical inference records have null gold fields; do not require gold.
    inputs = [{**record, "label": None, "target": None} for record in records]
    predictions = []
    for member in config["members"]:
        if member["kind"] == "qwen":
            from .classification import classify as member_classify
        else:
            from .encoder_classification import classify as member_classify
        # Each classifier frees its model before returning. Only CPU vectors
        # survive to the next member, so GPU models are never resident together.
        predictions.append(member_classify(member["checkpoint"], inputs, device, batch_size))
    return average_predictions(records, predictions)


def _canonical(records):
    if not isinstance(records, list):
        raise ValueError("Canonical data must be a list of records")
    _validate_records(records)
    return [(_id_key(row["id"]), row["source"], row["label"], row["target"]) for row in records]


def _load_run(name, kind, folder):
    folder = Path(folder).expanduser().resolve()
    train = _read_json(folder / "train.json")
    validation = _read_json(folder / "validation.json")
    canonical_train, canonical_validation = _canonical(train), _canonical(validation)
    if not validation:
        raise ValueError(f"Run {name} has empty validation data")
    manifest = _read_json(folder / "run_manifest.json")
    path = manifest.get("classifier_path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"Run {name} manifest must declare classifier_path")
    checkpoint = Path(path).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = folder / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Run {name} checkpoint does not exist: {checkpoint}")
    saved = _read_json(folder / "predictions.json")
    if not isinstance(saved, list):
        raise ValueError(f"Run {name} predictions must be a comparison-row list")
    indexed = {_id_key(row["id"]): row for row in validation}
    predictions = []
    for row in saved:
        if not isinstance(row, dict) or not all(field in row for field in
                                               ("id", "source", "target_class", "predicted_class", "probabilities")):
            raise ValueError(f"Run {name} predictions must use the comparison_rows schema")
        record = indexed.get(_id_key(row["id"]))
        if (record is None or row["source"] != record["source"]
                or type(row["target_class"]) is not int or row["target_class"] != record["label"]):
            raise ValueError(f"Run {name} prediction gold/source metadata does not match validation")
        probabilities = _probabilities(row["probabilities"])
        label = max(LABELS, key=probabilities.__getitem__)
        if type(row["predicted_class"]) is not int or row["predicted_class"] != label:
            raise ValueError(f"Run {name} predicted_class must equal argmax probabilities")
        predictions.append({"id": row["id"], "label": label, "probabilities": probabilities})
    predictions = average_predictions(validation, [predictions])
    return {"name": name, "kind": kind, "run_dir": str(folder), "checkpoint": str(checkpoint),
            "train": train, "validation": validation, "canonical_train": canonical_train,
            "canonical_validation": canonical_validation, "predictions": predictions}


def _paired_comparison(records, baseline, candidate, baseline_metrics, candidate_metrics):
    fixes, regressions, changed = [], [], []
    by_class = {str(label): {"fixes": 0, "regressions": 0, "changed": 0} for label in LABELS}
    details = []
    for record, before, after in zip(records, baseline, candidate):
        old, new, gold = before["label"], after["label"], record["label"]
        if old == new:
            continue
        changed.append(record["id"])
        by_class[str(gold)]["changed"] += 1
        outcome = "wrong_to_wrong"
        if old != gold and new == gold:
            fixes.append(record["id"])
            by_class[str(gold)]["fixes"] += 1
            outcome = "fix"
        elif old == gold and new != gold:
            regressions.append(record["id"])
            by_class[str(gold)]["regressions"] += 1
            outcome = "regression"
        details.append({"id": record["id"], "source": record["source"], "target_class": gold,
                        "baseline_class": old, "ensemble_class": new, "outcome": outcome})
    return {"count": len(records), "fixes": len(fixes), "regressions": len(regressions),
            "changed": len(changed), "net_fixes": len(fixes) - len(regressions),
            "fix_ids": fixes, "regression_ids": regressions, "changed_ids": changed,
            "macro_f1_delta": candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"],
            "per_class_f1_delta": {str(label): candidate_metrics["per_class"][str(label)]["f1"]
                                   - baseline_metrics["per_class"][str(label)]["f1"] for label in LABELS},
            "by_gold_class": by_class, "changes": details}


def _disagreement(records, members):
    evidence = {}
    for left, right in itertools.combinations(members, 2):
        distances, changed_ids = [], []
        for record, a, b in zip(records, left["predictions"], right["predictions"]):
            distances.append(0.5 * math.fsum(abs(x - y) for x, y in
                                            zip(a["probabilities"], b["probabilities"])))
            if a["label"] != b["label"]:
                changed_ids.append(record["id"])
        evidence[f"{left['name']}__{right['name']}"] = {
            "count": len(records), "argmax_disagreement_count": len(changed_ids),
            "argmax_disagreement_rate": len(changed_ids) / len(records),
            "argmax_disagreement_ids": changed_ids,
            "mean_total_variation": math.fsum(distances) / len(records),
            "max_total_variation": max(distances),
        }
    return evidence


def evaluate(qwen_run, bamibert_run, xlmr_run, output_dir):
    """Evaluate exactly three fixed combinations from saved artifacts, no models."""
    from .classification import _classification_metrics

    runs = [_load_run("qwen", "qwen", qwen_run), _load_run("bamibert", "encoder", bamibert_run),
            _load_run("xlmr", "encoder", xlmr_run)]
    qwen = runs[0]
    for run in runs[1:]:
        for split in ("train", "validation"):
            if run[f"canonical_{split}"] != qwen[f"canonical_{split}"]:
                raise ValueError(f"Ordered canonical {split} data differs: qwen vs {run['name']}")
    records = qwen["validation"]
    individuals = {}
    for run in runs:
        metrics = _classification_metrics(records, run["predictions"])
        run["metrics"] = metrics
        individuals[run["name"]] = {"run_dir": run["run_dir"], "checkpoint": run["checkpoint"],
                                     "kind": run["kind"], "count": len(records), "classification": metrics}
    # Stable ties prefer Qwen, then BamiBERT, then XLM-R; no architecture claim.
    best = max(runs, key=lambda run: run["metrics"]["macro_f1"])
    combinations = {"qwen_bamibert": [runs[0], runs[1]], "qwen_xlmr": [runs[0], runs[2]],
                    "all_three": runs}
    output = Path(output_dir).expanduser().resolve()
    results = {}
    for name, members in combinations.items():
        predictions = average_predictions(records, [run["predictions"] for run in members])
        classification = _classification_metrics(records, predictions)
        config = {"labels": list(LABELS), "aggregation": _AGGREGATION,
                  "members": [{"name": run["name"], "kind": run["kind"],
                               "checkpoint": run["checkpoint"], "weight": 1.0 / len(members)}
                              for run in members]}
        comparison = {
            "vs_qwen": {"baseline": "qwen", **_paired_comparison(records, qwen["predictions"], predictions,
                                                                 qwen["metrics"], classification)},
            "vs_best_individual": {"baseline": best["name"], **_paired_comparison(
                records, best["predictions"], predictions, best["metrics"], classification)},
            "probability_disagreement": _disagreement(records, members),
        }
        metrics = {"count": len(records), "classification": classification,
                   "individuals": {run["name"]: individuals[run["name"]] for run in members}}
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "predictions.json", comparison_rows(records, predictions))
        write_json(folder / "metrics.json", metrics)
        write_json(folder / "ensemble_config.json", config)
        write_json(folder / "comparison.json", comparison)
        results[name] = {"output_dir": str(folder), "members": [run["name"] for run in members],
                         "count": len(records), "classification": classification, "comparison": comparison}
    summary = {"labels": list(LABELS), "aggregation": _AGGREGATION,
               "matched_data": {"ordered_canonical_train_equal": True,
                                "ordered_canonical_validation_equal": True,
                                "train_count": len(qwen["train"]), "validation_count": len(records)},
               "individuals": individuals, "best_individual": best["name"],
               "best_individual_selection": "Highest standalone holdout Macro-F1 across all three; ties prefer qwen, bamibert, xlmr in that order.",
               "ensembles": results, "limitations": list(_LIMITATIONS)}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-run", required=True)
    parser.add_argument("--bamibert-run", required=True)
    parser.add_argument("--xlmr-run", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = evaluate(args.qwen_run, args.bamibert_run, args.xlmr_run, args.output_dir)
    print(json.dumps({"best_individual": summary["best_individual"],
                      "macro_f1": {name: run["classification"]["macro_f1"]
                                   for name, run in summary["ensembles"].items()}}), flush=True)


if __name__ == "__main__":
    main()
