"""Local PDF-aligned metrics; implementation assumptions in agents/docs/DATA_METRICS.md.

GLEU+ is source-aware, corpus-aggregated over orders 1..4 without smoothing.
It is NOT nltk.gleu, which has no source-retention penalty.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .data import LABELS, _id_key, _valid_label, _validate_records, load_records


def identity_predictions(records: list[dict], label: int | str = 0) -> list[dict]:
    """Return unchanged-source predictions with label 0, a fixed ID, or majority.

    'majority' uses these records' gold labels (ties choose lowest ID), and is a
    descriptive comparator, not a deployable majority estimate on held-out data.
    For a deployable comparator pass a label estimated from training records.
    """
    _validate_records(records, require_labels=label == "majority")
    if label == "majority":
        if not records:
            raise ValueError("Cannot estimate majority label from empty records")
        counts = Counter(row["label"] for row in records)
        label = min(LABELS, key=lambda value: (-counts[value], value))
    if not _valid_label(label):
        raise ValueError("Identity label must be an integer in 0..4 or 'majority'")
    return [{"id": row["id"], "label": label, "target": row["source"]} for row in records]


def _ngrams(tokens: list[str], order: int) -> Counter:
    return Counter(tuple(tokens[index:index + order]) for index in range(len(tokens) - order + 1))


def _gleu_plus(rows: list[tuple[str, str, str]]) -> float:
    """Implement PDF Eq. 7 literally, aggregating signed counts before clipping."""
    numerators = [0] * 4
    denominators = [0] * 4
    hypothesis_length = reference_length = 0
    for source, hypothesis, reference in rows:
        source_tokens, hypothesis_tokens, reference_tokens = source.split(), hypothesis.split(), reference.split()
        hypothesis_length += len(hypothesis_tokens)
        reference_length += len(reference_tokens)
        for index in range(4):
            source_counts = _ngrams(source_tokens, index + 1)
            hypothesis_counts = _ngrams(hypothesis_tokens, index + 1)
            reference_counts = _ngrams(reference_tokens, index + 1)
            for ngram, count in hypothesis_counts.items():
                matched = min(count, reference_counts[ngram])
                retained = min(count, source_counts[ngram])
                numerators[index] += matched - max(0, retained - matched)
                denominators[index] += count
    if not hypothesis_length or any(value <= 0 for value in numerators) or any(value == 0 for value in denominators):
        return 0.0
    brevity_penalty = 1.0 if hypothesis_length > reference_length else math.exp(1.0 - reference_length / hypothesis_length)
    return brevity_penalty * math.exp(sum(math.log(numerator / denominator) for numerator, denominator in zip(numerators, denominators)) / 4.0)


def _rouge_l(hypothesis: str, reference: str) -> float:
    hypothesis_tokens, reference_tokens = hypothesis.split(), reference.split()
    if not hypothesis_tokens or not reference_tokens:
        return float(not hypothesis_tokens and not reference_tokens)
    # LCS needs only one previous row; use the shorter side for its storage.
    if len(reference_tokens) > len(hypothesis_tokens):
        hypothesis_tokens, reference_tokens = reference_tokens, hypothesis_tokens
    previous = [0] * (len(reference_tokens) + 1)
    for token in hypothesis_tokens:
        current = [0]
        for index, reference_token in enumerate(reference_tokens, 1):
            current.append(previous[index - 1] + 1 if token == reference_token else max(previous[index], current[-1]))
        previous = current
    return 2.0 * previous[-1] / (len(hypothesis_tokens) + len(reference_tokens))


def _corrections(rows: list[tuple[str, str, str]]) -> dict:
    if not rows:
        return {"count": 0, "exact_match": None, "gleu_plus": None, "rouge_l": None}
    return {
        "count": len(rows),
        "exact_match": sum(hypothesis == reference for _, hypothesis, reference in rows) / len(rows),
        "gleu_plus": _gleu_plus(rows),
        "rouge_l": sum(_rouge_l(hypothesis, reference) for _, hypothesis, reference in rows) / len(rows),
    }


def score(records: list[dict], predictions: list[dict]) -> dict:
    """Align predictions by exact typed ID and score classification/correction.

    Prediction schema: {'id': original ID, 'label': int or None, 'target': str}.
    Every gold ID must occur exactly once; additional IDs fail. Invalid label
    values count as wrong, never get silently converted to a valid class, and
    occupy an explicit sixth confusion-matrix column. Empty targets are valid.
    """
    _validate_records(records)
    if not records:
        raise ValueError("Cannot score an empty dataset")
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a JSON array")
    aligned = {}
    for index, row in enumerate(predictions):
        if not isinstance(row, dict) or "id" not in row or "target" not in row:
            raise ValueError(f"Prediction {index} must contain id and target")
        key = _id_key(row["id"])
        if key in aligned:
            raise ValueError(f"Duplicate prediction ID: {row['id']!r}")
        if not isinstance(row["target"], str):
            raise ValueError(f"Prediction target for ID {row['id']!r} must be a string")
        aligned[key] = row
    gold_ids = {_id_key(row["id"]) for row in records}
    if set(aligned) != gold_ids:
        missing = [row["id"] for row in records if _id_key(row["id"]) not in aligned]
        extra = [row["id"] for row in predictions if _id_key(row["id"]) not in gold_ids]
        raise ValueError(f"Prediction IDs do not match gold IDs; missing={missing!r}, extra={extra!r}")
    confusion = [[0] * 6 for _ in LABELS]
    all_rows, error_rows = [], []
    invalid = 0
    for row in records:
        prediction = aligned[_id_key(row["id"])]
        predicted_label = prediction.get("label")
        valid = _valid_label(predicted_label)
        invalid += not valid
        confusion[row["label"]][predicted_label if valid else 5] += 1
        correction = (row["source"], prediction["target"], row["target"])
        all_rows.append(correction)
        if row["label"] != 0:
            error_rows.append(correction)
    per_class = {}
    for label in LABELS:
        true_positive = confusion[label][label]
        support = sum(confusion[label])
        predicted_count = sum(row[label] for row in confusion)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * true_positive / (support + predicted_count) if support + predicted_count else 0.0
        per_class[str(label)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    # Each row has one decision, including invalid rejections: wrong decisions
    # contribute one global FP and FN. Thus micro-F1 remains accuracy.
    correct = sum(confusion[label][label] for label in LABELS)
    return {
        "count": len(records),
        "classification": {
            "macro_f1": sum(value["f1"] for value in per_class.values()) / len(LABELS),
            "micro_f1": correct / len(records),
            "confusion_matrix": confusion,
            "labels": list(LABELS),
            "prediction_columns": [*LABELS, "invalid"],
            "per_class": per_class,
        },
        "corrections": {"all": _corrections(all_rows), "error_only": _corrections(error_rows)},
        "invalid_prediction_rate": invalid / len(records),
    }


def experiment_report(records: list[dict], predictions: list[dict]) -> dict:
    """Extend :func:`score` without changing its strict alignment or metrics.

    Clean preservation compares gold-class-0 predictions to the exact source,
    independently of their predicted label. Its rates are null without clean
    examples. Invalid JSON includes any non-null ``parse_error`` diagnostic,
    invalid label, or blank corrected sentence (the generation schema rejects
    blank text, although ``score`` deliberately permits it).

    Reference slices contain only gold error examples. Annotation-like targets
    start literally with ``(`` or contain literal ``...``; flags may overlap.
    References are never stripped, normalized, or rewritten. Empty slices have
    null metrics; nonempty slices use the unchanged scoring implementation.
    """
    report = score(records, predictions)
    aligned = {_id_key(row["id"]): row for row in predictions}
    clean_count = preserved = invalid_json = 0
    slices = {"annotation_like": [], "standalone_error": []}
    audit = []
    for row in records:
        prediction = aligned[_id_key(row["id"])]
        invalid_json += (
            prediction.get("parse_error") is not None
            or not _valid_label(prediction.get("label"))
            or not prediction["target"].strip()
        )
        if row["label"] == 0:
            clean_count += 1
            preserved += prediction["target"] == row["source"]
            continue
        flags = []
        if row["target"].startswith("("):
            flags.append("starts_with_parenthesis")
        if "..." in row["target"]:
            flags.append("contains_ellipsis")
        slices["annotation_like" if flags else "standalone_error"].append(row)
        if flags:
            audit.append({
                "id": row["id"],
                "source": row["source"],
                "reference": row["target"],
                "prediction": prediction["target"],
                "label": row["label"],
                "predicted_label": prediction.get("label"),
                "flags": flags,
            })
    report["clean_preservation"] = {
        "count": clean_count,
        "exact_source_preservation": preserved / clean_count if clean_count else None,
        "overcorrection_rate": (clean_count - preserved) / clean_count if clean_count else None,
    }
    report["invalid_json_rate"] = invalid_json / len(records)
    report["reference_slices"] = {
        name: {
            "count": len(subset),
            "metrics": score(subset, [aligned[_id_key(row["id"])] for row in subset]) if subset else None,
        }
        for name, subset in slices.items()
    }
    report["annotation_audit"] = audit
    return report


def _load_predictions(path: str | Path) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Predictions must be a JSON array")
    predictions = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"Prediction {index} must be an object")
        if "ID" in row:
            output = row.get("Output")
            if not isinstance(output, dict):
                raise ValueError(f"Prediction Output for ID {row['ID']!r} must be an object")
            if "Fixed Sentence" not in output:
                raise ValueError(f"Missing Fixed Sentence for ID {row['ID']!r}")
            predictions.append({"id": row["ID"], "label": output.get("Error_Type_Id"), "target": output["Fixed Sentence"]})
        elif "predicted_class" in row and "predicted_correction" in row:
            predictions.append({**row, "label": row["predicted_class"],
                                "target": row["predicted_correction"]})
        else:
            predictions.append(row)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Raw labeled dataset JSON")
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--predictions", help="Canonical or ID/Output prediction array")
    baseline.add_argument("--identity-label", choices=["0", "1", "2", "3", "4", "majority"])
    parser.add_argument("--output", help="Optional metrics JSON output")
    args = parser.parse_args()
    records = load_records(args.data)
    predictions = _load_predictions(args.predictions) if args.predictions else identity_predictions(records, "majority" if args.identity_label == "majority" else int(args.identity_label))
    report = score(records, predictions)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
