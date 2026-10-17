"""Strict raw-data loading and deterministic, leakage-free grouped splitting."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
from typing import Any

LABELS = tuple(range(5))


def _id_key(value: Any) -> tuple[type, Any]:
    """Keep integer and string IDs distinct; reject ambiguous JSON ID types."""
    if type(value) not in (int, str):
        raise ValueError(f"ID must be an integer or string, got {value!r}")
    return type(value), value


def _valid_label(value: Any) -> bool:
    return type(value) is int and value in LABELS


def _validate_records(records: list[dict], require_labels: bool = True) -> None:
    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not all(k in record for k in ("id", "source", "label", "target")):
            raise ValueError(f"Record {index} must contain id, source, label, target")
        key = _id_key(record["id"])
        if key in seen:
            raise ValueError(f"Duplicate ID: {record['id']!r}")
        seen.add(key)
        if not isinstance(record["source"], str):
            raise ValueError(f"Input for ID {record['id']!r} must be a string")
        label = record["label"]
        if not _valid_label(label) and (require_labels or label is not None):
            raise ValueError(f"Error_Type_Id for ID {record['id']!r} must be an integer in 0..4")
        target = record["target"]
        if not isinstance(target, str) and (require_labels or target is not None):
            raise ValueError(f"Fixed Sentence for ID {record['id']!r} must be a string")


def load_records(path: str | Path, require_labels: bool = True) -> list[dict]:
    """Load the raw JSON array without altering any source or non-null correction.

    With labels required, class 0's null correction is the exact input, whereas
    a null correction for classes 1..4 fails. Inference inputs may omit Output.
    Present labels must still be integers in 0..4 (booleans are not integers).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return _records_from_raw(raw, require_labels)


def _records_from_raw(raw: list, require_labels: bool = True) -> list[dict]:
    if not isinstance(raw, list):
        raise ValueError("Dataset must be a raw JSON array")
    records = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict) or "ID" not in row or "Input" not in row:
            raise ValueError(f"Row {index} must contain ID and Input")
        output = row.get("Output")
        if output is None:
            if require_labels:
                raise ValueError(f"Missing Output for ID {row['ID']!r}")
            output = {}
        if not isinstance(output, dict):
            raise ValueError(f"Output for ID {row['ID']!r} must be an object")
        label = output.get("Error_Type_Id")
        target = output.get("Fixed Sentence")
        if target is None and _valid_label(label) and label == 0:
            target = row["Input"]
        records.append({"id": row["ID"], "source": row["Input"], "label": label, "target": target})
    _validate_records(records, require_labels)
    return records


def _source_key(source: str) -> str:
    return " ".join(source.split()).casefold()


def _groups(records: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in records:
        groups[_source_key(row["source"])].append(row)
    return dict(groups)


def split_records(records: list[dict], validation_fraction: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Stratify by class, keeping normalized duplicate-source groups together.

    Each observed class must have at least two independent source groups. A
    deterministic subset-sum chooses the nearest feasible validation row count
    for each class, with at least one group in each partition. All rows remain.
    Output partitions are shuffled so prefix selection cannot follow label order.
    """
    if type(seed) is not int or seed != 42:
        raise ValueError("Split seed must be 42")
    _validate_records(records)
    if not records:
        raise ValueError("Cannot split an empty dataset")
    if not isinstance(validation_fraction, (int, float)) or not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    groups = _groups(records)
    by_class = defaultdict(list)
    for key, rows in groups.items():
        annotations = {(row["label"], row["target"]) for row in rows}
        if len(annotations) != 1:
            raise ValueError(f"Conflicting labels/targets for normalized source group: {[row['id'] for row in rows]}")
        by_class[rows[0]["label"]].append(key)
    rng = random.Random(seed)
    validation_keys = set()
    for label in sorted(by_class):
        keys = sorted(by_class[label])
        if len(keys) < 2:
            raise ValueError(f"Class {label} has fewer than two independent groups; cannot represent it in both splits")
        rng.shuffle(keys)
        total = sum(len(groups[key]) for key in keys)
        desired = max(1, min(total - 1, round(total * validation_fraction)))
        subsets = {0: ()}
        for key in keys:
            size = len(groups[key])
            for count, selected in list(subsets.items()):
                new_count = count + size
                if new_count not in subsets:
                    subsets[new_count] = selected + (key,)
        selected_count = min((count for count in subsets if 0 < count < total), key=lambda count: (abs(count - desired), count))
        validation_keys.update(subsets[selected_count])
    train, validation = [], []
    for row in records:
        (validation if _source_key(row["source"]) in validation_keys else train).append(row)
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def load_training_split(path: str | Path, validation_fraction: float, seed: int) -> tuple[list[dict], list[dict], dict]:
    """Use original grouped splits or a prebuilt split-before-augmentation bundle.

    A v3 bundle is never re-split: its validation records remain original gold.
    Retained originals define capped per-class targets, including at most 40
    exact-identity clean additions. Train-row provenance is optional, but its IDs,
    kinds and synthetic parents must agree with the original training partition.
    Only seed 42 is accepted, before opening the dataset; fractions must match.
    """
    if type(seed) is not int or seed != 42:
        raise ValueError("Dataset split seed must be 42")
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and raw.get("format") == "vgec-quality-mixture-v1":
        from .quality_data import load_quality
        return load_quality(raw, validation_fraction, seed)
    if isinstance(raw, dict) and raw.get("format") == "vgec-grouped-validation-v1":
        from .grouped_validation import load_grouped
        return load_grouped(raw, validation_fraction, seed)
    if isinstance(raw, dict) and raw.get("format") in ("vgec-retained-balanced-v1", "vgec-retained-balanced-v2"):
        from .retained_balanced import load_retained
        return load_retained(raw, validation_fraction, seed)
    if isinstance(raw, dict) and raw.get("format") == "vgec-final-training-v1":
        from .final_data import load_final
        return load_final(raw, validation_fraction, seed)
    if isinstance(raw, list):
        train, validation = split_records(_records_from_raw(raw), validation_fraction, seed)
        return train, validation, {"format": "raw", "original_rows": len(raw)}
    if isinstance(raw, dict) and raw.get("format") == "vgec-reviewed-v4":
        from .reviewed_data import load_reviewed
        return load_reviewed(raw, validation_fraction, seed)
    return _load_legacy_bundle(raw, validation_fraction, seed)


def _load_legacy_bundle(raw, validation_fraction, seed):
    """Validate the original v3 contract, including exact quality-cap counts."""
    if not isinstance(raw, dict) or raw.get("format") != "vgec-balanced-v3":
        raise ValueError("Expected raw dataset array or vgec-balanced-v3 bundle")
    if type(raw.get("seed")) is not int or raw["seed"] != 42:
        raise ValueError("Bundle seed must be 42")
    if raw.get("validation_fraction") != validation_fraction:
        raise ValueError("Bundle validation_fraction differs; rebuild the dataset for this fraction")
    train = _records_from_raw(raw.get("train"))
    validation = _records_from_raw(raw.get("validation"))
    if not train or not validation:
        raise ValueError("Bundle training and validation must both be nonempty")
    train_ids = {_id_key(r["id"]) for r in train}
    validation_ids = {_id_key(r["id"]) for r in validation}
    if train_ids & validation_ids:
        raise ValueError("Bundle has shared training/validation IDs")
    if {_source_key(r["source"]) for r in train} & {_source_key(r["source"]) for r in validation}:
        raise ValueError("Bundle has shared training/validation source groups")
    heldout_text = {_source_key(r[field]) for r in validation for field in ("source", "target")}
    for row in train:
        if any(_source_key(row[field]) in heldout_text for field in ("source", "target")):
            raise ValueError("Training source/target overlaps original validation text")
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Bundle must include provenance")
    if (provenance.get("balance_policy") != "quality_capped_2x"
            or ("balance_policy" in raw and raw["balance_policy"] != "quality_capped_2x")):
        raise ValueError("Bundle balance_policy must be quality_capped_2x")

    def provenance_ids(name: str) -> set:
        values = provenance.get(name)
        if not isinstance(values, list):
            raise ValueError(f"Bundle provenance {name} must be an array")
        keys = {_id_key(value) for value in values}
        if len(keys) != len(values):
            raise ValueError(f"Bundle provenance {name} has duplicate IDs")
        return keys

    original_ids = provenance_ids("original_train_ids")
    retained_ids = provenance_ids("retained_original_train_ids")
    synthetic_ids = provenance_ids("selected_synthetic_ids")
    quarantined_ids = provenance_ids("quarantined_ids")
    if original_ids & validation_ids:
        raise ValueError("Original training partition includes validation IDs")
    if not retained_ids <= original_ids:
        raise ValueError("Retained originals must belong to original training partition")
    if quarantined_ids != original_ids - retained_ids:
        raise ValueError("Quarantined IDs must exactly cover excluded original training IDs")
    if retained_ids & synthetic_ids:
        raise ValueError("Retained original and selected synthetic IDs must be disjoint")
    if synthetic_ids & original_ids:
        raise ValueError("Selected synthetic IDs must not belong to original training partition")
    if retained_ids | synthetic_ids != train_ids:
        raise ValueError("Retained original and selected synthetic IDs must exactly cover training IDs")
    original_counts = Counter(r["label"] for r in train if _id_key(r["id"]) in retained_ids)
    if set(original_counts) != set(LABELS):
        raise ValueError("Retained originals must represent all labels 0..4")
    if any(r["label"] == 0 and r["source"] != r["target"]
           for r in train if _id_key(r["id"]) in synthetic_ids):
        raise ValueError("Synthetic clean rows must have exact source/target identity")
    reference_counts = provenance.get("augmentation_reference_counts")
    partition_counts = provenance.get("original_train_class_counts")
    for name, counts in (("augmentation_reference_counts", reference_counts),
                         ("original_train_class_counts", partition_counts)):
        if (not isinstance(counts, dict) or set(counts) != {str(label) for label in LABELS}
                or any(type(value) is not int or value < 0 for value in counts.values())):
            raise ValueError(f"Bundle provenance {name} must contain nonnegative integer counts for labels 0..4")
    if sum(partition_counts.values()) != len(original_ids):
        raise ValueError("Original training class counts must cover the original partition")
    if any(not original_counts[label] <= reference_counts[str(label)] <= partition_counts[str(label)]
           for label in LABELS):
        raise ValueError("Augmentation reference counts must lie between retained and original class counts")
    largest = max(original_counts.values())
    targets = {str(label): (original_counts[label] + min(40, original_counts[label]) if label == 0
                            else original_counts[label] + min(reference_counts[str(label)],
                                                              max(0, largest - original_counts[label])))
               for label in LABELS}
    declared_targets = provenance.get("balance_targets_by_class")
    if (not isinstance(declared_targets, dict) or set(declared_targets) != set(targets)
            or any(type(value) is not int for value in declared_targets.values())
            or declared_targets != targets):
        raise ValueError("Declared balance_targets_by_class must match quality_capped_2x targets")
    if _counts(train) != targets:
        raise ValueError("Bundle training class counts must match quality_capped_2x targets")
    if "train_rows" in provenance:
        metadata_rows = provenance["train_rows"]
        if not isinstance(metadata_rows, list):
            raise ValueError("Bundle provenance train_rows must be an array")
        metadata_ids = set()
        for metadata in metadata_rows:
            if not isinstance(metadata, dict) or "id" not in metadata:
                raise ValueError("Bundle train_rows entries must include an ID")
            key = _id_key(metadata["id"])
            if key in metadata_ids:
                raise ValueError("Bundle train_rows has duplicate metadata IDs")
            metadata_ids.add(key)
            expected_kind = "original" if key in retained_ids else "synthetic"
            if metadata.get("kind") != expected_kind:
                raise ValueError("Bundle train_rows kind differs from retained/selected IDs")
            if expected_kind == "synthetic":
                parents = metadata.get("parent_ids", [])
                if not isinstance(parents, list):
                    raise ValueError("Synthetic parent_ids must be an array")
                if any(_id_key(parent) not in original_ids or _id_key(parent) in validation_ids for parent in parents):
                    raise ValueError("Synthetic parent_ids must belong to original training partition, not holdout")
        if metadata_ids != train_ids:
            raise ValueError("Bundle train_rows metadata IDs must exactly cover training IDs")
    return train, validation, {"format": raw["format"], "original_sha256": raw.get("original_sha256"),
                               "synthetic_sha256": raw.get("synthetic_sha256"), "provenance": provenance}


def declares_correct_sentence(target: str) -> bool:
    """Recognize a sentence-correctness claim, not an instruction containing đúng."""
    return bool(re.match(r"^\s*\(\s*câu(?:\s+này)?\s+đúng\b", target, re.IGNORECASE))


def load_classifier_split(original_path, bundle_path, validation_fraction, seed):
    """Validate classifier augmentation against the untouched organizer split.

    New bundles quarantine explicit sentence-correctness claims. Pinned legacy
    bundles retain their recorded substring policy for reproducibility. Other
    annotations remain supervision; no original may be repaired or relabeled.
    """
    bundle_raw = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    if isinstance(bundle_raw, dict) and bundle_raw.get("format") == "vgec-quality-mixture-v1":
        from .quality_data import load_quality
        source = bundle_raw["source_bundle"]["source_packet"]["classification"]
        original_bytes = Path(original_path).read_bytes()
        if (bundle_raw.get("stage") != "classification" or source["original_snapshot"] != json.loads(original_bytes)
                or source["metadata"]["original_file_sha256"] != hashlib.sha256(original_bytes).hexdigest()):
            raise ValueError("Quality classifier original differs")
        train, _, info = load_quality(bundle_raw, validation_fraction, seed)
        return train, info
    if isinstance(bundle_raw, dict) and bundle_raw.get("format") == "vgec-grouped-validation-v1":
        from .grouped_validation import load_grouped
        original_bytes = Path(original_path).read_bytes()
        source = bundle_raw.get("source_packet", {}).get("classification", {})
        if (bundle_raw.get("stage") != "classification"
                or source.get("original_snapshot") != json.loads(original_bytes)
                or source.get("metadata", {}).get("original_file_sha256")
                != hashlib.sha256(original_bytes).hexdigest()):
            raise ValueError("Grouped classifier original snapshot/hash differs")
        train, _, info = load_grouped(bundle_raw, validation_fraction, seed)
        return train, info
    if isinstance(bundle_raw, dict) and bundle_raw.get("format") in ("vgec-retained-balanced-v1", "vgec-retained-balanced-v2"):
        from .retained_balanced import load_retained
        original_bytes = Path(original_path).read_bytes()
        if (bundle_raw.get("stage") != "classification"
                or bundle_raw.get("original_snapshot") != json.loads(original_bytes)
                or bundle_raw.get("metadata", {}).get("original_file_sha256")
                != hashlib.sha256(original_bytes).hexdigest()):
            raise ValueError("Retained classifier original snapshot/hash differs")
        train, _, info = load_retained(bundle_raw, validation_fraction, seed)
        return train, info
    if isinstance(bundle_raw, dict) and bundle_raw.get("format") == "vgec-final-training-v1":
        from .final_data import load_final_classifier
        return load_final_classifier(original_path, bundle_raw, validation_fraction, seed)
    original_raw = json.loads(Path(original_path).read_text(encoding="utf-8"))
    originals, validation = split_records(_records_from_raw(original_raw), validation_fraction, seed)
    train, heldout, info = load_training_split(bundle_path, validation_fraction, seed)
    if info["format"] == "vgec-reviewed-v4":
        if (info["provenance"]["stage"] != "classification"
                or info["provenance"]["original_snapshot"] != original_raw
                or info["original_sha256"] != hashlib.sha256(Path(original_path).read_bytes()).hexdigest()
                or heldout != validation):
            raise ValueError("Reviewed classifier original snapshot/hash/holdout differs")
        return train, info
    if info["format"] != "vgec-balanced-v3":
        raise ValueError("Classifier requires a mixed v3 training bundle")
    provenance = info["provenance"]
    policy = provenance.get("original_retention_policy", "exclude_explicit_correct_sentence_annotation_contradictions")
    if policy not in ("exclude_explicit_correct_sentence_annotation_contradictions",
                      "exclude_explicit_correct_sentence_annotation_contradictions_v2"):
        raise ValueError("Unknown classifier original annotation policy")
    expected_originals = {_id_key(row["id"]): row for row in originals}
    expected_quarantine = {_id_key(row["id"]) for row in originals
                           if row["label"] != 0 and (
                               declares_correct_sentence(row["target"]) if policy.endswith("_v2") else
                               row["target"].lstrip().startswith("(") and "đúng" in row["target"].casefold())}
    expected = {key: row for key, row in expected_originals.items() if key not in expected_quarantine}
    retained = {_id_key(value) for value in provenance["retained_original_train_ids"]}
    declared = {_id_key(value) for value in provenance["original_train_ids"]}
    quarantined = {_id_key(value) for value in provenance["quarantined_ids"]}
    if declared != set(expected_originals) or retained != set(expected) or quarantined != expected_quarantine:
        raise ValueError("Classifier bundle retention/quarantine differs from original annotation policy")
    if (provenance["augmentation_reference_counts"] != _counts(list(expected.values()))
            or provenance["original_train_class_counts"] != _counts(originals)):
        raise ValueError("Classifier augmentation reference/original counts differ from organizer split")
    actual = {_id_key(row["id"]): row for row in train if _id_key(row["id"]) in retained}
    if actual != expected or heldout != validation:
        raise ValueError("Classifier bundle originals/holdout differ from original grouped split")
    bundle_raw = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    raw_by_id = {_id_key(row["ID"]): row for row in original_raw}
    retained_raw = {_id_key(row["ID"]): row for row in bundle_raw["train"] if _id_key(row["ID"]) in retained}
    if (retained_raw != {key: raw_by_id[key] for key in retained}
            or bundle_raw["validation"] != [raw_by_id[_id_key(row["id"])] for row in validation]):
        raise ValueError("Classifier bundle raw originals/holdout differ from organizer JSON")
    if info.get("original_sha256") != hashlib.sha256(Path(original_path).read_bytes()).hexdigest():
        raise ValueError("Classifier bundle original SHA256 differs")
    if "train_rows" not in provenance:
        raise ValueError("Classifier bundle requires synthetic-parent provenance")
    return train, info


def _length_summary(values: list[int]) -> dict:
    if not values:
        return {"min": None, "median": None, "mean": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {"min": ordered[0], "median": statistics.median(ordered), "mean": statistics.mean(ordered), "p95": ordered[math.floor(0.95 * (len(ordered) - 1))], "max": ordered[-1]}


def _counts(records: list[dict]) -> dict[str, int]:
    counts = Counter(row["label"] for row in records)
    return {str(label): counts[label] for label in LABELS}


def analyze(path: str | Path, output_dir: str | Path, seed: int = 42, validation_fraction: float = 0.2) -> dict:
    """Write analysis.json plus canonical train.json/validation.json and return it."""
    if type(seed) is not int or seed != 42:
        raise ValueError("Analysis seed must be 42")
    path = Path(path)
    records = load_records(path)
    train, validation = split_records(records, validation_fraction, seed)
    raw = json.loads(path.read_text(encoding="utf-8"))
    groups = _groups(records)
    duplicate_groups = [{"ids": [row["id"] for row in rows], "count": len(rows), "label": rows[0]["label"]} for rows in groups.values() if len(rows) > 1]
    label_runs = []
    for index, row in enumerate(records, 1):
        if not label_runs or label_runs[-1]["label"] != row["label"]:
            label_runs.append({"label": row["label"], "start_row": index, "end_row": index, "start_id": row["id"], "end_id": row["id"]})
        else:
            label_runs[-1].update(end_row=index, end_id=row["id"])
    report = {
        "data_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(records), "class_counts": _counts(records), "unique_ids": len(records),
        "null_input_count": sum(row["Input"] is None for row in raw),
        "null_label_count": sum(row["Output"]["Error_Type_Id"] is None for row in raw),
        "null_correction_count": sum(row["Output"].get("Fixed Sentence") is None for row in raw),
        "null_corrections_by_class": {str(label): sum(row["Output"]["Error_Type_Id"] == label and row["Output"].get("Fixed Sentence") is None for row in raw) for label in LABELS},
        "unique_source_groups": len(groups), "duplicate_group_count": len(duplicate_groups),
        "duplicate_extra_rows": sum(group["count"] - 1 for group in duplicate_groups),
        "duplicate_groups": duplicate_groups, "conflict_groups": [],
        "text_lengths": {field: {"characters": _length_summary([len(row[field]) for row in records]), "whitespace_tokens": _length_summary([len(row[field].split()) for row in records])} for field in ("source", "target")},
        "ordered_label_runs": label_runs,
        "parenthesized_target_ids": [row["id"] for row in records if row["target"].startswith("(")],
        "ellipsis_target_count": sum("..." in row["target"] for row in records),
        "split": {"seed": seed, "validation_fraction": validation_fraction, "train_rows": len(train), "validation_rows": len(validation), "train_class_counts": _counts(train), "validation_class_counts": _counts(validation), "train_source_groups": len(_groups(train)), "validation_source_groups": len(_groups(validation)), "shared_source_groups": 0},
        "assumptions": {"class_0_null_target": "Exact input; inferred no-error identity target", "group_normalization": "Whitespace collapse then casefold; no Unicode or punctuation normalization", "p95": "Sorted value at floor(0.95 * (N - 1))", "split_method": "Seeded class-stratified grouped subset-sum, nearest feasible row count; partition rows shuffled"},
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in (("analysis.json", report), ("train.json", train), ("validation.json", validation)):
        (destination / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    print(json.dumps(analyze(args.data, args.output_dir, 42, args.validation_fraction), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
