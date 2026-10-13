#!/usr/bin/env python3
"""Build private quality-capped VGEC training bundles without changing gold.

Correction excludes incomplete/annotation references; classification excludes
explicit correct-sentence contradictions. Validation stays raw and seed 42 fixed.
Clean additions are capped at 40; error additions at classifier-eligible originals.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import unicodedata
from typing import Any

# Support `python tools/build_dataset.py` without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.data import declares_correct_sentence, load_records, split_records


LABELS = tuple(range(5))
GENERATED_FILENAMES = ("VGEC_balanced.json", "VGEC_classifier.json", "audit.json", "manifest.json")


def normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def id_key(value: Any) -> tuple[type, Any]:
    if type(value) not in (int, str):
        raise ValueError(f"ID must be an integer or string, got {value!r}")
    return type(value), value


def counts(records: list[dict]) -> dict[str, int]:
    counter = Counter(row["label"] for row in records)
    return {str(label): counter[label] for label in LABELS}


def punctuation(text: str) -> str:
    return "".join(char for char in text if unicodedata.category(char).startswith("P"))


def without_punctuation(text: str) -> str:
    return normalized("".join(char for char in text if not unicodedata.category(char).startswith("P")))


def audit_record(row: dict, raw: dict, *, original: bool = False) -> dict:
    source, target, label = row["source"], row["target"], row["label"]
    flags, quarantine = [], []
    if target.lstrip().startswith("("):
        quarantine.append("parenthesized_target_annotation")
        if label != 0 and declares_correct_sentence(target):
            flags.append("annotation_declares_source_correct")
    if "..." in target or "…" in target:
        quarantine.append("elliptical_target_not_full_sentence")
    if label != 0 and normalized(source) == normalized(target):
        quarantine.append("nonzero_label_identity_target")
    if label == 0 and source != target:
        quarantine.append("class0_nonidentity_target")
    if label in (1, 2, 3) and normalized(source) != normalized(target) and without_punctuation(source) == without_punctuation(target):
        quarantine.append("punctuation_only_change_with_nonpunctuation_label")
    if punctuation(source) != punctuation(target):
        flags.append("source_target_punctuation_changed")
    if label == 4 and punctuation(source) == punctuation(target):
        flags.append("punctuation_label_without_punctuation_change")
    if "..." in source or "…" in source:
        quarantine.append("elliptical_source")
    if not target.strip():
        quarantine.append("empty_target")
    elif target.rstrip()[-1] not in ".!?。！？\"”’':;":
        flags.append("target_without_terminal_punctuation")
    if raw["Output"].get("Fixed Sentence") is None:
        flags.append("class0_null_target_identity_for_training_only")
    return {
        "original_id": row["id"], "label": label,
        "flags": flags + quarantine, "quarantine_reasons": quarantine,
        "source_punctuation": punctuation(source), "target_punctuation": punctuation(target),
        "raw_target_is_null": raw["Output"].get("Fixed Sentence") is None,
        "normalized_identity": normalized(source) == normalized(target),
    }


def read_input(path: Path) -> tuple[bytes, list[dict], list[dict]]:
    content = path.read_bytes()
    raw = json.loads(content)
    records = load_records(path)
    if path.read_bytes() != content:
        raise ValueError(f"Input changed while being read: {path}")
    return content, raw, records


def validate_pool(raw: list[dict], records: list[dict], original_ids: set) -> None:
    for raw_row, row in zip(raw, records):
        if id_key(row["id"]) in original_ids:
            raise ValueError(f"Synthetic ID collides with original ID: {row['id']!r}")
        metadata = raw_row.get("Provenance")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("template_rule"), str) or not metadata["template_rule"].strip():
            raise ValueError(f"Synthetic ID {row['id']!r} requires Provenance.template_rule")
        parents = metadata.get("parent_ids", [])
        if not isinstance(parents, list):
            raise ValueError(f"Synthetic ID {row['id']!r} parent_ids must be an array")
        for parent in parents:
            id_key(parent)
        if row["label"] == 0 and row["source"] != row["target"]:
            raise ValueError("Synthetic clean examples require exact identity targets")


def prepare_destination(destination: Path, inputs: tuple[Path, Path]) -> None:
    missing = []
    ancestor = destination
    while not ancestor.exists():
        missing.append(ancestor)
        ancestor = ancestor.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    if destination.is_symlink():
        raise ValueError(f"Refusing output-directory symlink: {destination}")
    os.chmod(destination, 0o700)
    if not destination.is_dir():
        raise ValueError(f"Output directory is not a directory: {destination}")
    for filename in GENERATED_FILENAMES:
        path = destination / filename
        if path.is_symlink():
            raise ValueError(f"Refusing to overwrite generated-path symlink: {path}")
        if path.exists() and not path.is_file():
            raise ValueError(f"Generated output is not a regular file: {path}")
        for source in inputs:
            if path.resolve() == source.resolve() or (path.exists() and os.path.samefile(path, source)):
                raise ValueError(f"Output would overwrite an input: {path}")
        if path.exists() and path.stat().st_nlink != 1:
            raise ValueError(f"Refusing to overwrite hard-linked output: {path}")
        if path.exists():
            expected_format = {
                "VGEC_balanced.json": {"vgec-balanced-v3"},
                "VGEC_classifier.json": {"vgec-balanced-v3"},
                "audit.json": {"vgec-audit-v2"},
                "manifest.json": {"vgec-manifest-v3"},
            }[filename]
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"Refusing to overwrite unrelated output: {path}") from exc
            if not isinstance(existing, dict) or existing.get("format") not in expected_format:
                raise ValueError(f"Refusing to overwrite unrelated output: {path}")


def build_split(
    records: list[dict], raw_by_id: dict, audits: dict,
    synthetic: list[dict], synthetic_raw: dict,
    validation_fraction: float, *, classifier: bool = False,
) -> tuple[dict, dict]:
    seed = 42
    original_train, original_validation = split_records(records, validation_fraction, seed)
    validation_text = {normalized(row[field]) for row in original_validation for field in ("source", "target")}
    leaking_sources = {
        normalized(row["source"]) for row in original_train
        if any(normalized(row[field]) in validation_text for field in ("source", "target"))
    }
    exclusion_reasons = {}
    for row in original_train:
        audit = audits[id_key(row["id"])]
        if classifier:
            reasons = ["annotation_declares_source_correct"] if "annotation_declares_source_correct" in audit["flags"] else []
        else:
            reasons = list(audit["quarantine_reasons"])
        if normalized(row["source"]) in leaking_sources:
            reasons.append("original_source_group_overlaps_validation_source_or_target")
        exclusion_reasons[id_key(row["id"])] = reasons
    retained = [row for row in original_train if not exclusion_reasons[id_key(row["id"])]]
    quarantined = [row for row in original_train if exclusion_reasons[id_key(row["id"])]]
    if classifier and any("original_source_group_overlaps_validation_source_or_target" in reasons for reasons in exclusion_reasons.values()):
        raise ValueError("Classifier cannot retain training text overlapping validation")
    class_counts = counts(retained)
    if any(value == 0 for value in class_counts.values()):
        raise ValueError(f"Seed {seed}: each class needs retained original training examples")
    largest = max(class_counts.values())
    reference_counts = counts([row for row in original_train
                               if "annotation_declares_source_correct" not in audits[id_key(row["id"])]["flags"]
                               and normalized(row["source"]) not in leaking_sources])
    targets = {str(label): class_counts[str(label)] + (
        min(40, class_counts[str(label)]) if label == 0 else
        min(reference_counts[str(label)], max(0, largest - class_counts[str(label)]))) for label in LABELS}

    train_ids = {id_key(row["id"]) for row in original_train}
    original_sources = defaultdict(list)
    for row in original_train:
        original_sources[normalized(row["source"])].append(row)
    pool_sources = defaultdict(set)
    for row in synthetic:
        pool_sources[normalized(row["source"])].add((row["label"], normalized(row["target"])))
    leaking_families = set()
    for row in synthetic:
        if any(normalized(row[field]) in validation_text for field in ("source", "target")):
            metadata = synthetic_raw[id_key(row["id"])]["Provenance"]
            leaking_families.update((field, metadata[field]) for field in ("family_id", "template_id") if metadata.get(field))

    eligible = defaultdict(list)
    rejections = []
    for row in synthetic:
        raw = synthetic_raw[id_key(row["id"])]
        metadata = raw["Provenance"]
        reasons = []
        if normalized(row["source"]) in validation_text or normalized(row["target"]) in validation_text:
            reasons.append("source_or_target_overlaps_original_validation_source_or_target")
        if any((field, metadata.get(field)) in leaking_families for field in ("family_id", "template_id")):
            reasons.append("authored_family_overlaps_validation_text")
        if any(id_key(parent) not in train_ids for parent in metadata.get("parent_ids", [])):
            reasons.append("parent_not_in_original_training_partition")
        key = normalized(row["source"])
        if key in original_sources:
            annotations = {(other["label"], normalized(other["target"])) for other in original_sources[key]}
            reasons.append("original_train_source_collision" if annotations == {(row["label"], normalized(row["target"]))} else "original_train_source_annotation_conflict")
        if len(pool_sources[key]) > 1:
            reasons.append("synthetic_source_annotation_conflict")
        reasons.extend(audit_record(row, raw)["quarantine_reasons"])
        if reasons:
            rejections.append({"synthetic_id": row["id"], "reasons": reasons})
        else:
            eligible[row["label"]].append(row)

    rng = random.Random(seed)
    selected = []
    selected_sources = {}
    for label in LABELS:
        families = defaultdict(list)
        for row in sorted(eligible[label], key=lambda row: (type(row["id"]).__name__, str(row["id"]))):
            metadata = synthetic_raw[id_key(row["id"])]["Provenance"]
            families[metadata["template_rule"]].append(row)
        rules = sorted(families)
        rng.shuffle(rules)
        for family in families.values():
            rng.shuffle(family)
        candidates = []
        while any(families.values()):
            for rule in rules:
                if families[rule]:
                    candidates.append(families[rule].pop())
        needed = targets[str(label)] - class_counts[str(label)]
        for row in candidates:
            key = normalized(row["source"])
            if key in selected_sources:
                previous = selected_sources[key]
                conflict = (previous["label"], normalized(previous["target"])) != (row["label"], normalized(row["target"]))
                rejections.append({"synthetic_id": row["id"], "reasons": ["selected_synthetic_annotation_conflict" if conflict else "selected_synthetic_source_collision"]})
                continue
            if needed:
                selected.append(row)
                selected_sources[key] = row
                needed -= 1
        if needed:
            raise ValueError(f"Seed {seed}: class {label} has insufficient eligible unique synthetic sources; missing {needed} of {targets[str(label)] - class_counts[str(label)]} required")

    train_entries = []
    for row in retained:
        raw = copy.deepcopy(raw_by_id[id_key(row["id"])])
        transformations = []
        train_entries.append((raw, {
            "id": row["id"], "kind": "original", "original_id": row["id"],
            "transformations": transformations, "audit_flags": audits[id_key(row["id"])]["flags"],
        }))
    for row in selected:
        raw = copy.deepcopy(synthetic_raw[id_key(row["id"])])
        metadata = raw["Provenance"]
        train_entries.append((raw, {
            "id": row["id"], "kind": "synthetic", "template_rule": metadata["template_rule"],
            "parent_ids": metadata.get("parent_ids", []), "authored_provenance": copy.deepcopy(metadata),
            "transformations": [],
        }))
    rng.shuffle(train_entries)
    train = [raw for raw, _ in train_entries]
    validation = [copy.deepcopy(raw_by_id[id_key(row["id"])]) for row in original_validation]
    balanced_counts = counts(retained + selected)
    if balanced_counts != targets:
        raise ValueError(f"Seed {seed}: capped augmentation invariant failed")
    provenance = {
        "split_method": "model.data.split_records on original corpus before quarantine or augmentation",
        "seed": seed, "balance_policy": "quality_capped_2x", "balance_targets_by_class": targets,
        "augmentation_reference_counts": reference_counts,
        "original_train_class_counts": counts(original_train),
        "original_train_ids": [row["id"] for row in original_train],
        "retained_original_train_ids": [row["id"] for row in retained],
        "quarantined_ids": [row["id"] for row in quarantined],
        "quarantine": [{"original_id": row["id"], "reasons": exclusion_reasons[id_key(row["id"])]} for row in quarantined],
        "selected_synthetic_ids": [row["id"] for row in selected],
        "class_counts": {"original_train": counts(original_train), "retained_original_train": class_counts, "selected_synthetic": counts(selected), "balanced_train": balanced_counts, "validation": counts(original_validation)},
        "rejections": rejections,
        "rejection_reason_counts": dict(sorted(Counter(reason for item in rejections for reason in item["reasons"]).items())),
        "eligible_pool_class_counts_before_selection": counts([row for rows in eligible.values() for row in rows]),
        "train_rows": [metadata for _, metadata in train_entries],
        "validation_rows": [{"id": row["id"], "kind": "original", "original_id": row["id"], "transformations": []} for row in original_validation],
        "audit_rows": [copy.deepcopy(audits[id_key(row["id"])]) for row in original_train + original_validation],
        "authored_overrides": [], "validation_raw_unchanged": True,
        "quarantine_repairs_gold": False,
    }
    if classifier:
        provenance["training_task"] = "classifier"
        provenance["original_retention_policy"] = "exclude_explicit_correct_sentence_annotation_contradictions_v2"
    summary = {
        "train_rows": len(train), "validation_rows": len(validation),
        "class_counts": provenance["class_counts"], "quarantined_ids": provenance["quarantined_ids"],
        "selected_synthetic_ids": provenance["selected_synthetic_ids"],
        "rejected_synthetic_count": len(rejections),
        "rejection_reason_counts": provenance["rejection_reason_counts"],
        "selected_rule_counts": dict(sorted(Counter(
            f"{row['label']}:{synthetic_raw[id_key(row['id'])]['Provenance']['template_rule']}"
            for row in selected).items())),
    }
    if classifier:
        summary["training_task"] = "classifier"
    return {"train": train, "validation": validation, "provenance": provenance}, summary


def build(original: Path, synthetic_path: Path, output_dir: Path, validation_fraction: float) -> dict:
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    original_bytes, raw, records = read_input(original)
    synthetic_bytes, pool_raw, pool = read_input(synthetic_path)
    original_sha = hashlib.sha256(original_bytes).hexdigest()
    synthetic_sha = hashlib.sha256(synthetic_bytes).hexdigest()
    raw_by_id = {id_key(row["ID"]): row for row in raw}
    synthetic_raw = {id_key(row["ID"]): row for row in pool_raw}
    validate_pool(pool_raw, pool, set(raw_by_id))
    audit_rows = [audit_record(row, raw_by_id[id_key(row["id"])], original=True) for row in records]
    audits = {id_key(row["original_id"]): row for row in audit_rows}
    privacy = {"private_dataset": True, "external_teacher_used_by_builder": False,
               "synthetic_policy": "Deterministic local corpus-derived pool; original training partition only; no private corpus sent externally",
               "publication_allowed": False, "redistribution_allowed": False,
               "terms": "Noncommercial research-only; acknowledge VLSP 2026 and VGEC organizers. Third-party hosting requires organizer authorization."}
    audit = {
        "format": "vgec-audit-v2", "original_sha256": original_sha,
        "rows": audit_rows, "original_class_counts": counts(records),
        "flag_counts": dict(sorted(Counter(flag for row in audit_rows for flag in row["flags"]).items())),
        "ambiguous_original_ids": [row["original_id"] for row in audit_rows if row["quarantine_reasons"]],
        "quarantine_policy": "Correction excludes annotations and ellipses in source/target; classifier excludes explicit correct-sentence annotation contradictions. Gold and validation stay raw.",
        "punctuation_policy": "Punctuation-only changes with nonpunctuation labels are excluded from correction training, not relabeled. Synthetic equivalents are rejected.",
        "authored_overrides": [], "quarantine_repairs_gold": False, "privacy": privacy,
    }
    split, summary = build_split(records, raw_by_id, audits, pool, synthetic_raw, validation_fraction)
    bundle = {"format": "vgec-balanced-v3", "seed": 42, "balance_policy": "quality_capped_2x", "validation_fraction": validation_fraction, "original_sha256": original_sha, "synthetic_sha256": synthetic_sha, "privacy": privacy, **split}
    classifier_split, classifier_summary = build_split(
        records, raw_by_id, audits, pool, synthetic_raw, validation_fraction, classifier=True,
    )
    classifier_bundle = {**bundle, **classifier_split, "training_task": "classifier"}
    manifest = {"format": "vgec-manifest-v3", "seed": 42, "balance_policy": "quality_capped_2x", "inputs": {"original": {"path": str(original), "sha256": original_sha, "rows": len(raw)}, "synthetic": {"path": str(synthetic_path), "sha256": synthetic_sha, "rows": len(pool_raw)}}, "validation_fraction": validation_fraction, "privacy": privacy, "split": summary}
    manifest["classifier_split"] = classifier_summary
    if original.read_bytes() != original_bytes or synthetic_path.read_bytes() != synthetic_bytes:
        raise ValueError("An input changed during building; no generated outputs were written")
    payloads = {"VGEC_balanced.json": bundle, "VGEC_classifier.json": classifier_bundle, "audit.json": audit}
    serialized = {name: (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8") for name, value in payloads.items()}
    manifest["outputs"] = {name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)} for name, content in serialized.items()}
    serialized["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    # Known generated filenames only; unrelated files (including the pool) stay intact.
    prepare_destination(output_dir, (original, synthetic_path))
    for name in GENERATED_FILENAMES:
        path = output_dir / name
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(serialized[name])
    if hashlib.sha256(original.read_bytes()).hexdigest() != original_sha:
        raise ValueError("Original changed while writing generated outputs")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("dataset/original/VGEC 2026_v1.json"))
    parser.add_argument("--synthetic", type=Path, default=Path("dataset/newdataset/synthetic_pool.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/newdataset"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    try:
        manifest = build(args.original, args.synthetic, args.output_dir, args.validation_fraction)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Dataset build failed: {exc}\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
