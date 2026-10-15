"""Read-only public-output ellipsis audit and complete private dataset inventory."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def has_ellipsis(text):
    return isinstance(text, str) and bool(re.search(r"\.{3,}|…", text))


def audit_predictions(sources, predictions):
    byid = {(type(r["ID"]), r["ID"]): r["Input"] for r in sources}
    ids = [(type(r["ID"]), r["ID"]) for r in predictions]
    if len(set(ids)) != len(ids) or set(ids) != set(byid):
        raise ValueError("Prediction IDs must match source IDs exactly")
    counts = Counter()
    introduced = []
    for row, k in zip(predictions, ids):
        value = row["Output"]["Fixed Sentence"]
        targets = [value] if isinstance(value, str) else value
        if not isinstance(targets, list) or any(
            not isinstance(t, str) for t in targets
        ):
            raise ValueError("Invalid correction value")
        source_has = has_ellipsis(byid[k])
        target_has = any(map(has_ellipsis, targets))
        counts["source_ellipsis_rows"] += source_has
        counts["target_ellipsis_rows"] += target_has
        counts["inherited_ellipsis_rows"] += source_has and target_has
        counts["introduced_ellipsis_rows"] += not source_has and target_has
        counts["source_ellipsis_target_without_marker_rows"] += (
            source_has and not target_has
        )
        if not source_has and target_has:
            introduced.append(row["ID"])
    return {
        "rows": len(predictions),
        **{
            k: counts[k]
            for k in (
                "source_ellipsis_rows",
                "target_ellipsis_rows",
                "inherited_ellipsis_rows",
                "introduced_ellipsis_rows",
                "source_ellipsis_target_without_marker_rows",
            )
        },
        "introduced_ellipsis_ids": introduced,
    }


def inventory(rows, original):
    original_ids = {r["ID"] for r in original}
    groups = defaultdict(list)
    for r in rows:
        p = r.get("Provenance", {})
        sampling = p.get("resampling", {})
        identity = sampling.get("source_id", r["ID"])
        if identity in original_ids:
            origin = "organizer_original"
            mechanism = (
                "organizer_reference_with_ellipsis"
                if has_ellipsis(r["Output"]["Fixed Sentence"])
                else "organizer_reference_without_ellipsis"
            )
        else:
            origin = p.get("origin", "unrecorded_non_original_origin")
            mechanism = p.get(
                "template_rule",
                (
                    "external_clean_identity"
                    if origin.startswith("human_annotated_ud_")
                    else "unrecorded_mechanism"
                ),
            )
        exposure = "repeat" if sampling else "canonical"
        groups[(r["Output"]["Error_Type_Id"], origin, mechanism, exposure)].append(r)
    return [
        {
            "label": label,
            "origin": origin,
            "mechanism": mechanism,
            "exposure": exposure,
            "count": len(values),
            "distinct_sources": len(
                {" ".join(r["Input"].split()).casefold() for r in values}
            ),
        }
        for (label, origin, mechanism, exposure), values in sorted(groups.items())
    ]


def run(output):
    base = ROOT / "dataset/kaggle-sailor-qwen-lora-20261014"
    original = json.loads((base / "VGEC 2026_v1.json").read_text())
    public_bytes = (base / "public_test.json").read_bytes()
    public = json.loads(public_bytes)
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    public_files = []
    errors = []
    correction_groups = defaultdict(list)
    roots = [ROOT / "runs", ROOT.parent / "test/runs", ROOT.parent / "alpha/runs"]
    for root in roots:
        for p in sorted(root.rglob("results.*")):
            if p.name not in ("results.json", "results.zip") or any(
                s in str(p).lower() for s in ("smoke", "fixture", "mechanics")
            ):
                continue
            try:
                if p.suffix == ".zip":
                    with zipfile.ZipFile(p) as z:
                        payload = z.read("results.json")
                else:
                    payload = p.read_bytes()
                rows = json.loads(payload)
                if not isinstance(rows, list) or len(rows) != len(public):
                    continue
                result = audit_predictions(public, rows)
                targets = [r["Output"]["Fixed Sentence"] for r in rows]
                targets = [v if isinstance(v, list) else [v] for v in targets]
                correction_sha = hashlib.sha256(
                    json.dumps(
                        targets, ensure_ascii=False, separators=(",", ":")
                    ).encode()
                ).hexdigest()
                public_files.append(
                    {
                        "path": str(p.resolve()),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "corrections_sha256": correction_sha,
                        **result,
                    }
                )
                correction_groups[correction_sha].append(str(p.resolve()))
            except (ValueError, KeyError, OSError, zipfile.BadZipFile) as e:
                errors.append({"path": str(p), "error": str(e)})
    summary = {
        "public_input_sha256": hashlib.sha256(public_bytes).hexdigest(),
        "public_input_rows": len(public),
        "public_input_ellipsis_rows": sum(has_ellipsis(r["Input"]) for r in public),
        "public_source_exact_training_overlap": sum(
            " ".join(r["Input"].split()).casefold()
            in {" ".join(t["Input"].split()).casefold() for t in original}
            for r in public
        ),
        "files": public_files,
        "distinct_correction_payloads": len(correction_groups),
        "read_errors": errors,
        "interpretation": "Ellipsis marker audit, not gold-reference accuracy; generated public predictions are not training supervision.",
    }
    (output / "public_ellipsis_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    entries = []
    datasets = []
    examples = {}
    paths = [
        p
        for p in (ROOT / "dataset").glob("*/VGEC_*.json")
        if p.name in ("VGEC_classifier.json", "VGEC_balanced.json")
    ]
    paths += [
        ROOT / "dataset/full-training-including-validation-20261014" / name
        for name in ("classification_train.json", "correction_train.json")
    ]
    paths += [ROOT / "dataset/validation-only-20261014/validation.json"]
    for p in sorted(p for p in paths if p.exists()):
        raw = json.loads(p.read_text())
        stage = (
            "classification"
            if p.name in ("VGEC_classifier.json", "classification_train.json")
            else "evaluation" if p.name == "validation.json" else "correction"
        )
        views = (
            {"validation" if stage == "evaluation" else "train": raw}
            if isinstance(raw, list)
            else {"train": raw["train"], "validation": raw.get("validation", [])}
        )
        if isinstance(raw, dict) and "unique" in raw:
            views["canonical"] = raw["unique"]
        for view, rows in views.items():
            items = inventory(rows, original)
            prefix = {"dataset": p.parent.name, "stage": stage, "view": view}
            entries.extend({**prefix, **v} for v in items)
            datasets.append(
                {
                    **prefix,
                    "rows": len(rows),
                    "counts": {
                        str(i): sum(r["Output"]["Error_Type_Id"] == i for r in rows)
                        for i in range(5)
                    },
                    "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
            )
            for row in rows:
                prov = row.get("Provenance", {})
                if type(row["ID"]) is int or prov.get("resampling"):
                    continue
                identity = (
                    row["Output"]["Error_Type_Id"],
                    prov.get("origin", "unknown"),
                    prov.get("template_rule", "external_clean_identity"),
                )
                examples.setdefault("|".join(map(str, identity)), row)
    with (output / "all_dataset_types.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "stage",
                "view",
                "label",
                "origin",
                "mechanism",
                "exposure",
                "count",
                "distinct_sources",
            ],
        )
        writer.writeheader()
        writer.writerows(entries)
    (output / "all_dataset_types.json").write_text(
        json.dumps({"datasets": datasets, "types": entries}, indent=2) + "\n"
    )
    (output / "private_examples_by_type.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# Complete saved-dataset extra-data inventory\n",
        "PRIVATE: contains corpus-derived examples in the linked JSON; do not redistribute.\n",
        "Counts separate canonical records, explicit repeated exposures and validation. Original duplicates are retained. Unpromoted candidate directories are build iterations. Public predictions remain audit evidence only.\n",
        "Every origin/mechanism/class in every saved stage bundle under dataset/ is included in the CSV/JSON. private_examples_by_type.json gives one actual example of every recorded extra-data type.\n",
        "## Public-output ellipsis audit\n",
        "| File | Rows | Output ellipses | Already present in input | Introduced |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in public_files:
        lines.append(
            f"| {Path(row['path']).relative_to(ROOT.parent)} | {row['rows']} | {row['target_ellipsis_rows']} | {row['inherited_ellipsis_rows']} | {row['introduced_ellipsis_rows']} |"
        )
    for dataset in sorted({v["dataset"] for v in entries}):
        lines.extend(
            [
                "",
                f"## {dataset}\n",
                "| Stage/view | Class | Origin | Type | Exposure | Count |",
                "|---|---:|---|---|---|---:|",
            ]
        )
        for r in entries:
            if r["dataset"] == dataset:
                lines.append(
                    f"| {r['stage']}/{r['view']} | {r['label']} | {r['origin']} | {r['mechanism']} | {r['exposure']} | {r['count']} |"
                )
    (output / "ALL_DATASET_TYPES.md").write_text("\n".join(lines) + "\n")
    return {
        "output": str(output),
        "public_files": len(public_files),
        "correction_payloads": len(correction_groups),
        "dataset_bundles": len({(v["dataset"], v["stage"]) for v in entries}),
        "inventory_rows": len(entries),
        "extra_types": len(examples),
        "public_read_errors": errors,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    print(json.dumps(run(p.parse_args().output_dir), indent=2))
