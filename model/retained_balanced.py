"""Separate private retained/balanced format with canonical rows and explicit repeats."""

import copy
from collections import Counter
from augmentation.review import digest
from model.data import _records_from_raw

FORMAT = "vgec-retained-balanced-v1"
FORMAT_V2 = "vgec-retained-balanced-v2"


def seal(
    stage, original, baseline, unique, train, external, metadata, format_version=FORMAT
):
    payload = {
        "format": format_version,
        "stage": stage,
        "seed": 42,
        "validation_fraction": 0,
        "validation": [],
        "original_snapshot": copy.deepcopy(original),
        "baseline_snapshot": copy.deepcopy(baseline),
        "unique": copy.deepcopy(unique),
        "train": copy.deepcopy(train),
        "external_clean": copy.deepcopy(external),
        "metadata": copy.deepcopy(metadata),
    }
    payload["content_sha256"] = digest(payload)
    return payload


def validate(raw):
    if (
        raw.get("format") not in (FORMAT, FORMAT_V2)
        or raw.get("stage") not in ("classification", "correction")
        or raw.get("seed") != 42
        or raw.get("validation_fraction") != 0
        or raw.get("validation") != []
    ):
        raise ValueError("Invalid retained balanced protocol")
    if digest({k: v for k, v in raw.items() if k != "content_sha256"}) != raw.get(
        "content_sha256"
    ):
        raise ValueError("Retained balanced content changed")
    for name in (
        "original_snapshot",
        "baseline_snapshot",
        "unique",
        "train",
        "external_clean",
    ):
        _records_from_raw(raw[name])
    canonical = {r["ID"]: r for r in raw["unique"]}
    if len(canonical) != len(raw["unique"]):
        raise ValueError("Duplicate canonical ID")
    if raw["train"][: len(raw["unique"])] != raw["unique"]:
        raise ValueError("Canonical prefix changed")
    for row in raw["train"][len(raw["unique"]) :]:
        sampling = row.get("Provenance", {}).get("resampling", {})
        source = canonical.get(sampling.get("source_id"))
        if source is None or sampling.get("is_new_unique_example") is not False:
            raise ValueError("Repeat must name a canonical example")
        if row["Input"] != source["Input"] or row["Output"] != source["Output"]:
            raise ValueError("Repeated source/label/target changed")
    byid = {r["ID"]: r for r in raw["original_snapshot"]}
    for row in raw["unique"]:
        if type(row["ID"]) is int:
            source = byid.get(row["ID"])
            if source is None or row["Input"] != source["Input"]:
                raise ValueError("Original source changed")
            if row["Output"]["Error_Type_Id"] != source["Output"]["Error_Type_Id"]:
                raise ValueError("Original label changed")
    from tools.build_retained_balanced import (
        balanced_view,
        classifier_base,
        corrector_originals,
        corpus_candidates,
        eligible_tree,
        make_row,
        tree_candidates,
        ellipsis,
    )

    baseline = raw["baseline_snapshot"]
    external = raw["external_clean"]
    metadata = raw["metadata"]
    if raw["stage"] == "classification":
        expected = classifier_base(raw["original_snapshot"], baseline) + external
        for row in expected:
            if type(row["ID"]) is int and row != byid[row["ID"]]:
                raise ValueError(
                    "Classifier must preserve the entire original annotation"
                )
    else:
        retained, _ = corrector_originals(
            raw["original_snapshot"],
            baseline,
            retain_abbreviated=raw["format"] == FORMAT_V2,
        )
        expected = (
            retained + [r for r in baseline if type(r["ID"]) is not int] + external
        )
    if raw["unique"][: len(expected)] != expected:
        raise ValueError("Retained original/baseline/external prefix differs")
    trees = metadata.get("external_parent_trees", [])
    by_external = {"ud:" + t["id"]: t for t in trees}
    for row in external:
        tree = by_external.get(row["ID"])
        if (
            tree is None
            or not eligible_tree(tree)
            or row["Input"] != tree["text"]
            or row["Output"] != {"Error_Type_Id": 0, "Fixed Sentence": tree["text"]}
        ):
            raise ValueError("External clean parent differs from eligible tree")
    additions = raw["unique"][len(expected) :]
    if additions:
        allowed = {}
        for row in raw["original_snapshot"]:
            if row["Output"]["Error_Type_Id"] == 0:
                for c in corpus_candidates(row["Input"], row["ID"]):
                    approved = make_row(c)
                    allowed[approved["ID"]] = approved
        for tree in trees:
            if not eligible_tree(tree):
                raise ValueError("Ineligible external parent")
            for c in tree_candidates(tree):
                approved = make_row(c)
                allowed[approved["ID"]] = approved
        seen = {" ".join(r["Input"].split()).casefold() for r in expected}
        parents = Counter()
        rules = Counter()
        for row in additions:
            source = " ".join(row["Input"].split()).casefold()
            if row != allowed.get(row["ID"]) or source in seen:
                raise ValueError("Unsupported or duplicated new deterministic example")
            seen.add(source)
            parents.update(row["Provenance"]["parent_ids"])
            rules.update([row["Provenance"]["template_rule"]])
        if max(parents.values(), default=0) > 4:
            raise ValueError("Parent edit cap exceeded")
        if max(rules.values(), default=0) > 32:
            raise ValueError("New rule cap exceeded")
    balance = metadata.get("stage_balance", {}).get(raw["stage"])
    if balance:
        if balance["target"] != sum(
            r["Output"]["Error_Type_Id"] == 0 for r in raw["unique"]
        ):
            raise ValueError("Balance target must equal clean population")
        expected_train, expected_info = balanced_view(
            raw["unique"],
            balance["target"],
            4,
            no_repeat_ids=(
                [
                    r["ID"]
                    for r in raw["original_snapshot"]
                    if r["Output"]["Error_Type_Id"]
                    and ellipsis(r["Output"]["Fixed Sentence"])
                ]
                if raw["format"] == FORMAT_V2 and raw["stage"] == "correction"
                else ()
            ),
        )
        if raw["train"] != expected_train or balance != expected_info:
            raise ValueError("Training exposures or declared balance differs")
    return raw


def load_retained(raw, fraction, seed):
    if fraction != 0 or seed != 42:
        raise ValueError("Retained final fit has no independent validation")
    validate(raw)
    originals = {r["ID"] for r in raw["original_snapshot"]}
    provenance = {
        "stage": raw["stage"],
        "quarantined_ids": [
            r["ID"]
            for r in raw["original_snapshot"]
            if r["ID"] not in {c["ID"] for c in raw["unique"]}
        ],
        "selected_synthetic_ids": [
            r["ID"]
            for r in raw["unique"]
            if type(r["ID"]) is not int
            and r["ID"] not in {c["ID"] for c in raw["external_clean"]}
        ],
        "external_train_ids": [r["ID"] for r in raw["external_clean"]],
        "resampled_train_ids": [r["ID"] for r in raw["train"][len(raw["unique"]) :]],
        "original_train_ids": [r["ID"] for r in raw["unique"] if r["ID"] in originals],
        "retained_original_train_ids": [
            r["ID"] for r in raw["unique"] if r["ID"] in originals
        ],
        "train_rows": [
            {
                "id": r["ID"],
                "kind": "original" if type(r["ID"]) is int else "synthetic",
                "original_id": r["ID"] if type(r["ID"]) is int else None,
                "parent_ids": r.get("Provenance", {}).get("parent_ids", []),
            }
            for r in raw["train"]
        ],
    }
    return (
        _records_from_raw(raw["train"]),
        [],
        {
            "format": raw["format"],
            "provenance": provenance,
            "independent_validation_rows": 0,
            "canonical_counts": dict(
                Counter(r["Output"]["Error_Type_Id"] for r in raw["unique"])
            ),
            "exposure_counts": dict(
                Counter(r["Output"]["Error_Type_Id"] for r in raw["train"])
            ),
        },
    )
