"""Organizer-anchored sampling of verified historical rows; never rewrites gold."""

import copy
from collections import Counter

from .retained_rules import digest, key, ellipsis, NOTES

FORMAT = "vgec-organizer-anchored-v1"
POLICY = {
    "seed": 42,
    "original_exposures": {"0": 1, "1": 2, "2": 3, "3": 3, "4": 3},
    "synthetic_class_cap": 32,
    "rule_cap": 6,
    "parent_cap": 2,
    "parent_class_cap": 1,
    "external_clean_parent_cap": 24,
    "synthetic_max_exposures": 1,
    "exclude_shortcut_prefixes": ["quoted_", "complete_training_gold_identity"],
}


def _canonical(rows):
    return [r for r in rows if not r.get("Provenance", {}).get("resampling")]


def curate(rows, stage):
    if stage not in ("classification", "correction"):
        raise ValueError("Unknown anchored stage")
    canonical = _canonical(rows)
    originals = [copy.deepcopy(r) for r in canonical if type(r["ID"]) is int]
    available = {r["ID"]: r for r in canonical}
    chosen = list(originals)
    used = {key(r["Input"]) for r in chosen}
    original_ids = {r["ID"] for r in originals}
    picked = set(original_ids)
    rules, parents, parent_class, classes = Counter(), Counter(), Counter(), Counter()
    external_parents = set()
    candidates = [
        r
        for r in canonical
        if type(r["ID"]) is not int and r["Output"]["Error_Type_Id"] != 0
    ]
    frequencies = Counter(
        r.get("Provenance", {}).get("template_rule", "") for r in candidates
    )
    for r in sorted(
        candidates,
        key=lambda r: (
            frequencies[r.get("Provenance", {}).get("template_rule", "")],
            digest(r["ID"]),
        ),
    ):
        p, label = r.get("Provenance", {}), r["Output"]["Error_Type_Id"]
        rule = p.get("template_rule", "")
        ids = p.get("parent_ids", [])
        target = r["Output"]["Fixed Sentence"]
        if (
            not rule
            or not ids
            or any(rule.startswith(x) for x in POLICY["exclude_shortcut_prefixes"])
            or key(r["Input"]) in used
            or not isinstance(target, str)
            or not target.strip()
            or ellipsis(target)
            or target.lstrip().startswith("(")
        ):
            continue
        if (
            rules[rule] >= POLICY["rule_cap"]
            or classes[label] >= POLICY["synthetic_class_cap"]
            or any(
                parents[pid] >= POLICY["parent_cap"]
                or parent_class[(pid, label)] >= POLICY["parent_class_cap"]
                for pid in ids
            )
        ):
            continue
        missing = [pid for pid in ids if pid not in picked]
        if any(
            pid not in available
            or available[pid]["Output"]["Error_Type_Id"] != 0
            or available[pid].get("Provenance", {}).get("origin")
            != "human_annotated_ud_vietnamese_vtb_train"
            for pid in missing
        ):
            continue
        if len(external_parents | set(missing)) > POLICY["external_clean_parent_cap"]:
            continue
        if any(
            key(available[pid]["Input"]) in used or ellipsis(available[pid]["Input"])
            for pid in missing
        ):
            continue
        for pid in missing:
            parent = copy.deepcopy(available[pid])
            chosen.append(parent)
            picked.add(pid)
            external_parents.add(pid)
            used.add(key(parent["Input"]))
        chosen.append(copy.deepcopy(r))
        picked.add(r["ID"])
        used.add(key(r["Input"]))
        classes[label] += 1
        rules[rule] += 1
        for pid in ids:
            parents[pid] += 1
            parent_class[(pid, label)] += 1
    train = copy.deepcopy(chosen)
    for r in originals:
        target = r["Output"]["Fixed Sentence"] or ""
        protected = r["ID"] in NOTES or (
            stage == "correction"
            and (ellipsis(target) or target.lstrip().startswith("("))
        )
        count = (
            1
            if protected
            else POLICY["original_exposures"][str(r["Output"]["Error_Type_Id"])]
        )
        for draw in range(2, count + 1):
            repeated = copy.deepcopy(r)
            repeated["ID"] = "anchor-repeat:" + digest([r["ID"], draw])[:24]
            repeated.setdefault("Provenance", {})["resampling"] = {
                "source_id": r["ID"],
                "draw": draw,
                "is_new_unique_example": False,
            }
            train.append(repeated)
    counts = lambda group: {
        str(k): sum(r["Output"]["Error_Type_Id"] == k for r in group) for k in range(5)
    }
    report = {
        "original_canonical_rows": len(originals),
        "original_exposures": sum(
            type(r["ID"]) is int or str(r["ID"]).startswith("anchor-repeat:")
            for r in train
        ),
        "canonical_counts": counts(chosen),
        "exposure_counts": counts(train),
        "unique_source_counts": {
            str(k): len(
                {key(r["Input"]) for r in chosen if r["Output"]["Error_Type_Id"] == k}
            )
            for k in range(5)
        },
        "selected_synthetic_errors": sum(classes.values()),
        "selected_rules": dict(rules),
        "external_clean_parents": len(external_parents),
        "withheld_synthetic_rows": len(canonical) - len(chosen),
        "maximum_synthetic_exposures": 1,
        "labels_or_targets_rewritten": 0,
        "scope": "Existing evidence/provenance and conservative sampling only; not human adjudication or measured improvement",
    }
    return train, report


def seal(base):
    train, report = curate(base["train"], base["stage"])
    result = {
        "format": FORMAT,
        "stage": base["stage"],
        "seed": 42,
        "source_bundle": base,
        "source_sha256": digest(base),
        "policy": copy.deepcopy(POLICY),
        "train": train,
        "validation": copy.deepcopy(base["validation"]),
        "report": report,
    }
    result["content_sha256"] = digest(result)
    return result


def load_anchored(raw, fraction, seed):
    from .data import _records_from_raw
    from .grouped_validation import audit
    from .quality_data import load_quality
    from .diversity_data import load_diverse

    if (
        raw.get("format") != FORMAT
        or raw.get("policy") != POLICY
        or raw.get("seed") != 42
        or digest({k: v for k, v in raw.items() if k != "content_sha256"})
        != raw.get("content_sha256")
    ):
        raise ValueError("Anchored mixture protocol/content differs")
    base = raw["source_bundle"]
    if digest(base) != raw.get("source_sha256") or base["stage"] != raw["stage"]:
        raise ValueError("Anchored source differs")
    loader = {
        "vgec-quality-mixture-v1": load_quality,
        "vgec-diverse-mixture-v1": load_diverse,
    }.get(base.get("format"))
    if loader is None:
        raise ValueError("Anchored mixture requires a verified quality/diverse source")
    _, validation, info = loader(base, fraction, seed)
    expected, report = curate(base["train"], raw["stage"])
    if (
        raw["train"] != expected
        or raw["validation"] != base["validation"]
        or raw["report"] != report
    ):
        raise ValueError("Anchored mixture differs from deterministic policy")
    leakage = audit({raw["stage"]: raw["train"]}, raw["validation"])
    info.update(format=FORMAT, anchored_report=report, leakage_audit=leakage)
    return _records_from_raw(raw["train"]), validation, info


def organizer_source(raw):
    """Resolve the existing immutable organizer snapshot, never a fit population."""
    base = raw["source_bundle"]
    if base.get("format") == "vgec-diverse-mixture-v1":
        base = base["source_quality"]
    return base["source_bundle"]["source_packet"]["classification"]
