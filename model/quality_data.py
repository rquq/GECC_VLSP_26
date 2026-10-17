"""Versioned training-only mixture; organizer labels and grouped validation stay fixed."""

import copy
from collections import Counter
from .retained_rules import digest, key, ellipsis, NOTES

FORMAT = "vgec-quality-mixture-v1"
POLICY = {
    "operator_rule_cap": 8,
    "operator_parent_cap": 2,
    "original_legacy_max_exposures": 4,
    "operator_max_exposures": 2,
    "clean_control_budget": 37,
    "seed": 42,
}


def curate(rows, stage):
    if stage not in ("classification", "correction"):
        raise ValueError("Unknown quality stage")
    canonical = [
        copy.deepcopy(r) for r in rows if not r.get("Provenance", {}).get("resampling")
    ]
    operators = [r for r in canonical if str(r["ID"]).startswith("balanced:")]
    frequency = Counter(r["Provenance"]["template_rule"] for r in operators)
    chosen, rules, parents = [], Counter(), Counter()
    for r in sorted(
        operators,
        key=lambda r: (
            [2, 1, 3, 4].index(r["Output"]["Error_Type_Id"]),
            frequency[r["Provenance"]["template_rule"]],
            digest(r["ID"]),
        ),
    ):
        p = r["Provenance"]
        rule = p["template_rule"]
        family = str(p.get("parent_ids", []))
        if (
            rules[rule] >= POLICY["operator_rule_cap"]
            or parents[family] >= POLICY["operator_parent_cap"]
        ):
            continue
        chosen.append(r)
        rules[rule] += 1
        parents[family] += 1
    retained = [r for r in canonical if not str(r["ID"]).startswith("balanced:")]
    known = {key(r["Input"]) for r in retained}
    controls = []
    for r in chosen:
        text = r["Output"]["Fixed Sentence"]
        if key(text) in known or ellipsis(text) or not text.endswith("."):
            continue
        controls.append(
            {
                "ID": "control:" + digest(text)[:24],
                "Input": text,
                "Output": {"Error_Type_Id": 0, "Fixed Sentence": text},
                "Provenance": {
                    "origin": "exact_existing_corrected_parent_control",
                    "parent_ids": r["Provenance"]["parent_ids"],
                    "operator_example_id": r["ID"],
                    "source_target_sha256": digest(text),
                    "uses_external_teacher": False,
                    "review_status": "existing_reference_copy_not_new_human_adjudication",
                },
            }
        )
        known.add(key(text))
        if len(controls) >= POLICY["clean_control_budget"]:
            break
    # Replace legacy synthetic clean controls, never organizer or licensed UD rows.
    replace = [
        r
        for r in retained
        if r["Output"]["Error_Type_Id"] == 0
        and type(r["ID"]) is not int
        and not str(r["ID"]).startswith("ud:")
    ]
    removed = {
        r["ID"] for r in sorted(replace, key=lambda r: digest(r["ID"]))[: len(controls)]
    }
    retained = [r for r in retained if r["ID"] not in removed]
    selected = retained + chosen + controls
    train = copy.deepcopy(selected)
    groups = {
        k: [r for r in selected if r["Output"]["Error_Type_Id"] == k] for k in range(5)
    }
    target = len(groups[0])
    exposure = Counter(r["Output"]["Error_Type_Id"] for r in train)

    def cap(r):
        if r["ID"] in NOTES or str(r["ID"]).startswith("control:"):
            return 1
        if stage == "correction" and ellipsis(r["Output"]["Fixed Sentence"] or ""):
            return 1
        return (
            POLICY["operator_max_exposures"]
            if str(r["ID"]).startswith("balanced:")
            else POLICY["original_legacy_max_exposures"]
        )

    for k in range(5):
        group = sorted(
            groups[k],
            key=lambda r: (
                (
                    0
                    if type(r["ID"]) is int
                    else 2 if str(r["ID"]).startswith("balanced:") else 1
                ),
                digest(r["ID"]),
            ),
        )
        for cycle in range(1, POLICY["original_legacy_max_exposures"]):
            for r in group:
                if exposure[k] >= target:
                    break
                if cycle >= cap(r):
                    continue
                repeated = copy.deepcopy(r)
                repeated["ID"] = "quality-repeat:" + digest([r["ID"], cycle])[:24]
                repeated.setdefault("Provenance", {})["resampling"] = {
                    "source_id": r["ID"],
                    "draw": cycle + 1,
                    "is_new_unique_example": False,
                }
                train.append(repeated)
                exposure[k] += 1
    report = {
        "canonical_counts": {str(k): len(groups[k]) for k in groups},
        "unique_source_counts": {
            str(k): len({key(r["Input"]) for r in groups[k]}) for k in groups
        },
        "exposure_counts": {str(k): exposure[k] for k in groups},
        "shortfalls": {str(k): max(0, target - exposure[k]) for k in groups},
        "operator_rules": dict(rules),
        "operator_rows_before": len(operators),
        "operator_rows_after": len(chosen),
        "matched_clean_controls": len(controls),
        "legacy_clean_replaced": len(removed),
        "policy": POLICY,
        "quality_scope": "controlled repetition/mechanism mix; no measured score or human-adjudication claim",
    }
    return selected, train, report


def seal(base):
    _, train, report = curate(base["train"], base["stage"])
    result = {
        "format": FORMAT,
        "stage": base["stage"],
        "seed": 42,
        "source_bundle": base,
        "source_sha256": digest(base),
        "policy": POLICY,
        "train": train,
        "validation": base["validation"],
        "report": report,
    }
    result["content_sha256"] = digest(result)
    return result


def load_quality(raw, fraction, seed):
    from .grouped_validation import load_grouped, audit
    from .data import _records_from_raw

    if (
        raw.get("format") != FORMAT
        or raw.get("policy") != POLICY
        or raw.get("seed") != 42
    ):
        raise ValueError("Quality mixture protocol differs")
    if digest({k: v for k, v in raw.items() if k != "content_sha256"}) != raw.get(
        "content_sha256"
    ):
        raise ValueError("Quality mixture content changed")
    base = raw["source_bundle"]
    if digest(base) != raw["source_sha256"] or raw["stage"] != base["stage"]:
        raise ValueError("Quality source changed")
    _, validation, info = load_grouped(base, fraction, seed)
    _, expected, report = curate(base["train"], raw["stage"])
    if (
        raw["train"] != expected
        or raw["validation"] != base["validation"]
        or raw["report"] != report
    ):
        raise ValueError("Quality mixture differs from deterministic policy")
    leakage = audit({raw["stage"]: raw["train"]}, raw["validation"])
    info.update(format=FORMAT, quality_report=report, leakage_audit=leakage)
    return _records_from_raw(raw["train"]), validation, info
