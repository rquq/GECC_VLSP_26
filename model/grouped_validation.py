"""Shared organizer holdout with cross-stage lineage and textual exclusion."""

import copy
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
import random
import re
import unicodedata

from .data import _id_key, _records_from_raw, _source_key
from .retained_rules import (
    digest,
    balanced_view,
    ellipsis,
    NOTES,
    CORRECTOR_REVIEW_EXCLUSIONS,
)

FORMAT = "vgec-grouped-validation-v1"
POLICY = {
    "normalization": "NFKC/casefold/word-tokens",
    "equal_token_bags": True,
    "shared_span_tokens": 10,
    "near_min_tokens": 8,
    "near_length_ratio": 0.8,
    "near_token_jaccard": 0.8,
    "near_sequence_ratio": 0.9,
}


def tokens(text):
    return tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def text_values(row):
    values = [row["Input"], row["Output"].get("Fixed Sentence")]
    return [tokens(t) for t in values if isinstance(t, str) and tokens(t)]


def typed_sort(key):
    return json.dumps([key[0].__name__, key[1]], ensure_ascii=False)


def parents(row):
    p = row.get("Provenance", {})
    result = list(p.get("parent_ids", []))
    if "resampling" in p:
        result.append(p["resampling"]["source_id"])
    return result


def near(a, b):
    if min(len(a), len(b)) < POLICY["near_min_tokens"]:
        return False
    if min(len(a), len(b)) / max(len(a), len(b)) < POLICY["near_length_ratio"]:
        return False
    sa, sb = set(a), set(b)
    return (
        len(sa & sb) / len(sa | sb) >= POLICY["near_token_jaccard"]
        and SequenceMatcher(None, a, b, autojunk=False).ratio()
        >= POLICY["near_sequence_ratio"]
    )


def components(rows):
    nodes = {_id_key(r["ID"]) for r in rows}
    parent = {k: k for k in nodes}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def join(a, b):
        if b not in parent:
            raise ValueError("Unresolved augmentation parent")
        parent[find(b)] = find(a)

    owners = {}
    text_owners = {}
    for r in rows:
        k = _id_key(r["ID"])
        for pid in parents(r):
            join(k, _id_key(pid))
        p = r.get("Provenance", {})
        signatures = [("family", p["family_id"])] if p.get("family_id") else []
        if p.get("template_id"):
            signatures.append(("template", p["template_id"]))
        for value in text_values(r):
            signatures += [("text", value), ("bag", tuple(sorted(value)))]
            signatures += [("span", value[i : i + 10]) for i in range(len(value) - 9)]
            text_owners.setdefault(value, k)
        for sig in signatures:
            if sig in owners:
                join(k, owners[sig])
            else:
                owners[sig] = k
    texts = sorted(text_owners, key=lambda v: (len(v), v))
    for i, a in enumerate(texts):
        for b in texts[i + 1 :]:
            if len(a) / len(b) < POLICY["near_length_ratio"]:
                break
            ka, kb = text_owners[a], text_owners[b]
            if find(ka) != find(kb) and near(a, b):
                join(ka, kb)
    groups = defaultdict(set)
    for k in nodes:
        groups[find(k)].add(k)
    return sorted(groups.values(), key=lambda g: min(map(typed_sort, g)))


def audit(training, validation):
    if not validation or any(
        type(r["ID"]) is not int or r.get("Provenance", {}).get("resampling")
        for r in validation
    ):
        raise ValueError("Validation must be nonempty original rows without resampling")
    train = [r for rows in training.values() for r in rows]
    tk = {_id_key(r["ID"]) for r in train}
    vk = {_id_key(r["ID"]) for r in validation}
    tt = {v for r in train for v in text_values(r)}
    vt = {v for r in validation for v in text_values(r)}
    bags = lambda values: {tuple(sorted(v)) for v in values}
    spans = lambda values: {v[i : i + 10] for v in values for i in range(len(v) - 9)}
    details = {
        "id_overlap": len(tk & vk),
        "normalized_source_or_target_overlap": len(tt & vt),
        "token_bag_overlap": len(bags(tt) & bags(vt)),
        "shared_10_token_spans": len(spans(tt) & spans(vt)),
        "direct_validation_parent_hits": sum(
            _id_key(p) in vk for r in train for p in parents(r)
        ),
        "mixed_lineage_or_text_components": sum(
            bool(g & tk) and bool(g & vk) for g in components(train + validation)
        ),
    }
    if any(details.values()):
        raise ValueError("Train/validation leakage: " + json.dumps(details))
    return {
        **details,
        "passed": True,
        "validation_rows": len(validation),
        "policy": POLICY,
        "scope": "Exported fit/evaluation records only; past model or human exposure is not erased.",
    }


def source_groups(packet):
    if set(packet) != {"classification", "correction"}:
        raise ValueError("Source packet must contain exactly both training stages")
    a, b = packet["classification"], packet["correction"]
    if a.get("stage") != "classification" or b.get("stage") != "correction":
        raise ValueError("Source packet stages differ")
    if a["original_snapshot"] != b["original_snapshot"]:
        raise ValueError("Original source packets differ")
    rows = a["original_snapshot"] + a["unique"] + b["unique"]
    return components(rows)


def eligible_groups(packet, groups):
    originals = {
        _id_key(r["ID"]): r for r in packet["classification"]["original_snapshot"]
    }
    correction = {
        _id_key(r["ID"]) for r in packet["correction"]["unique"] if type(r["ID"]) is int
    }
    result = []
    for group in groups:
        original_keys = group & originals.keys()
        if not original_keys or not original_keys <= correction:
            continue
        if any(k[1] in NOTES | set(CORRECTOR_REVIEW_EXCLUSIONS) for k in original_keys):
            continue
        bysource = defaultdict(list)
        for k in sorted(original_keys, key=typed_sort):
            bysource[_source_key(originals[k]["Input"])].append(k)
        representatives = []
        for keys in bysource.values():
            annotations = {
                (
                    originals[k]["Output"]["Error_Type_Id"],
                    _source_key(
                        originals[k]["Output"]["Fixed Sentence"]
                        or originals[k]["Input"]
                    ),
                )
                for k in keys
            }
            if len(annotations) != 1:
                break
            representatives.append(keys[0])
        else:
            result.append((group, representatives))
    return result


def select_validation(packet, fraction=0.2, seed=42):
    if type(seed) is not int or seed != 42 or fraction != 0.2:
        raise ValueError("Grouped selection requires fraction 0.2 and seed 42")
    groups = source_groups(packet)
    eligible = eligible_groups(packet, groups)
    originals = {
        _id_key(r["ID"]): r for r in packet["classification"]["original_snapshot"]
    }
    totals = Counter(
        r["Output"]["Error_Type_Id"]
        for r in packet["correction"]["unique"]
        if type(r["ID"]) is int
    )
    target = {i: max(2, round(totals[i] * fraction)) for i in range(5)}
    candidates = list(eligible)
    random.Random(seed).shuffle(candidates)
    counts = Counter()
    chosen = []
    remaining = list(candidates)
    cost = lambda c: sum(((c[i] - target[i]) / target[i]) ** 2 for i in range(5))
    while remaining:
        additions = [
            Counter(originals[k]["Output"]["Error_Type_Id"] for k in reps)
            for _, reps in remaining
        ]
        index = min(range(len(remaining)), key=lambda i: cost(counts + additions[i]))
        if cost(counts + additions[index]) >= cost(counts):
            break
        counts += additions[index]
        chosen.append(remaining.pop(index))
    if any(counts[i] < 2 or counts[i] >= totals[i] for i in range(5)):
        raise ValueError(
            "No supported five-class validation partition at requested grouping strength"
        )
    ids = sorted(k[1] for _, reps in chosen for k in reps)
    return ids, {
        "total_components": len(groups),
        "eligible_components": len(eligible),
        "validation_components": len(chosen),
        "target_counts": target,
        "actual_counts": dict(counts),
        "requested_fraction": fraction,
    }


def derive(packet, validation_ids):
    if (
        not validation_ids
        or any(type(i) is not int for i in validation_ids)
        or len(set(validation_ids)) != len(validation_ids)
    ):
        raise ValueError("Validation requires distinct organizer integer IDs")
    groups = source_groups(packet)
    vk = {_id_key(i) for i in validation_ids}
    eligible = eligible_groups(packet, groups)
    held_groups = [g for g in groups if g & vk]
    expected = {k for group, reps in eligible if group & vk for k in reps}
    if expected != vk or len(held_groups) != sum(
        bool(group & vk) for group, _ in eligible
    ):
        raise ValueError(
            "Validation cuts an ineligible family or omits an original counterpart"
        )
    held = set().union(*held_groups) if held_groups else set()
    source = packet["classification"]["original_snapshot"]
    lookup = {r["ID"]: r for r in source}
    validation = [copy.deepcopy(lookup[i]) for i in validation_ids]
    training = {}
    canonical = {}
    withheld = {}
    balance = {}
    for stage in ("classification", "correction"):
        rows = packet[stage]["unique"]
        canonical[stage] = [r for r in rows if _id_key(r["ID"]) not in held]
        withheld[stage] = [
            r
            for r in rows
            if _id_key(r["ID"]) in held and r["ID"] not in validation_ids
        ]
        target = sum(r["Output"]["Error_Type_Id"] == 0 for r in canonical[stage])
        protected = (
            [
                r["ID"]
                for r in canonical[stage]
                if type(r["ID"]) is int
                and r["Output"]["Error_Type_Id"]
                and ellipsis(r["Output"]["Fixed Sentence"])
            ]
            if stage == "correction"
            else []
        )
        training[stage], balance[stage] = balanced_view(
            canonical[stage], target, 4, protected
        )
    report = audit(training, validation)
    return training, validation, canonical, withheld, balance, report


def seal(stage, packet, validation_ids, train, validation, metadata):
    payload = {
        "format": FORMAT,
        "stage": stage,
        "seed": 42,
        "validation_fraction": 0.2,
        "source_packet": packet,
        "source_packet_sha256": digest(packet),
        "validation_original_ids": validation_ids,
        "train": train,
        "validation": validation,
        "metadata": metadata,
    }
    payload["content_sha256"] = digest(payload)
    return payload


def load_grouped(raw, fraction, seed):
    if (
        raw.get("format") != FORMAT
        or raw.get("stage") not in ("classification", "correction")
        or raw.get("seed") != 42
        or seed != 42
        or fraction != 0.2
        or raw.get("validation_fraction") != 0.2
    ):
        raise ValueError("Grouped validation protocol differs")
    if digest({k: v for k, v in raw.items() if k != "content_sha256"}) != raw.get(
        "content_sha256"
    ):
        raise ValueError("Grouped validation content changed")
    packet = raw["source_packet"]
    if digest(packet) != raw["source_packet_sha256"]:
        raise ValueError("Source packet changed")
    from .retained_balanced import validate

    for stage in packet:
        validate(packet[stage])
    expected_ids, _ = select_validation(packet)
    if raw["validation_original_ids"] != expected_ids:
        raise ValueError(
            "Validation IDs differ from the committed deterministic selection"
        )
    training, validation, canonical, withheld, balance, report = derive(
        packet, raw["validation_original_ids"]
    )
    stage = raw["stage"]
    if raw["train"] != training[stage] or raw["validation"] != validation:
        raise ValueError(
            "Derived train/validation records differ from grouped exclusion"
        )
    train_originals = {
        r["ID"] for r in canonical["classification"] if type(r["ID"]) is int
    }
    retained = {r["ID"] for r in canonical[stage] if type(r["ID"]) is int}
    provenance = {
        "stage": stage,
        "original_train_ids": sorted(train_originals),
        "retained_original_train_ids": [
            r["ID"] for r in canonical[stage] if type(r["ID"]) is int
        ],
        "quarantined_ids": sorted(train_originals - retained),
        "withheld_original_ids": sorted(
            r["ID"] for r in withheld[stage] if type(r["ID"]) is int
        ),
        "selected_synthetic_ids": [
            r["ID"]
            for r in canonical[stage]
            if type(r["ID"]) is not int and not str(r["ID"]).startswith("ud:")
        ],
        "external_train_ids": [
            r["ID"] for r in canonical[stage] if str(r["ID"]).startswith("ud:")
        ],
        "resampled_train_ids": [
            r["ID"] for r in training[stage][len(canonical[stage]) :]
        ],
    }
    records = _records_from_raw(validation)
    return (
        _records_from_raw(training[stage]),
        records,
        {
            "format": FORMAT,
            "provenance": provenance,
            "original_sha256": packet["classification"]["metadata"][
                "original_file_sha256"
            ],
            "validation_records": records,
            "leakage_audit": report,
            "balance": balance[stage],
        },
    )
