"""Evidence and leakage gates for independent grammar examples; never repairs organizer gold."""

from collections import Counter
from difflib import SequenceMatcher
import re

from .grouped_validation import tokens, near, POLICY
from .retained_rules import key, ellipsis, balanced_delimiters

FUNCTION_WORDS = frozenset(
    "là đã đang sẽ từng vừa mới được bị của với bằng trong tại ở vào ra từ đến cho về và nhưng hoặc mà thì nếu tuy dù nên vì bởi để do cũng vẫn chỉ những các một mỗi mọi rất quá lắm hơn nhất".split()
)


def qualification(events):
    expected = [e["task"]["expected_label"] for e in events]
    actual = [
        (
            e.get("parsed", {}).get("label", -1)
            if isinstance(e.get("parsed"), dict)
            else -1
        )
        for e in events
    ]
    counts = Counter(expected)
    successes = Counter(a for a, b in zip(expected, actual) if a == b)
    correct = sum(a == b for a, b in zip(expected, actual))
    passed = (
        len(events) == 10
        and counts == Counter({k: 2 for k in range(5)})
        and correct >= 8
        and all(successes[k] >= 1 for k in range(5))
    )
    return {
        "passed": passed,
        "calls": len(events),
        "correct": correct,
        "correct_by_class": dict(successes),
        "expected": expected,
        "predicted": actual,
        "scope": "Fixed training-only10-case operational qualification, not independent model-quality evidence",
    }


def blocked_text(text, validation):
    a = tokens(text)
    bag = Counter(a)
    spans = {a[i : i + 10] for i in range(len(a) - 9)}
    for row in validation:
        for text in [row["Input"], row["Output"].get("Fixed Sentence")]:
            if not isinstance(text, str):
                continue
            b = tokens(text)
            if (
                a == b
                or bag == Counter(b)
                or spans & {b[i : i + 10] for i in range(len(b) - 9)}
                or near(a, b)
            ):
                return True
    return False


def pair_gate(source, target, label, review):
    if (
        not isinstance(source, str)
        or not isinstance(target, str)
        or label not in (1, 2, 3, 4)
    ):
        return False, "invalid_fields"
    if not review or review.get("label") != label or review.get("certain") is not True:
        return False, "blind_subtype_or_uncertainty"
    if not isinstance(review.get("corrected"), str) or key(review["corrected"]) != key(
        target
    ):
        return False, "blind_correction_differs"
    if (
        key(source) == key(target)
        or ellipsis(source)
        or ellipsis(target)
        or not balanced_delimiters(target)
    ):
        return False, "identity_or_incomplete"
    s, t = tokens(source), tokens(target)
    if min(len(s), len(t)) < 3 or max(len(source), len(target)) > 350:
        return False, "length"
    if Counter(re.findall(r"\d+(?:[.,]\d+)*", source)) != Counter(
        re.findall(r"\d+(?:[.,]\d+)*", target)
    ):
        return False, "changed_numbers"
    if label == 2 and Counter(s) != Counter(t):
        return False, "phrase_changed_vocabulary"
    if label == 4 and s != t:
        return False, "punctuation_changed_words"
    if label != 4 and "".join(re.findall(r"[^\w\s]", source)) != "".join(
        re.findall(r"[^\w\s]", target)
    ):
        return False, "multiple_punctuation_change"
    if label in (1, 3):
        added = set((Counter(s) - Counter(t)))
        if (label == 1 and added) or (label == 3 and not added <= FUNCTION_WORDS):
            return False, "unsupported_lexical_change"
        if label == 3 and not set((Counter(t) - Counter(s))) <= FUNCTION_WORDS:
            return False, "removed_lexical_content"
    if SequenceMatcher(None, s, t, autojunk=False).ratio() < 0.6:
        return False, "large_edit"
    return True, "accepted"


FORMAT = "vgec-diverse-mixture-v1"
SOURCE_SHA256 = "1df180722db99b724f7368aca50597133bee28606de3394cd8d12a57da89978d"
DIVERSITY_POLICY = {
    "new_clean_parents": 64,
    "reserved_punctuation_parents": 12,
    "new_errors_per_parent": 4,
    "max_exposures": 2,
    "new_caps": {"1": 32, "2": 96, "3": 64, "4": 32},
    "same_function_form_cap": 8,
    "same_rule_cap": 32,
    "seed": 42,
    "organizer_prefix_evidence": True,
}


ADDITIVE_POLICY = {
    **DIVERSITY_POLICY,
    "exposure_strategy": "preserve_baseline_append_new_once",
    "max_exposures": 4,
}


def read_trees(path_or_text, *, is_text=False):
    """Align UD words with its exact sentence text; only normalize punctuation spacing."""
    from pathlib import Path

    source = path_or_text if is_text else Path(path_or_text).read_text()
    result = []
    for block in source.strip().split("\n\n"):
        sid = None
        text = None
        data = []
        for line in block.splitlines():
            if line.startswith("# sent_id ="):
                sid = line.split("=", 1)[1].strip()
            elif line.startswith("# text ="):
                text = line.split("=", 1)[1].strip()
            elif line and not line.startswith("#"):
                fields = line.split("\t")
                if fields[0].isdigit():
                    data.append(fields)
        if not sid or not text:
            continue
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        cursor = 0
        ts = []
        for fields in data:
            start = text.find(fields[1], cursor)
            if start < 0:
                break
            end = start + len(fields[1])
            cursor = end
            ts.append(
                {
                    "id": int(fields[0]),
                    "form": fields[1],
                    "upos": fields[3],
                    "head": int(fields[6]),
                    "relation": fields[7],
                    "start": start,
                    "end": end,
                }
            )
        if len(ts) == len(data):
            result.append({"id": sid, "text": text, "tokens": ts})
    return result


def observed_prefixes(rows):
    """Use exact human-attested prefix deletions only when their training labels agree."""
    from .retained_rules import NOTES

    labels = {}
    evidence = {}
    spelling = {}
    for row in rows:
        if type(row["ID"]) is not int or row["ID"] in NOTES:
            continue
        target = row["Output"]["Fixed Sentence"]
        if not isinstance(target, str) or ellipsis(target):
            continue
        source = row["Input"]
        a = tokens(source)
        b = tokens(target)
        for n in (1, 2):
            if len(a) > n and a[n:] == b:
                matches = list(re.finditer(r"\w+", source))
                prefix = source[: matches[n].start()].strip()
                normalized = prefix.casefold()
                labels.setdefault(normalized, set()).add(row["Output"]["Error_Type_Id"])
                evidence.setdefault(normalized, []).append(row["ID"])
                spelling[normalized] = prefix
    return {spelling[k]: evidence[k] for k in sorted(labels) if labels[k] == {3}}


def organizer_prefix_candidates(tree, catalog):
    from .retained_rules import edit_at

    text = tree["text"]
    roots = [t for t in tree["tokens"] if t["head"] == 0]
    if len(roots) != 1 or roots[0]["upos"] != "VERB" or any(c in text for c in ",;:"):
        return
    subjects = [
        t
        for t in tree["tokens"]
        if t["head"] == roots[0]["id"]
        and t["relation"] == "nsubj"
        and t["upos"] in ("NOUN", "PROPN")
        and t["start"] == 0
    ]
    if len(subjects) != 1:
        return
    if any(
        t["head"] == roots[0]["id"] and t["relation"] in ("conj", "parataxis")
        for t in tree["tokens"]
    ):
        return
    first = text[0] if subjects[0]["upos"] == "PROPN" else text[0].lower()
    for prefix, ids in catalog.items():
        c = edit_at(
            text,
            3,
            "organizer_redundant_prefix_" + prefix.casefold().replace(" ", "_"),
            0,
            1,
            prefix + " " + first,
            "ud:" + tree["id"],
        )
        c["organizer_evidence_ids"] = ids
        yield c


def extra_candidates(tree):
    from .retained_rules import edit_at, validate_edit

    text = tree["text"]
    parent = "ud:" + tree["id"]
    by = {t["id"]: t for t in tree["tokens"]}
    for t in tree["tokens"]:
        h = by.get(t["head"])
        if (
            t["form"].casefold() == "rất"
            and t["relation"] == "advmod"
            and h
            and h["upos"] == "ADJ"
            and t["end"] < h["start"]
            and not text[t["end"] : h["start"]].strip()
        ):
            c = edit_at(
                text,
                2,
                "degree_modifier_after_adjective",
                t["start"],
                h["end"],
                h["form"] + " " + t["form"],
                parent,
            )
            validate_edit(c)
            yield c
    yield from real_delimiter_candidates(text, parent)


def real_delimiter_candidates(text, parent):
    from .retained_rules import edit_at

    # Remove a real closing delimiter: no arbitrary punctuation added at sentence edges.
    for left, right, name in [
        ("(", ")", "parenthesis"),
        ("[", "]", "bracket"),
        ("“", "”", "quotation"),
        ('"', '"', "ascii_quote"),
    ]:
        if (left == right and text.count(left) == 2) or (
            left != right
            and text.count(left) == text.count(right) == 1
            and text.index(left) < text.index(right)
        ):
            i = text.rindex(right)
            yield edit_at(
                text, 4, "real_" + name + "_missing_closer", i, i + 1, "", parent
            )


def surface(text):
    return key(re.sub(r"\s+([,.!?;:])", r"\1", text))


def derive_diverse(base, conllu, *, policy=DIVERSITY_POLICY):
    import copy
    from .retained_rules import (
        digest,
        NOTES,
        CORRECTOR_REVIEW_EXCLUSIONS,
        eligible_tree,
        tree_candidates,
        corpus_candidates,
        make_row,
    )

    stage = base["stage"]
    validation = base["validation"]
    canonical = [
        copy.deepcopy(r)
        for r in base["train"]
        if not r.get("Provenance", {}).get("resampling")
    ]
    originals = [r for r in canonical if type(r["ID"]) is int]
    known = {surface(r["Input"]) for r in originals}
    selected = copy.deepcopy(originals)
    dropped = []
    for r in canonical:
        if type(r["ID"]) is int:
            continue
        if surface(r["Input"]) in known:
            dropped.append(r["ID"])
            continue
        known.add(surface(r["Input"]))
        selected.append(r)
    byid = {r["ID"]: r for r in selected}
    parents = {}
    pool = []
    prefix_catalog = observed_prefixes(originals)
    for r in originals:
        if r["ID"] in NOTES or r["ID"] in CORRECTOR_REVIEW_EXCLUSIONS:
            continue
        text = (
            r["Input"]
            if r["Output"]["Error_Type_Id"] == 0
            else r["Output"]["Fixed Sentence"]
        )
        if (
            not isinstance(text, str)
            or ellipsis(text)
            or not text.endswith(".")
            or not balanced_delimiters(text)
        ):
            continue
        pid = (
            r["ID"]
            if r["Output"]["Error_Type_Id"] == 0
            else "reference:" + str(r["ID"])
        )
        parents[pid] = {
            "ID": pid,
            "Input": text,
            "Output": {"Error_Type_Id": 0, "Fixed Sentence": text},
            "Provenance": {
                "origin": "exact_human_training_reference_control",
                "parent_ids": [] if type(pid) is int else [r["ID"]],
            },
        }
        pool.extend(corpus_candidates(text, pid))
        pool.extend(real_delimiter_candidates(text, pid))
    trees = {t["id"]: t for t in read_trees(conllu, is_text=True) if eligible_tree(t)}
    old_trees = base["source_bundle"]["source_packet"][stage]["metadata"].get(
        "external_parent_trees", []
    )
    for t in old_trees:
        trees[t["id"]] = t
    for t in trees.values():
        pid = "ud:" + t["id"]
        if blocked_text(t["text"], validation):
            continue
        parents[pid] = byid.get(pid) or {
            "ID": pid,
            "Input": t["text"],
            "Output": {"Error_Type_Id": 0, "Fixed Sentence": t["text"]},
            "Provenance": {
                "origin": "human_annotated_ud_vietnamese_vtb_train",
                "parent_ids": [],
                "tree_id": t["id"],
                "source_file_sha256": SOURCE_SHA256,
                "license": "CC-BY-SA-4.0",
            },
        }
        pool.extend(tree_candidates(t))
        pool.extend(extra_candidates(t))
        pool.extend(organizer_prefix_candidates(t, prefix_catalog))
    # Corpus/tree historical operators that add dangling punctuation are not expanded.
    pool = [
        c
        for c in pool
        if c["label"] != 4
        or c["rule"].startswith("real_")
        or c["rule"] == "missing_parenthesis_closing"
    ]
    if stage == "correction":
        pool = [c for c in pool if c["label"] != 1]
    frequency = Counter(c["rule"] for c in pool)
    pool.sort(
        key=lambda c: (
            [2, 4, 1, 3].index(c["label"]),
            0 if c["parent"] in byid else 1,
            frequency[c["rule"]],
            digest(c),
        )
    )
    counts = Counter()
    parent_counts = Counter()
    parent_class = Counter()
    rule_counts = Counter()
    function_forms = Counter()
    new_parents = {}
    new_rows = []
    for c in pool:
        label = c["label"]
        pid = c["parent"]
        p = parents.get(pid)
        if (
            p is None
            or surface(c["source"]) in known
            or blocked_text(c["source"], validation)
            or blocked_text(c["target"], validation)
        ):
            continue
        if (
            counts[label] >= policy["new_caps"][str(label)]
            or parent_counts[pid] >= policy["new_errors_per_parent"]
            or parent_class[(pid, label)] >= (2 if label == 2 else 1)
            or rule_counts[c["rule"]] >= policy["same_rule_cap"]
        ):
            continue
        function_form = (
            c["rule"]
            if c["rule"].startswith("organizer_redundant_prefix_")
            else c["edit"]["clean"].casefold()
        )
        if (
            label == 3
            and function_forms[function_form] >= policy["same_function_form_cap"]
        ):
            continue
        if pid not in byid and pid not in new_parents:
            if (
                len(new_parents) >= policy["new_clean_parents"]
                or surface(p["Input"]) in known
            ):
                continue
            if (
                label == 2
                and len(new_parents)
                >= policy["new_clean_parents"] - policy["reserved_punctuation_parents"]
            ):
                continue
            new_parents[pid] = p
            known.add(surface(p["Input"]))
        row = make_row(c)
        row["ID"] = "diverse:" + digest(c)[:24]
        row["Provenance"]["origin"] = "dependency_supported_diversity_expansion"
        row["Provenance"]["mechanical_evidence"] = c
        row["Provenance"]["source_is_human_written"] = True
        new_rows.append(row)
        known.add(surface(c["source"]))
        counts[label] += 1
        parent_counts[pid] += 1
        parent_class[(pid, label)] += 1
        rule_counts[c["rule"]] += 1
        if label == 3:
            function_forms[function_form] += 1
    selected += list(new_parents.values()) + new_rows
    if policy.get("exposure_strategy") == "preserve_baseline_append_new_once":
        train = copy.deepcopy(base["train"]) + copy.deepcopy(
            list(new_parents.values()) + new_rows
        )
        exposures = Counter(r["Output"]["Error_Type_Id"] for r in train)
    else:
        train = copy.deepcopy(selected)
        target = sum(r["Output"]["Error_Type_Id"] == 0 for r in selected)
        exposures = Counter(r["Output"]["Error_Type_Id"] for r in train)
        for r in sorted(
            selected, key=lambda r: (type(r["ID"]) is not int, digest(r["ID"]))
        ):
            k = r["Output"]["Error_Type_Id"]
            if (
                exposures[k] >= target
                or k == 0
                or r["ID"] in NOTES
                or (
                    stage == "correction"
                    and ellipsis(r["Output"]["Fixed Sentence"] or "")
                )
            ):
                continue
            repeated = copy.deepcopy(r)
            repeated["ID"] = "diverse-repeat:" + digest(r["ID"])[:24]
            repeated.setdefault("Provenance", {})["resampling"] = {
                "source_id": r["ID"],
                "draw": 2,
                "is_new_unique_example": False,
            }
            train.append(repeated)
            exposures[k] += 1
    report = {
        "canonical_counts": {
            str(k): sum(r["Output"]["Error_Type_Id"] == k for r in selected)
            for k in range(5)
        },
        "unique_source_counts": {
            str(k): len(
                {
                    surface(r["Input"])
                    for r in selected
                    if r["Output"]["Error_Type_Id"] == k
                }
            )
            for k in range(5)
        },
        "exposure_counts": {str(k): exposures[k] for k in range(5)},
        "new_errors": {str(k): counts[k] for k in range(1, 5)},
        "new_clean_controls": len(new_parents),
        "new_rules": dict(rule_counts),
        "function_forms": dict(function_forms),
        "organizer_prefix_catalog": prefix_catalog,
        "duplicate_synthetic_rows_removed": len(dropped),
        "dropped_synthetic_ids": dropped,
        "new_code_rows": len(new_rows),
        "new_external_parents": sum(str(p).startswith("ud:") for p in new_parents),
        "teacher_authored_rows": 0,
        "maximum_exposures_per_example": policy["max_exposures"],
        "scope": "Deterministic dependency/span checks, not human adjudication or measured model improvement; new lossy sentence-fragment targets excluded from corrector",
    }
    if policy.get("exposure_strategy"):
        report["exposure_strategy"] = policy["exposure_strategy"]
        report["new_examples_max_exposures"] = 1
    return train, report


def seal_diverse(base, conllu, *, policy=DIVERSITY_POLICY):
    from .retained_rules import digest

    train, report = derive_diverse(base, conllu, policy=policy)
    raw = {
        "format": FORMAT,
        "stage": base["stage"],
        "source_quality": base,
        "source_sha256": digest(base),
        "external_conllu": conllu,
        "external_sha256": SOURCE_SHA256 if conllu else None,
        "policy": policy,
        "train": train,
        "validation": base["validation"],
        "report": report,
    }
    raw["content_sha256"] = digest(raw)
    return raw


def load_diverse(raw, fraction, seed):
    import hashlib
    from .retained_rules import digest
    from .quality_data import load_quality
    from .grouped_validation import audit
    from .data import _records_from_raw

    if (
        raw.get("format") != FORMAT
        or raw.get("policy") not in (DIVERSITY_POLICY, ADDITIVE_POLICY)
        or digest({k: v for k, v in raw.items() if k != "content_sha256"})
        != raw.get("content_sha256")
    ):
        raise ValueError("Diverse mixture protocol/content differs")
    base = raw["source_quality"]
    if raw["stage"] != base["stage"] or digest(base) != raw["source_sha256"]:
        raise ValueError("Diverse source differs")
    _, validation, info = load_quality(base, fraction, seed)
    text = raw["external_conllu"]
    if text and (
        raw["external_sha256"] != SOURCE_SHA256
        or hashlib.sha256(text.encode()).hexdigest() != SOURCE_SHA256
    ):
        raise ValueError("External human corpus differs")
    expected, report = derive_diverse(base, text, policy=raw["policy"])
    if (
        raw["train"] != expected
        or raw["validation"] != base["validation"]
        or raw["report"] != report
    ):
        raise ValueError("Diverse mixture differs from deterministic evidence")
    leakage = audit({raw["stage"]: raw["train"]}, raw["validation"])
    info.update(format=FORMAT, diversity_report=report, leakage_audit=leakage)
    return _records_from_raw(raw["train"]), validation, info
