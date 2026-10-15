"""Pure provenance and resampling rules for retained training datasets."""
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from model.data import _source_key
from tools.author_synthetic import structural_candidates

PREVIOUSLY_EXCLUDED = {104, 105, 116, 121, 141, 143, 147, 168}
NOTES = {103, 104, 105, 108, 116, 121, 141, 143, 147, 168, 200}
CORRECTOR_REVIEW_EXCLUSIONS = {
    11: "Inline optional wording in the original target; previous control also quarantined this row."
}
EXTERNAL_REVIEW_EXCLUSIONS = {
    "train-s367": "Fragment-like bridge-list main clause; clean status uncertain.",
    "train-s351": "Ambiguous sentence-front punctuation; clean status uncertain.",
    "train-s206": "Questionable complement construction; clean status uncertain.",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def key(text):
    return _source_key(text)


def ellipsis(text):
    return "..." in text or "…" in text


def balanced_delimiters(text):
    if text.count('"') % 2:
        return False
    stack = []
    pairs = {")": "(", "]": "[", "}": "{", "”": "“", "»": "«"}
    for char in text:
        if char in "([{“«":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack


def eligible_tree(row):
    text = row["text"]
    tokens = row["tokens"]
    roots = [t for t in tokens if t["head"] == 0]
    return (
        row["id"] not in EXTERNAL_REVIEW_EXCLUSIONS
        and 8 <= len(tokens) <= 40
        and 35 <= len(text) <= 280
        and text.endswith(".")
        and not ellipsis(text)
        and balanced_delimiters(text)
        and len(roots) == 1
        and roots[0]["upos"] in ("VERB", "ADJ")
        and any(
            t["head"] == roots[0]["id"] and t["relation"] == "nsubj" for t in tokens
        )
        and not re.match(r"^(?:Nếu|Khi|Tuy|Mặc dù|Bởi vì|Vì)\b", text)
    )


def candidate(target, source, label, rule, start, clean, error, parent):
    return {
        "target": target,
        "source": source,
        "label": label,
        "rule": rule,
        "edit": {"start": start, "clean": clean, "error": error},
        "parent": parent,
    }


def edit_at(target, label, rule, start, end, replacement, parent):
    source = target[:start] + replacement + target[end:]
    return candidate(
        target, source, label, rule, start, target[start:end], replacement, parent
    )


def restore(c):
    e = c["edit"]
    i = e["start"]
    if c["source"][i : i + len(e["error"])] != e["error"]:
        raise ValueError("Error span differs")
    return c["source"][:i] + e["clean"] + c["source"][i + len(e["error"]) :]


def validate_edit(c):
    if (
        not 1 <= c["label"] <= 4
        or c["source"] == c["target"]
        or restore(c) != c["target"]
    ):
        raise ValueError("Non-reversible or identity edit")
    e = c["edit"]
    i = e["start"]
    if c["target"][i : i + len(e["clean"])] != e["clean"]:
        raise ValueError("Clean span differs")
    if c["source"] != c["target"][:i] + e["error"] + c["target"][i + len(e["clean"]) :]:
        raise ValueError("Source differs from edit")
    if c["label"] == 2 and Counter(re.findall(r"\w+", key(c["source"]))) != Counter(
        re.findall(r"\w+", key(c["target"]))
    ):
        raise ValueError("Phrase reordering changed words")
    if c["label"] == 4 and re.findall(r"\w+", c["source"]) != re.findall(
        r"\w+", c["target"]
    ):
        raise ValueError("Punctuation changed words")
    if any(
        not isinstance(c[k], str) or not c[k].strip()
        for k in ("source", "target", "rule")
    ):
        raise ValueError("Missing example evidence")


def punctuation_candidates(text, parent):
    # Real bounded-delimiter violations; never add more terminal-period/comma artifacts.
    for opening, closing, name in [
        ("“", "”", "quotation"),
        ("(", ")", "parenthesis"),
        ("[", "]", "bracket"),
    ]:
        if opening not in text and closing not in text:
            yield edit_at(
                text, 4, "unmatched_" + name + "_opening", 0, 0, opening, parent
            )
            yield edit_at(
                text,
                4,
                "unmatched_" + name + "_closing",
                len(text),
                len(text),
                closing,
                parent,
            )
        elif text.count(opening) == text.count(closing) == 1:
            index = text.index(closing)
            yield edit_at(
                text, 4, "missing_" + name + "_closing", index, index + 1, "", parent
            )


def corpus_candidates(text, parent):
    if ellipsis(text) or not text.endswith(".") or not balanced_delimiters(text):
        return
    for label, rule, source, evidence in structural_candidates(text):
        if label == 4:
            continue
        yield candidate(
            text,
            source,
            label,
            rule,
            evidence["start"],
            evidence["clean_span"],
            evidence["error_span"],
            parent,
        )
    yield from punctuation_candidates(text, parent)


def tree_candidates(row):
    text = row["text"]
    tokens = row["tokens"]
    by_id = {t["id"]: t for t in tokens}
    parent = "ud:" + row["id"]
    results = []
    for t in tokens:
        head = by_id.get(t["head"])
        form = t["form"].casefold()
        # Overt plural markers after an adjacent complete annotated nominal head.
        if (
            form in ("các", "những")
            and t["start"] > 0
            and head
            and head["upos"] == "NOUN"
            and t["relation"] in ("det", "nummod", "nummod:det")
            and t["end"] < head["start"]
            and not text[t["end"] : head["start"]].strip()
        ):
            results.append(
                edit_at(
                    text,
                    2,
                    "quantifier_after_annotated_noun",
                    t["start"],
                    head["end"],
                    head["form"] + " " + t["form"],
                    parent,
                )
            )
        # Only classifier forms that cannot become an ordinary postnominal modifier.
        if (
            form in ("chiếc", "ngôi", "căn", "cuốn", "quyển", "bức", "tấm", "tòa")
            and t["start"] > 0
            and head
            and head["upos"] == "NOUN"
            and t["relation"].startswith("clf")
            and not text[t["end"] : head["start"]].strip()
            and t["end"] < head["start"]
        ):
            results.append(
                edit_at(
                    text,
                    2,
                    "classifier_after_annotated_noun",
                    t["start"],
                    head["end"],
                    head["form"] + " " + t["form"],
                    parent,
                )
            )
        rule = None
        if form == "là" and t["relation"] == "cop":
            rule = "duplicated_copula"
        elif form in ("đã", "đang", "sẽ") and t["upos"] in ("AUX", "ADV"):
            rule = "duplicated_aspect_marker"
        elif form in ("của", "với", "bằng", "trong", "tại") and t["upos"] == "ADP":
            rule = "duplicated_annotated_preposition"
        elif form in ("và", "nhưng", "hoặc") and t["upos"] in ("CCONJ", "SCONJ"):
            rule = "duplicated_annotated_conjunction"
        if rule:
            results.append(
                edit_at(
                    text,
                    3,
                    rule,
                    t["start"],
                    t["end"],
                    t["form"] + " " + t["form"],
                    parent,
                )
            )
    roots = [t for t in tokens if t["head"] == 0]
    if len(roots) == 1:
        root = roots[0]
        subjects = [
            t for t in tokens if t["head"] == root["id"] and t["relation"] == "nsubj"
        ]
        objects = [
            t
            for t in tokens
            if t["head"] == root["id"]
            and t["relation"] == "obj"
            and t["upos"] == "NOUN"
        ]
        extra = [
            t
            for t in tokens
            if t["head"] == root["id"]
            and t["relation"]
            in ("aux", "aux:pass", "xcomp", "ccomp", "conj", "parataxis")
        ]
        if (
            root["upos"] == "VERB"
            and len(subjects) == 1
            and objects
            and not extra
            and subjects[0]["upos"] in ("NOUN", "PROPN")
        ):
            # Delete the whole predicate, not just the verb (zero-copula nominal
            # predication is possible in Vietnamese). Retain an initial nominal
            # subject subtree only, without embedded clauses or sentence fronts.
            subtree = {subjects[0]["id"]}
            for _ in tokens:
                subtree.update(t["id"] for t in tokens if t["head"] in subtree)
            nominal = [t for t in tokens if t["id"] in subtree]
            end = max(t["end"] for t in nominal)
            initial = [t for t in tokens if t["start"] < end]
            if (
                min(t["start"] for t in nominal) == 0
                and {t["id"] for t in initial} == subtree
                and not any(
                    t["upos"] in ("VERB", "AUX")
                    or t["relation"].startswith(("acl", "cop", "ccomp", "xcomp"))
                    for t in nominal
                )
            ):
                results.append(
                    edit_at(
                        text,
                        1,
                        "missing_annotated_main_predicate",
                        end,
                        len(text) - 1,
                        "",
                        parent,
                    )
                )
    results.extend(punctuation_candidates(text, parent))
    for c in results:
        validate_edit(c)
    return results


def make_row(c):
    validate_edit(c)
    identity = digest(c)
    family = digest(c["target"])[:24]
    return {
        "ID": "balanced:" + identity[:24],
        "Input": c["source"],
        "Output": {"Error_Type_Id": c["label"], "Fixed Sentence": c["target"]},
        "Provenance": {
            "origin": "retained_deterministic_minimal_edit",
            "parent_ids": [c["parent"]],
            "family_id": "clean:" + family,
            "template_rule": c["rule"],
            "edit": c["edit"],
            "review_status": "deterministic_applicability_checked_not_human_adjudicated",
            "uses_external_teacher": False,
            "external_treebank": str(c["parent"]).startswith("ud:"),
            "evidence_sha256": identity,
            "private_dataset": True,
        },
    }


def balanced_view(rows, target=None, max_repeat=4, no_repeat_ids=()):
    result = copy.deepcopy(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r["Output"]["Error_Type_Id"]].append(r)
    target = target or len(groups[0])
    repeats = Counter()
    for label in range(5):
        group = sorted(
            (
                r
                for r in groups[label]
                if r["ID"] not in NOTES and r["ID"] not in no_repeat_ids
            ),
            key=lambda r: digest([r["ID"], r["Input"]]),
        )
        count = len(groups[label])
        # Fill from organizer supervision first, then legacy pairs, then new
        # operators; never amplify explanatory-note annotations.
        tiers = [
            [r for r in group if type(r["ID"]) is int],
            [
                r
                for r in group
                if type(r["ID"]) is not int and not str(r["ID"]).startswith("balanced:")
            ],
            [r for r in group if str(r["ID"]).startswith("balanced:")],
        ]
        for tier in tiers:
            for cycle in range(1, max_repeat):
                for r in tier:
                    if count >= target:
                        break
                    repeated = copy.deepcopy(r)
                    repeated["ID"] = (
                        "repeat:"
                        + digest([type(r["ID"]).__name__, r["ID"], cycle])[:24]
                    )
                    repeated.setdefault("Provenance", {})["resampling"] = {
                        "source_id": r["ID"],
                        "draw": cycle + 1,
                        "is_new_unique_example": False,
                    }
                    result.append(repeated)
                    count += 1
                    repeats[label] += 1
    counts = Counter(r["Output"]["Error_Type_Id"] for r in result)
    info = {
        "target": target,
        "max_exposures_per_row": max_repeat,
        "repeat_priority": "organizer_then_legacy_then_new_code; notes_never_repeated",
        "canonical_counts": {str(i): len(groups[i]) for i in range(5)},
        "distinct_source_counts": {
            str(i): len({key(r["Input"]) for r in groups[i]}) for i in range(5)
        },
        "exposure_counts": {str(i): counts[i] for i in range(5)},
        "repeated_counts": {str(i): repeats[i] for i in range(5)},
        "shortfalls": {str(i): max(0, target - counts[i]) for i in range(5)},
    }
    if no_repeat_ids:
        info["non_repeated_original_ids"] = sorted(no_repeat_ids)
    return result, info


def corrector_originals(original, baseline, retain_abbreviated=False):
    """Versioned retention of organizer references; no invented completions."""
    existing = {r["ID"]: r for r in baseline if type(r["ID"]) is int}
    correction = []
    pending = []
    for r in original:
        if r["ID"] in NOTES:
            continue
        if r["ID"] in CORRECTOR_REVIEW_EXCLUSIONS:
            pending.append(
                {
                    "id": r["ID"],
                    "reason": "inline_optional_annotation_not_a_final_sentence",
                    "original": r,
                }
            )
            continue
        row = copy.deepcopy(r)
        label = r["Output"]["Error_Type_Id"]
        if label == 0:
            row["Output"]["Fixed Sentence"] = r["Input"]
        else:
            if retain_abbreviated and ellipsis(row["Output"]["Fixed Sentence"]):
                correction.append(row)
                continue
            previous = existing.get(r["ID"])
            if (
                previous
                and previous["Input"] == r["Input"]
                and previous["Output"]["Error_Type_Id"] == label
                and not ellipsis(previous["Output"]["Fixed Sentence"])
                and not previous["Output"]["Fixed Sentence"].lstrip().startswith("(")
            ):
                row = copy.deepcopy(previous)
            if ellipsis(row["Output"]["Fixed Sentence"]):
                pending.append(
                    {
                        "id": r["ID"],
                        "reason": "abbreviated_target_not_newly_adjudicated",
                        "original": r,
                    }
                )
                continue
        correction.append(row)
    return correction, pending


def classifier_base(original, baseline):
    """Keep the previous prefix and restore all previously omitted originals."""
    present = {r["ID"] for r in baseline if type(r["ID"]) is int}
    return copy.deepcopy(baseline) + [
        copy.deepcopy(r) for r in original if r["ID"] not in present
    ]

