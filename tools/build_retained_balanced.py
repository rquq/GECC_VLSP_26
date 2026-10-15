"""Private retention-first corpus builder; code-derived edits, explicit resampling."""

from __future__ import annotations
import argparse, copy, hashlib, json, re, zipfile
from collections import Counter, defaultdict
from pathlib import Path
from model.runtime import write_json
from model.data import _source_key
from tools.author_synthetic import structural_candidates

ROOT = Path(__file__).resolve().parents[1]
PREVIOUSLY_EXCLUDED = {104, 105, 116, 121, 141, 143, 147, 168}
NOTES = {103, 104, 105, 108, 116, 121, 141, 143, 147, 168, 200}
CORRECTOR_REVIEW_EXCLUSIONS = {
    11: "Inline optional wording in the original target; previous control also quarantined this row."
}
NAMES = {
    0: "clean",
    1: "sentence_structure",
    2: "phrase_structure",
    3: "function_word",
    4: "punctuation",
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


def parse_treebank(text):
    result = []
    for block in text.strip().split("\n\n"):
        row = {"tokens": []}
        for line in block.splitlines():
            if line.startswith("# sent_id ="):
                row["id"] = line.split("=", 1)[1].strip()
            elif line.startswith("# text ="):
                row["text"] = line.split("=", 1)[1].strip()
            elif line and not line.startswith("#"):
                f = line.split("\t")
                if len(f) == 10 and f[0].isdigit():
                    row["tokens"].append(
                        {
                            "id": int(f[0]),
                            "form": f[1],
                            "upos": f[3],
                            "xpos": f[4],
                            "head": int(f[6]),
                            "relation": f[7],
                        }
                    )
        if not row.get("text") or not row.get("id"):
            continue
        cursor = 0
        try:
            for token in row["tokens"]:
                token["start"] = row["text"].index(token["form"], cursor)
                if row["text"][cursor : token["start"]].strip():
                    raise ValueError("Unmapped source gap")
                token["end"] = token["start"] + len(token["form"])
                cursor = token["end"]
                if (
                    row["text"][cursor:].strip()
                    and cursor < len(row["text"])
                    and not row["text"][cursor].isspace()
                ):
                    # Only whitespace between the supplied UD token forms is supported.
                    raise ValueError("Non-space token boundary")
            if row["text"][cursor:].strip():
                raise ValueError("Trailing unmapped source")
        except ValueError:
            continue
        result.append(row)
    return result


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


def select_candidates(base, candidates, target, max_parent=4, max_rule=32):
    counts = Counter(r["Output"]["Error_Type_Id"] for r in base)
    seen = {key(r["Input"]) for r in base}
    parent_counts = Counter()
    rule_counts = Counter()
    picked = []
    rejected = []
    # Prefer underrepresented mechanism/parent exposure, with stable tie breaking.
    remaining = sorted(candidates, key=digest)
    while remaining:
        remaining.sort(
            key=lambda c: (
                counts[c["label"]] / max(1, target),
                parent_counts[str(c["parent"])],
                rule_counts[(c["label"], c["rule"])],
                digest(c),
            )
        )
        c = remaining.pop(0)
        reason = None
        try:
            validate_edit(c)
        except ValueError:
            reason = "invalid_edit"
        if counts[c["label"]] >= target:
            reason = "class_target_filled"
        elif key(c["source"]) in seen:
            reason = "duplicate_source"
        elif parent_counts[str(c["parent"])] >= max_parent:
            reason = "parent_cap"
        elif rule_counts[(c["label"], c["rule"])] >= max_rule:
            reason = "new_rule_cap"
        if reason:
            rejected.append({"candidate_digest": digest(c), "reason": reason})
            continue
        row = make_row(c)
        picked.append(row)
        seen.add(key(c["source"]))
        counts[c["label"]] += 1
        parent_counts[str(c["parent"])] += 1
        rule_counts[(c["label"], c["rule"])] += 1
    return picked, rejected


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


def build(output, external_limit=32):
    from model.retained_balanced import FORMAT_V2, seal, validate

    base_dir = ROOT / "dataset/kaggle-sailor-qwen-lora-20261014"
    ext_dir = ROOT / "dataset/external/ud-vietnamese-vtb"
    recipe = json.loads((ROOT / "configs/augmentation.json").read_text())
    names = {
        "original": "VGEC 2026_v1.json",
        "classification": "VGEC_classifier.json",
        "correction": "VGEC_balanced.json",
    }
    payloads = {k: (base_dir / name).read_bytes() for k, name in names.items()}
    if any(
        hashlib.sha256(v).hexdigest() != recipe["data"]["sha256"][k]
        for k, v in payloads.items()
    ):
        raise ValueError("Pinned baseline changed")
    original = json.loads(payloads["original"])
    baseline = {
        k: json.loads(payloads[k])["train"] for k in ("classification", "correction")
    }
    ext_manifest = json.loads((ext_dir / "source_manifest.json").read_text())
    for name, sha in ext_manifest["files"].items():
        if hashlib.sha256((ext_dir / name).read_bytes()).hexdigest() != sha:
            raise ValueError("External snapshot changed")
    treebank = parse_treebank((ext_dir / "vi_vtb-ud-train.conllu").read_text())
    # Require external parents to support a phrase edit, avoiding a new clean-only domain.
    trees = [t for t in treebank if eligible_tree(t)]
    trees = [t for t in trees if any(c["label"] == 2 for c in tree_candidates(t))]
    eligible_count = len(trees)
    trees = sorted(trees, key=lambda t: digest(t["id"]))[:external_limit]
    external = []
    candidates = []
    for t in trees:
        external.append(
            {
                "ID": "ud:" + t["id"],
                "Input": t["text"],
                "Output": {"Error_Type_Id": 0, "Fixed Sentence": t["text"]},
                "Provenance": {
                    "origin": "human_annotated_ud_vietnamese_vtb_train",
                    "parent_ids": ["ud:" + t["id"]],
                    "family_id": "clean:" + digest(t["text"])[:24],
                    "license": "CC-BY-SA-4.0",
                    "revision": ext_manifest["revision"],
                    "review_status": "complete_dependency_tree_and_delimiter_filter_not_human_gec_adjudication",
                },
            }
        )
        candidates.extend(tree_candidates(t))
    for r in original:
        if r["Output"]["Error_Type_Id"] == 0:
            candidates.extend(corpus_candidates(r["Input"], r["ID"]))
    classifier = classifier_base(original, baseline["classification"]) + external
    # Keep original labels; all note targets are corrector-ineligible. Keep supported prior target repairs.
    correction, pending = corrector_originals(
        original, baseline["correction"], retain_abbreviated=True
    )
    abbreviated_ids = [
        r["ID"]
        for r in correction
        if r["Output"]["Error_Type_Id"] and ellipsis(r["Output"]["Fixed Sentence"])
    ]
    legacy = [
        copy.deepcopy(r) for r in baseline["correction"] if type(r["ID"]) is not int
    ]
    correction += legacy + copy.deepcopy(external)
    target = sum(r["Output"]["Error_Type_Id"] == 0 for r in classifier)
    added, rejected = select_candidates(classifier, candidates, target)
    unique = {
        "classification": classifier + added,
        "correction": correction + copy.deepcopy(added),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    metadata = {
        "original_file_sha256": recipe["data"]["sha256"]["original"],
        "baseline_sha256": recipe["data"]["sha256"],
        "external_source": ext_manifest,
        "external_clean_parents": len(external),
        "eligible_external_parents_before_selection": eligible_count,
        "external_quality_exclusions": EXTERNAL_REVIEW_EXCLUSIONS,
        "external_parent_trees": trees,
        "user_policy": "Each error class approaches clean count; earlier 45-50% clean-exposure proposal superseded.",
        "classifier_original_exclusions": [],
        "restored_unchanged_classifier_ids": sorted(PREVIOUSLY_EXCLUDED),
        "corrector_note_exclusions": sorted(NOTES),
        "corrector_review_exclusions": CORRECTOR_REVIEW_EXCLUSIONS,
        "correction_target_policy": "retain_organizer_abbreviated_references_v2",
        "retained_abbreviated_target_ids": abbreviated_ids,
        "abbreviated_reference_policy": "Exact organizer targets, not newly validated full sentences; one exposure each, no inferred completion.",
        "new_code_examples": len(added),
        "new_code_class_counts": dict(
            Counter(str(r["Output"]["Error_Type_Id"]) for r in added)
        ),
        "new_code_rule_counts": dict(
            Counter(r["Provenance"]["template_rule"] for r in added)
        ),
        "new_code_distinct_parents": len(
            {p for r in added for p in r["Provenance"]["parent_ids"]}
        ),
        "new_parent_cap": 4,
        "new_rule_cap": 32,
        "source_code_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "tools/build_retained_balanced.py",
                "tools/author_synthetic.py",
                "model/retained_balanced.py",
                "model/data.py",
            )
        },
        "model_authored_new_examples": 0,
        "new_target_repairs": 0,
        "quality_scope": "Conservative deterministic applicability/lineage checks; no human adjudication or public score gain established.",
        "unresolved_corrector_targets": len(pending),
        "kaggle_started": False,
        "stage_balance": {},
    }
    views = {}
    for stage in unique:
        views[stage], metadata["stage_balance"][stage] = balanced_view(
            unique[stage],
            target=target,
            max_repeat=4,
            no_repeat_ids=abbreviated_ids if stage == "correction" else (),
        )
    for stage, filename in [
        ("classification", "VGEC_classifier.json"),
        ("correction", "VGEC_balanced.json"),
    ]:
        train = views[stage]
        bundle = seal(
            stage,
            original,
            baseline[stage],
            unique[stage],
            train,
            external,
            metadata,
            format_version=FORMAT_V2,
        )
        validate(bundle)
        write_json(output / filename, bundle)
        write_json(output / (stage + "_canonical.json"), unique[stage])
        write_json(output / (stage + "_train.json"), train)
    write_json(output / "original_master.json", original)
    (output / "VGEC 2026_v1.json").write_bytes(payloads["original"])
    write_json(output / "unresolved_targets.json", pending)
    write_json(output / "new_code_examples.json", added)
    write_json(output / "generation_rejections.json", rejected)
    write_json(output / "external_training_parents.json", trees)
    (output / "UD_VTB_LICENSE.txt").write_bytes((ext_dir / "LICENSE.txt").read_bytes())
    (output / "UD_VTB_README.md").write_bytes((ext_dir / "README.md").read_bytes())
    manifest = dict(
        metadata,
        format=FORMAT_V2,
        data_sha256={
            k: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for k, name in names.items()
        },
        training_counts={k: len(v) for k, v in views.items()},
        independent_validation_rows=0,
    )
    write_json(output / "training_manifest.json", manifest)
    write_json(output / "external_source_manifest.json", ext_manifest)
    readme = (
        "# Private retained and balanced VGEC training data\n\n"
        "Classification: VGEC_classifier.json. Correction: VGEC_balanced.json. "
        "Use the vgec-retained-balanced-v2 loader, seed 42, validation fraction 0. "
        "Raw exports are also included. This package is not a submission ZIP.\n\n"
        "Canonical (non-resampled) examples and explicit repeated training exposures are separate. "
        "Original duplicates are preserved; new generated sources are deduplicated. "
        "See training_manifest.json for exact counts and unresolved targets. "
        "V2 retains 112 original organizer references containing ellipses verbatim, once per epoch. "
        "They are not claimed to be reconstructed complete sentences. Public predictions are audit-only. "
        "No model benefit or human adjudication is claimed. No Kaggle launch is authorized by this export.\n\n"
        "Do not publish or redistribute VGEC data or this mixed archive. Acknowledge VLSP 2026 and the VGEC organizers.\n\n"
        "External component only: UD Vietnamese VTB, train split, CC BY-SA 4.0. "
        "Attribution: Vietnam Treebank creators and UD Vietnamese VTB contributors; "
        "https://github.com/UniversalDependencies/UD_Vietnamese-VTB . "
        "See the included upstream README, LICENSE and source manifest for authors, revision and checksums. "
        "Changes: filtered complete sentences, copied clean controls, and deterministic local error edits. "
        "The external component and its adaptations retain CC BY-SA 4.0; "
        "that license does not relicense the private VGEC corpus.\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir())
        if p.is_file()
    }
    write_json(
        output / "file_manifest.json",
        {"format": "private-vgec-file-manifest-v1", "files": files},
    )
    archive = output / "VGEC26-retained-balanced-v2-20261015.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.iterdir()):
            if p.is_file() and p != archive:
                z.write(p, p.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Archive CRC failure")
        for name, sha in files.items():
            if hashlib.sha256(z.read(name)).hexdigest() != sha:
                raise ValueError("Archive payload changed")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--external-limit", type=int, default=32)
    args = parser.parse_args()
    info = build(Path(args.output_dir), args.external_limit)
    print(
        json.dumps(
            {k: v for k, v in info.items() if k != "external_parent_trees"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
