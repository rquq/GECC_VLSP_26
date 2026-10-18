"""Build one private upload: anchored data, known baselines and hash-bound evidence."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from model.anchored_data import seal, load_anchored
from model.data import load_records, _source_key
from model.grouped_validation import audit
from model.retained_rules import ellipsis
from model.run_reuse import copy_classifier_run, validate_classifier_run
from model.runtime import write_json

MANIFEST = "focused_package_manifest.json"
ARCHIVE = "VGEC26-focused-private-20261018.zip"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def review_training(rows):
    """Audit unchanged organizer training targets; proposals never enter fitting."""
    from tools.propose_training_repairs import trailing_reference_candidate
    from model.data import declares_correct_sentence

    originals = [r for r in rows if type(r["ID"]) is int]
    grouped = defaultdict(list)
    for r in originals:
        grouped[_source_key(r["Input"])].append(r)
    results = []
    for r in originals:
        target = r["Output"]["Fixed Sentence"] or r["Input"]
        flags = []
        if ellipsis(target):
            flags.append("abbreviated_reference")
        if target.lstrip().startswith("("):
            flags.append("annotation_style_target")
        if r["Output"]["Error_Type_Id"] != 0 and declares_correct_sentence(target):
            flags.append("explicit_correctness_note")
        peers = grouped[_source_key(r["Input"])]
        if len({x["Output"]["Error_Type_Id"] for x in peers}) > 1:
            flags.append("duplicate_source_label_conflict")
        candidate = trailing_reference_candidate(r["Input"], target)
        results.append(
            {
                "original": r,
                "flags": flags,
                "mechanical_proposal": candidate,
                "status": (
                    "pending_evidence_not_applied"
                    if flags
                    else "retained_original_no_change"
                ),
                "eligible_training_parent": True,
                "labels_or_targets_changed": False,
            }
        )
    return {
        "scope": "training originals only; no inferred labels or accepted target rewrites",
        "training_rows_reviewed": len(results),
        "flagged_rows": sum(bool(r["flags"]) for r in results),
        "mechanical_proposals_not_admitted": sum(
            r["mechanical_proposal"] is not None for r in results
        ),
        "accepted_repairs": 0,
        "rows": results,
    }


def package(source, classifier, corrector, output):
    source, classifier, corrector, output = map(
        Path, (source, classifier, corrector, output)
    )
    validate_classifier_run(classifier, require_hints=False)
    old = json.loads((corrector / "run_manifest.json").read_text())
    if not old.get("saved_adapter_reload_verified"):
        raise ValueError("Baseline corrector was not reload verified")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    bundles = {}
    for stage, name in [
        ("classification", "VGEC_classifier.json"),
        ("correction", "VGEC_balanced.json"),
    ]:
        b = seal(json.loads((source / name).read_text()))
        load_anchored(b, 0.2, 42)
        write_json(output / name, b)
        bundles[stage] = b
    validation = bundles["classification"]["validation"]
    if bundles["correction"]["validation"] != validation:
        raise ValueError("Stage validation differs")
    from model.data import _records_from_raw

    gold = _records_from_raw(validation)
    if gold != json.loads(
        (classifier / "validation.json").read_text()
    ) or gold != json.loads((corrector / "validation.json").read_text()):
        raise ValueError("Baselines and prepared development rows differ")
    leakage = audit({k: b["train"] for k, b in bundles.items()}, validation)
    for name in ["VGEC 2026_v1.json", "private_test.json"]:
        shutil.copy2(source / name, output / name)
    test = load_records(output / "private_test.json", require_labels=False)
    if any(r["label"] is not None or r["target"] is not None for r in test):
        raise ValueError("Private input must be unlabeled")
    forbidden = {_source_key(r["source"]) for r in test}
    overlap = {
        k: len(
            forbidden
            & {
                _source_key(t)
                for r in b["train"]
                for t in [r["Input"], r["Output"]["Fixed Sentence"]]
                if t
            }
        )
        for k, b in bundles.items()
    }
    if any(overlap.values()):
        raise ValueError("Private input overlaps fitting text")
    copy_classifier_run(classifier, output / "baseline/classification")
    validate_classifier_run(output / "baseline/classification", require_hints=False)
    shutil.copytree(corrector / "adapter", output / "baseline/correction/adapter")
    for name in [
        "run_manifest.json",
        "config.json",
        "validation.json",
        "metrics.json",
        "predictions.json",
        "epoch_history.json",
        "source_manifest.json",
    ]:
        if (corrector / name).is_file():
            shutil.copy2(corrector / name, output / "baseline/correction" / name)
    licenses = source / "external-ud-vtb"
    if licenses.is_dir():
        shutil.copytree(licenses, output / "external-ud-vtb")
    review = review_training(bundles["classification"]["source_bundle"]["train"])
    write_json(output / "training_review_not_applied.json", review)
    (output / "README.md").write_text(
        "# Private focused Kaggle upload\n\n"
        "Use the existing kaggle-linux notebook launcher with Internet enabled and GPU T4 x2. "
        "Attach this package only; its Kaggle dataset name and extraction prefix may vary. "
        "No older notebook output or notebook toggles are needed. Save Version / Run All.\n\n"
        "The committed plan fits one Sailor classifier and one Qwen corrector for five full epochs, "
        "retains their best checkpoints, compares the bundled quality Sailor / old Qwen baselines, "
        "screens greedy, beam2, beam4 and beam4 edit reranking, and fully evaluates greedy plus one finalist per corrector. "
        "The best observed development components are selected; new challengers are not forced to win. "
        "The shared78 development rows and2034 private inference inputs never enter fitting. "
        "Original annotations are preserved; automated repair proposals are not admitted.\n\n"
        "Download results.zip for submission and retain the whole run output. selected_models/ and selected.json "
        "contain the selected adapters and exact decoder contract. Use tools.replay_focused to replay a relocated selection; "
        "ordinary greedy pipeline_predict does not reproduce a beam/reranking selection. "
        "Pretrained base weights remain pinned Hugging Face downloads. "
        "All data, generated comparisons, checkpoints and this ZIP remain private.\n"
    )
    data_hashes = {
        role: sha256(output / name)
        for role, name in [
            ("original", "VGEC 2026_v1.json"),
            ("classification", "VGEC_classifier.json"),
            ("correction", "VGEC_balanced.json"),
        ]
    }
    # Every package member is bound, including adapter tensors, configuration and private inputs.
    files = {
        str(p.relative_to(output)): sha256(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    manifest = {
        "format": "vgec-focused-package-v1",
        "archive": ARCHIVE,
        "data_sha256": data_hashes,
        "submission_sha256": sha256(output / "private_test.json"),
        "inference_rows": len(test),
        "validation_rows": len(gold),
        "stages": {k: v["report"] for k, v in bundles.items()},
        "training_review": {k: v for k, v in review.items() if k != "rows"},
        "leakage_audit": leakage,
        "private_exact_training_overlap": overlap,
        "baselines": {
            "classification": "baseline/classification",
            "correction": "baseline/correction",
        },
        "baseline_source_paths": {
            "classification": str(classifier.resolve()),
            "correction": str(corrector.resolve()),
        },
        "file_sha256": files,
        "privacy": "Private research derivatives/checkpoints; do not publish",
        "scope": "Development selection; no measured gain claimed for this new mixture",
    }
    write_json(output / MANIFEST, manifest)
    with zipfile.ZipFile(
        output / ARCHIVE, "w", zipfile.ZIP_DEFLATED, compresslevel=1
    ) as z:
        for name in sorted([*files, MANIFEST]):
            z.write(output / name, name)
    with zipfile.ZipFile(output / ARCHIVE) as z:
        if z.testzip() is not None:
            raise ValueError("Private package ZIP CRC failure")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "classifier", "corrector", "output"]:
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    m = package(a.source, a.classifier, a.corrector, a.output)
    print(
        json.dumps(
            {
                k: v
                for k, v in m.items()
                if k not in ("file_sha256", "baseline_source_paths")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
