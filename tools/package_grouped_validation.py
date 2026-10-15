"""Build a private grouped holdout from retained-v2 without changing full-fit data."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from model.data import _records_from_raw, load_classifier_split, load_training_split
from model.grouped_validation import (
    FORMAT,
    POLICY,
    audit,
    derive,
    seal,
    select_validation,
    source_groups,
    typed_sort,
)
from model.retained_balanced import FORMAT_V2, validate
from model.retained_rules import ellipsis
from model.runtime import write_json

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "VGEC26-grouped-validation-20261015.zip"
FILENAMES = {
    "classification": "VGEC_classifier.json",
    "correction": "VGEC_balanced.json",
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(rows):
    labels = Counter(r["Output"]["Error_Type_Id"] for r in rows)
    return {str(i): labels[i] for i in range(5)}


def population(rows):
    kinds = Counter(
        (
            "resampled"
            if r.get("Provenance", {}).get("resampling")
            else (
                "original"
                if type(r["ID"]) is int
                else (
                    "external_clean"
                    if str(r["ID"]).startswith("ud:")
                    else (
                        "new_code" if str(r["ID"]).startswith("balanced:") else "legacy"
                    )
                )
            )
        )
        for r in rows
    )
    return {"rows": len(rows), "class_counts": counts(rows), "kinds": dict(kinds)}


def verify_archive(archive, files):
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Archive CRC check failed")
        if set(z.namelist()) != set(files) | {"file_manifest.json"}:
            raise ValueError("Archive member list differs")
        for name, expected in files.items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError("Archive payload differs: " + name)


def package(source, inference, output):
    source, inference, output = map(Path, (source, inference, output))
    packet = {
        s: json.loads((source / name).read_text(encoding="utf-8"))
        for s, name in FILENAMES.items()
    }
    for stage, bundle in packet.items():
        if bundle.get("format") != FORMAT_V2 or bundle.get("stage") != stage:
            raise ValueError("Expected both retained-v2 source stages")
        validate(bundle)
    original = source / "VGEC 2026_v1.json"
    if json.loads(original.read_text(encoding="utf-8")) != packet["classification"][
        "original_snapshot"
    ] or any(
        b["metadata"]["original_file_sha256"] != file_hash(original)
        for b in packet.values()
    ):
        raise ValueError("Original file differs from source snapshots or hashes")
    public = json.loads(inference.read_text(encoding="utf-8"))
    _records_from_raw(public, require_labels=False)
    ids, selection = select_validation(packet)
    training, validation, canonical, withheld, balance, report = derive(packet, ids)
    report = {
        **report,
        "source_bundle_validation": "passed",
        "shared_original_validation": True,
        "source_preservation": "Source files are read only; organizer validation annotations are verbatim.",
    }
    stats = {
        stage: {
            "canonical": population(canonical[stage]),
            "training_exposures": population(training[stage]),
            "withheld": population(withheld[stage]),
            "balance": balance[stage],
            "abbreviated_original_targets_once_each": (
                sum(
                    type(r["ID"]) is int
                    and r["Output"]["Error_Type_Id"] != 0
                    and ellipsis(r["Output"]["Fixed Sentence"])
                    for r in canonical[stage]
                )
                if stage == "correction"
                else None
            ),
        }
        for stage in FILENAMES
    }
    metadata = {
        "policy": POLICY,
        "selection": selection,
        "stage_statistics": stats,
        "privacy": "PRIVATE research artifact; no corpus redistribution.",
        "historical_exposure": "Originals have appeared in earlier project full fits/reviews. "
        "Use fresh pretrained bases; this is a new internal split, not a historically blind test.",
        "fit_contract": "Only top-level train records may enter fitting. Embedded source_packet, "
        "validation and withheld records are audit/evaluation material, never training input.",
        "augmentation_scope": "Reuse deterministic existing examples only. No generation, repair or relabeling.",
        "selection_timing": "Split existing family graph, remove held families, then resample training only.",
    }
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    shutil.copy2(original, output / original.name)
    shutil.copy2(inference, output / "public_test.json")
    for name in (
        "UD_VTB_LICENSE.txt",
        "UD_VTB_README.md",
        "external_source_manifest.json",
    ):
        if (source / name).exists():
            shutil.copy2(source / name, output / name)
    for stage, filename in FILENAMES.items():
        bundle = seal(stage, packet, ids, training[stage], validation, metadata)
        write_json(output / filename, bundle)
        write_json(output / f"{stage}_train.json", training[stage])
        write_json(output / f"{stage}_canonical.json", canonical[stage])
    write_json(output / "validation.json", validation)
    write_json(output / "withheld_family_rows.json", withheld)
    groups = source_groups(packet)
    vk = {(int, i) for i in ids}
    write_json(
        output / "split_manifest.json",
        {
            "seed": 42,
            "fraction": 0.2,
            "validation_original_ids": ids,
            "selection": selection,
            "components": [
                {
                    "partition": "held_out" if g & vk else "training_side",
                    "members": [
                        json.loads(typed_sort(k)) for k in sorted(g, key=typed_sort)
                    ],
                }
                for g in groups
            ],
        },
    )
    hashes = {
        "original": file_hash(output / original.name),
        **{s: file_hash(output / n) for s, n in FILENAMES.items()},
    }
    manifest = {
        "format": FORMAT,
        "seed": 42,
        "validation_fraction": 0.2,
        "data_archive": ARCHIVE,
        "data_sha256": hashes,
        "source_dataset": source.name,
        "source_data_sha256": {s: file_hash(source / n) for s, n in FILENAMES.items()},
        "training_counts": {s: len(training[s]) for s in FILENAMES},
        "canonical_counts": {s: len(canonical[s]) for s in FILENAMES},
        "validation": population(validation),
        "validation_abbreviated_targets": sum(
            r["Output"]["Error_Type_Id"] != 0
            and ellipsis(r["Output"]["Fixed Sentence"])
            for r in validation
        ),
        "inference_inputs": {"public_test.json": file_hash(inference)},
        "metadata": metadata,
        "code_sha256": {
            str(p.relative_to(ROOT)): file_hash(p)
            for p in (
                ROOT / "model/grouped_validation.py",
                ROOT / "model/retained_rules.py",
                ROOT / "model/data.py",
                ROOT / "model/classification.py",
                Path(__file__).resolve(),
            )
        },
    }
    # Exercise the real consumer contract, including the classifier -> corrector holdout.
    normalized_validation = _records_from_raw(validation)
    for stage, filename in FILENAMES.items():
        train, heldout, info = load_training_split(output / filename, 0.2, 42)
        if (
            train != _records_from_raw(training[stage])
            or heldout != normalized_validation
            or info["provenance"]["stage"] != stage
        ):
            raise ValueError("Loader round-trip differs")
    _, classifier_info = load_classifier_split(
        output / original.name, output / FILENAMES["classification"], 0.2, 42
    )
    if classifier_info["validation_records"] != normalized_validation:
        raise ValueError("Classifier and corrector validation differ")
    audit(training, validation)
    report.update(
        runtime_loader_roundtrip="passed", classifier_corrector_holdout_match="passed"
    )
    write_json(output / "leakage_audit.json", report)
    write_json(output / "training_manifest.json", manifest)
    (output / "README.md").write_text(
        "# Private VGEC grouped validation dataset\n\n"
        "Separate approximately 80/20 organizer-row partition, seed 42. "
        f"Shared validation: {len(validation)} unchanged organizer rows; class counts {counts(validation)}. "
        "Validation has no synthetic examples or repeated exposures. "
        "The fraction refers to eligible original rows, not the resampled training array.\n\n"
        + "\n".join(
            f"- {s}: {len(canonical[s])} canonical training rows; "
            f"{len(training[s])} exposures; {len(withheld[s])} held-family rows withheld."
            for s in FILENAMES
        )
        + "\n\nThe graph connects typed IDs, augmentation parents, specific family/template IDs, "
        "normalized sources AND targets, equal token bags, any shared 10-token span and "
        "near duplicates of at least 8 tokens with length ratio >=0.8, token Jaccard >=0.8 "
        "and SequenceMatcher ratio >=0.9. Cross-stage training union is checked against validation. "
        "Generic grammar-rule names alone are not content-family identifiers. "
        "Balancing is done after exclusion; repeats are training-only, capped at four total exposures. "
        "Original abbreviated corrector references occur once, never oversampled. "
        "Annotation-note and inline optional-target exclusions are preserved.\n\n"
        "Only train is fit input. The embedded source_packet (including prior full-fit train arrays), "
        "validation and withheld files are audit/evaluation data. Do not concatenate them into training. "
        "Use configs/validation_experiment.json through the existing runner's --recipe argument. "
        "Default configs/experiment.json still selects the full-fit dataset; uploading this ZIP alone "
        "does not switch the notebook recipe. Notebook has no experiment toggles.\n\n"
        "This establishes zero overlap under the recorded checks for these exported fit/evaluation arrays. "
        "It cannot rule out arbitrary semantic paraphrases or historical model/human/pretraining exposure. "
        "Start both stages from fresh pretrained bases, not project-finetuned checkpoints. "
        "The small per-subtype holdout is a development set, not a stable public-score estimate. "
        "Organizer abbreviated references are preserved and still limit correction metrics. "
        "Public_test.json is inference-only and has supplied no training labels.\n\n"
        "Upload privately. Original/derived VGEC resources must not be redistributed. "
        "Retain UD Vietnamese VTB train-only source attribution and CC-BY-SA-4.0 notices. "
        "No new model generation, repairs, training or Kaggle run was performed by this build.\n",
        encoding="utf-8",
    )
    files = {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(
        output / "file_manifest.json",
        {"format": "private-vgec-file-manifest-v1", "files": files},
    )
    archive = output / ARCHIVE
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.iterdir()):
            if path.is_file() and path != archive:
                z.write(path, path.name)
    verify_archive(archive, files)
    if {s: file_hash(source / n) for s, n in FILENAMES.items()} != manifest[
        "source_data_sha256"
    ]:
        raise ValueError("Source dataset changed during packaging")
    return {
        "archive": str(archive.resolve()),
        "archive_sha256": file_hash(archive),
        "archive_bytes": archive.stat().st_size,
        "data_sha256": hashes,
        "training_counts": manifest["training_counts"],
        "canonical_counts": manifest["canonical_counts"],
        "validation": manifest["validation"],
        "selection": selection,
        "stage_statistics": stats,
        "leakage_audit": report,
        "archive_check": "CRC, member set and all payload hashes passed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "dataset/retained-balanced-v2-20261015"
    )
    parser.add_argument(
        "--inference",
        type=Path,
        default=ROOT / "dataset/kaggle-retained-v2-20261015/public_test.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dataset/kaggle-grouped-validation-20261015",
    )
    args = parser.parse_args()
    print(json.dumps(package(args.source, args.inference, args.output), indent=2))
