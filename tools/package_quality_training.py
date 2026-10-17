"""Private quality-mixture upload with an explicitly pinned inference-only input."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from model.quality_data import seal, load_quality
from model.grouped_validation import audit
from model.runtime import write_json
from model.data import load_records


def package(
    source,
    private_test,
    output,
    *,
    diverse=False,
    exposure_strategy="preserve-baseline"
):
    if exposure_strategy not in ("preserve-baseline", "cap-two"):
        raise ValueError("Unknown diversity exposure strategy")
    source, private_test, output = map(Path, (source, private_test, output))
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    shutil.copy2(source / "VGEC 2026_v1.json", output / "VGEC 2026_v1.json")
    shutil.copy2(private_test, output / "private_test.json")
    rows = load_records(private_test, require_labels=False)
    if any(r["label"] is not None or r["target"] is not None for r in rows):
        raise ValueError("Expected unlabeled private inference input")
    bundles, reports = {}, {}
    names = {
        "classification": "VGEC_classifier.json",
        "correction": "VGEC_balanced.json",
    }
    for stage, filename in names.items():
        raw = json.loads((source / filename).read_text())
        if diverse:
            from model.diversity_data import (
                seal_diverse,
                load_diverse,
                ADDITIVE_POLICY,
                DIVERSITY_POLICY,
            )

            conllu = (
                Path(__file__).resolve().parents[1]
                / "dataset/external/ud-vietnamese-vtb/vi_vtb-ud-train.conllu"
            ).read_text()
            b = seal_diverse(
                raw,
                conllu,
                policy=(
                    ADDITIVE_POLICY
                    if exposure_strategy == "preserve-baseline"
                    else DIVERSITY_POLICY
                ),
            )
            load_diverse(b, 0.2, 42)
        else:
            b = seal(raw)
            load_quality(b, 0.2, 42)
        write_json(output / filename, b)
        bundles[stage] = b
        reports[stage] = b["report"]
    if bundles["classification"]["validation"] != bundles["correction"]["validation"]:
        raise ValueError("Stage validation differs")
    leakage = audit(
        {stage: b["train"] for stage, b in bundles.items()},
        bundles["classification"]["validation"],
    )
    # Audit only: inference rows never supply generation contexts or labels.
    from model.data import _source_key

    test_text = {_source_key(r["source"]) for r in rows}
    overlap = {
        stage: len(
            test_text
            & {
                _source_key(t)
                for r in b["train"]
                for t in [r["Input"], r["Output"]["Fixed Sentence"]]
                if t
            }
        )
        for stage, b in bundles.items()
    }
    if any(overlap.values()):
        raise ValueError("Private input overlaps prepared training text")
    for name in ["LICENSE.txt", "README.md", "source_manifest.json"]:
        origin = (
            Path(__file__).resolve().parents[1]
            / "dataset/external/ud-vietnamese-vtb"
            / name
        )
        if origin.exists():
            target = output / "external-ud-vtb" / name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(origin, target)
    hashes = {
        role: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for role, name in {"original": "VGEC 2026_v1.json", **names}.items()
    }
    manifest = {
        "format": (
            "private-vgec-diverse-upload-v1"
            if diverse
            else "private-vgec-quality-upload-v1"
        ),
        "archive": (
            (
                "VGEC26-diverse-v3-private-20261017.zip"
                if exposure_strategy == "preserve-baseline"
                else "VGEC26-diverse-v2-private-20261017.zip"
            )
            if diverse
            else "VGEC26-quality-private-20261017.zip"
        ),
        "data_sha256": hashes,
        "submission_filename": "private_test.json",
        "submission_sha256": hashlib.sha256(private_test.read_bytes()).hexdigest(),
        "inference_rows": len(rows),
        "inference_role": "inference only; no labels, no generation parents",
        "validation_rows": len(bundles["classification"]["validation"]),
        "leakage_audit": leakage,
        "private_exact_training_overlap": overlap,
        "stages": reports,
        "source_training_package": str(source.resolve()),
        "external_data": "Pinned licensed human-annotated UD-VTB training parents and deterministic operations; no LLM-authored rows admitted",
        "scope": "Research candidate mixture; mechanical checks do not establish a model-score improvement",
    }
    write_json(output / "training_manifest.json", manifest)
    (output / "README.md").write_text(
        "PRIVATE research upload. Attach only this package. Fresh Qwen/Sailor/Mistral soft-vote training and Qwen correction. Keep78-row grouped development set. private_test.json is2034-row inference-only data; never train or synthesize from it. Synthetic rule/parent caps and matched controls are not a human quality guarantee. See training_manifest.json for exact counts and hashes. Keep original and derived VLSP resources private.\n"
    )
    files = {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*")
        if p.is_file()
    }
    write_json(
        output / "file_manifest.json",
        {"format": "private-vgec-file-manifest-v1", "files": files},
    )
    archive = output / manifest["archive"]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(output.rglob("*")):
            if p.is_file() and p != archive:
                z.write(p, p.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC failed")
        for name, digest in files.items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError("ZIP payload differs")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diverse",
        action="store_true",
        help="Expand a quality bundle using pinned human UD training data and deterministic rules",
    )
    parser.add_argument(
        "--exposure-strategy",
        choices=("preserve-baseline", "cap-two"),
        default="preserve-baseline",
        help="Dataset build policy only; the notebook has no switches",
    )
    a = parser.parse_args()
    print(
        json.dumps(
            package(
                a.source,
                a.private_test,
                a.output,
                diverse=a.diverse,
                exposure_strategy=a.exposure_strategy,
            ),
            indent=2,
        )
    )
