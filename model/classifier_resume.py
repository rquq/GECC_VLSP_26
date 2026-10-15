"""Hash-bound completed ensemble members, copied privately into a fresh run."""

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from .classifier_ensemble import NAMES
from .runtime import write_json

MANIFEST = "classifier_resume.json"
FORMAT = "vgec-classifier-resume-v1"
PATH_FIELDS = {"path", "input_root", "augmentation_path", "correction_path"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_recipe(config):
    """Mount locations change on portable reuse; dataset hashes/rows must not."""
    return {k: v for k, v in config["data"].items() if k not in PATH_FIELDS}


def _entries(manifest):
    names = manifest.get("members", [])
    files = manifest.get("file_sha256", {})
    if (
        manifest.get("format") != FORMAT
        or not names
        or names != [n for n in NAMES if n in names]
        or not files
    ):
        raise ValueError("Invalid classifier resume manifest")
    for name, digest in files.items():
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) < 4
            or tuple(path.parts[:2]) != ("classification", "members")
            or path.parts[2] not in names
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("Invalid classifier resume file path/hash")
    return files


def validate_files(root):
    root = Path(root).resolve()
    manifest = json.loads((root / MANIFEST).read_text())
    for name, expected in _entries(manifest).items():
        path = root / name
        if (
            not path.resolve().is_relative_to(root)
            or not path.is_file()
            or sha256(path) != expected
        ):
            raise ValueError(f"Classifier resume file missing or hash differs: {name}")
    return manifest


def materialize(
    input_root, archive_name, expected_manifest_sha256, output, source_manifest=None
):
    """Accept the private ZIP or its extracted contents; copy only hash-bound files."""
    input_root, output = Path(input_root), Path(output)
    archives = list(input_root.rglob(archive_name))
    if len(archives) > 1:
        raise ValueError("Multiple classifier resume archives attached")
    if archives:
        with zipfile.ZipFile(archives[0]) as archive:

            def read(name):
                entries = [
                    e
                    for e in archive.infolist()
                    if e.filename == name and not e.is_dir()
                ]
                if len(entries) != 1:
                    raise ValueError(
                        f"Missing or duplicate resume archive member: {name}"
                    )
                return archive.read(entries[0])

            raw = read(MANIFEST)
            if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
                raise ValueError(
                    "Classifier resume manifest differs from committed recipe"
                )
            value = json.loads(raw)
            for name, expected in _entries(value).items():
                content = read(name)
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError(f"Classifier resume archive hash differs: {name}")
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                path.write_bytes(content)
                path.chmod(0o600)
    else:
        candidates = list(input_root.rglob(MANIFEST))
        if (
            not candidates
            and source_manifest is not None
            and Path(source_manifest).is_file()
        ):
            # Original Kaggle output has no packet descriptor. Only accept the
            # exact completed-stage files bound by this committed source proof.
            value = json.loads(Path(source_manifest).read_text())
            if value.get("packet_manifest_sha256") != expected_manifest_sha256:
                raise ValueError(
                    "Original-output proof differs from requested recovery"
                )
            files = _entries(value)
            roots = [
                p.parent
                for p in input_root.rglob("classification")
                if p.is_dir() and all((p.parent / name).is_file() for name in files)
            ]
            if len(roots) != 1:
                raise ValueError(
                    "Attach exactly one original notebook output with both saved classifiers"
                )
            for name, expected in files.items():
                source = roots[0] / name
                if (
                    not source.resolve().is_relative_to(roots[0].resolve())
                    or sha256(source) != expected
                ):
                    raise ValueError(f"Original notebook output hash differs: {name}")
            for name in files:
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copy2(roots[0] / name, path)
            write_json(output / MANIFEST, value)
            validate_files(output)
            return output
        if len(candidates) != 1:
            raise ValueError("Attach exactly one extracted classifier resume package")
        source = candidates[0].parent
        raw = candidates[0].read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
            raise ValueError("Classifier resume manifest differs from committed recipe")
        value = validate_files(source)
        for name in _entries(value):
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copy2(source / name, path)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    (output / MANIFEST).write_bytes(raw)
    validate_files(output)
    return output


def resume_output(root):
    """Only a pristine validated packet can bypass the empty-output safeguard."""
    root = Path(root).expanduser().resolve()
    manifest = validate_files(root)
    expected = set(manifest["file_sha256"]) | {MANIFEST}
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError(
            "Resume output contains unexpected/prior-run files; use a fresh output directory"
        )
    return root


def plan(root, config, original, mixed):
    """Validate exact rows, pinned model, recipe and provenance before skipping a fit."""
    root = Path(root)
    if not (root / MANIFEST).exists():
        return {}
    from .run_reuse import validate_classifier_run
    from .data import load_classifier_split
    from .classification import _organizer_partition
    from .runtime import split_seed

    manifest = validate_files(root)
    requested = {e["name"]: e for e in config.get("ensemble", {}).get("members", [])}
    if not set(manifest["members"]) <= set(requested):
        raise ValueError("Resume members differ from requested classifier ensemble")
    seed, fraction = split_seed(config), config["data"]["validation_fraction"]
    train, info = load_classifier_split(original, mixed, fraction, seed)
    _, validation = _organizer_partition(original, fraction, seed, info)
    result = {}
    for name in manifest["members"]:
        member = root / "classification/members" / name
        old = validate_classifier_run(member, require_hints=False)
        stats = json.loads((member / "run_manifest.json").read_text())
        if (
            old["model"] != requested[name]["model"]
            or old["run"]["seed"] != 42
            or old.get("lora") != config.get("lora")
            or old["training"] != config["training"]
            or data_recipe(old) != data_recipe(config)
            or stats.get("data_sha256") != sha256(original)
            or stats.get("classifier_data_sha256") != sha256(mixed)
            or json.loads((member / "train.json").read_text()) != train
            or json.loads((member / "validation.json").read_text()) != validation
        ):
            raise ValueError(
                f"Completed resume member differs from requested model/data/recipe: {name}"
            )
        result[name] = member
    return result


def package(source, dataset_archive, output, selected=None):
    """Preserve completed members and canonical data; omit failed/incomplete members."""
    from .run_reuse import copy_classifier_run, validate_classifier_run

    source, output = Path(source), Path(output)
    stage = (
        source / "run/classification"
        if (source / "run/classification").is_dir()
        else source / "classification"
    )
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    names = []
    for name in NAMES if selected is None else selected:
        if name not in NAMES:
            raise ValueError("Unknown classifier resume member")
        member = stage / "members" / name
        if not (member / "run_manifest.json").exists():
            continue
        validate_classifier_run(member, require_hints=False)
        copy_classifier_run(member, output / "classification/members" / name)
        names.append(name)
    if not names:
        raise ValueError("No completed classifier members to preserve")
    files = {
        str(p.relative_to(output)): sha256(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    write_json(
        output / MANIFEST,
        {
            "format": FORMAT,
            "members": names,
            "source": str(source.resolve()),
            "file_sha256": files,
        },
    )
    with zipfile.ZipFile(dataset_archive) as archive:
        for name in [
            "VGEC 2026_v1.json",
            "VGEC_classifier.json",
            "VGEC_balanced.json",
            "public_test.json",
        ]:
            entries = [
                e
                for e in archive.infolist()
                if not e.is_dir() and Path(e.filename).name == name
            ]
            if len(entries) != 1:
                raise ValueError(f"Dataset package lacks unique {name}")
            (output / name).write_bytes(archive.read(entries[0]))
    destination = output.with_suffix(".zip")
    with zipfile.ZipFile(
        destination, "w", zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    return destination
