"""Package unchanged quality data for fresh fits, without baseline checkpoints."""
import argparse
from pathlib import Path
import shutil
import zipfile

from model.runtime import write_json
from tools.package_focused_training import MANIFEST, sha256

ARCHIVE = "VGEC26-fast-quality-private-20261020.zip"
DATA_FILES = ("VGEC 2026_v1.json", "VGEC_classifier.json", "VGEC_balanced.json", "private_test.json")


def package(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name in DATA_FILES:
        shutil.copy2(source / name, output / name)
    if (source / "external-ud-vtb").is_dir():
        shutil.copytree(source / "external-ud-vtb", output / "external-ud-vtb")
    (output / "README.txt").write_text(
        "PRIVATE research data. Do not publish or redistribute.\n"
        "Unchanged quality dataset:1128 classifier exposures,1011 corrector exposures,78 dev rows,2034 private inputs.\n"
        "For fresh Sailor20B/Qwen14B training. No baseline weights or predictions required.\n"
        "Upload this outer ZIP as a PRIVATE Kaggle dataset and attach it alone.\n"
        "No ZIP/TAR archives are nested inside this upload.\n")
    import json
    previous = json.loads((source / MANIFEST).read_text())
    files = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    if any(Path(name).suffix.lower() in {".zip", ".tar", ".gz", ".tgz"} for name in files):
        raise ValueError("Upload contents must not contain nested archives")
    write_json(output / MANIFEST, {
        "format": "vgec-focused-package-v1", "archive": ARCHIVE,
        "purpose": "fresh_training_only_no_baselines", "data_sha256": previous["data_sha256"],
        "submission_sha256": previous["submission_sha256"],
        "inference_rows": previous["inference_rows"], "validation_rows": previous["validation_rows"],
        "file_sha256": files, "privacy": "Private research derivatives; do not publish",
    })
    with zipfile.ZipFile(output / ARCHIVE, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != ARCHIVE:
                archive.write(path, path.relative_to(output).as_posix())
    return output / ARCHIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    archive = package(args.source, args.output)
    print(f"Created {archive}: {archive.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
