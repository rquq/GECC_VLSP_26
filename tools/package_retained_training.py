"""Assemble private retained-v2 data and its inference input for Kaggle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from model.runtime import write_json

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = 'VGEC26-sailor-qwen-retained-v2-20261015.zip'


def package(source, inference, output):
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for path in source.iterdir():
        if path.is_file() and path.suffix != '.zip':
            shutil.copy2(path, output / path.name)
    shutil.copy2(inference, output / 'public_test.json')
    names = {'original': 'VGEC 2026_v1.json',
             'classification': 'VGEC_classifier.json', 'correction': 'VGEC_balanced.json'}
    hashes = {role: hashlib.sha256((output / name).read_bytes()).hexdigest()
              for role, name in names.items()}
    manifest = json.loads((output / 'training_manifest.json').read_text())
    manifest.update(data_sha256=hashes, data_archive=ARCHIVE,
                    inference_inputs={'public_test.json': hashlib.sha256(inference.read_bytes()).hexdigest()},
                    source_training_manifest_sha256=hashlib.sha256((source / 'training_manifest.json').read_bytes()).hexdigest(),
                    packaging_checks='Not run at the user request; hashes recorded for runtime input matching.')
    write_json(output / 'training_manifest.json', manifest)
    with (output / 'README.md').open('a', encoding='utf-8') as f:
        f.write('\n## Kaggle training package\n\n'
                'Upload this archive as a PRIVATE Kaggle dataset, or upload its extracted contents. '
                'Attach only this dataset version. The enclosing Kaggle dataset name can vary. '
                'Import the kaggle-linux notebooks/kaggle_two_phase.ipynb launcher, enable Internet '
                'and T4 x2, enable the GITHUB_TOKEN repository secret, then Save & Run All. '
                'Both stages consume 1360 training exposures, 272 per class; '
                'Sailor classification runs 5 epochs and Qwen correction 6 with standard LoRA. '
                'public_test.json is inference-only. No local validation score is available. '
                'This packaging update was not tested at the user request.\n')
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output.iterdir()) if p.is_file() and p.name != 'file_manifest.json'}
    write_json(output / 'file_manifest.json', {'format': 'private-vgec-file-manifest-v1', 'files': files})
    archive = output / ARCHIVE
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.iterdir()):
            if path.is_file() and path != archive:
                z.write(path, path.name)
    return {'archive': str(archive.resolve()), 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'data_sha256': hashes, 'training_counts': manifest['training_counts'],
            'inference_inputs': manifest['inference_inputs'], 'verification': 'skipped_by_user_request'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'dataset/retained-balanced-v2-20261015')
    parser.add_argument('--inference', type=Path, default=ROOT / 'dataset/kaggle-sailor-qwen-lora-20261014/public_test.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'dataset/kaggle-retained-v2-20261015')
    args = parser.parse_args()
    print(json.dumps(package(args.source, args.inference, args.output), indent=2))
