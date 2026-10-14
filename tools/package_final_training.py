"""Package the approved full training populations for the committed Kaggle recipe."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from model.data import load_classifier_split, load_training_split
from model.final_data import FORMAT
from model.runtime import write_json

ROOT = Path(__file__).resolve().parents[1]


def final_bundle(path, stage):
    payload = Path(path).read_bytes()
    source = json.loads(payload)
    return {'format': FORMAT, 'stage': stage, 'seed': 42, 'validation_fraction': 0,
            'original_sha256': source.get('original_sha256'),
            'source_bundle_sha256': hashlib.sha256(payload).hexdigest(),
            'source_bundle_base64': base64.b64encode(payload).decode('ascii'),
            'train': source['train'] + source['validation'], 'validation': []}


def package(source, full_export, output):
    output.mkdir(parents=True, exist_ok=False)
    hashes, counts = {}, {}
    for stage, filename in [('classification', 'VGEC_classifier.json'), ('correction', 'VGEC_balanced.json')]:
        bundle = final_bundle(source / filename, stage)
        if bundle['train'] != json.loads((full_export / f'{stage}_train.json').read_text()):
            raise ValueError('Kaggle population differs from the requested full training export')
        write_json(output / filename, bundle)
        train, validation, _ = load_training_split(output / filename, 0, 42)
        counts[stage] = len(train)
        assert not validation
        hashes[stage] = hashlib.sha256((output / filename).read_bytes()).hexdigest()
    for filename in ['VGEC 2026_v1.json', 'public_test.json']:
        shutil.copy2(source / filename, output / filename)
    load_classifier_split(output / 'VGEC 2026_v1.json', output / 'VGEC_classifier.json', 0, 42)
    hashes['original'] = hashlib.sha256((output / 'VGEC 2026_v1.json').read_bytes()).hexdigest()
    shutil.copy2(full_export / 'added_correction_review_queue.json', output)
    manifest = {'format': FORMAT, 'training_counts': counts, 'data_sha256': hashes,
                'independent_validation_rows': 0, 'correction_review_flags_retained': len(json.loads((full_export / 'added_correction_review_queue.json').read_text())),
                'selection': 'fixed final epoch; no local validation score',
                'privacy': 'Private noncommercial corpus derivative; do not redistribute'}
    write_json(output / 'training_manifest.json', manifest)
    archive = output / 'VGEC26-sailor-qwen-lora-full-training-20261014.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.glob('*.json')):
            z.write(path, path.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            assert z.read(name) == (output / name).read_bytes()
    manifest['archive'] = {'path': str(archive), 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                           'bytes': archive.stat().st_size}
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT / 'dataset/kaggle-qwen-ready-20261013')
    p.add_argument('--full-export', type=Path, default=ROOT / 'dataset/full-training-including-validation-20261014')
    p.add_argument('--output', type=Path, default=ROOT / 'dataset/kaggle-sailor-qwen-lora-20261014')
    args = p.parse_args()
    print(json.dumps(package(args.source, args.full_export, args.output), indent=2))


if __name__ == '__main__':
    main()
