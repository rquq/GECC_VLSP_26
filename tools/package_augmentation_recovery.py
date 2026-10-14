"""Package unchanged baselines and saved eligible generations for private Kaggle recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from augmentation.recovery import FORMAT, load_payload, mechanical_pool
from augmentation.review import digest
from model.runtime import PROJECT_ROOT, write_json

FILENAMES = {'original': 'VGEC 2026_v1.json', 'classification': 'VGEC_classifier.json',
             'correction': 'VGEC_balanced.json'}


def package(source_run, baseline_dir, output):
    source_run, baseline_dir, output = map(Path, (source_run, baseline_dir, output))
    if output.exists():
        raise ValueError('Recovery package output already exists; preserve the previous artifact')
    source = json.loads((source_run / 'run_identity.json').read_text())
    inputs = {role: baseline_dir / name for role, name in FILENAMES.items()}
    hashes = {role: hashlib.sha256(path.read_bytes()).hexdigest() for role, path in inputs.items()}
    if hashes != source['recipe']['data']['sha256'] or any(hashes[k] != v['sha256'] for k, v in source['inputs'].items()):
        raise ValueError('Canonical baselines differ from the saved generation inputs')
    files = ('authored_candidates.json', 'calls.jsonl', 'run_identity.json')
    payload = {'format': FORMAT, 'source': source['source'], 'recipe': source['recipe'],
               'baseline_sha256': hashes,
               'source_file_sha256': {f: hashlib.sha256((source_run / f).read_bytes()).hexdigest() for f in files},
               'candidates': json.loads((source_run / 'authored_candidates.json').read_text()),
               'calls': [r for line in (source_run / 'calls.jsonl').read_text().splitlines()
                         if (r := json.loads(line))['role'] in ('translator', 'author')]}
    output.mkdir(parents=True, mode=0o700)
    packet = output / 'augmentation_recovery.json'; write_json(packet, payload)
    packet_sha = hashlib.sha256(packet.read_bytes()).hexdigest()
    load_payload(packet, packet_sha, source['recipe'], inputs)
    pool, rejected = mechanical_pool(payload['candidates'], source['recipe'])
    for role, path in inputs.items():
        shutil.copy2(path, output / FILENAMES[role])
    manifest = {'format': FORMAT, 'purpose': 'Private recovery inputs; not augmented training data',
                'source': source['source'], 'source_recipe_sha256': digest(source['recipe']),
                'payload': {'filename': packet.name, 'sha256': packet_sha},
                'baseline_sha256': hashes, 'saved_candidates': len(payload['candidates']),
                'mechanical_candidates': len(pool), 'mechanical_rejections': len(rejected),
                'translation_calls_planned': 0, 'authoring_calls_planned': 0}
    write_json(output / 'recovery_upload_manifest.json', manifest)
    archive = output / 'VGEC26-augmentation-recovery-20261015.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.glob('*.json')):
            z.write(path, path.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or any(z.read(p.name) != p.read_bytes() for p in output.glob('*.json')):
            raise ValueError('Recovery ZIP failed round-trip verification')
    manifest['archive'] = {'path': str(archive.resolve()), 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                           'bytes': archive.stat().st_size}
    write_json(output / 'package_verification.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--baseline-dir', default=str(PROJECT_ROOT / 'dataset/kaggle-sailor-qwen-lora-20261014'))
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.source_run, args.baseline_dir, args.output_dir), indent=2))


if __name__ == '__main__': main()
