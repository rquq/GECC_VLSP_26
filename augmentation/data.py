"""Baseline-preserving augmented datasets with embedded source and review evidence."""
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
import zipfile

from model.data import _id_key, _records_from_raw, _source_key, load_training_split
from model.runtime import write_json
from tools.build_dataset import audit_record
from .review import digest, pair_rows, select_pairs

FORMAT = 'vgec-augmented-final-v1'
FILES = {'classification': 'VGEC_classifier.json', 'correction': 'VGEC_balanced.json'}


def baselines(paths, recipe):
    values, infos = {}, {}
    for stage in FILES:
        train, validation, info = load_training_split(paths[stage], 0, 42)
        raw = json.loads(Path(paths[stage]).read_text())
        if info['format'] != 'vgec-final-training-v1' or info['stage'] != stage or validation:
            raise ValueError('Augmentation requires the approved final-training baseline for each stage')
        if len(train) != recipe['data']['counts'][stage]:
            raise ValueError('Baseline row count differs from committed augmentation recipe')
        values[stage], infos[stage] = raw, info
    return values, infos


def select_parents(paths, recipe):
    raw, info = baselines(paths, recipe)
    allowed = {_id_key(i) for i in info['classification']['provenance']['retained_original_train_ids']}
    allowed &= {_id_key(i) for i in info['correction']['provenance']['retained_original_train_ids']}
    parents, seen = [], set()
    for raw_row, row in zip(raw['correction']['train'], _records_from_raw(raw['correction']['train'])):
        if _id_key(row['id']) not in allowed or audit_record(row, raw_row, original=True)['quarantine_reasons']:
            continue
        clean = row['target']; key = _source_key(clean)
        if key in seen or not 20 <= len(clean) <= recipe['quality']['max_sentence_chars']:
            continue
        seen.add(key)
        parents.append({'id': row['id'], 'clean': clean, 'original_label': row['label'], 'source': row['source']})
    random.Random(recipe['seed']).shuffle(parents)
    return parents[:recipe['parents']['limit']]


def _verify_parents(pairs, baseline):
    rows = {_id_key(r['id']): r for r in _records_from_raw(baseline['train'])}
    source = json.loads(base64.b64decode(baseline['source_bundle_base64']))
    originals = set(source['provenance']['retained_original_train_ids']) | {r['ID'] for r in source['validation']}
    allowed = {_id_key(i) for i in originals}
    for pair in pairs:
        key = _id_key(pair['parent_id'])
        if key not in allowed or key not in rows or rows[key]['target'] != pair['parent_clean']:
            raise ValueError('Augmentation parent/reference differs from a retained baseline original')


def load_augmented(raw, fraction, seed):
    from .config import validate_recipe
    validate_recipe(raw['recipe'])
    if raw.get('format') != FORMAT:
        raise ValueError('Expected an augmented final-training bundle')
    if fraction != 0 or seed != 42 or raw.get('seed') != 42 or raw.get('validation_fraction') != 0 or raw.get('validation') != []:
        raise ValueError('Augmented final fits have no independent validation')
    payload = base64.b64decode(raw['baseline_base64'], validate=True)
    if hashlib.sha256(payload).hexdigest() != raw['baseline_sha256']:
        raise ValueError('Augmented baseline digest differs')
    base = json.loads(payload)
    stage = raw['stage']
    expected = raw['recipe']['data']
    if (stage not in FILES or raw['baseline_sha256'] != expected['sha256'][stage]
            or len(base['train']) != expected['counts'][stage]
            or base['original_sha256'] != expected['sha256']['original']):
        raise ValueError('Embedded baseline differs from recipe pinned digest/count/original')
    from model.final_data import load_final
    _, _, info = load_final(base, 0, 42)
    if raw['stage'] != base['stage'] or raw['original_sha256'] != base['original_sha256']:
        raise ValueError('Augmented stage/original differs from baseline')
    pairs = raw['accepted_pairs']; _verify_parents(pairs, base)
    selected, rejected = select_pairs(pairs, base['train'], raw['recipe'])
    if rejected or selected != pairs or raw['train'] != base['train'] + pair_rows(pairs):
        raise ValueError('Augmented rows/evidence/caps differ from reviewed baseline additions')
    train = _records_from_raw(raw['train'])
    provenance = info['provenance']
    new_rows = pair_rows(pairs)
    provenance['selected_synthetic_ids'] += [r['ID'] for r in new_rows]
    provenance['train_rows'] += [{'id': r['ID'], 'kind': 'synthetic', 'parent_ids': r['Provenance']['parent_ids'],
                                 'authored_provenance': r['Provenance']} for r in new_rows]
    return train, [], {**info, 'format': FORMAT, 'baseline_sha256': raw['baseline_sha256'],
                      'augmentation_rows': len(new_rows), 'accepted_error_pairs': len(pairs), 'provenance': provenance}


def load_augmented_classifier(original_path, raw, fraction, seed):
    train, _, info = load_augmented(raw, fraction, seed)
    if raw['stage'] != 'classification':
        raise ValueError('Expected augmented classification bundle')
    from model.data import load_classifier_split
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'base.json'; path.write_bytes(base64.b64decode(raw['baseline_base64']))
        load_classifier_split(original_path, path, 0, 42)
    return train, info


def export_dataset(paths, candidates, output, recipe, identity):
    raw, _ = baselines(paths, recipe)
    union = raw['classification']['train'] + raw['correction']['train']
    accepted, rejected = select_pairs(candidates, union, recipe)
    for stage in FILES: _verify_parents(accepted, raw[stage])
    output = Path(output)
    if output.exists():
        saved = json.loads((output / 'export_manifest.json').read_text())
        if (saved['source'] != identity or saved['recipe_sha256'] != digest(recipe)
                or json.loads((output / 'accepted_pairs.json').read_text()) != accepted
                or json.loads((output / 'rejected_candidates.json').read_text()) != rejected):
            raise ValueError('Published export differs from resumed inputs/review')
        for stage, entry in saved['files'].items():
            path = output / entry['filename']
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('Published dataset hash differs')
            load_augmented(json.loads(path.read_text()), 0, 42)
        archive = Path(saved['archive']['path'])
        if hashlib.sha256(archive.read_bytes()).hexdigest() != saved['archive']['sha256']:
            raise ValueError('Published archive hash differs')
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None: raise ValueError('Published ZIP failed CRC')
        return saved
    output.parent.mkdir(parents=True, exist_ok=True)
    # This peer directory belongs to this run's exporter. A hard interruption
    # can leave it behind; only completed output is ever visible to consumers.
    staging = output.with_name('.' + output.name + '-staging')
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir()
    try:
        manifest = _write_export(paths, raw, accepted, rejected, staging, output, recipe, identity)
        write_json(staging / 'export_manifest.json', manifest)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _write_export(paths, raw, accepted, rejected, staging, output, recipe, identity):
    added = pair_rows(accepted)
    manifest = {'format': FORMAT, 'status': 'completed' if accepted else 'completed_no_accepted_augmentation',
                'source': identity, 'recipe_sha256': digest(recipe), 'accepted_error_pairs': len(accepted),
                'added_clean_controls': sum(r['Output']['Error_Type_Id'] == 0 for r in added),
                'new_rows_per_stage': len(added), 'rejected_candidates': len(rejected), 'files': {},
                'independent_validation_rows': 0, 'human_adjudicated': False,
                'review_status': 'mechanical_gates_and_independent_model_review',
                'privacy': 'Private noncommercial derivative; organizer access upon request',
                'training_consumer': 'vgec-augmented-final-v1 supported on kaggle-augment; update training-branch consumer and committed hashes deliberately before use'}
    for stage, filename in FILES.items():
        payload = Path(paths[stage]).read_bytes()
        value = {'format': FORMAT, 'stage': stage, 'seed': 42, 'validation_fraction': 0, 'validation': [],
                 'original_sha256': raw[stage]['original_sha256'], 'baseline_base64': base64.b64encode(payload).decode(),
                 'baseline_sha256': hashlib.sha256(payload).hexdigest(), 'recipe': recipe,
                 'accepted_pairs': accepted, 'train': raw[stage]['train'] + added}
        path = staging / filename; write_json(path, value)
        load_augmented(value, 0, 42)
        write_json(staging / f'{stage}_train.json', value['train'])
        manifest['files'][stage] = {'path': str(output / filename), 'filename': filename, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                    'baseline_rows': len(raw[stage]['train']), 'total_rows': len(value['train']),
                                    'class_counts': {str(k): v for k, v in Counter(r['Output']['Error_Type_Id'] for r in value['train']).items()}}
    if paths.get('original'):
        shutil.copy2(paths['original'], staging / 'VGEC 2026_v1.json')
    write_json(staging / 'accepted_pairs.json', accepted)
    write_json(staging / 'rejected_candidates.json', rejected)
    write_json(staging / 'training_manifest.json', manifest)
    archive = staging / 'VGEC26-augmented-full-training.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(staging.glob('*.json')): z.write(path, path.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise ValueError('Augmentation ZIP failed CRC')
        for name in z.namelist():
            if z.read(name) != (staging / name).read_bytes(): raise ValueError('Archive member differs')
    manifest['archive'] = {'path': str(output / archive.name), 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'bytes': archive.stat().st_size}
    return manifest
