"""Verified final-fit datasets: source training plus every former holdout row."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import tempfile

FORMAT = 'vgec-final-training-v1'


def source_bytes(raw):
    payload = base64.b64decode(raw['source_bundle_base64'], validate=True)
    if hashlib.sha256(payload).hexdigest() != raw['source_bundle_sha256']:
        raise ValueError('Final training source bundle digest differs')
    return payload


def load_final(raw, fraction, seed):
    from .data import _records_from_raw, _load_legacy_bundle
    if fraction != 0 or raw.get('validation_fraction') != 0 or seed != 42 or raw.get('seed') != 42:
        raise ValueError('Final training requires seed42 and validation_fraction=0')
    if raw.get('stage') not in ('classification', 'correction') or raw.get('validation') != []:
        raise ValueError('Final training has no independent local validation')
    source = json.loads(source_bytes(raw))
    if source.get('format') == 'vgec-reviewed-v4':
        from .reviewed_data import load_reviewed
        _, _, info = load_reviewed(source, source['validation_fraction'], seed)
        if info['provenance']['stage'] != raw['stage']:
            raise ValueError('Final training stage differs from its reviewed source')
    else:
        _, _, info = _load_legacy_bundle(source, source['validation_fraction'], seed)
    if raw['train'] != source['train'] + source['validation']:
        raise ValueError('Final training must preserve exact source train and former validation rows')
    if raw.get('original_sha256') != source.get('original_sha256'):
        raise ValueError('Final training original digest differs from source')
    train = _records_from_raw(raw['train'])
    provenance = copy.deepcopy(info['provenance'])
    added = [r['ID'] for r in source['validation']]
    provenance['retained_original_train_ids'] += added
    provenance['original_train_ids'] += added
    provenance['train_rows'] += [{'id': i, 'kind': 'original', 'original_id': i} for i in added]
    return train, [], {'format': FORMAT, 'provenance': provenance,
                      'original_sha256': raw['original_sha256'], 'stage': raw['stage'],
                      'source_bundle_sha256': raw['source_bundle_sha256'],
                      'former_validation_rows_added': len(added), 'independent_validation_rows': 0}


def load_final_classifier(original_path, raw, fraction, seed):
    from .data import load_classifier_split
    train, _, info = load_final(raw, fraction, seed)
    if raw['stage'] != 'classification':
        raise ValueError('Classifier requires a classification final-training bundle')
    if hashlib.sha256(Path(original_path).read_bytes()).hexdigest() != raw['original_sha256']:
        raise ValueError('Final classifier original digest differs')
    # Reuse the complete existing source contract, including exact original
    # fields, parent groups and annotation quarantine, without relaxing it.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'source.json'
        path.write_bytes(source_bytes(raw))
        source = json.loads(path.read_text())
        load_classifier_split(original_path, path, source['validation_fraction'], seed)
    return train, info
