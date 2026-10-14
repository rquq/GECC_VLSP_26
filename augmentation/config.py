"""Validate the committed bounded augmentation recipe before any model load."""
import json
import math
from pathlib import Path


def validate_recipe(recipe):
    if recipe.get('seed') != 42 or type(recipe.get('seed')) is not int:
        raise ValueError('Augmentation seed must be42')
    run = recipe['run']
    if run['device'] not in ('cuda', 'cpu'):
        raise ValueError('Augmentation device must be cuda or cpu')
    for key, cap in [('max_calls', 1344), ('max_wall_minutes', 360)]:
        if type(run[key]) is not int or not 0 < run[key] <= cap: raise ValueError(f'Invalid bounded {key}')
    parents = recipe['parents']
    if type(parents['limit']) is not int or not 0 < parents['limit'] <= 96:
        raise ValueError('At most96 original parents are allowed')
    if parents['proposals_per_reference'] != 2 or not parents['pivots'] or len(set(parents['pivots'])) != len(parents['pivots']) or not set(parents['pivots']) <= {'English', 'Chinese'}:
        raise ValueError('Use two proposals and the committed English/Chinese pivots')
    for label, maximum in {'1':24, '2':64, '3':64, '4':48}.items():
        value = recipe['error_caps'][label]
        if type(value) is not int or not 0 <= value <= maximum: raise ValueError('Error quota exceeds bounded recipe')
    q = recipe['quality']
    if type(q['minimum_confidence']) not in (int, float) or not math.isfinite(q['minimum_confidence']) or not .9 <= q['minimum_confidence'] <= 1:
        raise ValueError('Review confidence threshold must be finite and at least0.9')
    if type(q['max_edit_fraction']) not in (int, float) or not 0 < q['max_edit_fraction'] <= .35:
        raise ValueError('Edits may cover at most35% of the sentence')
    for key, cap in [('max_parent_errors',4),('max_reference_errors',2),('max_sentence_chars',600)]:
        if type(q[key]) is not int or not 0 < q[key] <= cap: raise ValueError('Invalid quality cap')
    for role in ('translator', 'author', 'reviewer'):
        m = recipe['models'][role]
        if not isinstance(m['name'], str) or not isinstance(m['revision'], str) or not m['revision'] or type(m['parameters']) is not int or not 0 < m['parameters'] <= 8_000_000_000:
            raise ValueError('Each model needs a pinned native size<=8B')
    if recipe['models']['author']['name'] == recipe['models']['reviewer']['name']:
        raise ValueError('Independent reviewer must differ from author')
    for role in ('classification','correction'):
        if type(recipe['data']['counts'][role]) is not int or recipe['data']['counts'][role] < 1:
            raise ValueError('Baseline counts must be explicit positive integers')
    return recipe


def load_recipe(path): return validate_recipe(json.loads(Path(path).read_text()))
