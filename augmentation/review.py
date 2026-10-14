"""Pure review gates and quotas; author claims never establish acceptance."""
from collections import Counter
import hashlib
import json
import math
import re
import unicodedata

from model.data import _id_key, _source_key, _valid_label
from tools.local_teacher import MECHANISMS

FUNCTION_WORDS = set('là đã đang sẽ mà và nhưng hoặc của cho với từ tại đến ở vào bằng về vì nên nếu thì tuy dù cũng để do bởi được bị còn nữa rất quá hãy chớ đừng không chưa chẳng'.split())


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def words(text):
    return re.findall(r'\w+', _source_key(text), re.UNICODE)


def punctuation(text):
    return ''.join(c for c in text if unicodedata.category(c).startswith('P'))


def protected(text):
    numbers = Counter(re.findall(r'\d+(?:[.,]\d+)*', text))
    names = Counter(re.findall(r'\b(?:[A-ZĐ][\wÀ-ỹ]*\s+){1,}[A-ZĐ][\wÀ-ỹ]*\b', text))
    return numbers, names


def valid_confidence(value, minimum):
    return type(value) in (int, float) and math.isfinite(value) and minimum <= value <= 1


def apply_patch(reference, patch):
    clean, error = patch.get('clean_span'), patch.get('error_span')
    if not isinstance(clean, str) or not clean or not isinstance(error, str) or reference.count(clean) != 1 or clean == error:
        raise ValueError('Patch needs one unique nonempty clean span and a different replacement')
    return reference.replace(clean, error, 1)


def validate_pair(pair, recipe, *, require_reviews=True):
    """Return reasons, never coerce labels/flags or infer missing review evidence."""
    reasons = []
    try:
        label = pair['label']; _id_key(pair['parent_id'])
        if not _valid_label(label) or label == 0 or pair['mechanism'] not in MECHANISMS[label]:
            return ['unknown_label_or_mechanism']
        quality = recipe['quality']
        reference, source, parent = pair['reference'], pair['source'], pair['parent_clean']
        if any(not isinstance(x, str) or not x.strip() or len(x) > quality['max_sentence_chars'] for x in (reference, source, parent)):
            return ['invalid_sentence']
        if any(x.lstrip().startswith('(') or '...' in x or '…' in x for x in (reference, source, parent)):
            reasons.append('annotation_or_incomplete_sentence')
        if _source_key(reference) == _source_key(parent):
            reasons.append('unchanged_roundtrip')
        if protected(reference) != protected(parent) or protected(source) != protected(reference):
            reasons.append('names_or_numbers_changed')
        patch = pair['patch']
        if apply_patch(reference, patch) != source:
            reasons.append('source_differs_from_patch')
        if max(len(patch['clean_span']), len(patch['error_span'])) / len(reference) > quality['max_edit_fraction']:
            reasons.append('overbroad_edit')
        if _source_key(source) == _source_key(reference):
            reasons.append('identity_error')
        original_words, error_words = words(reference), words(source)
        if label == 4:
            if original_words != error_words or punctuation(reference) == punctuation(source):
                reasons.append('not_punctuation_only')
        else:
            if punctuation(reference) != punctuation(source):
                reasons.append('mixed_punctuation_error')
            if label == 2 and Counter(original_words) != Counter(error_words):
                reasons.append('phrase_edit_changes_vocabulary')
            if label == 3:
                changed = (Counter(original_words) - Counter(error_words)) + (Counter(error_words) - Counter(original_words))
                if not changed or not set(changed) <= FUNCTION_WORDS:
                    reasons.append('function_edit_changes_content')
            if label == 1 and Counter(error_words) - Counter(original_words):
                reasons.append('structure_edit_adds_vocabulary')
        models = pair['models']
        for role in ('translator', 'author', 'reviewer'):
            model = models[role]
            if model != recipe['models'][role] or type(model['parameters']) is not int or not 0 < model['parameters'] <= 8_000_000_000:
                reasons.append('model_identity_or_size')
        if models['author']['name'] == models['reviewer']['name']:
            reasons.append('reviewer_is_author')
        if pair['pivot'] not in recipe['parents']['pivots']:
            reasons.append('unknown_pivot')
        calls = pair['calls']
        required_calls = ('forward', 'backward', 'author') + (('clean_review', 'error_review') if require_reviews else ())
        if any(not isinstance(calls.get(key), str) or not calls[key] for key in required_calls):
            reasons.append('missing_call_provenance')
        if not require_reviews:
            return sorted(set(reasons))
        clean, review = pair['clean_review'], pair['error_review']
        if recipe.get('recovery'):
            if isinstance(review.get('evidence'), str) and words(review['evidence']) == words(source):
                reasons.append('reviewer_evidence_copies_source')
            try:
                span, replacement = review['source_span'], review['replacement']
                correction = apply_patch(source, {'clean_span': span, 'error_span': replacement})
                if (correction != review['correction'] or
                        max(len(span), len(replacement)) / len(source) > quality['max_edit_fraction']):
                    reasons.append('reviewer_patch_invalid')
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                reasons.append('reviewer_patch_invalid')
        if (not _valid_label(clean.get('parent_label')) or clean['parent_label'] != 0
                or not _valid_label(clean.get('translation_label')) or clean['translation_label'] != 0
                or clean.get('meaning_preserved') is not True or clean.get('names_and_numbers_preserved') is not True
                or not valid_confidence(clean.get('confidence'), quality['minimum_confidence'])
                or not isinstance(clean.get('evidence'), str) or len(clean['evidence'].strip()) < 10):
            reasons.append('clean_semantics_review_failed')
        if (not _valid_label(review.get('source_label')) or review['source_label'] != label
                or review.get('mechanism') != pair['mechanism'] or review.get('single_error') is not True or review.get('required_edit') is not True
                or review.get('meaning_preserved') is not True or review.get('other_error_types') != []
                or not valid_confidence(review.get('confidence'), quality['minimum_confidence'])
                or not isinstance(review.get('correction'), str) or _source_key(review['correction']) != _source_key(reference)
                or not isinstance(review.get('evidence'), str) or len(review['evidence'].strip()) < 10):
            reasons.append('blind_error_review_failed')
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        reasons.append('malformed_candidate')
    return sorted(set(reasons))


def pair_rows(pairs):
    """Emit one clean control per unique reference, retaining paired family IDs."""
    rows, clean_seen = [], set()
    for pair in pairs:
        identity = digest(pair)
        family = 'parent:' + digest([type(pair['parent_id']).__name__, pair['parent_id']])[:24]
        common = {'origin': 'roundtrip_cross_model_review', 'parent_ids': [pair['parent_id']],
                  'family_id': family, 'review_status': 'automated_independent_review_not_human_adjudicated',
                  'evidence_sha256': identity}
        key = _source_key(pair['reference'])
        if key not in clean_seen:
            clean_seen.add(key)
            rows.append({'ID': 'aug:clean:' + digest(key)[:24], 'Input': pair['reference'],
                         'Output': {'Error_Type_Id': 0, 'Fixed Sentence': pair['reference']},
                         'Provenance': {**common, 'template_rule': 'matched_clean_control'}})
        rows.append({'ID': 'aug:error:' + identity[:24], 'Input': pair['source'],
                     'Output': {'Error_Type_Id': pair['label'], 'Fixed Sentence': pair['reference']},
                     'Provenance': {**common, 'template_rule': pair['mechanism']}})
    return rows


def select_pairs(candidates, baseline_rows, recipe):
    accepted, rejected = [], []
    seen = {_source_key(r['Input']) for r in baseline_rows}
    references, parents, classes, mechanisms = Counter(), Counter(), Counter(), Counter()
    for pair in candidates:
        reasons = validate_pair(pair, recipe)
        if not reasons:
            label, mechanism = pair['label'], pair['mechanism']
            parent, reference = _id_key(pair['parent_id']), _source_key(pair['reference'])
            source = _source_key(pair['source'])
            cap = recipe['error_caps'][str(label)]
            if classes[label] >= cap or mechanisms[label, mechanism] >= math.ceil(cap / len(MECHANISMS[label])):
                reasons.append('class_or_mechanism_cap')
            if parents[parent] >= recipe['quality']['max_parent_errors'] or references[reference] >= recipe['quality']['max_reference_errors']:
                reasons.append('parent_or_reference_cap')
            if source in seen or reference in seen and references[reference] == 0:
                reasons.append('duplicate_or_baseline_collision')
            if not reasons:
                accepted.append(pair); seen.add(source); seen.add(reference)
                classes[label] += 1; mechanisms[label, mechanism] += 1
                parents[parent] += 1; references[reference] += 1
        if reasons:
            rejected.append({'candidate_sha256': digest(pair), 'reasons': reasons, 'candidate': pair})
    return accepted, rejected
