"""Translation, error authoring and independent blind review, in three model phases."""
import json
from pathlib import Path

from model.data import _source_key
from model.runtime import write_json
from tools.local_teacher import TAXONOMY, MECHANISMS
from .data import baselines, select_parents, export_dataset
from .journal import BudgetExhausted
from .review import apply_patch, digest, protected


def parse_object(text):
    if not isinstance(text, str): return None
    if text.startswith('```') and text.endswith('```'):
        lines = text.splitlines(); text = '\n'.join(lines[1:-1])
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError): return None


def run_pipeline(inputs, recipe, caller, output, identity):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    raw, _ = baselines(inputs, recipe)
    parents = select_parents(inputs, recipe)
    write_json(output / 'selected_parents.json', parents)
    baseline_text = {_source_key(r[field]) for rows in raw.values() for r in rows['train'] for field in ('Input',)}
    baseline_text |= {_source_key(r['Output']['Fixed Sentence']) for rows in raw.values() for r in rows['train'] if isinstance(r['Output']['Fixed Sentence'], str)}
    variants, proposals, completed, rejected = [], [], [], []
    seen = set(baseline_text)
    exhausted = False
    try:
        print(f'Augmentation: translating {len(parents)} parents through {len(recipe["parents"]["pivots"])} pivots', flush=True)
        for parent in parents:
            for pivot in recipe['parents']['pivots']:
                key = digest([type(parent['id']).__name__, parent['id'], pivot])
                forward = caller.call('translator', key + ':forward', f'Translate the following Vietnamese text into {pivot}. Preserve meaning, names and numbers. Output only the translation, without explanations.\n\n{parent["clean"]}', False)
                pivot_text = forward['text']
                if not isinstance(pivot_text, str) or not pivot_text.strip():
                    rejected.append({'phase': 'translation', 'parent_id': parent['id'], 'reason': 'empty_forward', 'call_id': forward['id']}); continue
                backward = caller.call('translator', key + ':backward', f'Translate the following {pivot} text into Vietnamese. Preserve meaning, names and numbers; use grammatical Vietnamese and retain the original factual content. Output only the translation, without explanations.\n\n{pivot_text}', True)
                reference = backward['text']
                if (not isinstance(reference, str) or not 20 <= len(reference) <= recipe['quality']['max_sentence_chars']
                        or _source_key(reference) in seen or protected(reference) != protected(parent['clean'])
                        or reference.lstrip().startswith('(') or '...' in reference or '…' in reference):
                    rejected.append({'phase': 'translation', 'parent_id': parent['id'], 'reason': 'duplicate_incomplete_or_changed_protected_content', 'call_id': backward['id']}); continue
                seen.add(_source_key(reference))
                variants.append({'parent_id': parent['id'], 'parent_clean': parent['clean'], 'pivot': pivot,
                                 'reference': reference, 'key': key, 'calls': {'forward': forward['id'], 'backward': backward['id']}})
        write_json(output / 'translated_references.json', variants)
        print(f'Augmentation: authoring from {len(variants)} novel references', flush=True)
        # Slots favor rare phrase/function/punctuation classes while rotating all
        # six mechanisms across parents. No repeated calls to manufacture quotas.
        slots = [2, 3, 4, 2, 3, 4, 1]
        for i, variant in enumerate(variants):
            for attempt in range(recipe['parents']['proposals_per_reference']):
                slot = i * recipe['parents']['proposals_per_reference'] + attempt
                label = slots[slot % len(slots)]
                mechanism = MECHANISMS[label][(slot // len(slots)) % len(MECHANISMS[label])]
                prompt = (TAXONOMY + '\nCreate exactly one minimal error of type ' + str(label) + ': ' + mechanism + '. '
                          'Use only one contiguous span replacement. Do not change names, numbers, facts or punctuation except for type4. '
                          'Type2 must reorder existing phrase words; type3 must change only grammatical function words; type1 must delete a required structural constituent. '
                          'If this mechanism does not apply, return {"applicable":false}. Otherwise return JSON with applicable=true, clean_span, error_span and explanation. '
                          'clean_span must occur exactly once in the reference; its replacement must be the complete erroneous span. Do not rewrite the reference.\nReference:\n' + variant['reference'])
                call = caller.call('author', variant['key'] + ':author:' + str(attempt), prompt, True)
                authored = parse_object(call['text'])
                try:
                    if not authored or authored.get('applicable') is not True: raise ValueError('not_applicable_or_invalid_json')
                    patch = {k: authored[k] for k in ('clean_span', 'error_span')}
                    source = apply_patch(variant['reference'], patch)
                except (ValueError, KeyError, TypeError) as e:
                    rejected.append({'phase': 'authoring', 'parent_id': variant['parent_id'], 'reason': str(e), 'call_id': call['id']}); continue
                proposals.append({**variant, 'label': label, 'mechanism': mechanism, 'patch': patch, 'source': source,
                                  'author_explanation': authored.get('explanation'), 'models': recipe['models'],
                                  'calls': {**variant['calls'], 'author': call['id']}})
        write_json(output / 'authored_candidates.json', proposals)
        print(f'Augmentation: independently reviewing {len(proposals)} candidates', flush=True)
        clean_reviews = {}
        for proposal in proposals:
            key = proposal['key']
            if key not in clean_reviews:
                prompt = (TAXONOMY + '\nIndependently label both sentences. Check whether they express the same facts, including negation, numbers and named entities. '
                          'Return JSON: parent_label, translation_label, meaning_preserved, names_and_numbers_preserved, confidence (0..1), evidence. '
                          'Reject awkward or grammatically ambiguous sentences; do not assume either is correct.\nSentence A:\n' + proposal['parent_clean'] + '\nSentence B:\n' + proposal['reference'])
                call = caller.call('reviewer', key + ':clean_review', prompt, False)
                clean_reviews[key] = (parse_object(call['text']) or {}, call['id'])
            clean, clean_id = clean_reviews[key]
            prompt = (TAXONOMY + '\nBlindly identify the single error type and minimally correct the sentence. '
                      'No author label, mechanism or desired correction is provided. Select one exact English mechanism name independently from this inventory; do not translate its name: '
                      + json.dumps(MECHANISMS, ensure_ascii=False) + '. Return JSON: source_label, mechanism, correction (complete sentence), single_error, required_edit, '
                      'meaning_preserved, other_error_types (array), confidence (0..1), evidence explaining the required grammatical edit. '
                      'If uncertain or only stylistic, report that accurately.\nSentence:\n' + proposal['source'])
            call = caller.call('reviewer', digest([key, proposal['label'], proposal['mechanism']]) + ':error_review', prompt, False)
            completed.append({**proposal, 'clean_review': clean, 'error_review': parse_object(call['text']) or {},
                              'calls': {**proposal['calls'], 'clean_review': clean_id, 'error_review': call['id']}})
    except BudgetExhausted:
        exhausted = True
        print('Augmentation budget reached; exporting only fully reviewed candidates', flush=True)
    write_json(output / 'translated_references.json', variants)
    write_json(output / 'authored_candidates.json', proposals)
    write_json(output / 'reviewed_candidates.json', completed)
    write_json(output / 'phase_rejections.json', rejected)
    result = export_dataset(inputs, completed, output / 'dataset', recipe, identity)
    result.update(budget_exhausted=exhausted, selected_parents=len(parents), translated_references=len(variants),
                  authored_candidates=len(proposals), fully_reviewed_candidates=len(completed), phase_rejections=len(rejected))
    write_json(output / 'pipeline_manifest.json', result)
    print(json.dumps({'status': result['status'], 'accepted_error_pairs': result['accepted_error_pairs'],
                      'new_rows_per_stage': result['new_rows_per_stage'], 'budget_exhausted': exhausted}), flush=True)
    return result
