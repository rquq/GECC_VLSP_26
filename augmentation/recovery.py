"""Review saved eligible generations in a fresh journal, without generating again."""
import copy
import hashlib
import json
import re
from pathlib import Path
import zipfile

from model.data import _records_from_raw, _source_key
from model.runtime import write_json
from tools.local_teacher import MECHANISMS
from .data import baselines, export_dataset, select_parents, _verify_parents
from .journal import BudgetExhausted
from .pipeline import parse_object, request_review, review_schema_valid
from .review import apply_patch, digest, validate_pair, valid_confidence, words

FORMAT = 'vgec-augmentation-recovery-v1'
TAXONOMY = (
    'Đánh giá ngữ pháp tiếng Việt theo đúng 5 mã: 0 = câu đúng; '
    '1 = lỗi cấu trúc toàn câu, thiếu chủ ngữ/vị ngữ/mệnh đề chính; '
    '2 = lỗi cấu trúc bên trong cụm từ; 3 = lỗi hư từ (giới từ/liên từ/trợ từ); '
    '4 = lỗi dấu câu. Không coi sửa phong cách, thay từ đồng nghĩa, chính tả hoặc '
    'thêm đã/đang/sẽ tùy ý là sửa lỗi ngữ pháp bắt buộc. '
    'Nếu một giới từ đầu câu biến toàn bộ chủ ngữ thành trạng ngữ và câu mất chủ ngữ, '
    'xét lỗi cấu trúc toàn câu. Không đoán chắc khi có nhiều cách hiểu.'
)


def clean_prompt(parent, reference):
    return (TAXONOMY + '\nKiểm tra HAI câu dưới đây. Không dịch sang tiếng Anh. '
            'parent_label là mã lỗi SỐ NGUYÊN 0..4 của câu A; translation_label là '
            'mã lỗi SỐ NGUYÊN 0..4 của câu B, KHÔNG phải bản dịch. '
            'Hai câu phải cùng sự việc, chủ thể, phủ định, số lượng và tên riêng. '
            'Chỉ trả một JSON object với đúng các trường và kiểu: '
            'parent_label: integer; translation_label: integer; meaning_preserved: boolean; '
            'names_and_numbers_preserved: boolean; confidence: number 0..1; evidence: string '
            '(giải thích bằng một câu tiếng Việt, tối đa 35 từ). confidence là độ chắc chắn '
            'của nhận định, KHÔNG phải điểm giống nhau của hai câu. Không có Markdown hoặc nội dung ngoài JSON. '
            'Nêu rõ khác nghĩa hoặc câu lỗi; không mặc định hai câu đều đúng.\n'
            'Câu A:\n' + parent + '\nCâu B:\n' + reference)


def error_prompt(source):
    return (TAXONOMY + '\nChỉ xem câu được cung cấp, tự xác định mã lỗi và sửa tối thiểu. '
            'Giữ từ vựng, ý nghĩa, tên và số; chỉ sửa lỗi ngữ pháp bắt buộc. '
            'KHÔNG sinh lại toàn bộ câu. Chỉ chỉ ra MỘT đoạn ngắn cần thay trong câu gốc '
            '(source_span, chép nguyên văn và xuất hiện đúng một lần), và đoạn thay thế (replacement). '
            'Chọn đoạn sửa NGẮN NHẤT có thể; 35% chỉ là giới hạn trên, không phải độ dài nên dùng. '
            'Ưu tiên xóa từ thừa trước khi thêm từ mới. Nếu giới từ đầu câu làm mất chủ ngữ, '
            'thử xóa giới từ thừa; không nhắc lại chủ thể đã có trong câu. '
            'Có thể dùng replacement="" để xóa đoạn thừa. '
            'Nếu câu đã đúng: source_label=0, mechanism="none", source_span="", replacement="", '
            'single_error=false, required_edit=false. '
            'Nếu có lỗi, chọn ĐÚNG MỘT tên mechanism tiếng Anh trong danh mục: '
            + json.dumps(MECHANISMS, ensure_ascii=False) + '. '
            'Chỉ trả một JSON object, không Markdown: source_label: integer 0..4; '
            'mechanism: string; source_span: string; replacement: string; '
            'single_error: boolean; required_edit: boolean; meaning_preserved: boolean; '
            'other_error_types: array of strings (rỗng nếu không có lỗi khác); '
            'confidence: number 0..1 (độ chắc chắn của nhận định); evidence: string '
            '(một câu tiếng Việt, tối đa 35 từ). '
            'Không trả cả câu hoặc giải thích vào source_span/replacement.\nCâu:\n' + source)


def error_schema(source, max_edit_fraction=.35):
    """Constrain the teacher to unique, bounded substrings, without a target."""
    limit = int(len(source) * max_edit_fraction)
    words = list(re.finditer(r'\S+', source))
    spans = {''}
    for i, word in enumerate(words):
        for last in words[i:i + 6]:
            for end in (last.end(), last.end() + 1):
                span = source[word.start():end]
                if 0 < len(span) <= limit and source.count(span) == 1:
                    spans.add(span)
    # Punctuation errors can require replacing a single attached character.
    spans.update(c for c in source if not c.isalnum() and not c.isspace() and source.count(c) == 1)
    properties = {
        'source_label': {'type': 'integer', 'enum': list(range(5))},
        'mechanism': {'type': 'string', 'enum': ['none'] + [m for ms in MECHANISMS.values() for m in ms]},
        'source_span': {'type': 'string', 'enum': sorted(spans)},
        'replacement': {'type': 'string', 'maxLength': limit},
        'single_error': {'type': 'boolean'}, 'required_edit': {'type': 'boolean'},
        'meaning_preserved': {'type': 'boolean'},
        'other_error_types': {'type': 'array', 'items': {'type': 'string'}},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'evidence': {'type': 'string', 'maxLength': 240},
    }
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def request_error(caller, key, source, max_edit_fraction=.35):
    """A blind model chooses the edit; code only copies/replaces its exact span."""
    prompt, ids = error_prompt(source), []
    for attempt in range(2):
        call = caller.call('reviewer', key + (':format_retry' if attempt else ''), prompt, False,
                           response_schema=error_schema(source, max_edit_fraction))
        ids.append(call['id'])
        review = parse_object(call['text'])
        try:
            span, replacement = review['source_span'], review['replacement']
            if not isinstance(span, str) or not isinstance(replacement, str):
                raise ValueError('Edit spans must be strings')
            if review['source_label'] == 0:
                if span or replacement: raise ValueError('A clean verdict must not edit the source')
                correction = source
            else:
                if max(len(span), len(replacement)) / len(source) > max_edit_fraction:
                    raise ValueError('Reviewer edit is overbroad')
                correction = apply_patch(source, {'clean_span': span, 'error_span': replacement})
            review = {**review, 'correction': correction}
            explanation = review.get('evidence', '')
            has_evidence = (review.get('source_label') == 0 or
                            (isinstance(explanation, str) and len(explanation.strip()) >= 10
                             and words(explanation) != words(source)))
            if review_schema_valid('error', review) and has_evidence:
                return review, ids
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass
        prompt = error_prompt(source) + '\nĐầu ra trước chưa đúng schema/đoạn sửa. Trả đúng một JSON object, '
        prompt += ('source_span phải được chép chính xác từ câu gốc, duy nhất và ngắn; không viết lại toàn bộ câu. '
                   'Nếu có lỗi, evidence phải giải thích ngữ pháp (ít nhất 10 ký tự), không chỉ chép lại câu gốc.')
    return {}, ids


def mechanical_pool(candidates, recipe):
    eligible, rejected = [], []
    for original in candidates:
        candidate = copy.deepcopy(original)
        candidate['models']['reviewer'] = recipe['models']['reviewer']
        candidate.pop('clean_review', None); candidate.pop('error_review', None)
        candidate['calls'] = {k: v for k, v in candidate['calls'].items() if k in ('forward', 'backward', 'author')}
        reasons = validate_pair(candidate, recipe, require_reviews=False)
        if reasons:
            rejected.append({'candidate_sha256': digest(original), 'reasons': reasons})
        else:
            candidate['source_candidate_sha256'] = digest(original)
            eligible.append(candidate)
    return eligible, rejected


def load_payload(path, expected_sha256, recipe, inputs):
    payload_bytes = Path(path).read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != expected_sha256:
        raise ValueError('Recovery payload digest differs from committed recipe')
    payload = json.loads(payload_bytes)
    if payload['format'] != FORMAT:
        raise ValueError('Unknown saved-generation recovery format')
    original = payload['recipe']
    for role in ('translator', 'author'):
        if original['models'][role] != recipe['models'][role]:
            raise ValueError('Recovery must preserve the original translator and author')
    for role in ('original', 'classification', 'correction'):
        actual = hashlib.sha256(Path(inputs[role]).read_bytes()).hexdigest()
        if actual != recipe['data']['sha256'][role] or actual != payload['baseline_sha256'][role]:
            raise ValueError('Recovery baseline differs from saved generation inputs')
    raw, _ = baselines(inputs, recipe)
    for stage in ('classification', 'correction'):
        _verify_parents(payload['candidates'], raw[stage])
    calls = {}
    for call in payload['calls']:
        if (call['entry_sha256'] != digest({k: v for k, v in call.items() if k != 'entry_sha256'})
                or call['output_sha256'] != digest(call['text']) or call['id'] in calls):
            raise ValueError('Saved generation journal digest/identity differs')
        identity = digest({'role': call['role'], 'key': call['key'], 'prompt': call['prompt'],
                           'sampling': call['sampling'], 'options': call['settings'],
                           'model': call['model'], 'seed': original['seed']})
        if call['id'] != identity:
            raise ValueError('Saved generation request identity differs')
        calls[call['id']] = call
    for p in payload['candidates']:
        for key, role in [('forward', 'translator'), ('backward', 'translator'), ('author', 'author')]:
            c = calls[p['calls'][key]]
            if c['role'] != role or c['model'] != original['models'][role] or c['status'] != 'completed':
                raise ValueError('Candidate generation model/call evidence differs')
        authored = parse_object(calls[p['calls']['author']]['text'])
        forward = calls[p['calls']['forward']]
        backward = calls[p['calls']['backward']]
        expected_key = digest([type(p['parent_id']).__name__, p['parent_id'], p['pivot']])
        if (p['key'] != expected_key
                or forward['key'] != expected_key + ':forward' or backward['key'] != expected_key + ':backward'
                or forward['prompt'] != f'Translate the following Vietnamese text into {p["pivot"]}. Preserve meaning, names and numbers. Output only the translation, without explanations.\n\n{p["parent_clean"]}'
                or backward['prompt'] != f'Translate the following {p["pivot"]} text into Vietnamese. Preserve meaning, names and numbers; use grammatical Vietnamese and retain the original factual content. Output only the translation, without explanations.\n\n{forward["text"]}'):
            raise ValueError('Candidate parent/pivot differs from saved translation prompts')
        if (calls[p['calls']['backward']]['text'] != p['reference']
                or not authored or authored.get('applicable') is not True
                or any(authored.get(k) != p['patch'][k] for k in ('clean_span', 'error_span'))
                or f'Create exactly one minimal error of type {p["label"]}: {p["mechanism"]}.'
                   not in calls[p['calls']['author']]['prompt']
                or not calls[p['calls']['author']]['prompt'].endswith('\nReference:\n' + p['reference'])):
            raise ValueError('Candidate differs from eligible author/translation output')
    return payload


def resolve_payload(input_root, work, recipe):
    filename = 'augmentation_recovery.json'
    expected = recipe['recovery']['sha256']
    root = Path(input_root)
    for path in sorted(root.rglob(filename)):
        if hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            return path
    for path in sorted(root.rglob(recipe['data']['archive'])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != recipe['recovery']['archive_sha256']:
            continue
        with zipfile.ZipFile(path) as z:
            info = z.getinfo(filename)
            if info.file_size > 32 * 1024**2:
                raise ValueError('Recovery payload exceeds the bounded input size')
            content = z.read(filename)
            baseline_names = ('VGEC 2026_v1.json', 'VGEC_classifier.json', 'VGEC_balanced.json')
            baselines = {}
            for name in baseline_names:
                if z.getinfo(name).file_size > 32 * 1024**2:
                    raise ValueError('Recovery baseline exceeds the bounded input size')
                baselines[name] = z.read(name)
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('Recovery archive payload digest differs')
        target = Path(work) / 'recovery_input' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        for name, data in baselines.items():
            (target.parent / name).write_bytes(data)
        return target
    raise ValueError('Attach the committed private recovery ZIP; no matching recovery payload was found')


def pilot_cases(inputs, recipe):
    raw, _ = baselines(inputs, recipe)
    parents = sorted((p for p in select_parents(inputs, recipe)
                      if p['original_label'] == 0 and _source_key(p['source']) == _source_key(p['clean'])),
                     key=lambda p: (len(p['clean']), str(p['id'])))
    if not parents:
        raise ValueError('No complete retained originals for the reviewer check')
    clean = parents[:4]
    cases = []
    for p in clean[:2]:
        cases.append({'kind': 'clean', 'parent': p['clean'], 'reference': p['clean'],
                      'expected_meaning': True, 'origin_ids': [p['id']]})
        cases.append({'kind': 'error', 'source': p['clean'], 'expected_label': 0,
                      'expected_correction': p['clean'], 'origin_ids': [p['id']]})
    if len(clean) >= 4:
        for a, b in [(clean[0], clean[2]), (clean[1], clean[3])]:
            cases.append({'kind': 'clean', 'parent': a['clean'], 'reference': b['clean'],
                          'expected_meaning': False, 'origin_ids': [a['id'], b['id']]})
    anchors = []
    for row in _records_from_raw(raw['correction']['train']):
        if type(row['id']) is not int or row['label'] != 1:
            continue
        for word in ('Qua', 'Bằng', 'Với'):
            if (row['source'].startswith(word + ' ')
                    and _source_key(row['target']) == _source_key(row['source'][len(word) + 1:])):
                anchors.append(row); break
    for row in sorted(anchors, key=lambda r: len(r['source']))[:2]:
        cases.append({'kind': 'error', 'source': row['source'], 'expected_label': row['label'],
                      'expected_correction': row['target'], 'origin_ids': [row['id']]})
    return cases


def check_reviewer(caller, cases):
    results = []
    for i, case in enumerate(cases):
        kind = case['kind']
        prompt = clean_prompt(case['parent'], case['reference']) if kind == 'clean' else error_prompt(case['source'])
        if kind == 'clean':
            review, call_ids = request_review(caller, f'pilot:{i}:{kind}', prompt, kind)
        else:
            review, call_ids = request_error(caller, f'pilot:{i}:{kind}', case['source'])
        if kind == 'clean':
            # A changed-fact control must be rejected, even if the reviewer also
            # finds a grammar issue. It is not a second grammar gold standard.
            passed = review.get('meaning_preserved') is case['expected_meaning']
            if case['expected_meaning']:
                passed = (passed and review.get('parent_label') == 0 and review.get('translation_label') == 0
                          and review.get('names_and_numbers_preserved') is True)
        else:
            label = case['expected_label']
            passed = (review.get('source_label') == label
                      and review.get('required_edit') is (label != 0)
                      and review.get('single_error') is (label != 0)
                      and review.get('meaning_preserved') is True and review.get('other_error_types') == []
                      and review.get('mechanism') in (['none'] if label == 0 else MECHANISMS[label])
                      and _source_key(review.get('correction', '')) == _source_key(case['expected_correction']))
        minimum = 0 if kind == 'clean' and not case['expected_meaning'] else .9
        evidence_minimum = 0 if kind == 'error' and case['expected_label'] == 0 else 10
        passed = bool(passed and valid_confidence(review.get('confidence'), minimum)
                      and len(review.get('evidence', '').strip()) >= evidence_minimum)
        results.append({'case': case, 'review': review, 'calls': call_ids, 'passed': passed})
        if not passed:
            break
    return {'passed': len(results) == len(cases) and all(r['passed'] for r in results),
            'scope': 'Small format/negative-control and recorded-original check; not independent quality validation',
            'planned_cases': len(cases), 'cases': results}


def clean_passes(review, recipe):
    return (review.get('parent_label') == 0 and review.get('translation_label') == 0
            and review.get('meaning_preserved') is True and review.get('names_and_numbers_preserved') is True
            and valid_confidence(review.get('confidence'), recipe['quality']['minimum_confidence'])
            and len(review.get('evidence', '').strip()) >= 10)


def run_recovery(inputs, recipe, caller, output, identity, candidates, provenance):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    pool, rejected = mechanical_pool(candidates, recipe)
    write_json(output / 'mechanical_rejections.json', rejected)
    write_json(output / 'recovery_pool.json', pool)
    print(f'Recovery: {len(pool)}/{len(candidates)} saved candidates pass mechanical checks; no translation/authoring', flush=True)
    pilot = check_reviewer(caller, pilot_cases(inputs, recipe))
    write_json(output / 'reviewer_check.json', pilot)
    if not pilot['passed']:
        result = {'format': 'vgec-augmentation-recovery-result-v1', 'status': 'reviewer_check_failed',
                  'mechanical_candidates': len(pool), 'accepted_error_pairs': 0, 'new_rows_per_stage': 0,
                  'recovery': provenance, 'diagnostic': str(output / 'reviewer_check.json')}
        write_json(output / 'pipeline_manifest.json', result)
        print('Reviewer check failed early. No dataset produced; inspect reviewer_check.json.', flush=True)
        return result
    completed, clean_reviews, exhausted = [], {}, False
    try:
        for i, p in enumerate(pool):
            key = p['key']
            if key not in clean_reviews:
                clean_reviews[key] = request_review(caller, key + ':clean_review', clean_prompt(p['parent_clean'], p['reference']), 'clean')
            clean, clean_calls = clean_reviews[key]
            calls = {**p['calls'], 'clean_review': clean_calls[-1]}
            if clean_passes(clean, recipe):
                error, error_calls = request_error(caller, digest([key, p['label'], p['mechanism']]) + ':error_review', p['source'], recipe['quality']['max_edit_fraction'])
                calls['error_review'] = error_calls[-1]
            else:
                error, error_calls = {}, []
            completed.append({**p, 'clean_review': clean, 'error_review': error, 'calls': calls,
                              'review_call_ids': {'clean': clean_calls, 'error': error_calls}, 'recovery': provenance})
            print(f'Recovery review {i + 1}/{len(pool)}', flush=True)
    except BudgetExhausted:
        exhausted = True
    write_json(output / 'reviewed_candidates.json', completed)
    result = export_dataset(inputs, completed, output / 'dataset', recipe, identity)
    result.update(recovery=provenance, mechanical_candidates=len(pool), mechanical_rejections=len(rejected),
                  fully_reviewed_candidates=len(completed), budget_exhausted=exhausted,
                  translation_calls=0, authoring_calls=0)
    write_json(output / 'pipeline_manifest.json', result)
    return result
