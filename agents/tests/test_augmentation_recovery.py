"""Recovery reuses eligible saved generations and fails before expensive review."""
import copy
import importlib.util
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from agents.tests.test_augmentation import FullDatasetTests, pair


class RecoveryTests(FullDatasetTests):
    def recovery(self):
        self.assertIsNotNone(importlib.util.find_spec('augmentation.recovery'),
                             'Saved-generation recovery is not implemented')
        from augmentation import recovery
        return recovery

    def test_prefilter_drops_invalid_edits_without_inference(self):
        recovery = self.recovery()
        good = pair()
        good.pop('clean_review'); good.pop('error_review')
        good['calls'] = {k: v for k, v in good['calls'].items() if k in ('forward', 'backward', 'author')}
        bad = copy.deepcopy(good)
        bad['patch'] = {'clean_span': 'useful', 'error_span': 'different'}
        bad['source'] = bad['reference'].replace('useful', 'different')
        eligible, rejected = recovery.mechanical_pool([good, bad], self.c)
        self.assertEqual(len(eligible), 1)
        self.assertIn('function_edit_changes_content', rejected[0]['reasons'])
        self.assertNotIn('clean_review', eligible[0])

    def test_failed_reviewer_check_never_reviews_candidates_or_exports(self):
        recovery = self.recovery()
        class Caller:
            def __init__(self): self.requests = []
            def call(self, role, key, prompt, sampling=False, **kwargs):
                self.requests.append(key)
                if not key.startswith('pilot:'):
                    raise AssertionError('A failed reviewer must not process the full pool')
                return {'id': key, 'text': 'Not machine-readable'}
        caller = Caller()
        out = self.root / 'recovery'
        result = recovery.run_recovery(self.paths, self.c, caller, out, {},
                                       [pair()], {'sha256': 'fixture'})
        self.assertEqual(result['status'], 'reviewer_check_failed')
        self.assertFalse((out / 'dataset').exists())
        self.assertTrue(caller.requests)

    def test_probe_rejects_valid_json_that_blindly_approves_every_pair(self):
        recovery = self.recovery()
        class Caller:
            def call(self, role, key, prompt, sampling=False, **kwargs):
                text = {'parent_label': 0, 'translation_label': 0, 'meaning_preserved': True,
                        'names_and_numbers_preserved': True, 'confidence': .99,
                        'evidence': 'Unconditional approval is not a useful reviewer.'}
                return {'id': key, 'text': json.dumps(text)}
        result = recovery.check_reviewer(Caller(), recovery.pilot_cases(self.paths, self.c))
        self.assertFalse(result['passed'])
        self.assertTrue(any(not x['passed'] for x in result['cases']))

    def test_changed_fact_control_can_reject_without_approving_grammar(self):
        recovery = self.recovery()
        review = {'parent_label': 0, 'translation_label': 4, 'meaning_preserved': False,
                  'names_and_numbers_preserved': False, 'confidence': .4,
                  'evidence': 'Fixture sentences express different facts and must be rejected.'}
        class Caller:
            def call(self, role, key, prompt, sampling=False, **kwargs):
                return {'id': key, 'text': json.dumps(review)}
        result = recovery.check_reviewer(Caller(), [{'kind': 'clean', 'parent': 'Fixture A.',
            'reference': 'Fixture B.', 'expected_meaning': False, 'origin_ids': [0, 1]}])
        self.assertTrue(result['passed'])
        self.assertFalse(recovery.clean_passes(review, self.c))

    def test_changed_recovery_packet_is_rejected_before_generation(self):
        recovery = self.recovery()
        path = self.root / 'packet.json'
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'digest'):
            recovery.load_payload(path, '0' * 64, self.c, self.paths)

    def test_blind_review_correction_is_derived_only_from_valid_model_span(self):
        recovery = self.recovery()
        p = pair()
        class Caller:
            def call(self, role, key, prompt, sampling=False, **kwargs):
                return {'id': key, 'text': json.dumps(p['error_review'])}
        review, _ = recovery.request_error(Caller(), 'test:span', p['source'])
        self.assertEqual(review['correction'], p['reference'])
        p['error_review']['source_span'] = 'absent text'
        review, ids = recovery.request_error(Caller(), 'test:span', p['source'])
        self.assertEqual(review, {})
        self.assertEqual(len(ids), 2)

    def test_blind_review_constrains_spans_without_exposing_reference(self):
        recovery = self.recovery()
        p = pair()
        class Caller:
            def call(self, role, key, prompt, sampling=False, **kwargs):
                self.schema = kwargs['response_schema']
                return {'id': key, 'text': json.dumps(p['error_review'])}
        caller = Caller()
        recovery.request_error(caller, 'constrained', p['source'])
        spans = caller.schema['properties']['source_span']['enum']
        self.assertIn('là là', spans)
        self.assertNotIn(p['source'], spans)
        self.assertTrue(all(not span or (p['source'].count(span) == 1
                         and len(span) <= len(p['source']) * .35) for span in spans))
        self.assertLessEqual(caller.schema['properties']['replacement']['maxLength'], len(p['source']) * .35)
        self.assertNotIn('correction', caller.schema['properties'])

    def test_constrained_schema_is_bound_into_journal_identity(self):
        self.recovery()
        from augmentation.journal import Journal
        class Engine:
            def __init__(self, *_): pass
            def generate(self, prompt, sampling, options, seed):
                self.options = options
                return '{}'
            def close(self): pass
        schema = {'type': 'object', 'properties': {}}
        with Journal(self.root / 'schema-journal', self.c, {}, Engine) as journal:
            row = journal.call('reviewer', 'schema', 'fixture', response_schema=schema)
            self.assertEqual(row['settings']['response_schema'], schema)
            self.assertEqual(journal.engine.options['response_schema'], schema)
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                journal.call('reviewer', 'schema', 'fixture', response_schema={'type': 'string'})

    def test_loader_rejects_a_review_patch_that_does_not_produce_its_correction(self):
        self.recovery()
        p = pair()
        p['error_review']['source_span'] = 'absent text'
        from augmentation.review import validate_pair
        self.assertIn('reviewer_patch_invalid', validate_pair(p, self.c))

    def test_recovery_evidence_cannot_just_repeat_the_source(self):
        self.recovery()
        from augmentation.review import validate_pair
        p = pair()
        p['error_review']['evidence'] = p['source'].rstrip(' .')
        self.assertIn('reviewer_evidence_copies_source', validate_pair(p, self.c))

    def test_default_export_still_requires_independent_review(self):
        recovery = self.recovery()
        p = pair(); p.pop('clean_review'); p.pop('error_review')
        eligible, _ = recovery.mechanical_pool([p], self.c)
        from augmentation.data import export_dataset
        result = export_dataset(self.paths, eligible, self.root / 'strict', self.c, {})
        self.assertEqual(result['accepted_error_pairs'], 0)

    def test_recovery_success_uses_only_reviewer_and_keeps_exact_baseline(self):
        recovery = self.recovery()
        p = pair()
        p['key'] = 'fixture-reference-key'
        cases = recovery.pilot_cases(self.paths, self.c)
        pilot_responses = []
        for case in cases:
            if case['kind'] == 'clean':
                pilot_responses.append({'parent_label': 0, 'translation_label': 0,
                    'meaning_preserved': case['expected_meaning'], 'names_and_numbers_preserved': True,
                    'confidence': .99, 'evidence': 'Fixture check with an explicitly known expected result.'})
            else:
                label = case['expected_label']
                pilot_responses.append({'source_label': label, 'mechanism': 'none' if label == 0 else 'missing required subject',
                    'source_span': '' if label == 0 else case['source'].split(' ', 1)[0] + ' ', 'replacement': '',
                    'single_error': label != 0, 'required_edit': label != 0,
                    'meaning_preserved': True, 'other_error_types': [], 'confidence': .99,
                    'evidence': 'Fixture result uses an existing recorded source and target.'})
        class Caller:
            def __init__(self): self.responses = iter(pilot_responses); self.roles = []
            def call(self, role, key, prompt, sampling=False, **kwargs):
                self.roles.append(role)
                response = (next(self.responses) if key.startswith('pilot:') else
                            p['clean_review'] if key.endswith(':clean_review') else
                            {**p['error_review'], 'source_span': 'là là', 'replacement': 'là'})
                return {'id': key, 'text': json.dumps(response)}
        caller = Caller()
        result = recovery.run_recovery(self.paths, self.c, caller, self.root / 'ok', {}, [p], {'sha256': 'fixture'})
        self.assertEqual(result['accepted_error_pairs'], 1)
        self.assertEqual(set(caller.roles), {'reviewer'})
        self.assertEqual(result['translation_calls'], 0)
        for stage in ('classification', 'correction'):
            original = json.loads(self.paths[stage].read_text())['train']
            augmented = json.loads(Path(result['files'][stage]['path']).read_text())['train']
            self.assertEqual(augmented[:len(original)], original)
            self.assertEqual(len(augmented), len(original) + 2)

    def test_recovery_budget_cannot_expand_to_original_full_run(self):
        from augmentation.config import validate_recipe
        self.recovery()
        recipe = copy.deepcopy(self.c)
        recipe['recovery'] = {'sha256': 'a' * 64, 'archive_sha256': 'b' * 64}
        recipe['run']['max_calls'] = 1536
        with self.assertRaisesRegex(ValueError, '256'):
            validate_recipe(recipe)

    def test_upload_folder_name_can_change_for_extracted_or_zipped_payload(self):
        recovery = self.recovery()
        content = b'{"fixture":true}'
        recipe = copy.deepcopy(self.c)
        recipe['recovery'] = {'sha256': hashlib.sha256(content).hexdigest()}
        root = self.root / 'arbitrary-user-upload-name'; root.mkdir()
        path = root / 'augmentation_recovery.json'; path.write_bytes(content)
        self.assertEqual(recovery.resolve_payload(self.root, self.root / 'work', recipe), path)
        path.unlink()
        archive = root / recipe['data']['archive']
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('augmentation_recovery.json', content)
            for name in ('VGEC 2026_v1.json', 'VGEC_classifier.json', 'VGEC_balanced.json'):
                z.writestr(name, '{}')
        recipe['recovery']['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
        resolved = recovery.resolve_payload(self.root, self.root / 'work', recipe)
        self.assertEqual(resolved.read_bytes(), content)
        self.assertEqual((resolved.parent / 'VGEC_classifier.json').read_text(), '{}')
        archive.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Attach'):
            recovery.resolve_payload(root, self.root / 'other-work', recipe)


if __name__ == '__main__': unittest.main()
