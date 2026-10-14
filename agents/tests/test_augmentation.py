"""Augmentation contracts preserve the full baseline and require independent evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from augmentation.review import validate_pair, select_pairs, pair_rows
from augmentation.data import select_parents, export_dataset, load_augmented
from model.data import _records_from_raw, load_training_split
from agents.tests.test_balanced_data import bundle
from tools.package_final_training import final_bundle

ROOT = Path(__file__).resolve().parents[2]


def recipe():
    return json.loads((ROOT / 'configs/augmentation.json').read_text())


def pair(parent=10000, suffix='x'):
    c = recipe()
    clean = f'Sample {suffix} là useful token .'
    source = clean.replace('là', 'là là')
    return {'parent_id': parent, 'parent_clean': 'Original clean sentence.', 'pivot': 'English',
            'reference': clean, 'label': 3, 'mechanism': 'redundant copula',
            'patch': {'clean_span': 'là', 'error_span': 'là là'}, 'source': source,
            'models': c['models'], 'calls': {'forward': 'f', 'backward': 'b', 'author': 'a', 'clean_review': 'c', 'error_review': 'e'},
            'clean_review': {'parent_label': 0, 'translation_label': 0, 'meaning_preserved': True,
                             'names_and_numbers_preserved': True, 'confidence': .95, 'evidence': 'Fixture evidence of equivalent clean references.'},
            'error_review': {'source_label': 3, 'mechanism': 'redundant copula', 'correction': clean, 'single_error': True,
                             'required_edit': True, 'meaning_preserved': True, 'other_error_types': [],
                             'confidence': .95, 'evidence': 'Fixture evidence of adjacent duplicated function marker.'}}


class ReviewTests(unittest.TestCase):
    def test_independent_review_and_exact_span_are_required(self):
        p = pair()
        self.assertEqual(validate_pair(p, recipe()), [])
        for field, value in [('source_label', 2), ('source_label', True), ('confidence', float('nan')),
                              ('other_error_types', [2]), ('mechanism', 'wrong preposition'), ('required_edit', False), ('correction', 'other')]:
            bad = copy.deepcopy(p); bad['error_review'][field] = value
            self.assertTrue(validate_pair(bad, recipe()), field)
        bad = copy.deepcopy(p); bad['models']['reviewer'] = bad['models']['author']
        self.assertTrue(validate_pair(bad, recipe()))
        bad = copy.deepcopy(p); bad['source'] += ' extra'
        self.assertTrue(validate_pair(bad, recipe()))

    def test_function_changes_cannot_add_content_or_change_numbers(self):
        p = pair(); p['patch'] = {'clean_span': 'token', 'error_span': 'different'}
        p['source'] = p['reference'].replace('token', 'different')
        self.assertTrue(validate_pair(p, recipe()))
        p = pair(); p['reference'] += ' 42'; p['source'] += ' 43'
        self.assertTrue(validate_pair(p, recipe()))

    def test_caps_and_duplicates_do_not_force_filler(self):
        c = recipe(); c['error_caps']['3'] = 2
        candidates = [pair(10000 + i, str(i)) for i in range(5)]
        accepted, rejected = select_pairs(candidates + [candidates[0]], [], c)
        self.assertLessEqual(len(accepted), 2)
        self.assertTrue(rejected)
        rows = pair_rows(accepted)
        self.assertEqual(sum(r['Output']['Error_Type_Id'] == 0 for r in rows), len(accepted))
        self.assertEqual(len(_records_from_raw(rows)), len(rows))


class FullDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = {}
        for stage in ('classification', 'correction'):
            src = self.root / f'{stage}-source.json'; src.write_text(json.dumps(bundle()))
            path = self.root / f'{stage}.json'; path.write_text(json.dumps(final_bundle(src, stage)))
            self.paths[stage] = path
        self.c = recipe()
        self.c['data']['counts'] = {stage: len(json.loads(path.read_text())['train']) for stage, path in self.paths.items()}
        self.c['data']['sha256'].update({stage: hashlib.sha256(path.read_bytes()).hexdigest() for stage, path in self.paths.items()})
        self.c['data']['sha256']['original'] = json.loads(self.paths['classification'].read_text())['original_sha256']

    def test_seed_selection_uses_original_complete_references_only(self):
        parents = select_parents(self.paths, self.c)
        self.assertTrue(parents)
        self.assertFalse(any(str(p['id']).startswith('syn:') for p in parents))
        self.assertFalse(any('...' in p['clean'] or p['clean'].startswith('(') for p in parents))
        self.assertTrue(any(p['id'] == 10000 for p in parents))
        self.assertFalse(any(p['id'] == 10001 for p in parents))

    def test_export_is_append_only_and_loader_rejects_tampering(self):
        result = export_dataset(self.paths, [pair()], self.root / 'export', self.c, {'source_commit': 'fixture'})
        for stage, path in self.paths.items():
            output = Path(result['files'][stage]['path'])
            raw = json.loads(output.read_text())
            base = json.loads(path.read_text())['train']
            self.assertEqual(raw['train'][:len(base)], base)
            train, validation, info = load_training_split(output, 0, 42)
            self.assertEqual(len(train), len(base) + 2); self.assertEqual(validation, [])
            raw['train'][0]['Input'] += ' tampered'
            with self.assertRaises(ValueError): load_augmented(raw, 0, 42)
        self.assertTrue(Path(result['archive']['path']).is_file())

    def test_parent_not_in_baseline_is_rejected(self):
        with self.assertRaises(ValueError):
            export_dataset(self.paths, [pair(123456)], self.root / 'export', self.c, {})

    def test_failed_export_can_resume_and_completed_publication_is_reusable(self):
        output = self.root / 'export'
        with patch('augmentation.data.load_augmented', side_effect=RuntimeError('interrupted validation')):
            with self.assertRaises(RuntimeError): export_dataset(self.paths, [pair()], output, self.c, {})
        self.assertFalse(output.exists())
        result = export_dataset(self.paths, [pair()], output, self.c, {})
        self.assertEqual(export_dataset(self.paths, [pair()], output, self.c, {}), result)
        Path(result['files']['classification']['path']).write_text('tampered')
        with self.assertRaises(ValueError): export_dataset(self.paths, [pair()], output, self.c, {})

    def test_augmented_loader_requires_recipe_pinned_baseline(self):
        result = export_dataset(self.paths, [pair()], self.root / 'export', self.c, {})
        raw = json.loads(Path(result['files']['classification']['path']).read_text())
        for section, key, value in [('sha256', 'classification', '0' * 64), ('counts', 'classification', 585),
                                    ('sha256', 'original', '0' * 64)]:
            bad = copy.deepcopy(raw); bad['recipe']['data'][section][key] = value
            with self.assertRaises(ValueError): load_augmented(bad, 0, 42)


if __name__ == '__main__': unittest.main()
