"""Final fits retire the local holdout without changing any source supervision."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents.tests.test_balanced_data import bundle
from model.data import load_training_split
from model.correction import placement_preflight
from model.tracking import result_metrics
from tools.package_final_training import final_bundle


class FinalTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source.json'
        self.source.write_text(json.dumps(bundle()))
        self.final = final_bundle(self.source, 'classification')
        self.path = self.root / 'final.json'

    def load(self, value=None, fraction=0):
        self.path.write_text(json.dumps(self.final if value is None else value))
        return load_training_split(self.path, fraction, 42)

    def test_every_former_holdout_row_is_training_and_no_rows_are_scored(self):
        train, validation, info = self.load()
        original = json.loads(self.source.read_text())
        self.assertEqual(len(train), len(original['train']) + len(original['validation']))
        self.assertEqual(validation, [])
        self.assertEqual(info['format'], 'vgec-final-training-v1')
        self.assertEqual({r['id'] for r in train[-2:]}, {10000, 10001})
        self.assertEqual(result_metrics({'classification': None, 'combined_score': None}), {})

    def test_removed_or_relabelled_rows_are_rejected(self):
        for mutate in (lambda v: v['train'].pop(),
                       lambda v: v['train'][-1]['Output'].update(Error_Type_Id=2),
                       lambda v: v.update(validation=v['train'][-1:])):
            value = copy.deepcopy(self.final)
            mutate(value)
            with self.assertRaises(ValueError):
                self.load(value)

    def test_final_bundle_cannot_be_used_as_a_new_holdout(self):
        with self.assertRaises(ValueError):
            self.load(fraction=.2)

    def test_embedded_source_tampering_is_rejected(self):
        value = copy.deepcopy(self.final)
        value['source_bundle_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'digest differs'):
            self.load(value)

    def test_raw_arrays_cannot_silently_resplit_a_final_fit(self):
        with self.assertRaises(ValueError):
            self.load(json.loads(self.source.read_text())['train'])

    def test_full_precision_balanced_classifier_budget_covers_score(self):
        from transformers import LlamaConfig
        LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=4, num_attention_heads=2,
                    num_key_value_heads=2).save_pretrained(self.root / 'base')
        settings = {'name': str(self.root / 'base'), 'revision': 'main',
                    'load_in_4bit': False, 'compute_dtype': 'float16', 'device_map': 'balanced'}
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2), \
             patch('torch.cuda.mem_get_info', return_value=(15*1024**3, 16*1024**3)):
            placement = placement_preflight(settings, task='classification')
        self.assertFalse(placement['quantized'])
        self.assertEqual(placement['device_map']['score'], 'cuda:1')
        self.assertEqual(set(placement['memory_budget']), {'cuda:0', 'cuda:1'})
        self.assertNotIn('lm_head', placement['device_map'])



class LoRAPrecisionTests(unittest.TestCase):
    def test_trainable_head_and_adapters_use_fp32_without_promoting_frozen_base(self):
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import Qwen2Config, Qwen2ForSequenceClassification
        from model.runtime import promote_trainable_parameters
        base = Qwen2ForSequenceClassification(Qwen2Config(vocab_size=64, hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, num_labels=5)).half()
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj'],
            task_type=TaskType.SEQ_CLS, modules_to_save=['score']))
        promote_trainable_parameters(model)
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad))
        self.assertTrue(all(p.dtype == torch.float16 for p in model.parameters() if not p.requires_grad))



class FinalNotebookTests(unittest.TestCase):
    def test_launcher_displays_submission_when_validation_scores_are_unavailable(self):
        import contextlib
        import io
        from tools.notebook import launch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for stage in ('classification', 'correction'):
                (root / stage).mkdir()
                (root / stage / 'metrics.json').write_text(json.dumps({'classification': None, 'corrections': {}}))
            (root / 'pipeline_manifest.json').write_text(json.dumps({'mode': 'both'}))
            (root / 'results.zip').touch()
            class Finished:
                stdout = iter([f'VGEC_RUN_DIR:{root}\n'])
                args = ['fixture']
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def wait(self): return 0
            with patch('tools.notebook.subprocess.Popen', return_value=Finished()), \
                    patch('IPython.display.display') as display, contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(launch(root, 'python'), root)
            self.assertIn('unavailable_full_training', output.getvalue())
            display.assert_called_once()


if __name__ == '__main__':
    unittest.main()
