"""Inference must fit without an optimizer reserve or live classifier weights."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from transformers import Qwen2Config
from model.correction import load_corrector, placement_preflight


class InferenceMemoryTests(unittest.TestCase):
    def test_inference_fits_when_training_reserve_does_not(self):
        # A real tiny Qwen architecture and simulated low-memory GPU; there is
        # room for inference workspace, but no room for an optimizer/gradients.
        with tempfile.TemporaryDirectory() as folder:
            Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2,
                        num_key_value_heads=2).save_pretrained(folder)
            settings = {'name': folder, 'revision': 'main', 'load_in_4bit': True,
                        'compute_dtype': 'float16', 'device_map': 'single'}
            with patch('torch.cuda.is_available', return_value=True), \
                    patch('torch.cuda.mem_get_info', return_value=(2*1024**3, 3*1024**3)):
                with self.assertRaises(RuntimeError):
                    placement_preflight(settings)
                inference = placement_preflight(settings, training=False)
            self.assertLess(inference['memory_budget']['cuda:0']['required_bytes'], 2*1024**3)
            self.assertEqual(inference['device_map'], {'': 'cuda:0'})

    def test_saved_corrector_reload_reaches_loader_without_training_reserve(self):
        class WeightLoaderReached(Exception):
            pass

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2,
                        num_key_value_heads=2).save_pretrained(root / 'base')
            settings = {'name': str(root / 'base'), 'revision': 'main', 'load_in_4bit': True,
                        'compute_dtype': 'float16', 'device_map': 'single'}
            (root / 'loading.json').write_text(json.dumps(settings))
            (root / 'inference.json').write_text(json.dumps({'prompt': 'correct', 'hint_mode': 'none'}))
            # Only the expensive external weight loader is replaced. The saved
            # config, real Qwen architecture and memory guard run normally.
            with patch('torch.cuda.is_available', return_value=True), \
                    patch('torch.cuda.mem_get_info', return_value=(2*1024**3, 3*1024**3)), \
                    patch('model.correction._load_base', side_effect=WeightLoaderReached), \
                    self.assertRaises(WeightLoaderReached):
                load_corrector(root, 'cuda')


if __name__ == '__main__':
    unittest.main()
