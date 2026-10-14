"""Bounded journals, resume identity and tied native decoder placement."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents.tests.test_augmentation import recipe
from augmentation.journal import Journal, BudgetExhausted
from augmentation.inference import layer_placement
from augmentation.pipeline import run_pipeline


class FakeEngine:
    closed = 0
    calls = 0
    def __init__(self, settings, device): self.settings = settings
    def generate(self, prompt, sampling, options, seed):
        type(self).calls += 1
        return '{"fixture":true}'
    def close(self): type(self).closed += 1


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.c = recipe(); self.identity = {'data': 'fixture', 'commit': 'same'}
        FakeEngine.calls = FakeEngine.closed = 0

    def test_resume_reuses_outputs_without_loading_models_and_checks_prompt(self):
        with Journal(self.root, self.c, self.identity, FakeEngine) as j:
            first = j.call('author', 'key', 'prompt', True)
        with Journal(self.root, self.c, self.identity, FakeEngine) as j:
            self.assertEqual(j.call('author', 'key', 'prompt', True), first)
            self.assertEqual(FakeEngine.calls, 1)
            with self.assertRaises(ValueError): j.call('author', 'key', 'changed prompt', True)
        self.assertEqual(FakeEngine.closed, 1)

    def test_limits_and_corrupt_journal_fail_without_extra_calls(self):
        self.c['run']['max_calls'] = 1
        with Journal(self.root, self.c, self.identity, FakeEngine) as j:
            j.call('translator', 'first', 'prompt', False)
            with self.assertRaises(BudgetExhausted): j.call('author', 'second', 'prompt', True)
        path = self.root / 'calls.jsonl'; rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[0]['text'] = 'tampered'; path.write_text(json.dumps(rows[0])+'\n')
        with self.assertRaises(ValueError): Journal(self.root, self.c, self.identity, FakeEngine)
        self.assertEqual(FakeEngine.calls, 1)

    def test_failed_generation_consumes_the_call_budget_and_keeps_request_evidence(self):
        class Crash(FakeEngine):
            def generate(self, *args): raise RuntimeError('fixture CUDA failure')
        self.c['run']['max_calls'] = 1
        with self.assertRaises(RuntimeError):
            with Journal(self.root, self.c, self.identity, Crash) as j:
                j.call('author', 'failed', 'private fixture prompt', True)
        with Journal(self.root, self.c, self.identity, FakeEngine) as j:
            with self.assertRaises(BudgetExhausted): j.call('author', 'next', 'prompt', True)
        rows = [json.loads(x) for x in (self.root / 'calls.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['status'], 'failed')
        self.assertEqual(rows[0]['prompt'], 'private fixture prompt')

    def test_role_transition_closes_previous_model(self):
        with Journal(self.root, self.c, self.identity, FakeEngine) as j:
            j.call('translator', 't', 'prompt', False)
            j.call('author', 'a', 'prompt', True)
            self.assertEqual(FakeEngine.closed, 1)
        self.assertEqual(FakeEngine.closed, 2)

    def test_interrupted_attempt_remains_reserved_after_resume(self):
        class Interrupted(FakeEngine):
            def generate(self, *args): raise KeyboardInterrupt()
        self.c['run']['max_calls'] = 1
        with self.assertRaises(KeyboardInterrupt):
            with Journal(self.root, self.c, self.identity, Interrupted) as j:
                j.call('author', 'interrupted', 'prompt', True)
        with Journal(self.root, self.c, self.identity, FakeEngine) as j:
            with self.assertRaises(BudgetExhausted): j.call('author', 'interrupted', 'prompt', True)
        self.assertFalse((self.root / 'inflight_call.json').exists())
        self.assertEqual(len((self.root / 'interrupted_attempts.jsonl').read_text().splitlines()), 1)

    def test_hard_interruption_without_cleanup_cannot_reset_elapsed_time(self):
        self.c['run']['max_wall_minutes'] = 1
        for phase in ('loading', 'generation'):
            with self.subTest(phase=phase):
                now = [100.0]
                class Interrupted(FakeEngine):
                    def __init__(self, *args):
                        if phase == 'loading': now[0] += 61; raise KeyboardInterrupt()
                        super().__init__(*args)
                    def generate(self, *args): now[0] += 61; raise KeyboardInterrupt()
                path = self.root / phase
                journal = Journal(path, self.c, self.identity, Interrupted, clock=lambda: now[0], wall_clock=lambda: now[0])
                with self.assertRaises(KeyboardInterrupt): journal.call('author', 'interrupted', 'prompt', True)
                # Deliberately no close()/__exit__: simulate notebook SIGTERM.
                with Journal(path, self.c, self.identity, FakeEngine, clock=lambda: now[0], wall_clock=lambda: now[0]) as resumed:
                    with self.assertRaises(BudgetExhausted): resumed.call('author', 'next', 'prompt', True)

    def test_resume_cannot_reset_the_elapsed_budget(self):
        now = [100.0]
        self.c['run']['max_wall_minutes'] = 1
        with Journal(self.root, self.c, self.identity, FakeEngine, clock=lambda: now[0]) as j:
            j.call('author', 'first', 'prompt', True)
            now[0] += 61
        now[0] = 500.0
        with Journal(self.root, self.c, self.identity, FakeEngine, clock=lambda: now[0]) as j:
            with self.assertRaises(BudgetExhausted): j.call('author', 'second', 'prompt', True)


class PlacementTests(unittest.TestCase):
    def test_tied_output_head_stays_with_embedding_and_weights_are_not_counted_twice(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
                            tie_word_embeddings=True)
        with torch.device('meta'): model = Qwen2ForCausalLM(config)
        model.tie_weights()
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=2), \
             patch('torch.cuda.mem_get_info', return_value=(15*1024**3,16*1024**3)):
            p = layer_placement(model, 'cuda')
        self.assertEqual(p['device_map']['lm_head'], p['device_map']['model.embed_tokens'])
        self.assertEqual(p['parameters'], sum(x.numel() for x in model.parameters()))
        self.assertEqual(set(p['memory_budget']), {'cuda:0','cuda:1'})



class LauncherTests(unittest.TestCase):
    def test_augmentation_launcher_links_dataset_without_training_reports(self):
        import contextlib, io
        from tools.notebook import launch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); archive = root / 'dataset.zip'; archive.touch()
            (root / 'pipeline_manifest.json').write_text(json.dumps({'format':'vgec-augmented-final-v1',
                'status':'completed', 'accepted_error_pairs':2, 'new_rows_per_stage':4,
                'files':{'classification':{'total_rows':589},'correction':{'total_rows':499}},
                'archive':{'path':str(archive)}}))
            class Finished:
                stdout = iter([f'VGEC_RUN_DIR:{root}\n']); args = ['fixture']
                def __enter__(self): return self
                def __exit__(self, *_): pass
                def wait(self): return 0
            with patch('tools.notebook.subprocess.Popen', return_value=Finished()) as process, \
                    patch('IPython.display.display') as display, contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(launch(root,'python'),root)
            self.assertIn('tools.run_augmentation', process.call_args.args[0])
            self.assertIn('accepted_error_pairs',out.getvalue()); display.assert_called_once()
            self.assertTrue((root / 'console.log').is_file())


if __name__ == '__main__': unittest.main()
