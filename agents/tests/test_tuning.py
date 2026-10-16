import copy
import unittest


class TuningTests(unittest.TestCase):
    def test_uniform_reproduces_soft_vote_and_alignment(self):
        from model.tuning import calibrated_vote

        rows = [
            [{"id": 1, "label": 0, "probabilities": [0.4, 0.3, 0.1, 0.1, 0.1]}],
            [{"id": 1, "label": 1, "probabilities": [0.2, 0.5, 0.1, 0.1, 0.1]}],
            [{"id": 1, "label": 1, "probabilities": [0.3, 0.4, 0.1, 0.1, 0.1]}],
        ]
        got = calibrated_vote(rows, [1, 1, 1], 1, 1)
        self.assertEqual(got[0]["label"], 1)
        self.assertAlmostEqual(got[0]["probabilities"][1], 0.4)
        with self.assertRaises(ValueError):
            calibrated_vote(rows, [0, 0, 0], 1, 1)
        with self.assertRaises(ValueError):
            calibrated_vote(rows, [1, 1, 1], 0, 1)
        bad = copy.deepcopy(rows)
        bad[1][0]["id"] = "1"
        with self.assertRaises(ValueError):
            calibrated_vote(bad, [1, 1, 1], 1, 1)

    def test_strategy_matrix_covers_distinct_decoders_with_bounded_beams(self):
        from model.tuning import decoding_strategies

        strategies = decoding_strategies()
        self.assertEqual(len({s["name"] for s in strategies}), len(strategies))
        self.assertEqual(strategies[0]["name"], "greedy_reference")
        self.assertTrue(any(s.get("schema") for s in strategies))
        self.assertTrue(any(s.get("edit_rerank") for s in strategies))
        self.assertTrue(
            any(s["generation"].get("num_beam_groups", 1) > 1 for s in strategies)
        )
        self.assertTrue(any(s["generation"].get("do_sample") for s in strategies))
        self.assertTrue(
            all(s["generation"].get("num_beams", 1) <= 8 for s in strategies)
        )

    def test_screening_is_seeded_stratified_and_gold_is_stripped(self):
        from model.tuning import panel, inference_rows

        rows = [
            {"id": 10 * k + i, "source": "toy", "target": "reference", "label": k}
            for k in range(5)
            for i in range(5)
        ]
        self.assertEqual(panel(rows, 2, 42), panel(rows, 2, 42))
        self.assertEqual(len(panel(rows, 2, 42)), 10)
        self.assertEqual(set(inference_rows(rows)[0]), {"id", "source"})


class CacheIdentityTests(unittest.TestCase):
    def test_saved_prompt_changes_invalidate_decode_cache(self):
        import tempfile, json
        from pathlib import Path
        from tools.decoding_sweep import cache_identity

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "adapter_model.safetensors").write_bytes(b"weights")
            (root / "inference.json").write_text(json.dumps({"prompt": "first"}))
            first = cache_identity(root, [], False)
            (root / "inference.json").write_text(json.dumps({"prompt": "changed"}))
            self.assertNotEqual(first, cache_identity(root, [], False))


class CandidateLikelihoodTests(unittest.TestCase):
    def test_partial_final_chunk_matches_full_shifted_likelihood(self):
        import json
        import types
        import torch
        from tools.decoding_sweep import _candidate_score

        class Tokenizer:
            eos_token_id = 2

            def encode(self, text, add_special_tokens=False):
                self.asserted_text = json.loads(text)
                return [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        class Model:
            def get_input_embeddings(self):
                return types.SimpleNamespace(weight=torch.empty(1))

            def __call__(self, input_ids, **kwargs):
                length = input_ids.shape[1]
                self.logits = (
                    torch.arange(length * 16).reshape(1, length, 16).float() / 13
                )
                return types.SimpleNamespace(logits=self.logits)

        model, tokenizer = Model(), Tokenizer()
        prefix = [1, 2, 3]
        actual = _candidate_score(model, tokenizer, prefix, "toy")
        expected = -torch.nn.functional.cross_entropy(
            model.logits[0, len(prefix) - 1 : -1],
            torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 2]),
        ).item()
        self.assertAlmostEqual(actual, expected, places=6)


class OptionalDecoderFailureTests(unittest.TestCase):
    def test_transformers_contrastive_type_error_is_recorded_and_next_decoder_runs(
        self,
    ):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        import torch
        from transformers import Qwen2Config
        from transformers.generation.utils import stack_model_outputs
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from model.runtime import write_json
        from model.tuning import decoding_strategies
        from tools.decoding_sweep import sweep

        def decode(model, tokenizer, row, inference, strategy):
            if strategy["name"] == "contrastive":
                # Reproduce the exact upstream dictionary-concatenation failure.
                stack_model_outputs(
                    [
                        CausalLMOutputWithPast(
                            logits=torch.zeros(1, 1, 8), past_key_values={}
                        )
                    ],
                    Qwen2Config(),
                )
            return {
                "id": row["id"],
                "target": row["source"],
                "raw_generation": "toy",
                "parse_error": None,
            }

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            adapter, classifier = root / "adapter", root / "classification"
            adapter.mkdir()
            classifier.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"test-only")
            gold = [
                {"id": i, "source": "toy", "target": "toy", "label": i}
                for i in range(5)
            ]
            write_json(classifier / "validation.json", gold)
            write_json(
                classifier / "predictions.json",
                [{"id": i, "predicted_class": i} for i in range(5)],
            )
            matrix = {s["name"]: s for s in decoding_strategies()}
            strategies = [
                matrix[n] for n in ["greedy_reference", "contrastive", "beam2"]
            ]
            with patch(
                "tools.decoding_sweep._load", return_value=(object(), object(), {})
            ), patch("tools.decoding_sweep.decode", side_effect=decode):
                result = sweep(
                    adapter,
                    classifier,
                    root / "output",
                    device="cpu",
                    per_class=1,
                    finalists=2,
                    strategies=strategies,
                )
                with self.assertRaisesRegex(
                    RuntimeError, "No decoding strategy completed"
                ):
                    sweep(
                        adapter,
                        classifier,
                        root / "all-failed",
                        device="cpu",
                        per_class=1,
                        finalists=1,
                        strategies=[matrix["contrastive"]],
                    )
            reports = {r["strategy"]["name"]: r for r in result["screen"]}
            failed = reports["contrastive"]
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["error_type"], "TypeError")
            self.assertIn("dict", failed["error"])
            self.assertNotIn("metrics", failed)
            self.assertEqual(reports["beam2"]["status"], "completed")
            self.assertEqual(
                {r["strategy"]["name"] for r in result["full"]},
                {"greedy_reference", "beam2"},
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(
                json.loads((root / "output/contrastive/screen.json").read_text())[
                    "error_type"
                ],
                "TypeError",
            )


class ContrastiveOutputCompatibilityTests(unittest.TestCase):
    def test_real_lora_contrastive_generation_matches_unhooked_model(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from accelerate.hooks import AlignDevicesHook, add_hook_to_module
        from peft import LoraConfig, get_peft_model
        from tools.decoding_sweep import _contrastive_output_compat

        torch.manual_seed(42)
        base = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                eos_token_id=2,
                pad_token_id=0,
            )
        )
        model = get_peft_model(
            base,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=2,
                lora_alpha=4,
                target_modules=["q_proj"],
            ),
        ).eval()
        ids = torch.tensor([[1, 3, 4]])
        options = dict(
            attention_mask=torch.ones_like(ids),
            max_new_tokens=4,
            penalty_alpha=0.6,
            top_k=4,
            low_memory=True,
            do_sample=False,
        )
        with torch.inference_mode():
            expected = model.generate(ids, **options)
            add_hook_to_module(
                base,
                AlignDevicesHook(
                    execution_device="cpu",
                    io_same_device=True,
                    skip_keys=["past_key_values", "logits"],
                ),
            )
            with self.assertRaisesRegex(TypeError, "Unexpected attribute type.*dict"):
                model.generate(ids, **options)
            before = len(base._forward_hooks)
            with _contrastive_output_compat(model):
                actual = model.generate(ids, **options)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(len(base._forward_hooks), before)
            self.assertGreater(actual.shape[1], ids.shape[1])
            from types import SimpleNamespace
            from unittest.mock import patch
            from model.tuning import decoding_strategies
            from tools.decoding_sweep import decode

            tokenizer = SimpleNamespace(
                pad_token_id=0, eos_token_id=2, decode=lambda *args, **kwargs: "toy"
            )
            strategy = next(
                s for s in decoding_strategies() if s["name"] == "contrastive"
            )
            with patch(
                "model.correction._prompt_ids", return_value=ids[0].tolist()
            ), patch("model.correction._parse_correction", return_value=("toy", None)):
                result = decode(
                    model,
                    tokenizer,
                    {"id": 1, "source": "toy"},
                    {"max_length": 32, "generation": {"max_new_tokens": 4}},
                    strategy,
                )
            self.assertEqual(result["target"], "toy")
            self.assertEqual(len(base._forward_hooks), before)

    def test_canonical_output_preserves_tensor_cache_and_actual_loss(self):
        import torch
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from transformers.cache_utils import DynamicCache
        from accelerate.hooks import AlignDevicesHook, add_hook_to_module
        from tools.decoding_sweep import _contrastive_output_compat

        class Forward(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.zeros(1, 1, 4)
                self.cache = DynamicCache()
                self.actual_loss = None

            def forward(self, ids):
                return CausalLMOutputWithPast(
                    loss=self.actual_loss,
                    logits=self.logits,
                    past_key_values=self.cache,
                )

        model = Forward()
        add_hook_to_module(
            model,
            AlignDevicesHook(
                execution_device="cpu",
                io_same_device=True,
                skip_keys=["logits", "past_key_values", "loss"],
            ),
        )
        ids = torch.tensor([[1]])
        self.assertIsInstance(model(ids).loss, dict)
        with _contrastive_output_compat(model):
            output = model(ids)
            self.assertIsNone(output.loss)
            self.assertIs(output.logits, model.logits)
            self.assertIs(output.past_key_values, model.cache)
            model.actual_loss = torch.tensor(2.0)
            self.assertIs(model(ids).loss, model.actual_loss)
        with self.assertRaisesRegex(RuntimeError, "test exception"):
            with _contrastive_output_compat(model):
                raise RuntimeError("test exception")
        self.assertEqual(len(model._forward_hooks), 0)
