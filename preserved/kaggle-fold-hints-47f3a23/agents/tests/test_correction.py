import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from model.correction import (_parse_correction, _reload_selected_corrector,
                              correct, encode_record, load_corrector, validate_hints)


class CorrectionGenerationTests(unittest.TestCase):
    def test_encoding_uses_predicted_hint_and_supervises_only_completion(self):
        record = {"id": 7, "source": "Câu gốc.", "label": 1, "target": "Câu đích riêng biệt."}
        inference = {"prompt": "correct", "hint_descriptions": [f"description {i}" for i in range(5)]}
        tokenizer = MagicMock(eos_token_id=99)
        tokenizer.apply_chat_template.return_value = [10, 11, 12]
        tokenizer.encode.return_value = [20, 21]

        encoded = encode_record(tokenizer, record, {"label": 3}, 20, inference)

        messages = tokenizer.apply_chat_template.call_args.args[0]
        prompt = "\n".join(message["content"] for message in messages)
        self.assertIn("Gợi ý nhãn dự đoán: 3", prompt)
        self.assertIn("description 3", prompt)
        self.assertIn(record["source"], prompt)
        self.assertNotIn("description 1", prompt)
        self.assertNotIn(record["target"], prompt)
        self.assertEqual(encoded["input_ids"], [10, 11, 12, 20, 21, 99])
        self.assertEqual(encoded["labels"], [-100, -100, -100, 20, 21, 99])
        self.assertEqual(encoded["attention_mask"], [1] * 6)

    def test_invalid_generation_has_no_success_fallback(self):
        for text in ('[]', 'null', '1', '"text"',
                     '{"corrected_sentence":true}', '{"corrected_sentence":""}',
                     '{"corrected_sentence":"a","explanation":"b"}',
                     "explanation, not JSON"):
            with self.subTest(text=text):
                target, error = _parse_correction(text)
                self.assertEqual(target, "")
                self.assertIsNotNone(error)

    def test_clean_hints_are_generated_and_invalid_parse_is_retained(self):
        import torch

        records = [{"id": i, "source": "Câu đúng.", "label": 0, "target": "Câu đúng."}
                   for i in (1, 2)]
        hints = [{"id": row["id"], "label": 0, "probabilities": [1., 0., 0., 0., 0.]}
                 for row in records]
        tokenizer = MagicMock(pad_token_id=0, eos_token_id=3)
        tokenizer.apply_chat_template.return_value = [1, 2]
        tokenizer.decode.side_effect = ['{"corrected_sentence":"Câu được sửa."}', "not JSON", "still not JSON"]
        model = MagicMock()
        model.get_input_embeddings.return_value = SimpleNamespace(weight=torch.empty(0))
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        inference = {"prompt": "correct", "hint_descriptions": ["clean"] * 5,
                     "generation": {"max_new_tokens": 8}, "max_length": 8}

        with patch("lmformatenforcer.integrations.transformers.build_transformers_prefix_allowed_tokens_fn", return_value=lambda *_: [3]):
            predictions = correct(model, tokenizer, records, hints, inference)

        self.assertEqual(model.generate.call_count, 3)
        self.assertEqual([row["label"] for row in predictions], [0, 0])
        self.assertEqual(predictions[0]["target"], "Câu được sửa.")
        self.assertIsNone(predictions[0]["parse_error"])
        self.assertEqual(predictions[1]["target"], "")
        self.assertEqual(predictions[1]["raw_generation"], "still not JSON")
        self.assertTrue(predictions[1]["regenerated"])
        self.assertEqual(model.generate.call_args.kwargs["num_beams"], 4)
        self.assertIsNotNone(predictions[1]["parse_error"])

    def test_invalid_json_recovers_without_changing_label_or_using_gold(self):
        import torch
        record = {"id": 9, "source": "Câu gốc.", "label": 4, "target": "Private gold"}
        tokenizer = MagicMock(pad_token_id=0, eos_token_id=3)
        tokenizer.apply_chat_template.return_value = [1, 2]
        tokenizer.decode.side_effect = ["bad JSON", '{"corrected_sentence":"Câu được sửa."}']
        model = MagicMock()
        model.get_input_embeddings.return_value = SimpleNamespace(weight=torch.empty(0))
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        inference = {"prompt": "correct", "hint_mode": "none", "generation": {}, "max_length": 32}
        with patch("lmformatenforcer.integrations.transformers.build_transformers_prefix_allowed_tokens_fn", return_value=lambda *_: [3]):
            prediction = correct(model, tokenizer, [record], [{"id": 9, "label": 2, "probabilities": [0., 0., 1., 0., 0.]}], inference)[0]
        self.assertEqual(prediction["label"], 2)
        self.assertEqual(prediction["target"], "Câu được sửa.")
        self.assertIsNone(prediction["parse_error"])
        self.assertTrue(prediction["regenerated"])
        self.assertNotIn("Private gold", str(tokenizer.apply_chat_template.call_args))


class HintContractTests(unittest.TestCase):
    def test_quarantined_original_partition_and_contrast_family_leakage(self):
        import copy
        import hashlib

        first = {"id": 1, "source": "first source", "label": 1, "target": "first fixed"}
        second = {"id": 2, "source": "second source", "label": 0, "target": "second source"}
        clean = {"id": "clean", "source": "legal construction", "label": 0, "target": "legal construction"}
        error = {"id": "error", "source": "broken construction", "label": 2, "target": "legal construction"}
        validation = [{"id": 9, "source": "holdout", "label": 0, "target": "holdout"}]
        correction = [first, second, error]
        classifier = [first, second, clean, error]

        def hint(row, fold=None):
            result = {"id": row["id"], "label": 0, "probabilities": [1., 0., 0., 0., 0.]}
            if fold is not None:
                result["fold"] = fold
            return result

        for field in ("family_id", "template_id"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "data.json"
                path.write_text("fixture", encoding="utf-8")
                metadata = [{"id": row["id"], "authored_provenance": {field: "shared"}}
                            for row in (clean, error)]
                provenance = {"provenance": {"original_train_ids": [1, 2, 3],
                                             "train_rows": metadata[1:]}}
                artifact = {
                    "seed": 42, "validation_fraction": .2,
                    "correction_data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "classifier_provenance": {
                        "original_train_ids": [1, 2, 3], "retained_original_train_ids": [1, 2],
                        "quarantined_ids": [3], "selected_synthetic_ids": ["clean", "error"],
                        "train_rows": metadata},
                    "train": [hint(first, 0), hint(error, 0), hint(second, 1)],
                    "validation": [hint(validation[0])],
                    "folds": [{"fold": 0, "train_ids": [2], "predict_ids": [1, "error"]},
                              {"fold": 1, "train_ids": [1, "clean", "error"], "predict_ids": [2]}],
                }
                indexed, _ = validate_hints(artifact, correction, validation, provenance,
                                            classifier, path, 42, .2)
                self.assertEqual({key[1] for key in indexed}, {1, 2, "error"})
                leaked = copy.deepcopy(artifact)
                leaked["folds"][0]["train_ids"].append("clean")
                with self.assertRaisesRegex(ValueError, "contrast/template family"):
                    validate_hints(leaked, correction, validation, provenance, classifier, path, 42, .2)
                missing_quarantine = copy.deepcopy(artifact)
                missing_quarantine["classifier_provenance"]["quarantined_ids"] = []
                with self.assertRaisesRegex(ValueError, "declared mixed partition"):
                    validate_hints(missing_quarantine, correction, validation, provenance,
                                   classifier, path, 42, .2)


class SelectedAdapterReloadTests(unittest.TestCase):
    def test_selected_weights_replace_last_epoch_without_merging_or_copying_base(self):
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(42)
        base = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
            pad_token_id=0, eos_token_id=2, tie_word_embeddings=False))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"<pad>": 0, "<unk>": 1, "<eos>": 2},
                                                unk_token="<unk>")),
            pad_token="<pad>", unk_token="<unk>", eos_token="<eos>")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path, adapter = root / "base", root / "adapter"
            base.save_pretrained(base_path)
            settings = {"name": str(base_path), "revision": "local", "load_in_4bit": False,
                        "device_map": "single", "compute_dtype": "float32"}
            inference = {"prompt": "saved prompt", "hint_mode": "none",
                         "max_length": 32, "generation": {"max_new_tokens": 4, "do_sample": False}}
            frozen_weight = base.model.layers[0].self_attn.q_proj.weight
            frozen_snapshot = frozen_weight.detach().clone()
            model = get_peft_model(base, LoraConfig(
                r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM))
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "lora_" in name:
                        parameter.fill_(0.1)
            model.eval()
            tokens = torch.tensor([[3, 4, 5]])
            with torch.no_grad():
                selected_logits = model(tokens).logits.clone()
            model.save_pretrained(adapter)
            tokenizer.save_pretrained(adapter)
            for name, value in (("loading.json", settings), ("inference.json", inference)):
                (adapter / name).write_text(json.dumps(value), encoding="utf-8")
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "lora_" in name:
                        parameter.fill_(0.9)
                last_epoch_logits = model(tokens).logits
            self.assertFalse(torch.allclose(selected_logits, last_epoch_logits))
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
            reloaded, _, saved_inference = _reload_selected_corrector(model, adapter)
            self.assertIs(reloaded.get_base_model(), base)
            self.assertIs(base.model.layers[0].self_attn.q_proj.base_layer.weight, frozen_weight)
            torch.testing.assert_close(frozen_weight, frozen_snapshot)
            self.assertFalse(base.is_gradient_checkpointing)
            self.assertFalse(base.get_input_embeddings()(tokens).requires_grad)
            self.assertFalse(any(parameter.requires_grad for parameter in reloaded.parameters()))
            self.assertEqual(saved_inference, inference)
            fresh, _, _ = load_corrector(adapter, "cpu")
            with torch.no_grad():
                torch.testing.assert_close(reloaded(tokens).logits, selected_logits)
                torch.testing.assert_close(reloaded(tokens).logits, fresh(tokens).logits)
                expected = fresh.generate(tokens, max_new_tokens=4, do_sample=False)
                actual = reloaded.generate(tokens, max_new_tokens=4, do_sample=False)
                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
