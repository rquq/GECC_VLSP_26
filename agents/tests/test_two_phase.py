"""Consumer-visible leakage boundaries for classifier-conditioned correction."""
import unittest
import tempfile
from pathlib import Path

from model.classification import (_configure_padding, _connected_folds, _encoded, _fit,
                                  _predict, _preprocessing, classify, _class_weights,
                                  _weighted_hierarchical_loss, _joint_log_probabilities, _architecture)
from model.data import _id_key, _source_key


class OutOfFoldGroupingTests(unittest.TestCase):
    def test_synthetic_rows_and_omitted_original_parents_never_train_their_hint_model(self):
        originals = [{"id": i, "source": f"source {i}", "label": i % 5, "target": f"target {i}"}
                     for i in range(20)]
        # Original0 is excluded from correction eligibility, but remains classifier-eligible.
        correction = originals[1:] + [{"id": "synthetic", "source": "new error", "label": 1, "target": "fixed"}]
        provenance = {"train_rows": [{"id": "synthetic", "kind": "synthetic", "parent_ids": [0, 1]}]}
        mixed = originals + correction[-1:]
        plans = _connected_folds(originals, correction, provenance, 3, 42, mixed)
        predictions = [row["id"] for _, _, rows in plans for row in rows]
        self.assertCountEqual(predictions, [row["id"] for row in correction])
        for train, heldout, predict in plans:
            train_ids = {_id_key(row["id"]) for row in train}
            self.assertFalse(train_ids & {_id_key(row["id"]) for row in heldout})
            self.assertFalse({_source_key(row["source"]) for row in train} &
                             {_source_key(row["source"]) for row in predict})
            if not any(row["id"] == "synthetic" for row in predict):
                self.assertIn("synthetic", [row["id"] for row in train])
            if any(row["id"] == "synthetic" for row in predict):
                self.assertNotIn(_id_key(0), train_ids)
                self.assertNotIn(_id_key(1), train_ids)
                self.assertIn(1, [row["id"] for row in predict])

    def test_normalized_source_duplicates_remain_one_fold_even_with_typed_ids(self):
        originals = [{"id": i, "source": f"source {i}", "label": i % 5, "target": "fixed"}
                     for i in range(20)]
        duplicate = {"id": "0", "source": "  SOURCE   0 ", "label": 0, "target": "fixed"}
        originals.append(duplicate)
        plans = _connected_folds(originals, originals, {}, 3, 42, originals)
        assignments = {_id_key(row["id"]): fold for fold, (_, _, rows) in enumerate(plans) for row in rows}
        self.assertEqual(assignments[_id_key(0)], assignments[_id_key("0")])
        for train, _, predict in plans:
            self.assertFalse({_source_key(row["source"]) for row in train} &
                             {_source_key(row["source"]) for row in predict})


class MixedClassifierFoldTests(unittest.TestCase):
    def test_classifier_only_synthetic_parent_components_and_fold_caps(self):
        from collections import Counter

        originals = [{"id": i, "source": f"original {i}", "label": i % 5, "target": "fixed"}
                     for i in range(40)]
        # Classifier-only row joins two correction source components.
        synthetic = [{"id": f"syn:{i}", "source": f"synthetic {i}", "label": 1, "target": "fixed"}
                     for i in range(8)]
        metadata = [{"id": r["id"], "parent_ids": [1, 6] if i == 0 else []}
                    for i, r in enumerate(synthetic)]
        mixed = originals + synthetic
        plans = _connected_folds(originals, originals, {"train_rows": metadata}, 3, 42, mixed)
        assignments = {r["id"]: fold for fold, (_, _, rows) in enumerate(plans) for r in rows}
        self.assertEqual(assignments[1], assignments[6])
        for fit, _, predicted in plans:
            keys = {r["id"] for r in fit}
            if any(r["id"] in (1, 6) for r in predicted):
                self.assertNotIn("syn:0", keys)
            counts = Counter(r["label"] for r in fit if type(r["id"]) is int)
            added = Counter(r["label"] for r in fit if type(r["id"]) is str)
            self.assertTrue(all(added[label] <= counts[label] for label in range(5)))


class ClassWeightingTests(unittest.TestCase):
    def test_weights_use_fit_partition_and_have_unit_expected_scale(self):
        counts = [16, 8, 3, 2, 4]
        records = [{"label": label} for label, count in enumerate(counts) for _ in range(count)]
        result = _class_weights(records, 0.5)
        weights = result["weights"]
        self.assertAlmostEqual(sum(n * w for n, w in zip(counts, weights)) / sum(counts), 1.0)
        self.assertAlmostEqual(weights[3] / weights[0], (counts[0] / counts[3]) ** 0.5)
        self.assertEqual(_class_weights(records, 0)["weights"], [1.0] * 5)
        fold = records[:-2]
        self.assertGreater(_class_weights(fold, 0.5)["weights"][4], weights[4])

    def test_split_accumulation_matches_full_weighted_hierarchical_loss_and_gradient(self):
        import torch

        weights = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])
        # Three rows represent a partial final accumulation window.
        labels = torch.tensor([0, 3, 4])
        logits = torch.tensor([[0.1, -0.2, 0.3, 0.5, -0.8],
                               [0.4, 0.3, -0.6, 0.8, 0.2],
                               [-0.7, 0.1, 0.4, -0.2, 0.5]], requires_grad=True)
        full = _weighted_hierarchical_loss(logits, labels, weights)
        expected = torch.autograd.grad(full, logits)[0]
        split_logits = logits.detach().clone().requires_grad_()
        split = sum(_weighted_hierarchical_loss(split_logits[i:i + 1], labels[i:i + 1], weights,
                                               num_items_in_batch=3) for i in range(3))
        actual = torch.autograd.grad(split, split_logits)[0]
        torch.testing.assert_close(split, full)
        torch.testing.assert_close(actual, expected)
        binary = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, 0], (labels > 0).float(), reduction="none")
        subtype = torch.nn.functional.cross_entropy(logits[1:, 1:], labels[1:] - 1, reduction="none")
        decomposed = (weights[labels] * (binary + 2 * torch.cat((torch.zeros(1), subtype)))).sum() / 3
        torch.testing.assert_close(full, decomposed)
        unweighted = _weighted_hierarchical_loss(logits, labels, torch.ones(5))
        torch.testing.assert_close(unweighted, (binary.sum() + 2 * subtype.sum()) / 3)
        # Doubling subtype strength must not double the detection gradient.
        original = torch.nn.functional.nll_loss(
            _joint_log_probabilities(logits), labels, weight=weights, reduction="sum") / 3
        original_gradient = torch.autograd.grad(original, logits)[0]
        torch.testing.assert_close(expected[:, 0], original_gradient[:, 0])
        torch.testing.assert_close(expected[:, 1:], 2 * original_gradient[:, 1:])

    def test_clean_loss_retains_sample_weight_at_microbatch_one(self):
        import torch

        logits = torch.zeros(1, 5, requires_grad=True)
        label = torch.tensor([0])
        loss = _weighted_hierarchical_loss(logits, label, torch.tensor([0.5, 1., 1., 1., 1.]))
        torch.testing.assert_close(loss, torch.log(torch.tensor(2.)) * 0.5)
        gradient = torch.autograd.grad(loss, logits)[0]
        torch.testing.assert_close(gradient[:, 0], torch.tensor([0.25]))
        torch.testing.assert_close(gradient[:, 1:], torch.zeros(1, 4), atol=0, rtol=0)

    def test_missing_classes_and_invalid_powers_fail(self):
        records = [{"label": label} for label in range(5)]
        for power in (-1, 2, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(power=power), self.assertRaises(ValueError):
                _class_weights(records, power)
        with self.assertRaises(ValueError):
            _class_weights(records[:-1], 0.5)


class HierarchicalProbabilityTests(unittest.TestCase):
    def test_composition_normalizes_and_matches_binary_conditional_product(self):
        import torch

        logits = torch.tensor([[0.8, -1., 0.3, 2., -0.4], [-0.6, 1., -0.5, 0.2, 0.4]])
        probabilities = _joint_log_probabilities(logits).exp()
        q = logits[:, :1].sigmoid()
        expected = torch.cat((1 - q, q * logits[:, 1:].softmax(-1)), dim=-1)
        torch.testing.assert_close(probabilities, expected)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))
        # A binary majority for errors does not force an error subtype prediction.
        soft = _joint_log_probabilities(torch.tensor([[0.4, 0., 0., 0., 0.]])).exp()
        self.assertGreater(float(soft[:, 1:].sum()), 0.5)
        self.assertEqual(int(soft.argmax(-1)), 0)

    def test_extreme_logits_keep_finite_log_likelihood_and_gradients(self):
        import torch

        logits = torch.tensor([[1000., 1000., -1000., -500., 0.],
                               [-1000., -1000., 1000., 0., 500.]], requires_grad=True)
        log_probabilities = _joint_log_probabilities(logits)
        self.assertTrue(torch.isfinite(log_probabilities).all().item())
        torch.testing.assert_close(log_probabilities.exp().sum(-1), torch.ones(2))
        loss = _weighted_hierarchical_loss(logits, torch.tensor([2, 4]), torch.ones(5))
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, logits)[0]).all().item())

    def test_clean_rows_have_zero_subtype_gradient_even_with_nonuniform_weights(self):
        import torch

        logits = torch.tensor([[2., -3., 1., 5., -2.], [-1., 4., 0., -6., 3.]], requires_grad=True)
        loss = _weighted_hierarchical_loss(logits, torch.zeros(2, dtype=torch.long),
                                           torch.tensor([0.7, 1., 2., 3., 4.]))
        gradient = torch.autograd.grad(loss, logits)[0]
        torch.testing.assert_close(gradient[:, 1:], torch.zeros(2, 4), atol=0, rtol=0)
        self.assertTrue((gradient[:, 0] > 0).all().item())

class QwenClassifierTests(unittest.TestCase):
    """Exercise actual decoder pooling and adapter training without downloads."""

    def tiny_components(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForSequenceClassification

        torch.manual_seed(42)
        vocab = {"<|endoftext|>": 0, "<unk>": 1, "<eos>": 2,
                 "Tôi": 3, "đọc": 4, "sách": 5, "rất": 6, "hay": 7}
        backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        backend.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                           eos_token="<eos>", pad_token="<eos>")
        self.assertFalse(_configure_padding(tokenizer))
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertNotEqual(tokenizer.pad_token_id, tokenizer.eos_token_id)
        config = Qwen3Config(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=8, max_position_embeddings=64, num_labels=5,
                             pad_token_id=0, eos_token_id=2, attention_dropout=0.0,
                             problem_type="single_label_classification")
        config._attn_implementation = "eager"
        return Qwen3ForSequenceClassification(config), tokenizer

    def test_last_real_token_pooling_and_batch_padding_invariance(self):
        import torch

        model, tokenizer = self.tiny_components()
        model.eval()
        records = [{"id": "short", "source": "Tôi <eos>", "label": 0},
                   {"id": "long", "source": "Tôi đọc sách rất hay", "label": 4}]
        preprocessing = _preprocessing({"training": {"max_length": 32}})
        tokens = tokenizer([row["source"] for row in records], padding=True, return_tensors="pt")
        tokens.pop("token_type_ids", None)
        with torch.inference_mode():
            hidden = model.model(**tokens).last_hidden_state
            expected = model.score(hidden)[torch.arange(2), tokens["attention_mask"].sum(-1) - 1]
            actual = model(**tokens).logits
        torch.testing.assert_close(actual, expected)
        alone = _predict(model, tokenizer, records, preprocessing, "cpu", 1)
        batched = _predict(model, tokenizer, records, preprocessing, "cpu", 2)
        torch.testing.assert_close(torch.tensor([row["probabilities"] for row in alone]),
                                   torch.tensor([row["probabilities"] for row in batched]),
                                   atol=1e-6, rtol=1e-6)

    def test_raw_text_encoding_never_includes_correction_targets(self):
        _, tokenizer = self.tiny_components()
        preprocessing = _preprocessing({"training": {"max_length": 512}})
        records = [{"id": 1, "source": "Tôi đọc sách", "label": 2, "target": "correction secret"}]
        encoded = _encoded(tokenizer, records, preprocessing)
        changed = _encoded(tokenizer, [{**records[0], "target": "different correction"}], preprocessing)
        self.assertEqual(encoded, changed)
        self.assertEqual(encoded[0]["input_ids"], [3, 4, 5])
        self.assertEqual(encoded[0]["labels"], 2)
        self.assertNotIn("target", encoded[0])

    def test_cpu_tiny_lora_score_updates_and_saved_adapter_reload(self):
        import json
        import torch
        from safetensors.torch import load_file

        model, tokenizer = self.tiny_components()
        initial_score = model.score.weight.detach().clone()
        records = [{"id": i, "source": "Tôi đọc sách", "label": i % 5, "target": "unused"}
                   for i in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            model.save_pretrained(base)
            tokenizer.save_pretrained(base)
            config = {
                "model": {"name": str(base), "revision": "local", "load_in_4bit": False,
                          "compute_dtype": "float32", "quant_type": "nf4", "use_double_quant": True,
                          "attn_implementation": "eager"},
                "lora": {"r": 2, "lora_alpha": 4, "lora_dropout": 0.3,
                         "target_modules": ["q_proj", "v_proj"], "bias": "none"},
                "training": {"max_length": 32, "effective_batch_size": 2,
                             "args": {"num_train_epochs": 1, "max_steps": 1,
                                      "per_device_train_batch_size": 1, "per_device_eval_batch_size": 2,
                                      "learning_rate": 0.05, "warmup_steps": 0, "weight_decay": 0.0,
                                      "gradient_checkpointing": True, "optim": "adamw_torch",
                                      "logging_steps": 1, "disable_tqdm": True}},
            }
            checkpoint = root / "classifier"
            stats = _fit(config, records, None, checkpoint, root, "cpu", 42, _preprocessing(config))
            self.assertEqual(stats["optimizer_steps"], 1)
            self.assertLess(stats["trainable_parameters"], stats["total_parameters"])
            saved = json.loads((checkpoint / "classifier_config.json").read_text())
            self.assertEqual(saved["model"], config["model"])
            self.assertEqual(saved["lora"]["modules_to_save"], ["score"])
            self.assertEqual(saved["architecture"], _architecture())
            self.assertEqual(saved["training_algorithm"]["beta"], 1.0)
            self.assertEqual(saved["training_algorithm"]["training_forwards"], 2)
            self.assertEqual(saved["training_algorithm"]["lora_dropout"], 0.3)
            state = load_file(str(checkpoint / "adapter_model.safetensors"))
            head = next(value for key, value in state.items() if key.endswith(".score.weight"))
            self.assertFalse(torch.equal(head, initial_score))
            self.assertTrue(any("lora_B" in key and torch.count_nonzero(value).item()
                                for key, value in state.items()))
            single = classify(checkpoint, records, "cpu", 1)
            batched = classify(checkpoint, records, "cpu", 5)
            self.assertEqual([row["id"] for row in batched], list(range(5)))
            values = torch.tensor([row["probabilities"] for row in batched])
            self.assertEqual(tuple(values.shape), (5, 5))
            self.assertTrue(torch.isfinite(values).all().item())
            torch.testing.assert_close(values.sum(-1), torch.ones(5))
            torch.testing.assert_close(values, torch.tensor([row["probabilities"] for row in single]),
                                       atol=1e-6, rtol=1e-6)
            # Reloaded raw scores must compose to precisely the public probabilities.
            from peft import PeftModel
            reloaded = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
            reloaded.eval()
            tokens = tokenizer([row["source"] for row in records], padding=True, return_tensors="pt")
            with torch.inference_mode():
                expected = _joint_log_probabilities(reloaded(**tokens).logits).exp()
            torch.testing.assert_close(values, expected, atol=1e-6, rtol=1e-6)
            # Old flat-head checkpoints must not silently acquire hierarchical semantics.
            for architecture in (None, {**_architecture(), "version": 0}):
                incompatible = {**saved}
                if architecture is None:
                    incompatible.pop("architecture")
                else:
                    incompatible["architecture"] = architecture
                (checkpoint / "classifier_config.json").write_text(json.dumps(incompatible), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "architecture metadata"):
                    classify(checkpoint, records, "cpu", 1)


if __name__ == "__main__":
    unittest.main()
