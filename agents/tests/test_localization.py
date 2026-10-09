"""CPU behavior/gradient/reload coverage; no downloads or mock model forwards."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from model.classification import _fit, _preprocessing, classify, _weighted_hierarchical_loss
from model.localization import (IGNORE, LocalizationCollator, LocalizationHead, align_pair,
                                encode_localization, lexical_units, localization_loss, load_head,
                                save_head)


def tiny_components():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForSequenceClassification
    vocab = {token: i for i, token in enumerate(["<pad>", "<unk>", "<eos>", "Tôi", "đọc", "sách", "báo", "rất", "hay", ".", "Bạn", "nhanh", "a", "b", "c", "é", "é", "!"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.padding_side = "right"
    config = Qwen3Config(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, max_position_embeddings=64, num_labels=5,
                        pad_token_id=0, eos_token_id=2, attention_dropout=0.0,
                        problem_type="single_label_classification")
    config._attn_implementation = "eager"
    return Qwen3ForSequenceClassification(config), tokenizer


def row(source, target, label=1, identifier=1):
    return {"id": identifier, "source": source, "target": target, "label": label}


class AlignmentTests(unittest.TestCase):
    def test_substitution_deletion_and_all_gap_positions(self):
        examples = [
            ("Tôi đọc sách.", "Tôi đọc báo.", [(8, 12)], []),
            ("Tôi rất đọc sách.", "Tôi đọc sách.", [(4, 7)], []),
            ("đọc sách.", "Tôi đọc sách.", [], [0]),
            ("Tôi sách.", "Tôi đọc sách.", [], [1]),
            ("Tôi đọc.", "Tôi đọc.!", [], [3]),
        ]
        _, tokenizer = tiny_components()
        for source, target, spans, gaps in examples:
            with self.subTest(target=target):
                alignment = align_pair(source, target, 1)
                self.assertTrue(alignment["eligible"], alignment)
                self.assertEqual(alignment["error_spans"], spans)
                self.assertEqual(alignment["insert_gaps"], gaps)
                encoded, audit = encode_localization(tokenizer, [row(source, target)], 32)
                self.assertTrue(audit["rows"][0]["eligible"])
                self.assertEqual(sum(x == 1 for x in encoded[0]["error_labels"]), len(spans))
                self.assertEqual(sum(x == 1 for x in encoded[0]["insert_labels"]), len(gaps))
                self.assertEqual(encoded[0]["input_ids"], tokenizer(source)["input_ids"])

    def test_unicode_offsets_are_original_codepoints_and_punctuation_separate(self):
        self.assertEqual(lexical_units("é é!"), [("é", 0, 1), ("é", 2, 4), ("!", 4, 5)])
        alignment = align_pair("é é!", "é c!", 1)
        self.assertEqual(alignment["error_spans"], [(2, 4)])
        _, tokenizer = tiny_components()
        encoded, audit = encode_localization(tokenizer, [row("é é!", "é c!")], 32)
        self.assertEqual(audit["rows"][0]["offsets"], [(0, 1), (2, 4), (4, 5)])
        self.assertEqual(encoded[0]["error_labels"], [0, 1, 0])

    def test_exclusions_never_become_negative_examples(self):
        examples = [row("Tôi đọc...", "Tôi đọc."), row("Tôi đọc…", "Tôi đọc."),
                    row("Tôi đọc.", "(Câu đúng ngữ pháp)."), row("Tôi đọc.", "Tôi (rất) đọc."),
                    row("Tôi đọc:", "Tôi rất đọc:"), row("Tôi đọc", "Tôi rất đọc"),
                    row("a a.", "a."), row("Tôi đọc.", "Tôi  đọc."),
                    row("Tôi đọc.", "Tôi đọc."), row("Tôi đọc.", "Tôi rất đọc.", label=0)]
        _, tokenizer = tiny_components()
        encoded, audit = encode_localization(tokenizer, examples, 32)
        self.assertEqual(audit["summary"]["eligible_rows"], 0)
        for feature in encoded:
            self.assertTrue(all(x == IGNORE for x in feature["error_labels"] + feature["insert_labels"]))
        clean, clean_audit = encode_localization(tokenizer, [row("Tôi đọc.", "Tôi đọc.", 0)], 32)
        self.assertTrue(clean_audit["rows"][0]["eligible"])
        self.assertTrue(all(x == 0 for x in clean[0]["error_labels"]))

    def test_truncation_hides_edits_without_false_clean(self):
        _, tokenizer = tiny_components()
        encoded, audit = encode_localization(tokenizer, [row("Tôi đọc sách.", "Tôi đọc báo.")], 2)
        self.assertEqual(audit["rows"][0]["reason"], "truncated_pair")
        self.assertEqual(encoded[0]["error_labels"], [IGNORE, IGNORE])
        self.assertEqual(encoded[0]["insert_labels"], [IGNORE] * 3)

    def test_padding_ignored_and_audit_reconciles(self):
        _, tokenizer = tiny_components()
        encoded, audit = encode_localization(tokenizer, [row("đọc.", "Tôi đọc."), row("Tôi đọc sách.", "Tôi đọc sách.", 0, 2)], 32)
        batch = LocalizationCollator(tokenizer)(encoded)
        self.assertEqual(batch["error_labels"][0].tolist(), [0, 0, IGNORE, IGNORE])
        self.assertEqual(batch["insert_labels"][0].tolist(), [1, 0, 0, IGNORE, IGNORE])
        self.assertEqual(audit["summary"]["eligible_rows"], 2)
        self.assertEqual(audit["summary"]["insert_counts"]["1"], 1)


class LossAndHeadTests(unittest.TestCase):
    def test_zero_ignored_gradient_and_full_window_accumulation_equivalence(self):
        torch.manual_seed(42)
        error = torch.randn(3, 4, requires_grad=True)
        insert = torch.randn(3, 5, requires_grad=True)
        errors = torch.tensor([[0, 1, IGNORE, IGNORE], [IGNORE] * 4, [1, 0, 0, 0]])
        gaps = torch.tensor([[1, 0, 0, IGNORE, IGNORE], [IGNORE] * 5, [0, 0, 1, 0, 0]])
        full = localization_loss(error, insert, errors, gaps)
        expected = torch.autograd.grad(full, (error, insert))
        split = sum(localization_loss(error[i:i+1], insert[i:i+1], errors[i:i+1], gaps[i:i+1], 3) for i in range(3))
        actual = torch.autograd.grad(split, (error, insert))
        torch.testing.assert_close(split, full)
        for gradient, reference, labels in zip(actual, expected, (errors, gaps)):
            torch.testing.assert_close(gradient, reference)
            self.assertEqual(torch.count_nonzero(gradient[labels == IGNORE]).item(), 0)
        loss = localization_loss(error, insert, torch.full_like(errors, IGNORE), torch.full_like(gaps, IGNORE))
        gradients = torch.autograd.grad(loss, (error, insert))
        self.assertEqual(loss.item(), 0)
        self.assertTrue(all(torch.count_nonzero(g).item() == 0 for g in gradients))

    def test_right_padded_end_gap_uses_boundary_not_padding_state(self):
        head = LocalizationHead(3)
        hidden = torch.randn(2, 5, 3, requires_grad=True)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
        valid = torch.tensor([[True, True, True, False, False, False], [True] * 6])
        _, gaps = head(hidden, mask, valid)
        alone = head(hidden[:1, :2], mask[:1, :2], valid[:1, :3])[1]
        torch.testing.assert_close(gaps[:1, :3], alone)
        expected = head.insert(torch.cat((hidden[0, 1], head.end)))
        torch.testing.assert_close(gaps[0, 2], expected.squeeze())
        gradients = torch.autograd.grad(gaps[0, 2], (hidden, head.end))
        self.assertEqual(torch.count_nonzero(gradients[0][0, 2:]).item(), 0)
        self.assertGreater(gradients[1].abs().sum().item(), 0)

    def test_no_eligible_rows_give_zero_backbone_and_head_gradients(self):
        head = LocalizationHead(3)
        hidden = torch.randn(2, 4, 3, requires_grad=True)
        error_labels = torch.full((2, 4), IGNORE)
        insert_labels = torch.full((2, 5), IGNORE)
        loss = localization_loss(*head(hidden, torch.ones(2, 4), insert_labels != IGNORE),
                                 error_labels, insert_labels)
        loss.backward()
        self.assertEqual(torch.count_nonzero(hidden.grad).item(), 0)
        self.assertTrue(all(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() == 0
                            for parameter in head.parameters()))

    def test_insertion_only_backbone_gradients_update_and_head_reload(self):
        from peft import LoraConfig, TaskType, get_peft_model
        torch.manual_seed(42)
        base, tokenizer = tiny_components()
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                              target_modules=["q_proj", "v_proj"], task_type=TaskType.SEQ_CLS,
                                              modules_to_save=["score"]))
        head = LocalizationHead(16)
        model.localization_head = head
        features, _ = encode_localization(tokenizer, [row("đọc sách.", "Tôi đọc sách.")], 32)
        batch = LocalizationCollator(tokenizer)(features)
        hidden = model.get_base_model().model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        logits = head(hidden, batch["attention_mask"], batch["insert_labels"] != IGNORE)
        loss = localization_loss(*logits, batch["error_labels"], batch["insert_labels"])
        before = head.insert.weight.detach().clone()
        loss.backward()
        self.assertGreater(head.start.grad.abs().sum().item(), 0)
        self.assertGreater(head.insert.weight.grad.abs().sum().item(), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item() > 0 for name, p in model.named_parameters() if "lora_B" in name))
        torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05).step()
        self.assertFalse(torch.equal(before, head.insert.weight))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "localization_head.pt"
            save_head(head, path)
            reloaded = load_head(path)
            for a, b in zip(head(hidden, batch["attention_mask"]), reloaded(hidden, batch["attention_mask"])):
                torch.testing.assert_close(a, b)

    def test_actual_trainer_partial_window_normalization_and_inference_reload(self):
        """Actual SGD updates match full batch with a partial accumulation window."""
        torch.manual_seed(42)
        model, tokenizer = tiny_components()
        records = [row("Tôi đọc sách.", "Tôi đọc sách.", 0, 0),
                   row("Tôi đọc sách.", "Tôi đọc báo.", 1, 1),
                   row("đọc sách.", "Tôi đọc sách.", 2, 2),
                   row("Tôi rất đọc sách.", "Tôi đọc sách.", 3, 3),
                   row("Tôi sách.", "Tôi đọc sách.", 4, 4)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            model.save_pretrained(base)
            tokenizer.save_pretrained(base)
            config = {"model": {"name": str(base), "revision": "local", "load_in_4bit": False,
                                 "compute_dtype": "float32", "quant_type": "nf4", "use_double_quant": True,
                                 "attn_implementation": "eager"},
                      "lora": {"r": 2, "lora_alpha": 4, "lora_dropout": 0,
                               "target_modules": ["q_proj", "v_proj"], "bias": "none"},
                      "training": {"max_length": 32, "effective_batch_size": 8,
                                   "args": {"num_train_epochs": 1, "per_device_train_batch_size": 1,
                                            "per_device_eval_batch_size": 2, "learning_rate": 0.01,
                                            "warmup_steps": 0, "weight_decay": 0, "gradient_checkpointing": True,
                                            "optim": "sgd", "max_grad_norm": 0,
                                            "logging_steps": 1, "disable_tqdm": True}}}
            checkpoint = root / "classifier"
            stats = _fit(config, records, None, checkpoint, root, "cpu", 42, _preprocessing(config))
            self.assertEqual(stats["optimizer_steps"], 1)
            self.assertEqual(stats["localization_audit"]["eligible_rows"], 5)
            auxiliary = load_head(checkpoint / "localization_head.pt")
            self.assertTrue(torch.isfinite(auxiliary.insert.weight).all())
            self.assertGreater(auxiliary.start.abs().sum().item(), 0)
            import copy
            from safetensors.torch import load_file
            whole_config = copy.deepcopy(config)
            whole_config["training"]["effective_batch_size"] = 5
            whole_config["training"]["args"]["per_device_train_batch_size"] = 5
            whole_checkpoint = root / "whole" / "classifier"
            whole_checkpoint.parent.mkdir()
            whole_stats = _fit(whole_config, records, None, whole_checkpoint, root / "whole",
                               "cpu", 42, _preprocessing(whole_config))
            whole_auxiliary = load_head(whole_checkpoint / "localization_head.pt")
            for key, value in auxiliary.state_dict().items():
                torch.testing.assert_close(value, whole_auxiliary.state_dict()[key], atol=1e-6, rtol=1e-5)
            split_adapter = load_file(str(checkpoint / "adapter_model.safetensors"))
            whole_adapter = load_file(str(whole_checkpoint / "adapter_model.safetensors"))
            for key, value in split_adapter.items():
                torch.testing.assert_close(value, whole_adapter[key], atol=1e-6, rtol=1e-5)
            self.assertAlmostEqual(stats["training_loss"], whole_stats["training_loss"], places=5)
            checkpoint.joinpath("localization_head.pt").unlink()
            predictions = classify(checkpoint, records, "cpu", 2)
            self.assertEqual(len(predictions), 5)
            torch.testing.assert_close(torch.tensor([r["probabilities"] for r in predictions]).sum(-1), torch.ones(5))
            audit = json.loads((root / "localization_audit.json").read_text())
            self.assertEqual(audit["summary"]["rows"], 5)


if __name__ == "__main__":
    unittest.main()
