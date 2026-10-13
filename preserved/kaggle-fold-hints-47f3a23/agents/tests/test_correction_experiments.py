import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from model.correction import (_parse_correction, _prompt_ids,
                              prepare_training_inference, validate_initial_adapter)
from model.correction_edits import apply_edits, encode_edits
from model.correction_experiments import experiment_variants, paired_rows
from model.correction_reranking import reranking_config, select_candidate


class LocalizedEditTests(unittest.TestCase):
    def test_round_trip_unicode_insert_delete_and_untouched_spans(self):
        pairs = [("Tôi 😀 đi học.", "Tôi 😀 đã đi học."), ("a\u0301😀b", "a\u0301b"),
                 ("Có lỗi!", "Có lỗi."), ("abc", ""), ("", "xin chào"), ("đúng", "đúng")]
        for source, target in pairs:
            with self.subTest(source=source, target=target):
                value = encode_edits(source, target)
                self.assertEqual(value, encode_edits(source, target))
                self.assertEqual(apply_edits(source, value), target)
        self.assertEqual(apply_edits("a😀bc", {"edits": [
            {"start": 1, "end": 2, "replacement": "đ"},
            {"start": 4, "end": 4, "replacement": "!"}]}), "ađbc!")

    def test_strict_schema_boundaries_never_fall_back(self):
        invalid = [[], {"edits": None}, {"edits": [], "extra": 1},
                   {"edits": [{"start": True, "end": 1, "replacement": "x"}]},
                   {"edits": [{"start": 0.0, "end": 1, "replacement": "x"}]},
                   {"edits": [{"start": -1, "end": 1, "replacement": "x"}]},
                   {"edits": [{"start": 0, "end": 5, "replacement": "x"}]},
                   {"edits": [{"start": 2, "end": 1, "replacement": "x"}]},
                   {"edits": [{"start": 0, "end": 1, "replacement": None}]},
                   {"edits": [{"start": 0, "end": 1, "replacement": "x", "extra": 0}]},
                   {"edits": [{"start": 2, "end": 3, "replacement": ""},
                              {"start": 0, "end": 1, "replacement": ""}]},
                   {"edits": [{"start": 0, "end": 2, "replacement": ""},
                              {"start": 1, "end": 3, "replacement": ""}]},
                   {"edits": [{"start": 1, "end": 1, "replacement": "a"},
                              {"start": 1, "end": 1, "replacement": "b"}]}]
        for value in invalid:
            with self.subTest(value=value):
                target, error = _parse_correction(json.dumps(value), "abcd", "edits")
                self.assertEqual(target, "")
                self.assertIsNotNone(error)
        self.assertEqual(_parse_correction('{"edits":[]}', "abcd", "edits"), ("abcd", None))
        target, error = _parse_correction('{"edits":[],"edits":[]}', "abcd", "edits")
        self.assertEqual(target, "")
        self.assertIsNotNone(error)



class HintAblationTests(unittest.TestCase):
    def test_no_hint_removes_system_label_reference_and_descriptions(self):
        inference = fixture_config()["inference"]
        inference["hint_mode"] = "none"
        tokenizer = MagicMock()
        _prompt_ids(tokenizer, "source only", 3, inference)
        messages = tokenizer.apply_chat_template.call_args.args[0]
        self.assertIn("source only", messages[1]["content"])
        joined = "\n".join(row["content"] for row in messages)
        self.assertNotIn("Nhãn", joined)
        self.assertNotIn("gợi ý", joined)
        for description in inference["hint_descriptions"]:
            self.assertNotIn(description, joined)


    def test_variants_do_not_mutate_saved_inference_and_pair_typed_ids(self):
        saved = prepare_training_inference(fixture_config())
        original = copy.deepcopy(saved)
        variants = experiment_variants(saved, hint_ablation=True, rerank=True)
        self.assertEqual(saved, original)
        self.assertEqual(variants["saved"], original)
        self.assertEqual(variants["hint_none"]["hint_mode"], "none")
        self.assertTrue(variants["rerank_none"]["reranking"]["enabled"])
        records = [{"id": 1, "source": "a", "label": 0, "target": "a"},
                   {"id": "1", "source": "b", "label": 1, "target": "c"}]
        predictions = {"saved": [{"id": "1", "target": "c"}, {"id": 1, "target": "a"}]}
        rows = paired_rows(records, predictions)
        self.assertEqual(rows[0]["variants"]["saved"]["target"], "a")
        with self.assertRaises(ValueError):
            paired_rows(records, {"saved": [{"id": 1}, {"id": 1}]})


class RankingContractTests(unittest.TestCase):
    def candidate(self, target, grammar, semantic):
        return {"target": target, "parse_error": None,
                "scores": {"grammar_probability": grammar, "semantic_cosine": semantic}}

    def test_quality_beats_edit_cost_and_source_has_no_gate(self):
        config = reranking_config({})
        candidates = [self.candidate("bad source", .1, 1.), self.candidate("good correction", .99, .99)]
        self.assertEqual(select_candidate("bad source", candidates, config), 1)
        candidates = [self.candidate("same", .9, .9), self.candidate("changed", .9, .9)]
        self.assertEqual(select_candidate("same", candidates, config), 0)

    def test_invalid_candidate_never_receives_fabricated_scores(self):
        candidates = [{"target": "", "parse_error": "invalid JSON"}, self.candidate("source", .7, 1.)]
        self.assertEqual(select_candidate("source", candidates, reranking_config({})), 1)
        self.assertNotIn("scores", candidates[0])
        with self.assertRaises(ValueError):
            select_candidate("source", candidates[:1], reranking_config({}))
        with self.assertRaises(ValueError):
            reranking_config({"num_candidates": True})
        with self.assertRaises(ValueError):
            select_candidate("source", [self.candidate("source", float("nan"), 1.)], reranking_config({}))


def fixture_config():
    return {"model": {"name": "base", "revision": "pinned"},
            "lora": {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"], "bias": "none"},
            "inference": {"prompt": "Nhãn dự đoán là gợi ý. Exact legacy prompt.",
                          "hint_descriptions": [f"description-{i}" for i in range(5)]},
            "generation": {"do_sample": False}, "training": {"max_length": 512}}


class WarmstartContractTests(unittest.TestCase):
    def test_legacy_sentence_and_early_incompatibility(self):
        config = fixture_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter_config = {**config["lora"], "task_type": "CAUSAL_LM", "peft_type": "LORA",
                              "base_model_name_or_path": "base", "revision": "pinned"}
            for name, value in (("loading.json", config["model"]), ("inference.json", config["inference"]),
                                ("adapter_config.json", adapter_config)):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            (root / "adapter_model.safetensors").write_bytes(b"weights-existence-fixture")
            self.assertIn("fresh optimizer", validate_initial_adapter(root, config)["continuation"])
            for section, field, value in (("model", "revision", "other"), ("lora", "r", 8),
                                          ("inference", "output_format", "edits"),
                                          ("inference", "hint_mode", "none"),
                                          ("inference", "prompt", "changed interface")):
                changed = copy.deepcopy(config)
                changed[section][field] = value
                with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                    validate_initial_adapter(root, changed)
            adapter_config["task_type"] = "SEQ_CLS"
            (root / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CAUSAL_LM"):
                validate_initial_adapter(root, config)


if __name__ == "__main__":
    unittest.main()
