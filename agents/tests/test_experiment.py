"""Committed recipes, private bundle identity, and native T4 precision."""
import contextlib
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from model.classification import placement_preflight
from model.correction import _dtype
from tools.run_experiment import model_size, prepare_run, resolve_inputs

PROJECT = Path(__file__).resolve().parents[2]


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "input"
        self.inputs.mkdir()
        self.recipe = json.loads((PROJECT / "configs/experiment.json").read_text())
        rows = [{"ID": 1, "Input": "private source", "Output": {"Error_Type_Id": 0, "Fixed Sentence": None}}]
        self.files = {"VGEC 2026_v1.json": json.dumps(rows).encode(), "VGEC_classifier.json": b'{"fixture":"classifier"}',
                      "VGEC_balanced.json": b'{"fixture":"corrector"}', "public_test.json": b'[{"ID":1,"Input":"public input"}]'}
        for role, name in (("original", "VGEC 2026_v1.json"), ("classification", "VGEC_classifier.json"), ("correction", "VGEC_balanced.json")):
            self.recipe["data_sha256"][role] = hashlib.sha256(self.files[name]).hexdigest()
        with zipfile.ZipFile(self.inputs / self.recipe["data_archive"], "w") as archive:
            for name, content in self.files.items():
                archive.writestr(name, content)
        self.recipe_path = self.root / "experiment.json"
        self.recipe_path.write_text(json.dumps(self.recipe))

    def test_canonical_archive_wins_over_stale_loose_json(self):
        (self.inputs / "VGEC_classifier.json").write_text("stale input")
        paths = resolve_inputs(self.recipe, self.inputs, self.root / "resolved")
        self.assertEqual(paths["classification"].read_bytes(), self.files["VGEC_classifier.json"])
        self.assertEqual(paths["submission"].read_bytes(), self.files["public_test.json"])

    def test_input_hash_mismatch_stops_before_model_loading(self):
        self.recipe["data_sha256"]["classification"] = "0" * 64
        self.recipe_path.write_text(json.dumps(self.recipe))
        with patch("tools.run_experiment.model_size") as load, self.assertRaisesRegex(ValueError, "differs from committed recipe"):
            prepare_run(self.recipe_path, self.inputs, self.root / "outputs", world_size=1)
        load.assert_not_called()

    def test_missing_submission_never_becomes_validation_export(self):
        (self.inputs / self.recipe["data_archive"]).unlink()
        for name, content in self.files.items():
            if name != "public_test.json":
                (self.inputs / name).write_bytes(content)
        with patch("tools.run_experiment.PROJECT_ROOT", self.root), self.assertRaises(FileNotFoundError):
            resolve_inputs(self.recipe, self.inputs, self.root / "resolved")

    def test_classifier_screening_recipe_does_not_require_submission_input(self):
        self.recipe["mode"] = "task1"
        self.recipe_path.write_text(json.dumps(self.recipe))
        (self.inputs / self.recipe["data_archive"]).unlink()
        for name, content in self.files.items():
            if name != "public_test.json":
                (self.inputs / name).write_bytes(content)
        with patch("tools.run_experiment.model_size", return_value={"total_upper_bound": 123}):
            _, command = prepare_run(self.recipe_path, self.inputs, self.root / "outputs", world_size=1)
        self.assertNotIn("--submission-data", command)

    def test_archive_member_paths_cannot_escape_destination(self):
        with zipfile.ZipFile(self.inputs / self.recipe["data_archive"], "w") as archive:
            for name, content in self.files.items():
                archive.writestr("../../" + name, content)
        paths = resolve_inputs(self.recipe, self.inputs, self.root / "resolved")
        self.assertTrue(all(path.parent == self.root / "resolved" for path in paths.values()))

    def test_prepared_run_keeps_recipe_and_effective_batch_and_sets_real_input(self):
        with patch("tools.run_experiment.model_size", return_value={"total_upper_bound": 123}):
            work, command = prepare_run(self.recipe_path, self.inputs, self.root / "outputs", world_size=1)
        config = json.loads((work / "classification_config.json").read_text())
        self.assertEqual(config["run"]["world_size"], 1)
        self.assertEqual(config["training"]["effective_batch_size"], 8)
        self.assertEqual(command[command.index("--submission-data") + 1], str(work / "inputs/public_test.json"))
        manifest = json.loads((work / "experiment_manifest.json").read_text())
        self.assertEqual(manifest["recipe_sha256"], hashlib.sha256(self.recipe_path.read_bytes()).hexdigest())
        self.assertEqual(manifest["recipe"], self.recipe)

    def test_real_meta_model_counts_adapter_and_head_and_does_not_count_packed_storage(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForSequenceClassification
        with tempfile.TemporaryDirectory() as folder:
            config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                 num_labels=5, tie_word_embeddings=False)
            config.save_pretrained(folder)
            settings = {"task": "classification", "model": {"name": folder, "revision": "main", "load_in_4bit": True},
                        "lora": {"r": 2, "target_modules": ["q_proj"], "modules_to_save": ["score"]}}
            result = model_size(settings)
            model = Qwen2ForSequenceClassification(config)
            self.assertEqual(result["base"], sum(p.numel() for p in model.parameters()))
            self.assertEqual(result["adapter"], 2 * (16 + 16))
            self.assertEqual(result["saved_module_copies"], 16 * 5)
            self.assertGreater(result["total_upper_bound"], result["base"])


class PrecisionTests(unittest.TestCase):
    def test_t4_emulated_bf16_is_not_selected_for_either_stage(self):
        import torch
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.device", side_effect=lambda *_: contextlib.nullcontext()), \
                patch("torch.cuda.is_bf16_supported", side_effect=lambda including_emulation=True: including_emulation):
            placement = placement_preflight({"load_in_4bit": True, "compute_dtype": "bfloat16"}, "cuda:0")
            self.assertEqual(placement["compute_dtype"], "float16")
            self.assertEqual(_dtype({"compute_dtype": "auto", "device_map": "single"}, "cuda"), torch.float16)
            self.assertEqual(_dtype({"compute_dtype": "bfloat16"}, "cuda"), torch.float16)


if __name__ == "__main__":
    unittest.main()
