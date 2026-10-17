"""Per-model training choices must survive fitting, validation and artifact reuse."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from model.runtime import write_json


class MemberTrainingTests(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).resolve().parents[2] /
                           "configs/retrain_best_classification.json").read_text())

    def test_overrides_are_applied_without_mutating_shared_recipe(self):
        from model.runtime import classifier_member_config
        config = self.config()
        entry = config["ensemble"]["members"][0]
        entry["training_args"] = {"learning_rate": 2e-5,
                                  "num_train_epochs": 3, "warmup_steps": 70}
        before = copy.deepcopy(config)
        child = classifier_member_config(config, entry)
        self.assertEqual(child["training"]["args"]["learning_rate"], 2e-5)
        self.assertEqual(child["training"]["args"]["num_train_epochs"], 3)
        self.assertEqual(child["run"]["seed"], 42)
        self.assertNotIn("ensemble", child)
        child["lora"]["modules_to_save"].append("mutation")
        child["model"]["revision"] = "mutation"
        self.assertEqual(config, before)
        sibling = classifier_member_config(config, config["ensemble"]["members"][1])
        self.assertEqual(sibling["training"], before["training"])

    def test_invalid_or_unrelated_overrides_fail_during_config_loading(self):
        from model.runtime import load_config
        for override in ({"seed": 43}, {"max_length": 32},
                         {"learning_rate": float("nan")}, {"learning_rate": True},
                         {"num_train_epochs": 0}, {"warmup_steps": -1},
                         {"warmup_steps": 1.5}, [2e-5]):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as folder:
                config = self.config()
                config["ensemble"]["members"][0]["training_args"] = override
                path = Path(folder) / "config.json"
                write_json(path, config)
                with self.assertRaises(ValueError):
                    load_config(path, "classification")

    def test_native_size_audit_does_not_mutate_lora_configuration(self):
        from transformers import Qwen2Config
        from tools.run_experiment import model_size
        with tempfile.TemporaryDirectory() as folder:
            Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=2).save_pretrained(folder)
            config = self.config()
            config.pop("ensemble")
            config["model"].update(name=folder, revision="main")
            before = copy.deepcopy(config)
            self.assertGreater(model_size(config)["total_upper_bound"], 0)
            self.assertEqual(config, before)

    def test_ensemble_accepts_requested_member_training_and_rejects_drift(self):
        from agents.tests.test_voting_classifier import EnsembleArtifactTests
        from model.classifier_ensemble import build, validate
        from model.runtime import classifier_member_config
        for drift in (False, True):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                config = EnsembleArtifactTests().fixture(root)
                entry = config["ensemble"]["members"][0]
                entry["training_args"] = {"learning_rate": 2e-5}
                child_path = root / "members/qwen7b/config.json"
                child = json.loads(child_path.read_text())
                child["training"] = classifier_member_config(config, entry)["training"]
                if drift:
                    child["training"]["args"]["learning_rate"] = 3e-5
                write_json(child_path, child)
                if drift:
                    with self.assertRaisesRegex(ValueError, "training recipe"):
                        build(root, config)
                else:
                    build(root, config)
                    self.assertEqual(validate(root)["ensemble"], config["ensemble"])


if __name__ == "__main__":
    unittest.main()
