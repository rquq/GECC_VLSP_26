"""Track isolation and split-GPU NF4 contracts without native weight allocation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model.runtime import parameter_limit, load_config
from model.classification import placement_preflight
from tools.run_experiment import model_size

PROJECT = Path(__file__).resolve().parents[2]


class UnconstrainedTests(unittest.TestCase):
    def test_track_limit_is_explicit_and_defaults_to_constrained(self):
        self.assertEqual(parameter_limit({}), 8_000_000_000)
        self.assertEqual(parameter_limit({"competition_id": 1038}), 8_000_000_000)
        self.assertIsNone(parameter_limit({"competition_id": 1039}))
        for value in ["1039", True, 0, 1040]:
            with self.assertRaisesRegex(ValueError, "competition_id"):
                parameter_limit({"competition_id": value})

    def test_native_count_stays_over_8b_despite_quantization(self):
        from transformers import Qwen2Config

        with tempfile.TemporaryDirectory() as folder:
            # Large embedding on meta only: over 8B, without allocating weights.
            Qwen2Config(vocab_size=1_000_000, hidden_size=8192,
                        intermediate_size=128, num_hidden_layers=1,
                        num_attention_heads=64, num_key_value_heads=8,
                        tie_word_embeddings=False).save_pretrained(folder)
            config = {"task": "classification", "competition_id": 1039,
                      "model": {"name": folder, "revision": "main", "load_in_4bit": True},
                      "lora": {"r": 2, "target_modules": ["q_proj"], "modules_to_save": ["score"]}}
            result = model_size(config)
            self.assertGreater(result["total_upper_bound"], 8_000_000_000)
            self.assertEqual(result["competition_id"], 1039)
            self.assertIsNone(result["limit"])
            config.pop("competition_id")
            with self.assertRaisesRegex(ValueError, "8B limit"):
                model_size(config)

    def test_balanced_nf4_classifier_uses_two_gpus_with_conservative_reserve(self):
        from transformers import Qwen2Config

        with tempfile.TemporaryDirectory() as folder:
            Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=4, num_attention_heads=2,
                        num_key_value_heads=2, tie_word_embeddings=False).save_pretrained(folder)
            settings = {"name": folder, "revision": "main", "load_in_4bit": True,
                        "compute_dtype": "float16", "device_map": "balanced"}
            with patch("torch.cuda.is_available", return_value=True), \
                    patch("torch.cuda.device_count", return_value=2), \
                    patch("torch.cuda.mem_get_info", return_value=(15_526_068_224, 15_636_037_632)):
                placement = placement_preflight(settings)
                self.assertTrue(placement["quantized"])
                self.assertEqual(placement["compute_dtype"], "float16")
                self.assertEqual(set(placement["device_map"].values()), {"cuda:0", "cuda:1"})
                self.assertEqual(placement["device_map"]["score"], "cuda:1")
                for budget in placement["memory_budget"].values():
                    self.assertGreater(budget["required_bytes"] - budget["weight_bytes"], 4 * 1024**3)
                with patch("torch.cuda.mem_get_info", return_value=(3 * 1024**3, 16 * 1024**3)):
                    with self.assertRaisesRegex(RuntimeError, "no offload"):
                        placement_preflight(settings)

    def test_both_tracks_keep_separate_committed_plans(self):
        from tools.run_focused import validate_plan

        for filename, track, quantized in [("focused_plan.json", 1038, False),
                                           ("unconstrained_plan.json", 1039, True)]:
            recipe = validate_plan(json.loads((PROJECT / "configs" / filename).read_text()))
            for field in ["classification_config", "correction_config"]:
                config = load_config(PROJECT / recipe[field])
                self.assertEqual(config.get("competition_id", 1038), track)
                self.assertEqual(config["model"]["load_in_4bit"], quantized)
                self.assertGreaterEqual(config["training"]["args"]["num_train_epochs"], 5)

    def test_cache_cleanup_is_private_and_keeps_the_current_base(self):
        from model.weight_cache import MARKER, prepare_weight_cache
        import os

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / MARKER).write_text("vgec-temporary-base-cache-v1\n")
            for name in ["models--owner--current", "models--owner--previous", "adapters"]:
                (root / name).mkdir()
                (root / name / "payload").write_text("fixture")
            with patch.dict(os.environ, {"HF_HUB_CACHE": folder, "VGEC_WEIGHT_CACHE_ROOT": folder}):
                prepare_weight_cache({"name": "owner/current"})
                self.assertTrue((root / "models--owner--current/payload").is_file())
                self.assertTrue((root / "adapters/payload").is_file())
                self.assertFalse((root / "models--owner--previous").exists())
                (root / MARKER).unlink()
                with self.assertRaisesRegex(ValueError, "controller-owned"):
                    prepare_weight_cache({"name": "owner/current"})

    def test_cache_cleanup_rejects_symlink_targets(self):
        from model.weight_cache import MARKER, prepare_weight_cache
        import os

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache, external = root / "cache", root / "outside"
            cache.mkdir(); external.mkdir()
            (external / "keep").write_text("protected")
            (cache / MARKER).write_text("vgec-temporary-base-cache-v1\n")
            (cache / "models--owner--previous").symlink_to(external, target_is_directory=True)
            with patch.dict(os.environ, {"HF_HUB_CACHE": str(cache), "VGEC_WEIGHT_CACHE_ROOT": str(cache)}):
                with self.assertRaisesRegex(ValueError, "Unexpected entry"):
                    prepare_weight_cache({"name": "owner/current"})
            self.assertTrue((external / "keep").is_file())


if __name__ == "__main__":
    unittest.main()
