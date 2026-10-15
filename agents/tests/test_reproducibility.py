"""Replay real RNG streams and keep model seeds separate from partition seeds."""

import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest


class ReproducibilityTests(unittest.TestCase):
    def test_global_rng_streams_replay_and_differ_across_training_seeds(self):
        import numpy as np
        import torch
        from model.reproducibility import seed_everything

        def draw(seed):
            policy = seed_everything(seed)
            self.assertTrue(policy["strict_determinism"])
            return random.random(), np.random.rand(), torch.rand(4)

        a = draw(42)
        b = draw(42)
        c = draw(43)
        self.assertEqual(a[:2], b[:2])
        torch.testing.assert_close(a[2], b[2], atol=0, rtol=0)
        self.assertNotEqual(a[:2], c[:2])
        self.assertFalse(torch.equal(a[2], c[2]))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_child_startup_env_forces_hash_seed_and_cuda_workspace(self):
        from model.reproducibility import process_environment

        env = process_environment(
            43,
            {
                "PYTHONHASHSEED": "random",
                "CUBLAS_WORKSPACE_CONFIG": "bad",
                "KEEP": "yes",
            },
        )
        self.assertEqual(env["PYTHONHASHSEED"], "43")
        self.assertEqual(env["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(env["KEEP"], "yes")
        command = [
            sys.executable,
            "-c",
            "import json;print(json.dumps(sorted({'apple','pear','cherry','mango'},key=hash)))",
        ]
        first = subprocess.check_output(command, env={**os.environ, **env}, text=True)
        self.assertEqual(
            first,
            subprocess.check_output(command, env={**os.environ, **env}, text=True),
        )

    def test_classifier_accepts_three_fit_seeds_but_keeps_one_split_seed(self):
        from model.runtime import load_config, split_seed

        config = json.loads(
            (Path(__file__).resolve().parents[2] / "configs/baseline.json").read_text()
        )
        config.pop(
            "ensemble", None
        )  # Individual-fit API retains historical seed support.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for seed in (42, 43, 44):
                config["run"]["seed"] = seed
                config["data"]["split_seed"] = 42
                path.write_text(json.dumps(config))
                loaded = load_config(path, "classification")
                self.assertEqual(split_seed(loaded), 42)
            for seed in (True, 42.0, "42", 45):
                config["run"]["seed"] = seed
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    load_config(path)
            config["run"]["seed"] = 43
            config["data"]["split_seed"] = 43
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
