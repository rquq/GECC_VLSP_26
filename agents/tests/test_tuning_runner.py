import json
from pathlib import Path
import tempfile
import unittest
from model.runtime import write_json


class TuningRunnerTests(unittest.TestCase):
    def test_four_model_coordinator_uses_remaining_wall_budget(self):
        from tools.run_tuning import stage_timeout

        plan = {"max_wall_seconds": 32400, "max_stage_seconds": 21600}
        self.assertEqual(stage_timeout(plan, "retuned-models", 3600), 28800)
        self.assertEqual(stage_timeout(plan, "baseline-decoders", 3600), 21600)
        self.assertEqual(stage_timeout(plan, "retuned-models", 30000), 2400)
        with self.assertRaisesRegex(RuntimeError, "wall budget"):
            stage_timeout(plan, "retuned-models", 32400)

    def test_deferred_challenger_has_no_greedy_public_predictions_to_reuse(self):
        from tools.run_tuning import reuse_public_predictions

        self.assertFalse(
            reuse_public_predictions(
                "greedy_reference", "challenger", {}, Path("not-exported.json")
            )
        )

    def test_source_discovery_is_recursive_hash_bound_and_rejects_tampering(self):
        from tools.run_tuning import discover_source
        from model.classifier_resume import sha256

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input/datasets/weird/nested/run"
            source.mkdir(parents=True)
            write_json(source / "pipeline_manifest.json", {"mode": "both"})
            write_json(source / "model.json", {"version": 1})
            proof = {"file_sha256": {p.name: sha256(p) for p in source.iterdir()}}
            self.assertEqual(discover_source(root / "input", proof), source)
            (source / "model.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash"):
                discover_source(root / "input", proof)

    def test_changed_public_input_cannot_reuse_baseline_greedy_outputs(self):
        from tools.run_tuning import reuse_public_predictions
        from model.classifier_resume import sha256

        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "public.json"
            p.write_text("original inputs")
            proof = {"public_data_sha256": sha256(p)}
            self.assertTrue(
                reuse_public_predictions("greedy_reference", "baseline", proof, p)
            )
            p.write_text("changed inputs with the same IDs")
            self.assertFalse(
                reuse_public_predictions("greedy_reference", "baseline", proof, p)
            )
            self.assertFalse(reuse_public_predictions("beam2", "baseline", proof, p))

    def test_committed_handoff_binds_ensemble_and_all_four_adapters(self):
        root = Path(__file__).resolve().parents[2]
        proof = json.loads((root / "configs/tuning_source.json").read_text())[
            "file_sha256"
        ]
        self.assertIn("classification/classifier/ensemble.json", proof)
        self.assertIn("correction/adapter/adapter_model.safetensors", proof)
        for name in ["qwen7b", "sailor7b", "mistral7b"]:
            self.assertIn(
                f"classification/members/{name}/classifier/adapter_model.safetensors",
                proof,
            )

    def test_tuned_recipes_keep_standard_lora_shared_split_and_reduce_learning_rate(
        self,
    ):
        from tools.run_tuning import validate_plan

        root = Path(__file__).resolve().parents[2]
        plan = json.loads((root / "configs/tuning_plan.json").read_text())
        validate_plan(plan, root)
        config = json.loads((root / "configs/tuned_baseline.json").read_text())
        self.assertFalse(config["model"]["load_in_4bit"])
        self.assertEqual(config["run"]["seed"], 42)
        self.assertLess(config["training"]["args"]["learning_rate"], 5e-5)


class TuningNotebookTests(unittest.TestCase):
    def test_launcher_displays_tuning_without_a_classification_directory(self):
        import types
        from unittest.mock import patch
        from tools.notebook import launch
        import contextlib, io

        project = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "pipeline_manifest.json", {"mode": "tuning"})
            write_json(
                root / "tuning_summary.json",
                {
                    "selected": {},
                    "selected_metrics": {
                        "annotation_audit": [{"source": "PRIVATE_CORPUS"}],
                        "combined_score": 0.5,
                    },
                },
            )
            (root / "results.zip").write_bytes(b"fixture")

            class Finished:
                stdout = ["VGEC_RUN_DIR:" + str(root) + "\n"]
                args = []

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def wait(self):
                    return 0

            display = types.ModuleType("IPython.display")
            display.FileLink = str
            display.display = lambda *args: None
            captured = io.StringIO()
            with patch.dict("sys.modules", {"IPython.display": display}), patch(
                "tools.notebook.subprocess.Popen", return_value=Finished()
            ), contextlib.redirect_stdout(captured):
                self.assertEqual(launch(project, "python"), root)
            self.assertNotIn("PRIVATE_CORPUS", captured.getvalue())
