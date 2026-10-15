"""A saved classifier is complete only after an independent reload succeeds."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from model.runtime import write_json


class FinalizeTests(unittest.TestCase):
    def fixture(self, root):
        rows = [{"id": 1, "source": "toy", "target": "toy", "label": 0}]
        write_json(root / "train.json", rows)
        write_json(root / "validation.json", rows)
        write_json(
            root / "training_complete.json",
            {
                "task": "classification",
                "final_fit": False,
                "best_epoch": 1,
                "selection_macro_f1": 0.2,
                "saved_classifier_reload_verified": False,
                "config": {"training": {"args": {"per_device_eval_batch_size": 1}}},
            },
        )
        return [{"id": 1, "label": 0, "probabilities": [1, 0, 0, 0, 0]}]

    def test_failed_reload_does_not_publish_completion(self):
        from model.classifier_finalize import finalize

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            with patch(
                "model.classification.classify", side_effect=RuntimeError("memory")
            ):
                with self.assertRaisesRegex(RuntimeError, "memory"):
                    finalize(root, "cpu")
            self.assertFalse((root / "run_manifest.json").exists())
            self.assertTrue((root / "training_complete.json").exists())

    def test_success_publishes_actual_predictions_and_verified_manifest(self):
        from model.classifier_finalize import finalize

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictions = self.fixture(root)
            with patch("model.classification.classify", return_value=predictions):
                metrics = finalize(root, "cpu")
            self.assertEqual(metrics["classification"]["micro_f1"], 1)
            self.assertTrue(
                json.loads((root / "run_manifest.json").read_text())[
                    "saved_classifier_reload_verified"
                ]
            )
            self.assertEqual(
                len(json.loads((root / "predictions.json").read_text())), 1
            )

    def test_selection_score_mismatch_does_not_publish_completion(self):
        from model.classifier_finalize import finalize

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictions = self.fixture(root)
            receipt = json.loads((root / "training_complete.json").read_text())
            receipt["selection_macro_f1"] = 0.9
            write_json(root / "training_complete.json", receipt)
            with patch("model.classification.classify", return_value=predictions):
                with self.assertRaisesRegex(ValueError, "selected epoch"):
                    finalize(root, "cpu")
            self.assertFalse((root / "run_manifest.json").exists())
