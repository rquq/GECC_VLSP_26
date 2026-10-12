"""Submission schema and diagnostic exports preserve exact typed IDs without fallback."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from model.metrics import _load_predictions, experiment_report
from model.prediction_export import comparison_rows, submission_rows, SubmissionExportError
from model.pipeline_predict import main as predict_main


class PredictionExportTests(unittest.TestCase):
    def test_shuffled_typed_ids_join_without_gold_prediction_confusion(self):
        records = [
            {"id": 1, "source": "a wrong sentence here", "label": 1, "target": "a correct sentence here"},
            {"id": "1", "source": "another clean sentence here", "label": 0, "target": "another clean sentence here"},
        ]
        predictions = [
            {"id": "1", "label": None, "target": "", "parse_error": "invalid JSON"},
            {"id": 1, "label": 0, "target": records[0]["source"]},
        ]
        rows = comparison_rows(records, predictions)
        self.assertEqual([row["id"] for row in rows], [1, "1"])
        self.assertEqual(rows[0]["target_class"], 1)
        self.assertEqual(rows[0]["predicted_class"], 0)
        self.assertNotEqual(rows[0]["target_correction"], rows[0]["predicted_correction"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            report = experiment_report(records, _load_predictions(path))
        self.assertEqual(report, experiment_report(records, predictions))
        self.assertEqual(report["classification"]["micro_f1"], 0.0)
        self.assertEqual(report["corrections"]["all"]["exact_match"], 0.0)

    def test_ambiguous_missing_and_extra_predictions_fail(self):
        records = [{"id": 1, "source": "source", "label": 0, "target": "source"}]
        prediction = {"id": 1, "label": 0, "target": "source"}
        for predictions in ([], [prediction, prediction], [{**prediction, "id": "1"}],
                            [prediction, {**prediction, "id": 2}]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                comparison_rows(records, predictions)

    def test_unlabeled_input_does_not_invent_gold(self):
        record = {"id": 7, "source": "input", "label": None, "target": None}
        row = comparison_rows([record], [{"id": 7, "label": 1, "target": "corrected"}])[0]
        self.assertIsNone(row["target_class"])
        self.assertIsNone(row["target_correction"])
        self.assertEqual(row["predicted_correction"], "corrected")

    def test_submission_has_exact_schema_typed_ids_and_input_order(self):
        records = [
            {"id": 1, "source": "gold must not leak", "label": 2, "target": "gold target"},
            {"id": "1", "source": "second source", "label": None, "target": None},
        ]
        predictions = [
            {"id": "1", "label": 4, "target": " second correction  ", "raw_generation": "diagnostics"},
            {"id": 1, "label": 0, "target": "predicted text", "parse_error": None, "probabilities": [1]},
        ]
        self.assertEqual(submission_rows(records, predictions), [
            {"ID": 1, "Output": {"Error_Type_Id": 0, "Fixed Sentence": "predicted text"}},
            {"ID": "1", "Output": {"Error_Type_Id": 4, "Fixed Sentence": " second correction  "}},
        ])

    def test_submission_rejects_invalid_generations_without_gold_or_source_fallback(self):
        record = {"id": "x", "source": "available source", "label": 1, "target": "available gold"}
        valid = {"id": "x", "label": 2, "target": "prediction"}
        cases = ([{**valid, "label": label}] for label in (None, True, -1, 5, "2", 2.0))
        for predictions in list(cases) + [
            [{**valid, "target": None}], [{**valid, "target": ""}], [{**valid, "target": " \t\n"}],
            [{**valid, "parse_error": "invalid JSON"}], [{**valid, "parse_error": ""}],
        ]:
            with self.subTest(predictions=predictions), self.assertRaises(SubmissionExportError):
                submission_rows([record], predictions)

    def test_submission_rejects_missing_extra_duplicate_and_wrong_typed_ids(self):
        record = {"id": 1, "source": "source", "label": None, "target": None}
        valid = {"id": 1, "label": 0, "target": "prediction"}
        for predictions in ([], [valid, valid], [{**valid, "id": "1"}],
                            [valid, {**valid, "id": 2}]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                submission_rows([record], predictions)

    def _inference_cli(self, data, output, predictions):
        with patch("sys.argv", ["pipeline_predict", "--classifier", "classifier", "--adapter", "adapter",
                                "--data", str(data), "--output", str(output), "--device", "cpu", "--evaluate"]), \
                patch("model.classification.classify", return_value=[{"id": 1, "label": 1}]) as classify, \
                patch("model.correction.load_corrector", return_value=(object(), object(), {})), \
                patch("model.correction.correct", return_value=predictions) as correct, \
                patch.dict("sys.modules", {"torch": SimpleNamespace()}), \
                contextlib.redirect_stdout(io.StringIO()):
            predict_main()
            self.assertEqual(classify.call_args.args[1],
                             [{"id": 1, "source": "a wrong sentence with words", "label": None, "target": None}])
            self.assertEqual(correct.call_args.args[2], classify.call_args.args[1])

    def test_cli_writes_submission_comparison_and_labeled_composite_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data, output = root / "input.json", root / "submission.json"
            data.write_text(json.dumps([{"ID": 1, "Input": "a wrong sentence with words",
                                        "Output": {"Error_Type_Id": 1, "Fixed Sentence": "a right sentence with words"}}]))
            predictions = [{"id": 1, "label": 1, "target": "a right sentence with words", "raw_generation": "private"}]
            self._inference_cli(data, output, predictions)
            self.assertEqual(json.loads(output.read_text()), [
                {"ID": 1, "Output": {"Error_Type_Id": 1, "Fixed Sentence": "a right sentence with words"}},
            ])
            comparison = json.loads((root / "submission.comparisons.json").read_text())[0]
            self.assertEqual(comparison["source"], "a wrong sentence with words")
            self.assertEqual(comparison["raw_generation"], "private")
            metrics = json.loads((root / "submission.metrics.json").read_text())
            self.assertAlmostEqual(metrics["combined_score"], 0.6)
            metadata = json.loads((root / "submission.evaluation.json").read_text())
            self.assertFalse(metadata["combined_score_metadata"]["official"])
            self.assertEqual(metadata["status"], "completed")

    def test_cli_invalid_generation_fails_and_retains_diagnostics_not_submission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data, output = root / "input.json", root / "submission.json"
            data.write_text(json.dumps([{"ID": 1, "Input": "a wrong sentence with words",
                                        "Output": {"Error_Type_Id": 1, "Fixed Sentence": "a right sentence with words"}}]))
            predictions = [{"id": 1, "label": 1, "target": "", "parse_error": "malformed JSON"}]
            with self.assertRaises(SubmissionExportError):
                self._inference_cli(data, output, predictions)
            self.assertFalse(output.exists())
            diagnostics = json.loads((root / "submission.comparisons.json").read_text())[0]
            self.assertEqual(diagnostics["parse_error"], "malformed JSON")
            self.assertEqual(diagnostics["predicted_correction"], "")
            metadata = json.loads((root / "submission.evaluation.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertIn("parse_error", metadata["failure"])



if __name__ == "__main__":
    unittest.main()
