"""Comparison exports must join exact IDs and never score gold as predictions."""
import json
from pathlib import Path
import tempfile
import unittest

from model.metrics import _load_predictions, experiment_report
from model.prediction_export import comparison_rows


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


if __name__ == "__main__":
    unittest.main()
