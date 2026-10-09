import copy
import unittest

from model.metrics import experiment_report, score


class ExperimentMetricsTests(unittest.TestCase):
    def test_clean_preservation_is_exact_and_independent_of_predicted_label(self):
        records = [
            {"id": 1, "source": "Giữ nguyên câu.", "label": 0, "target": "Giữ nguyên câu."},
            {"id": 2, "source": "Hai  dấu cách.", "label": 0, "target": "Hai  dấu cách."},
            {"id": 3, "source": "sai", "label": 1, "target": "đúng"},
        ]
        predictions = [
            {"id": 1, "label": 4, "target": records[0]["source"]},
            {"id": 2, "label": 0, "target": "Hai dấu cách."},
            {"id": 3, "label": 1, "target": "đúng"},
        ]
        report = experiment_report(records, predictions)
        self.assertEqual(report["clean_preservation"], {
            "count": 2, "exact_source_preservation": 0.5, "overcorrection_rate": 0.5,
        })
        self.assertEqual(report["invalid_json_rate"], 0.0)
        usual = score(records, predictions)
        self.assertEqual({key: report[key] for key in usual}, usual)

    def test_invalid_json_is_separate_from_wrong_valid_label(self):
        records = [
            {"id": index, "source": "source", "label": 1, "target": "reference"}
            for index in range(7)
        ]
        predictions = [
            {"id": 0, "label": 2, "target": "wrong", "parse_error": None},
            {"id": 1, "label": 1, "target": "reference", "parse_error": "malformed JSON"},
            {"id": 2, "label": None, "target": "", "parse_error": "missing fields"},
            {"id": 3, "label": True, "target": "reference"},
            {"id": 4, "label": "1", "target": "reference"},
            {"id": 5, "label": 1, "target": " \t\n"},
            {"id": 6, "label": 1, "target": "reference", "parse_error": ""},
        ]
        report = experiment_report(records, predictions)
        self.assertEqual(report["invalid_json_rate"], 6 / 7)
        self.assertEqual(report["invalid_prediction_rate"], 3 / 7)
        self.assertEqual(report["classification"]["micro_f1"], 3 / 7)

    def test_reference_flags_overlap_without_rewriting_gold(self):
        records = [
            {"id": "both", "source": "first source", "label": 1, "target": "(ghi chú ... giữ nguyên)"},
            {"id": "parenthesis", "source": "second source", "label": 2, "target": "(ghi chú)"},
            {"id": "ellipsis", "source": "third source", "label": 3, "target": "câu ..."},
            {"id": "literal", "source": "fourth source", "label": 4, "target": " (not a prefix) …"},
            {"id": "clean", "source": "(clean ...)", "label": 0, "target": "(clean ...)"},
        ]
        predictions = [
            {"id": row["id"], "label": row["label"], "target": row["target"]}
            for row in reversed(records)
        ]
        original_records, original_predictions = copy.deepcopy(records), copy.deepcopy(predictions)
        report = experiment_report(records, predictions)
        self.assertEqual(records, original_records)
        self.assertEqual(predictions, original_predictions)
        annotation = report["reference_slices"]["annotation_like"]
        standalone = report["reference_slices"]["standalone_error"]
        self.assertEqual(annotation["count"], 3)
        self.assertEqual(standalone["count"], 1)
        self.assertEqual(annotation["metrics"]["corrections"]["all"]["exact_match"], 1.0)
        self.assertEqual(standalone["metrics"], score([records[3]], [predictions[1]]))
        self.assertEqual(report["annotation_audit"][0], {
            "id": "both", "source": "first source", "reference": "(ghi chú ... giữ nguyên)",
            "prediction": "(ghi chú ... giữ nguyên)", "label": 1, "predicted_label": 1,
            "flags": ["starts_with_parenthesis", "contains_ellipsis"],
        })
        self.assertEqual(report["annotation_audit"][1]["flags"], ["starts_with_parenthesis"])
        self.assertEqual(report["annotation_audit"][2]["flags"], ["contains_ellipsis"])

    def test_typed_ids_align_report_and_slice_predictions(self):
        records = [
            {"id": 1, "source": "integer source", "label": 1, "target": "(integer reference)"},
            {"id": "1", "source": "string source", "label": 0, "target": "string source"},
        ]
        predictions = [
            {"id": "1", "label": 0, "target": "changed string"},
            {"id": 1, "label": 1, "target": "(integer reference)"},
        ]
        report = experiment_report(records, predictions)
        self.assertEqual(report["clean_preservation"]["overcorrection_rate"], 1.0)
        self.assertEqual(report["annotation_audit"][0]["prediction"], "(integer reference)")
        self.assertEqual(report["reference_slices"]["annotation_like"]["metrics"],
                         score([records[0]], [predictions[1]]))

    def test_empty_subsets_have_null_rates_and_metrics(self):
        clean = [{"id": 0, "source": "clean", "label": 0, "target": "clean"}]
        report = experiment_report(clean, [{"id": 0, "label": 0, "target": "clean"}])
        self.assertEqual(report["annotation_audit"], [])
        for subset in report["reference_slices"].values():
            self.assertEqual(subset, {"count": 0, "metrics": None})
        errors = [{"id": 1, "source": "wrong", "label": 1, "target": "right"}]
        report = experiment_report(errors, [{"id": 1, "label": 1, "target": "right"}])
        self.assertEqual(report["clean_preservation"], {
            "count": 0, "exact_source_preservation": None, "overcorrection_rate": None,
        })
        self.assertEqual(report["reference_slices"]["annotation_like"], {"count": 0, "metrics": None})
        self.assertEqual(report["reference_slices"]["standalone_error"]["count"], 1)
        with self.assertRaisesRegex(ValueError, "empty dataset"):
            experiment_report([], [])

    def test_strict_alignment_and_target_validation_are_retained(self):
        records = [{"id": 1, "source": "wrong", "label": 1, "target": "right"}]
        cases = [
            [],
            [{"id": "1", "label": 1, "target": "right"}],
            [{"id": 1, "label": 1, "target": "right"}] * 2,
            [{"id": 1, "label": 1, "target": "right"}, {"id": 2, "label": 1, "target": "right"}],
            [{"id": 1, "label": 1}],
            [{"id": 1, "label": 1, "target": None}],
            [{"id": True, "label": 1, "target": "right"}],
        ]
        for predictions in cases:
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                experiment_report(records, predictions)


if __name__ == "__main__":
    unittest.main()
