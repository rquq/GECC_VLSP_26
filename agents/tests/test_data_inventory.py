"""Saved-output audits use toy data and distinguish reference data from repeats."""

import unittest


class DataInventoryTests(unittest.TestCase):
    def row(self, i, text, target):
        return {
            "ID": i,
            "Input": text,
            "Output": {"Error_Type_Id": 2, "Fixed Sentence": target},
        }

    def test_audit_distinguishes_inherited_and_introduced_ellipses(self):
        from tools.audit_dataset_inventory import audit_predictions

        source = [
            self.row(1, "source ...", "unused"),
            self.row("1", "complete source.", "unused"),
        ]
        predictions = [
            self.row(1, "ignored", ["corrected …"]),
            self.row("1", "ignored", "short ..."),
        ]
        result = audit_predictions(source, predictions)
        self.assertEqual(result["source_ellipsis_rows"], 1)
        self.assertEqual(result["target_ellipsis_rows"], 2)
        self.assertEqual(result["inherited_ellipsis_rows"], 1)
        self.assertEqual(result["introduced_ellipsis_rows"], 1)

    def test_audit_rejects_incomplete_prediction_id_coverage(self):
        from tools.audit_dataset_inventory import audit_predictions

        with self.assertRaises(ValueError):
            audit_predictions([self.row(1, "x", "x")], [])

    def test_inventory_does_not_call_resampling_new_synthetic(self):
        from tools.audit_dataset_inventory import inventory

        source = self.row(1, "fixture source", "fixture target")
        repeat = self.row("repeat:fixture", "fixture source", "fixture target")
        repeat["Provenance"] = {
            "resampling": {"source_id": 1, "is_new_unique_example": False}
        }
        result = inventory([source, repeat], [source])
        self.assertEqual(sum(r["count"] for r in result), 2)
        self.assertEqual({r["origin"] for r in result}, {"organizer_original"})
        self.assertEqual({r["exposure"] for r in result}, {"canonical", "repeat"})


if __name__ == "__main__":
    unittest.main()
