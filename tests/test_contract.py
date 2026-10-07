import json
from pathlib import Path
import tempfile
import unittest

from vgec.data import load_records, split_records
from vgec.metrics import score
from vgec.modeling import parse_completion


class BaselineContractTests(unittest.TestCase):
    def test_duplicate_sources_never_cross_split(self):
        records = [{"id": label * 10 + i, "source": f"source {label} {i}", "label": label, "target": f"target {label} {i}"}
                   for label in range(5) for i in range(5)]
        records.append({**records[0], "id": 100, "source": "  SOURCE 0 0  "})
        train, validation = split_records(records, validation_fraction=0.2, seed=42)
        normalize = lambda r: " ".join(r["source"].split()).casefold()
        self.assertFalse({normalize(r) for r in train} & {normalize(r) for r in validation})
        self.assertEqual({r["label"] for r in validation}, set(range(5)))
        self.assertEqual({r["id"] for r in train + validation}, {r["id"] for r in records})

    def test_null_gold_only_means_identity_for_class_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.json"
            raw = [{"ID": 1, "Input": "Giữ nguyên câu.", "Output": {"Error_Type_Id": 0, "Fixed Sentence": None}}]
            path.write_text(json.dumps(raw))
            self.assertEqual(load_records(path)[0]["target"], raw[0]["Input"])
            raw[0]["Output"]["Error_Type_Id"] = 1
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                load_records(path)

    def test_source_preservation_penalty_and_invalid_label(self):
        records = [{"id": 1, "source": "a wrong c d e", "label": 1, "target": "a right c d e"}]
        perfect = score(records, [{"id": 1, "label": 1, "target": records[0]["target"]}])
        copied = score(records, [{"id": 1, "label": None, "target": records[0]["source"]}])
        self.assertEqual(perfect["corrections"]["all"]["gleu_plus"], 1.0)
        self.assertEqual(copied["corrections"]["all"]["exact_match"], 0.0)
        self.assertEqual(copied["corrections"]["all"]["gleu_plus"], 0.0)
        self.assertEqual(copied["classification"]["micro_f1"], 0.0)
        self.assertEqual(copied["invalid_prediction_rate"], 1.0)

    def test_invalid_generation_has_no_success_fallback(self):
        for text in ['{"error_type_id":true,"corrected_sentence":"a"}',
                     '{"error_type_id":5,"corrected_sentence":"a"}',
                     '{"error_type_id":0,"corrected_sentence":""}', "explanation, not JSON"]:
            label, target, error = parse_completion(text)
            self.assertIsNone(label)
            self.assertEqual(target, "")
            self.assertIsNotNone(error)


if __name__ == "__main__":
    unittest.main()
