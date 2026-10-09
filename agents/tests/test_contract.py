import json
from pathlib import Path
import tempfile
import unittest

from model.two_phase import load_stage_config
from model.data import analyze, load_records, split_records
from model.metrics import score
from model.correction import _parse_correction


class PipelineContractTests(unittest.TestCase):
    def test_duplicate_sources_never_cross_split(self):
        records = [{"id": label * 10 + i, "source": f"source {label} {i}", "label": label, "target": f"target {label} {i}"}
                   for label in range(5) for i in range(5)]
        records.append({**records[0], "id": 100, "source": "  SOURCE 0 0  "})
        train, validation = split_records(records, validation_fraction=0.2, seed=42)
        normalize = lambda r: " ".join(r["source"].split()).casefold()
        self.assertFalse({normalize(r) for r in train} & {normalize(r) for r in validation})
        self.assertEqual({r["label"] for r in validation}, set(range(5)))
        self.assertEqual({r["id"] for r in train + validation}, {r["id"] for r in records})

    def test_non42_analysis_and_split_are_rejected_before_loading_rows(self):
        for seed in (43, "42", 42.0, True):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    split_records([], seed=seed)
                with self.assertRaises(ValueError):
                    analyze("missing-data.json", "unused-output", seed=seed)

    def test_config_rejects_non42_and_legacy_seed_lists(self):
        config = json.loads((Path(__file__).resolve().parents[2] / "configs/classification.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for seed in (43, "42", 42.0, True):
                with self.subTest(seed=seed):
                    config["run"]["seed"] = seed
                    path.write_text(json.dumps(config))
                    with self.assertRaises(ValueError):
                        load_stage_config(path, "classification")
            del config["run"]["seed"]
            config["run"]["seeds"] = [42]
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_stage_config(path, "classification")

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
        for text in ('[]', 'null', '1', '"text"',
                     '{"corrected_sentence":true}', '{"corrected_sentence":""}',
                     '{"corrected_sentence":"a","explanation":"b"}',
                     "explanation, not JSON"):
            target, error = _parse_correction(text)
            self.assertEqual(target, "")
            self.assertIsNotNone(error)


if __name__ == "__main__":
    unittest.main()
