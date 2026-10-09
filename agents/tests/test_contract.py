import json
from pathlib import Path
import tempfile
import unittest

from model.two_phase import load_stage_config
from model.data import analyze, load_records, split_records
from model.metrics import score
from model.correction import _parse_correction, validate_hints


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


class HintContractTests(unittest.TestCase):
    def test_quarantined_original_partition_and_contrast_family_leakage(self):
        import copy
        import hashlib

        first = {"id": 1, "source": "first source", "label": 1, "target": "first fixed"}
        second = {"id": 2, "source": "second source", "label": 0, "target": "second source"}
        clean = {"id": "clean", "source": "legal construction", "label": 0, "target": "legal construction"}
        error = {"id": "error", "source": "broken construction", "label": 2, "target": "legal construction"}
        validation = [{"id": 9, "source": "holdout", "label": 0, "target": "holdout"}]
        correction = [first, second, error]
        classifier = [first, second, clean, error]

        def hint(row, fold=None):
            result = {"id": row["id"], "label": 0, "probabilities": [1., 0., 0., 0., 0.]}
            if fold is not None:
                result["fold"] = fold
            return result

        for field in ("family_id", "template_id"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "data.json"
                path.write_text("fixture", encoding="utf-8")
                metadata = [{"id": row["id"], "authored_provenance": {field: "shared"}}
                            for row in (clean, error)]
                provenance = {"provenance": {"original_train_ids": [1, 2, 3],
                                             "train_rows": metadata[1:]}}
                artifact = {
                    "seed": 42, "validation_fraction": .2,
                    "correction_data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "classifier_provenance": {
                        "original_train_ids": [1, 2, 3], "retained_original_train_ids": [1, 2],
                        "quarantined_ids": [3], "selected_synthetic_ids": ["clean", "error"],
                        "train_rows": metadata},
                    "train": [hint(first, 0), hint(error, 0), hint(second, 1)],
                    "validation": [hint(validation[0])],
                    "folds": [{"fold": 0, "train_ids": [2], "predict_ids": [1, "error"]},
                              {"fold": 1, "train_ids": [1, "clean", "error"], "predict_ids": [2]}],
                }
                indexed, _ = validate_hints(artifact, correction, validation, provenance,
                                            classifier, path, 42, .2)
                self.assertEqual({key[1] for key in indexed}, {1, 2, "error"})
                leaked = copy.deepcopy(artifact)
                leaked["folds"][0]["train_ids"].append("clean")
                with self.assertRaisesRegex(ValueError, "contrast/template family"):
                    validate_hints(leaked, correction, validation, provenance, classifier, path, 42, .2)
                missing_quarantine = copy.deepcopy(artifact)
                missing_quarantine["classifier_provenance"]["quarantined_ids"] = []
                with self.assertRaisesRegex(ValueError, "declared mixed partition"):
                    validate_hints(missing_quarantine, correction, validation, provenance,
                                   classifier, path, 42, .2)


if __name__ == "__main__":
    unittest.main()
