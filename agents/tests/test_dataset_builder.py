"""Audit and synthetic selection operate on training only."""
import json
from pathlib import Path
import tempfile
import unittest

from model.data import load_records, load_training_split, split_records
from tools.build_dataset import audit_record, build, build_split, id_key, normalized


class DatasetBuilderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original = self.root / "original.json"
        self.pool = self.root / "pool.json"
        self.raw = [{"ID": label * 10 + i, "Input": f"original {label} {i}.",
                     "Output": {"Error_Type_Id": label,
                                "Fixed Sentence": None if label == 0 else f"corrected {label} {i}."}}
                    for label in range(5) for i in range(5)]
        self.original.write_text(json.dumps(self.raw))
        train, validation = split_records(load_records(self.original), 0.2, 42)
        self.quarantined_id = next(r["id"] for r in train if r["label"] == 2)
        next(r for r in self.raw if r["ID"] == self.quarantined_id)["Output"]["Fixed Sentence"] = "(Ambiguous annotation) ..."
        self.original.write_text(json.dumps(self.raw))
        self.validation_ids = [r["id"] for r in validation]
        heldout = next(r for r in validation if r["label"] == 2)
        self.pool_rows = [
            {"ID": "syn:leak", "Input": "  " + heldout["source"].upper() + "  ",
             "Output": {"Error_Type_Id": 2, "Fixed Sentence": "Synthetic corrected sentence."},
             "Provenance": {"template_rule": "forbidden-heldout-copy", "parent_ids": []}},
            {"ID": "syn:safe", "Input": "Safe synthetic incorrect sentence.",
             "Output": {"Error_Type_Id": 2, "Fixed Sentence": "Safe synthetic corrected sentence."},
             "Provenance": {"template_rule": "local-pair", "parent_ids": []}},
        ]
        self.pool.write_text(json.dumps(self.pool_rows))

    def add_pool(self):
        self.pool_rows.extend(
            {"ID": f"syn:{label}:{i}", "Input": f"synthetic error {label} {i}.",
             "Output": {"Error_Type_Id": label, "Fixed Sentence": f"synthetic repair {label} {i}."},
             "Provenance": {"template_rule": "local-pair", "parent_ids": []}}
            for label in range(1, 5) for i in range(10)
        )
        self.pool.write_text(json.dumps(self.pool_rows))

    def test_original_abbreviations_and_punctuation_labels_are_retained(self):
        train, _ = split_records(load_records(self.original), 0.2, 42)
        ids = [r["id"] for r in train if r["label"] == 1][:2]
        by_id = {r["ID"]: r for r in self.raw}
        by_id[ids[0]]["Output"]["Fixed Sentence"] = "Original shortened correction..."
        by_id[ids[1]]["Output"]["Fixed Sentence"] = by_id[ids[1]]["Input"].rstrip(".") + "?"
        self.original.write_text(json.dumps(self.raw))
        output = self.root / "retained"
        build(self.original, self.pool, output, 0.2)
        split = json.loads((output / "VGEC_balanced.json").read_text())
        actual = {r["ID"]: r for r in split["train"]}
        for identifier in ids:
            self.assertEqual(actual[identifier], by_id[identifier])
        self.assertNotIn(self.quarantined_id, actual)

    def test_low_original_class_gets_only_one_synthetic_per_original(self):
        self.raw.extend({"ID": 100 + i, "Input": f"extra clean {i}.",
                         "Output": {"Error_Type_Id": 0, "Fixed Sentence": None}} for i in range(5))
        self.original.write_text(json.dumps(self.raw))
        train, _ = split_records(load_records(self.original), 0.2, 42)
        class2 = [r["id"] for r in train if r["label"] == 2]
        for r in self.raw:
            if r["ID"] in class2[1:]:
                r["Output"]["Fixed Sentence"] = "(Explanatory note)"
        # Keep exactly one usable class-2 original, irrespective of shuffle order.
        next(r for r in self.raw if r["ID"] == class2[0])["Output"]["Fixed Sentence"] = "Usable correction."
        self.original.write_text(json.dumps(self.raw))
        self.add_pool()
        output = self.root / "capped"
        report = build(self.original, self.pool, output, 0.2)["split"]["class_counts"]
        self.assertEqual(report["retained_original_train"]["2"], 1)
        self.assertEqual(report["selected_synthetic"]["2"], 1)
        self.assertEqual(report["balanced_train"]["2"], 2)
        self.assertEqual(report["balanced_train"]["0"], 8)

    def test_original_target_overlap_excludes_whole_source_group(self):
        train, validation = split_records(load_records(self.original), 0.2, 42)
        original = next(r for r in train if r["label"] == 1)
        heldout = next(r for r in validation if r["label"] == 3)
        next(r for r in self.raw if r["ID"] == original["id"])["Output"]["Fixed Sentence"] = "  " + heldout["source"].upper() + "  "
        self.original.write_text(json.dumps(self.raw))
        self.add_pool()
        output = self.root / "no-leak"
        records = load_records(self.original)
        raw_by_id = {id_key(r["ID"]): r for r in self.raw}
        audits = {id_key(r["id"]): audit_record(r, raw_by_id[id_key(r["id"])], original=True) for r in records}
        split, _ = build_split(records, raw_by_id, audits, load_records(self.pool),
                               {id_key(r["ID"]): r for r in self.pool_rows}, 0.2)
        self.assertNotIn(original["id"], [r["ID"] for r in split["train"]])
        exclusion = next(r for r in split["provenance"]["quarantine"] if r["original_id"] == original["id"])
        self.assertIn("original_source_group_overlaps_validation_source_or_target", exclusion["reasons"])
        with self.assertRaisesRegex(ValueError, "full original training retention"):
            build(self.original, self.pool, output, 0.2)
        self.assertFalse(output.exists())

    def test_quarantine_and_balance_preserve_raw_gold_and_are_reproducible(self):
        before = self.original.read_bytes()
        output = self.root / "balanced"
        report = build(self.original, self.pool, output, 0.2)
        bundle = json.loads((output / "VGEC_balanced.json").read_text())
        self.assertEqual(bundle["format"], "vgec-balanced-v3")
        self.assertEqual(bundle["seed"], 42)
        self.assertEqual(report["format"], "vgec-manifest-v3")
        self.assertEqual(report["seed"], 42)
        self.assertNotIn("splits", bundle)
        self.assertNotIn("splits", report)
        self.assertNotIn("seeds", report)
        split = bundle
        by_id = {r["ID"]: r for r in self.raw}
        self.assertEqual(split["validation"], [by_id[i] for i in self.validation_ids])
        self.assertEqual(report["split"]["class_counts"]["balanced_train"], {str(i): 4 for i in range(5)})
        self.assertEqual(split["provenance"]["quarantined_ids"], [self.quarantined_id])
        self.assertEqual(split["provenance"]["selected_synthetic_ids"], ["syn:safe"])
        self.assertTrue(any(r["synthetic_id"] == "syn:leak" for r in split["provenance"]["rejections"]))
        first = (output / "VGEC_balanced.json").read_bytes()
        build(self.original, self.pool, output, 0.2)
        self.assertEqual((output / "VGEC_balanced.json").read_bytes(), first)
        self.assertEqual(self.original.read_bytes(), before)
        classifier_first = (output / "VGEC_classifier.json").read_bytes()
        manifest_first = (output / "manifest.json").read_bytes()
        build(self.original, self.pool, output, 0.2)
        self.assertEqual((output / "VGEC_classifier.json").read_bytes(), classifier_first)
        self.assertEqual((output / "manifest.json").read_bytes(), manifest_first)

    def test_classifier_retains_labeled_annotations_and_loads_capped_mix(self):
        self.raw.extend({"ID": 100 + i, "Input": f"extra clean {i}.",
                         "Output": {"Error_Type_Id": 0, "Fixed Sentence": None}} for i in range(5))
        self.original.write_text(json.dumps(self.raw))
        train, validation = split_records(load_records(self.original), 0.2, 42)
        by_id = {r["ID"]: r for r in self.raw}
        annotated = []
        for label in range(1, 5):
            identifier = next(r["id"] for r in train if r["label"] == label)
            by_id[identifier]["Output"]["Fixed Sentence"] = "(Organizer explanatory annotation)"
            annotated.append(identifier)
        self.original.write_text(json.dumps(self.raw))
        self.add_pool()
        self.pool_rows.append({"ID": "syn:bad-parent", "Input": "Bad parent input.",
                               "Output": {"Error_Type_Id": 2, "Fixed Sentence": "Bad parent correction."},
                               "Provenance": {"template_rule": "local-pair", "parent_ids": [validation[0]["id"]]}})
        self.pool_rows.append({"ID": "syn:target-leak", "Input": "Unique leaking target input.",
                               "Output": {"Error_Type_Id": 2, "Fixed Sentence": validation[0]["source"]},
                               "Provenance": {"template_rule": "local-pair", "parent_ids": []}})
        self.pool.write_text(json.dumps(self.pool_rows))
        output = self.root / "classifier"
        manifest = build(self.original, self.pool, output, 0.2)
        bundle = json.loads((output / "VGEC_classifier.json").read_text())
        correction = json.loads((output / "VGEC_balanced.json").read_text())
        actual = {r["ID"]: r for r in bundle["train"]}
        correction_ids = {r["ID"] for r in correction["train"]}
        for identifier in annotated:
            self.assertEqual(actual[identifier], by_id[identifier])
            self.assertNotIn(identifier, correction_ids)
        provenance = bundle["provenance"]
        self.assertEqual(provenance["retained_original_train_ids"], provenance["original_train_ids"])
        self.assertEqual(provenance["quarantined_ids"], [])
        self.assertEqual(bundle["validation"], correction["validation"])
        self.assertEqual(bundle["validation"], [by_id[r["id"]] for r in validation])
        self.assertEqual(manifest["classifier_split"]["class_counts"]["balanced_train"], {str(i): 8 for i in range(5)})
        self.assertEqual(provenance["class_counts"]["selected_synthetic"], {"0": 0, "1": 4, "2": 4, "3": 4, "4": 4})
        rejected = {r["synthetic_id"]: r["reasons"] for r in provenance["rejections"]}
        self.assertIn("source_or_target_overlaps_original_validation_source_or_target", rejected["syn:leak"])
        self.assertIn("source_or_target_overlaps_original_validation_source_or_target", rejected["syn:target-leak"])
        self.assertIn("parent_not_in_original_training_partition", rejected["syn:bad-parent"])
        self.assertIn("VGEC_classifier.json", manifest["outputs"])
        loaded_train, loaded_validation, _ = load_training_split(output / "VGEC_classifier.json", 0.2, 42)
        self.assertEqual(len(loaded_train), 40)
        self.assertEqual([r["id"] for r in loaded_validation], [r["id"] for r in validation])

    def test_classifier_output_refuses_unrelated_file(self):
        output = self.root / "unrelated"
        output.mkdir()
        path = output / "VGEC_classifier.json"
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "unrelated output"):
            build(self.original, self.pool, output, 0.2)
        self.assertEqual(path.read_text(), "{}")

    def test_insufficient_eligible_pool_creates_no_dataset(self):
        self.pool.write_text(json.dumps(self.pool_rows[:1]))
        output = self.root / "insufficient"
        with self.assertRaisesRegex(ValueError, "insufficient eligible"):
            build(self.original, self.pool, output, 0.2)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
