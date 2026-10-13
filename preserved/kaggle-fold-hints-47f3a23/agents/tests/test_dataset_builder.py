"""Audit and synthetic selection operate on training only."""
import json
from collections import Counter
import hashlib
import re
import stat
from pathlib import Path
import tempfile
import unittest

from model.data import load_classifier_split, load_records, load_training_split, split_records
from tools.build_dataset import audit_record, build, build_split, id_key
from tools.author_synthetic import author_pool, structural_candidates, write_private_pool


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
        self.pool_rows.extend(
            {"ID": f"syn:clean:{i}", "Input": f"Authored clean sentence {i}.",
             "Output": {"Error_Type_Id": 0, "Fixed Sentence": f"Authored clean sentence {i}."},
             "Provenance": {"template_rule": "clean-identity", "parent_ids": []}}
            for i in range(10)
        )
        self.pool.write_text(json.dumps(self.pool_rows))

    def add_pool(self):
        self.pool_rows.extend(
            {"ID": f"syn:{label}:{i}", "Input": f"synthetic error {label} {i}.",
             "Output": {"Error_Type_Id": label, "Fixed Sentence": f"synthetic repair {label} {i}."},
             "Provenance": {"template_rule": "local-pair", "parent_ids": []}}
            for label in range(1, 5) for i in range(10)
        )
        self.pool.write_text(json.dumps(self.pool_rows))

    def test_incomplete_and_mislabeled_punctuation_originals_are_not_repaired(self):
        train, validation = split_records(load_records(self.original), 0.2, 42)
        ids = [r["id"] for r in train if r["label"] == 1][:2]
        by_id = {r["ID"]: r for r in self.raw}
        by_id[ids[0]]["Output"]["Fixed Sentence"] = "Original shortened correction..."
        by_id[ids[1]]["Output"]["Fixed Sentence"] = by_id[ids[1]]["Input"].rstrip(".") + "?"
        self.original.write_text(json.dumps(self.raw))
        self.add_pool()
        before = self.original.read_bytes()
        output = self.root / "strict-correction"
        build(self.original, self.pool, output, 0.2)
        correction = json.loads((output / "VGEC_balanced.json").read_text())
        classifier = json.loads((output / "VGEC_classifier.json").read_text())
        correction_ids = {r["ID"] for r in correction["train"]}
        classifier_rows = {r["ID"]: r for r in classifier["train"]}
        for identifier in ids:
            self.assertNotIn(identifier, correction_ids)
            self.assertEqual(classifier_rows[identifier], by_id[identifier])
        self.assertNotIn(self.quarantined_id, correction_ids)
        expected_validation = [by_id[r["id"]] for r in validation]
        self.assertEqual(correction["validation"], expected_validation)
        self.assertEqual(classifier["validation"], expected_validation)
        self.assertEqual(self.original.read_bytes(), before)

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
        with self.assertRaisesRegex(ValueError, "overlapping validation"):
            build(self.original, self.pool, output, 0.2)
        self.assertFalse(output.exists())

    def test_quarantine_and_balance_preserve_raw_gold_and_are_reproducible(self):
        before = self.original.read_bytes()
        output = self.root / "balanced"
        report = build(self.original, self.pool, output, 0.2)
        bundle = json.loads((output / "VGEC_balanced.json").read_text())
        split = bundle
        by_id = {r["ID"]: r for r in self.raw}
        self.assertEqual(split["validation"], [by_id[i] for i in self.validation_ids])
        self.assertEqual(report["split"]["class_counts"]["balanced_train"],
                         {"0": 8, "1": 4, "2": 4, "3": 4, "4": 4})
        self.assertEqual(split["provenance"]["quarantined_ids"], [self.quarantined_id])
        selected_errors = [r["ID"] for r in split["train"]
                           if isinstance(r["ID"], str) and r["Output"]["Error_Type_Id"] != 0]
        self.assertEqual(selected_errors, ["syn:safe"])
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
        self.assertEqual(manifest["classifier_split"]["class_counts"]["balanced_train"],
                         {"0": 16, "1": 8, "2": 8, "3": 8, "4": 8})
        self.assertEqual(provenance["class_counts"]["selected_synthetic"],
                         {"0": 8, "1": 4, "2": 4, "3": 4, "4": 4})
        rejected = {r["synthetic_id"]: r["reasons"] for r in provenance["rejections"]}
        self.assertIn("source_or_target_overlaps_original_validation_source_or_target", rejected["syn:leak"])
        self.assertIn("source_or_target_overlaps_original_validation_source_or_target", rejected["syn:target-leak"])
        self.assertIn("parent_not_in_original_training_partition", rejected["syn:bad-parent"])
        self.assertIn("VGEC_classifier.json", manifest["outputs"])
        loaded_train, loaded_validation, _ = load_training_split(output / "VGEC_classifier.json", 0.2, 42)
        self.assertEqual(len(loaded_train), 48)
        self.assertEqual([r["id"] for r in loaded_validation], [r["id"] for r in validation])
        classifier_train, _ = load_classifier_split(self.original, output / "VGEC_classifier.json", 0.2, 42)
        self.assertEqual(classifier_train, loaded_train)

    def test_correction_excludes_annotations_and_both_ellipsis_fields_without_changing_holdout(self):
        train, validation = split_records(load_records(self.original), 0.2, 42)
        ids = [r["id"] for r in train if r["label"] == 3]
        by_id = {r["ID"]: r for r in self.raw}
        by_id[ids[0]]["Input"] += "..."
        by_id[ids[1]]["Output"]["Fixed Sentence"] = "Incomplete correction…"
        by_id[ids[2]]["Output"]["Fixed Sentence"] = "(Organizer note)"
        heldout = next(r for r in validation if r["label"] == 3)
        by_id[heldout["id"]]["Output"]["Fixed Sentence"] = "(Heldout note) ..."
        self.original.write_text(json.dumps(self.raw))
        before = self.original.read_bytes()
        expected_train, expected_validation = split_records(load_records(self.original), 0.2, 42)
        self.assertTrue(set(ids[:3]) <= {r["id"] for r in expected_train})
        self.add_pool()
        output = self.root / "complete-corrections"
        build(self.original, self.pool, output, 0.2)
        correction = json.loads((output / "VGEC_balanced.json").read_text())
        classifier = json.loads((output / "VGEC_classifier.json").read_text())
        correction_ids = {r["ID"] for r in correction["train"]}
        classifier_rows = {r["ID"]: r for r in classifier["train"]}
        for identifier in ids[:3] + [self.quarantined_id]:
            self.assertNotIn(identifier, correction_ids)
            self.assertEqual(classifier_rows[identifier], by_id[identifier])
        expected_holdout = [by_id[r["id"]] for r in expected_validation]
        self.assertEqual(correction["validation"], expected_holdout)
        self.assertEqual(classifier["validation"], expected_holdout)
        self.assertEqual(self.original.read_bytes(), before)
        load_classifier_split(self.original, output / "VGEC_classifier.json", 0.2, 42)

    def test_classifier_quarantines_only_annotation_explicitly_declaring_source_correct(self):
        train, validation = split_records(load_records(self.original), 0.2, 42)
        contradictory = next(r for r in train if r["label"] == 1)
        by_id = {r["ID"]: r for r in self.raw}
        by_id[contradictory["id"]]["Output"]["Fixed Sentence"] = "  (Câu này ĐÚNG.)"
        self.original.write_text(json.dumps(self.raw))
        self.add_pool()
        output = self.root / "contradictory"
        build(self.original, self.pool, output, 0.2)
        classifier = json.loads((output / "VGEC_classifier.json").read_text())
        self.assertEqual(classifier["provenance"]["quarantined_ids"], [contradictory["id"]])
        actual = {r["ID"]: r for r in classifier["train"]}
        self.assertNotIn(contradictory["id"], actual)
        self.assertEqual(actual[self.quarantined_id], by_id[self.quarantined_id])
        self.assertEqual(classifier["validation"], [by_id[r["id"]] for r in validation])
        load_classifier_split(self.original, output / "VGEC_classifier.json", 0.2, 42)

    def test_entire_authored_family_is_rejected_if_any_member_matches_holdout(self):
        for field in ("family_id", "template_id"):
            with self.subTest(field=field):
                self.pool_rows[0]["Provenance"] = {
                    "template_rule": "forbidden-heldout-copy", "parent_ids": [], field: "leaking-family"}
                companion = {"ID": "syn:family-companion", "Input": "Distinct companion error.",
                             "Output": {"Error_Type_Id": 2, "Fixed Sentence": "Distinct companion repair."},
                             "Provenance": {"template_rule": "local-pair", "parent_ids": [], field: "leaking-family"}}
                self.pool.write_text(json.dumps(self.pool_rows + [companion]))
                output = self.root / f"family-{field}"
                build(self.original, self.pool, output, 0.2)
                for filename in ("VGEC_balanced.json", "VGEC_classifier.json"):
                    split = json.loads((output / filename).read_text())
                    rejections = {r["synthetic_id"]: r["reasons"] for r in split["provenance"]["rejections"]}
                    for identifier in ("syn:leak", companion["ID"]):
                        self.assertIn("authored_family_overlaps_validation_text", rejections[identifier])
                        self.assertNotIn(identifier, split["provenance"]["selected_synthetic_ids"])

    def test_malformed_clean_pool_label_is_rejected_before_outputs_are_written(self):
        clean = next(r for r in self.pool_rows if r["Output"]["Error_Type_Id"] == 0)
        clean["Output"]["Fixed Sentence"] = "Different correction for a clean label."
        self.pool.write_text(json.dumps(self.pool_rows))
        output = self.root / "malformed-clean"
        with self.assertRaises(ValueError):
            build(self.original, self.pool, output, 0.2)
        self.assertFalse(output.exists())

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

class CorpusAugmentationTests(unittest.TestCase):
    def test_rules_keep_optional_fronts_semantics_and_phrase_boundaries(self):
        sentence = "Với sự hỗ trợ của nhà trường, học sinh đã đọc các tác phẩm."
        candidates = list(structural_candidates(sentence))
        self.assertFalse(any(label == 1 for label, _, _, _ in candidates))
        phrase_errors = [source for label, _, source, _ in candidates if label == 2]
        self.assertEqual(phrase_errors, ["Với sự hỗ trợ của nhà trường, học sinh đã đọc tác phẩm các."])
        self.assertTrue(all(source.startswith("Với sự hỗ trợ của nhà trường,")
                            for _, _, source, _ in candidates))
        self.assertFalse(any(label == 2 for label, _, _, _ in
                             structural_candidates("Năm 1980, văn học phát triển.")))
        self.assertFalse(any(label == 3 for label, _, _, _ in
                             structural_candidates("Tiếng chim ríu ríu trên cành.")))
        self.assertFalse(any(label == 2 for label, _, _, _ in
                             structural_candidates("Học sinh đọc ba tác phẩm.")))
        copular = list(structural_candidates("Ngôn từ là yếu tố của văn bản."))
        self.assertEqual([source for label, _, source, _ in copular if label == 1], ["Ngôn từ."])
        for label, _, source, _ in candidates:
            if label == 4:
                self.assertEqual(re.findall(r"\w+", source), re.findall(r"\w+", sentence))

    def test_real_corpus_pool_is_deterministic_train_only_and_originals_immutable(self):
        original = Path(__file__).resolve().parents[2] / "dataset/original/VGEC 2026_v1.json"
        if not original.is_file():
            import zipfile
            archive = original.parents[1] / "kaggle-input/VGEC26-rebuilt.zip"
            if not archive.is_file():
                self.skipTest("Private corpus integration requires the original data or canonical ZIP")
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            original = Path(temporary.name) / original.name
            archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.addCleanup(lambda: self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), archive_hash))
            with zipfile.ZipFile(archive) as handle:
                original.write_bytes(handle.read(original.name))
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in original.parent.iterdir() if path.is_file()}
        rows = author_pool(original)
        self.assertEqual(rows, author_pool(original))
        train, validation = split_records(load_records(original), 0.2, 42)
        train_ids = {id_key(row["id"]) for row in train}
        normalize = lambda text: " ".join(text.split()).casefold()
        heldout = {normalize(row[field]) for row in validation for field in ("source", "target")}
        original_sources = {normalize(row["source"]) for row in train}
        seen = set()
        for row in rows:
            metadata = row["Provenance"]
            self.assertTrue(metadata["parent_ids"])
            self.assertTrue(all(id_key(parent) in train_ids for parent in metadata["parent_ids"]))
            self.assertEqual(metadata["origin"], "corpus_derived_deterministic_augmentation")
            self.assertEqual(metadata["review_status"], "deterministic_rule_checked_not_human_adjudicated")
            self.assertTrue(metadata["uses_source_corpus"])
            self.assertFalse(metadata["uses_external_teacher"])
            self.assertTrue(metadata["family_id"] and metadata["template_id"])
            self.assertNotIn(normalize(row["Input"]), heldout | original_sources | seen)
            self.assertNotIn(normalize(row["Output"]["Fixed Sentence"]), heldout)
            seen.add(normalize(row["Input"]))
            label = row["Output"]["Error_Type_Id"]
            if label == 0:
                self.assertEqual(row["Input"], row["Output"]["Fixed Sentence"])
            if metadata["template_rule"].startswith("quoted_original_clause_edit"):
                parent = next(parent for parent in train if parent["id"] == metadata["parent_ids"][0])
                evidence = metadata["evidence"]
                self.assertEqual(row["Input"], evidence["quote_open"] + parent["source"] + evidence["quote_close"])
                self.assertEqual(row["Output"]["Fixed Sentence"], evidence["quote_open"] + parent["target"] + evidence["quote_close"])
        counts = Counter(row["Output"]["Error_Type_Id"] for row in rows)
        self.assertTrue(all(counts[label] >= cap for label, cap in enumerate((40, 78, 23, 23, 28))))
        self.assertEqual({path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}, before)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new" / "private" / "synthetic_pool.json"
            write_private_pool(output, rows, original)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(output.parent.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(output.parent.parent.stat().st_mode), 0o700)
            manifest = build(original, output, output.parent, 0.2)
            raw = json.loads(original.read_text(encoding="utf-8"))
            by_id = {id_key(row["ID"]): row for row in raw}
            expected_validation = [by_id[id_key(row["id"])] for row in validation]
            for filename, summary_key in (("VGEC_balanced.json", "split"), ("VGEC_classifier.json", "classifier_split")):
                bundle = json.loads((output.parent / filename).read_text(encoding="utf-8"))
                self.assertEqual(bundle["validation"], expected_validation)
                for row in bundle["train"]:
                    if id_key(row["ID"]) in by_id:
                        self.assertEqual(row, by_id[id_key(row["ID"])])
                counts = manifest[summary_key]["class_counts"]
                reference = bundle["provenance"]["augmentation_reference_counts"]
                self.assertLessEqual(counts["selected_synthetic"]["0"], 40)
                for label in map(str, range(1, 5)):
                    self.assertLessEqual(counts["selected_synthetic"][label], reference[label])
            for filename in ("VGEC_balanced.json", "VGEC_classifier.json", "audit.json", "manifest.json"):
                self.assertEqual(stat.S_IMODE((output.parent / filename).stat().st_mode), 0o600)
            self.assertEqual({path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}, before)



if __name__ == "__main__":
    unittest.main()
