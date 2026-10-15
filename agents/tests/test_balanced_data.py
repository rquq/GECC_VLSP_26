"""Capped augmentation must preserve originals without leaking into validation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from model.data import load_classifier_split, load_records, load_training_split, split_records


def row(identifier, label, source, target):
    return {"ID": identifier, "Input": source,
            "Output": {"Error_Type_Id": label, "Fixed Sentence": target}}


def bundle(original_counts=(4, 1, 2, 3, 1)):
    originals = [row(label * 1000 + index, label, f"original {label} {index}",
                     None if label == 0 else f"correction {label} {index} ...")
                 for label, count in enumerate(original_counts) for index in range(count)]
    largest = max(original_counts)
    targets = {str(label): count + min(40, count) if label == 0 else min(largest, 2 * count)
               for label, count in enumerate(original_counts)}
    synthetics = [row(f"syn:{label}:{index}", label, f"synthetic {label} {index}",
                      f"synthetic {label} {index}" if label == 0 else f"synthetic correction {label} {index}")
                  for label in range(5)
                  for index in range(targets[str(label)] - original_counts[label])]
    validation = [row(10000, 0, "Original clean sentence.", None),
                  row(10001, 1, "Original incorrect sentence.", "(Annotation) ...")]
    provenance = {
        "balance_policy": "quality_capped_2x", "balance_targets_by_class": targets,
        "augmentation_reference_counts": {str(label): count for label, count in enumerate(original_counts)},
        "original_train_class_counts": {str(label): count + (label == 2)
                                        for label, count in enumerate(original_counts)},
        "original_train_ids": [r["ID"] for r in originals] + [999],
        "retained_original_train_ids": [r["ID"] for r in originals],
        "quarantined_ids": [999],
        "selected_synthetic_ids": [r["ID"] for r in synthetics],
        "train_rows": [{"id": r["ID"], "kind": "original", "original_id": r["ID"]}
                       for r in originals] +
                      [{"id": r["ID"], "kind": "synthetic", "parent_ids": [r["Output"]["Error_Type_Id"] * 1000]}
                       for r in synthetics],
    }
    return {"format": "vgec-balanced-v3", "seed": 42, "validation_fraction": 0.2,
            "train": originals + synthetics, "validation": validation,
            "provenance": provenance}


class BalancedSplitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "bundle.json"
        self.reset_bundle()

    def reset_bundle(self, original_counts=(4, 1, 2, 3, 1)):
        self.bundle = bundle(original_counts)
        self.split = self.bundle
        self.train = self.split["train"]
        self.provenance = self.split["provenance"]

    def load(self, seed=42, fraction=0.2):
        self.path.write_text(json.dumps(self.bundle), encoding="utf-8")
        return load_training_split(self.path, fraction, seed)

    def add_synthetic(self, label):
        identifier = f"extra:{label}"
        self.train.append(row(identifier, label, f"extra source {label}", f"extra target {label}"))
        self.provenance["selected_synthetic_ids"].append(identifier)
        self.provenance["train_rows"].append({"id": identifier, "kind": "synthetic", "parent_ids": []})

    def test_capped_unequal_training_preserves_original_targets_and_validation(self):
        train, validation, metadata = self.load()
        self.assertEqual([r["id"] for r in validation], [10000, 10001])
        self.assertEqual(validation[0]["target"], "Original clean sentence.")
        self.assertEqual(validation[1]["target"], "(Annotation) ...")
        self.assertEqual({r["id"] for r in train}, {r["ID"] for r in self.train})
        self.assertEqual(next(r["target"] for r in train if r["id"] == 1000), "correction 1 0 ...")
        counts = {str(label): sum(r["label"] == label for r in train) for label in range(5)}
        self.assertEqual(counts, {"0": 8, "1": 2, "2": 4, "3": 4, "4": 2})
        originals = self.provenance["retained_original_train_ids"]
        self.assertTrue(all(sum(r["label"] == label for r in train) <=
                            2 * sum(r["label"] == label and r["id"] in originals for r in train)
                            for label in range(1, 5)))
        self.assertEqual(metadata["provenance"], self.provenance)
        self.assertEqual(metadata["format"], "vgec-balanced-v3")

    def test_majority_can_be_an_error_class_and_clean_additions_are_capped(self):
        self.reset_bundle((2, 5, 1, 3, 1))
        train, _, _ = self.load()
        self.assertEqual([sum(r["label"] == label for r in train) for label in range(5)],
                         [4, 5, 2, 5, 2])
        self.reset_bundle((50, 1, 2, 3, 1))
        train, _, _ = self.load()
        self.assertEqual(sum(r["label"] == 0 for r in train), 90)

    def test_seed_and_fraction_mismatch_cannot_resplit_augmented_rows(self):
        with self.assertRaisesRegex(ValueError, "Dataset split seed must be 42"):
            self.load(seed=43)
        with self.assertRaisesRegex(ValueError, "validation_fraction differs"):
            self.load(fraction=0.25)

    def test_old_bundle_format_is_rejected(self):
        for old_format in ("vgec-balanced-v1", "vgec-balanced-v2"):
            with self.subTest(format=old_format):
                self.bundle["format"] = old_format
                with self.assertRaises(ValueError):
                    self.load()

    def test_bundle_seed_must_be_exactly_42(self):
        for seed in (None, 43, "42", 42.0, True):
            with self.subTest(seed=seed):
                self.bundle["seed"] = seed
                with self.assertRaisesRegex(ValueError, "Bundle seed must be 42"):
                    self.load()

    def test_non42_seed_is_rejected_before_opening_dataset(self):
        with self.assertRaisesRegex(ValueError, "Dataset split seed must be 42"):
            load_training_split(self.path, 0.2, 43)

    def test_all_original_and_synthetic_cross_text_validation_leaks_rejected(self):
        for kind in ("original", "synthetic"):
            for training_field in ("Input", "Fixed Sentence"):
                for validation_text in ("  ORIGINAL clean sentence.  ", " ORIGINAL incorrect SENTENCE. ",
                                        " (ANNOTATION)   ... "):
                    with self.subTest(kind=kind, field=training_field, validation_text=validation_text):
                        self.reset_bundle()
                        record = next(r for r in self.train if (isinstance(r["ID"], str)) == (kind == "synthetic"))
                        if training_field == "Input":
                            record["Input"] = validation_text
                        else:
                            record["Output"]["Fixed Sentence"] = validation_text
                        with self.assertRaises(ValueError):
                            self.load()

    def test_shared_training_validation_ids_rejected(self):
        self.train[0]["ID"] = 10000
        with self.assertRaisesRegex(ValueError, "shared training/validation IDs"):
            self.load()

    def test_duplicate_training_ids_are_rejected(self):
        self.train.append(dict(self.train[0]))
        with self.assertRaises(ValueError):
            self.load()

    def test_normalized_source_groups_cannot_cross_holdout(self):
        self.train[0]["Input"] = "  ORIGINAL   incorrect SENTENCE. "
        with self.assertRaisesRegex(ValueError, "source groups"):
            self.load()

    def test_over_and_under_augmented_classes_rejected(self):
        self.add_synthetic(1)
        with self.assertRaises(ValueError):
            self.load()
        self.reset_bundle()
        removed = self.train.pop()
        self.provenance["selected_synthetic_ids"].remove(removed["ID"])
        self.provenance["train_rows"].pop()
        with self.assertRaises(ValueError):
            self.load()

    def test_synthetic_clean_labels_require_exact_identity(self):
        for target in ("different correction", " SYNTHETIC 0 0 ", "synthetic 0 0 "):
            with self.subTest(target=target):
                self.reset_bundle()
                clean = next(r for r in self.train if r["ID"] == "syn:0:0")
                clean["Output"]["Fixed Sentence"] = target
                with self.assertRaisesRegex(ValueError, "exact source/target identity"):
                    self.load()

    def test_all_five_classes_require_retained_originals(self):
        self.train[:] = [r for r in self.train if r["ID"] != 4000]
        self.provenance["retained_original_train_ids"].remove(4000)
        self.provenance["quarantined_ids"].append(4000)
        self.provenance["train_rows"][:] = [r for r in self.provenance["train_rows"] if r["id"] != 4000]
        with self.assertRaises(ValueError):
            self.load()

    def test_policy_and_declared_targets_must_match_actual_original_counts(self):
        for policy in ("original_capped_2x", "equal_classes"):
            with self.subTest(policy=policy):
                self.provenance["balance_policy"] = policy
                with self.assertRaises(ValueError):
                    self.load()
        for targets in ({"0": 4, "1": 2, "2": 4, "3": 4, "4": 2},
                        {"0": 8, "1": 2, "2": 4, "3": 4},
                        {"0": 8, "1": 2.0, "2": 4, "3": 4, "4": 2}):
            with self.subTest(targets=targets):
                self.reset_bundle()
                self.provenance["balance_targets_by_class"] = targets
                with self.assertRaises(ValueError):
                    self.load()

    def test_legacy_top_level_policy_cannot_override_quality_provenance(self):
        self.bundle["balance_policy"] = "original_capped_2x"
        with self.assertRaisesRegex(ValueError, "balance_policy"):
            self.load()

    def test_correction_augmentation_can_use_classifier_eligible_reference_counts(self):
        self.reset_bundle((8, 2, 1, 1, 1))
        excluded = [900, 901, 902, 903]
        self.provenance["original_train_ids"].extend(excluded)
        self.provenance["quarantined_ids"].extend(excluded)
        self.provenance["original_train_class_counts"]["2"] += len(excluded)
        self.provenance["augmentation_reference_counts"]["2"] = 5
        self.provenance["balance_targets_by_class"]["2"] = 6
        for index in range(4):
            identifier = f"reference:2:{index}"
            self.train.append(row(identifier, 2, f"reference error {index}", f"reference repair {index}"))
            self.provenance["selected_synthetic_ids"].append(identifier)
            self.provenance["train_rows"].append({"id": identifier, "kind": "synthetic", "parent_ids": []})
        train, _, _ = self.load()
        self.assertEqual(sum(r["label"] == 2 for r in train), 6)

    def test_augmentation_reference_counts_are_bounded_by_original_partition(self):
        for value in (0, 4, 2.0, True):
            with self.subTest(value=value):
                self.reset_bundle()
                self.provenance["augmentation_reference_counts"]["2"] = value
                with self.assertRaises(ValueError):
                    self.load()

    def test_partition_id_lists_require_arrays_and_unique_ids(self):
        for field in ("original_train_ids", "retained_original_train_ids", "selected_synthetic_ids", "quarantined_ids"):
            for invalid in (None, "not an array", "duplicate"):
                with self.subTest(field=field, invalid=invalid):
                    self.reset_bundle()
                    self.provenance[field] = (self.provenance[field] + [self.provenance[field][0]]
                                              if invalid == "duplicate" else invalid)
                    with self.assertRaises(ValueError):
                        self.load()

    def test_retained_and_synthetic_ids_must_be_disjoint_and_cover_train(self):
        for field, action in (("retained_original_train_ids", "remove"),
                              ("selected_synthetic_ids", "remove"),
                              ("retained_original_train_ids", "add"),
                              ("selected_synthetic_ids", "overlap")):
            with self.subTest(field=field, action=action):
                self.reset_bundle()
                if action == "remove":
                    self.provenance[field].pop()
                else:
                    self.provenance[field].append(999 if action == "add" else 0)
                with self.assertRaises(ValueError):
                    self.load()

    def test_retained_originals_must_belong_to_non_holdout_original_partition(self):
        self.provenance["original_train_ids"].remove(1000)
        with self.assertRaises(ValueError):
            self.load()
        self.reset_bundle()
        self.provenance["original_train_ids"].append(10001)
        with self.assertRaises(ValueError):
            self.load()
        self.reset_bundle()
        self.provenance["original_train_ids"].append(self.provenance["selected_synthetic_ids"][0])
        with self.assertRaises(ValueError):
            self.load()

    def test_supplied_train_row_metadata_requires_exact_unique_ids_and_kinds(self):
        for action in ("duplicate", "missing", "unknown", "kind", "invalid"):
            with self.subTest(action=action):
                self.reset_bundle()
                metadata = self.provenance["train_rows"]
                if action == "duplicate":
                    metadata.append(dict(metadata[0]))
                elif action == "missing":
                    metadata.pop()
                elif action == "unknown":
                    metadata[0]["id"] = "unknown"
                elif action == "kind":
                    metadata[0]["kind"] = "synthetic"
                else:
                    metadata[0] = None
                with self.assertRaises(ValueError):
                    self.load()

    def test_synthetic_parent_ids_require_original_training_partition(self):
        for parents in ([10001], ["unknown"], "not an array", [True]):
            with self.subTest(parents=parents):
                self.reset_bundle()
                metadata = next(r for r in self.provenance["train_rows"] if r["kind"] == "synthetic")
                metadata["parent_ids"] = parents
                with self.assertRaises(ValueError):
                    self.load()

    def test_parent_can_be_quarantined_original_in_training_partition(self):
        metadata = next(r for r in self.provenance["train_rows"] if r["kind"] == "synthetic")
        metadata["parent_ids"] = [999]
        self.load()

    def test_optional_train_row_metadata_and_parentless_synthetics_are_accepted(self):
        for metadata in self.provenance["train_rows"]:
            metadata.pop("parent_ids", None)
        self.load()
        del self.provenance["train_rows"]
        self.load()


class ClassifierRetentionTests(unittest.TestCase):
    def test_explicit_sentence_claim_excludes_procedural_annotation(self):
        from model.data import declares_correct_sentence
        for text in ("(Câu đúng ngữ pháp).", "  (Câu này ĐÚNG, không cần sửa.)"):
            self.assertTrue(declares_correct_sentence(text))
        for text in ("(Dùng dấu phẩy để liệt kê đúng quy cách).", "(Câu này không đúng.)", "Câu đúng nhưng không phải chú thích."):
            self.assertFalse(declares_correct_sentence(text))

    def test_annotation_policy_versions_preserve_existing_bundle(self):
        identifier = self.annotation_ids[1]
        raw = next(r for r in self.raw if r["ID"] == identifier)
        raw["Output"]["Fixed Sentence"] = "(Dùng dấu phẩy để liệt kê đúng quy cách)."
        self.original.write_text(json.dumps(self.raw), encoding="utf-8")
        self.make_bundle(self.excluded | {identifier})
        legacy, _ = self.load()
        self.assertNotIn(identifier, {r["id"] for r in legacy})
        self.make_bundle()
        self.bundle["provenance"]["original_retention_policy"] = "exclude_explicit_correct_sentence_annotation_contradictions_v2"
        strict, _ = self.load()
        self.assertIn(identifier, {r["id"] for r in strict})

    def test_unknown_annotation_policy_is_rejected(self):
        self.bundle["provenance"]["original_retention_policy"] = "unchecked"
        with self.assertRaisesRegex(ValueError, "annotation policy"):
            self.load()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original = self.root / "original.json"
        self.path = self.root / "classifier.json"
        self.raw = [row(label * 10 + index, label, f"organizer {label} {index}",
                        None if label == 0 else f"organizer repair {label} {index}")
                    for label in range(5) for index in range(5)]
        self.original.write_text(json.dumps(self.raw), encoding="utf-8")
        train, _ = split_records(load_records(self.original), 0.2, 42)
        self.annotation_ids = [r["id"] for r in train if r["label"] == 2]
        targets = ("  (Câu này ĐÚNG, không cần sửa.)", "(Ghi chú chưa đủ)",
                   "Câu đúng nhưng không phải chú thích.", "Bản sửa ...")
        by_id = {r["ID"]: r for r in self.raw}
        for identifier, target in zip(self.annotation_ids, targets):
            by_id[identifier]["Output"]["Fixed Sentence"] = target
        self.original.write_text(json.dumps(self.raw), encoding="utf-8")
        self.excluded = {self.annotation_ids[0]}
        self.make_bundle()

    def make_bundle(self, excluded=None):
        excluded = self.excluded if excluded is None else excluded
        train, validation = split_records(load_records(self.original), 0.2, 42)
        by_id = {r["ID"]: r for r in self.raw}
        retained = [r for r in train if r["id"] not in excluded]
        counts = [sum(r["label"] == label for r in retained) for label in range(5)]
        targets = {str(label): count + min(40, count) if label == 0 else min(max(counts), 2 * count)
                   for label, count in enumerate(counts)}
        synthetics = [row(f"syn:{label}:{index}", label, f"authored {label} {index}",
                          f"authored {label} {index}" if label == 0 else f"authored repair {label} {index}")
                      for label in range(5) for index in range(targets[str(label)] - counts[label])]
        self.bundle = {
            "format": "vgec-balanced-v3", "seed": 42, "validation_fraction": 0.2,
            "original_sha256": hashlib.sha256(self.original.read_bytes()).hexdigest(),
            "train": [copy.deepcopy(by_id[r["id"]]) for r in retained] + synthetics,
            "validation": [copy.deepcopy(by_id[r["id"]]) for r in validation],
            "provenance": {
                "balance_policy": "quality_capped_2x", "balance_targets_by_class": targets,
                "augmentation_reference_counts": {str(label): count for label, count in enumerate(counts)},
                "original_train_class_counts": {str(label): sum(r["label"] == label for r in train)
                                                for label in range(5)},
                "original_train_ids": [r["id"] for r in train],
                "retained_original_train_ids": [r["id"] for r in retained],
                "quarantined_ids": [r["id"] for r in train if r["id"] in excluded],
                "selected_synthetic_ids": [r["ID"] for r in synthetics],
                "train_rows": [{"id": r["id"], "kind": "original"} for r in retained]
                              + [{"id": r["ID"], "kind": "synthetic", "parent_ids": []} for r in synthetics],
            },
        }

    def load(self):
        self.path.write_text(json.dumps(self.bundle), encoding="utf-8")
        return load_classifier_split(self.original, self.path, 0.2, 42)

    def test_only_contradictory_nonzero_correct_annotation_is_quarantined(self):
        train, _ = self.load()
        actual = {r["id"]: r for r in train}
        self.assertNotIn(self.annotation_ids[0], actual)
        originals = {r["id"]: r for r in load_records(self.original)}
        for identifier in self.annotation_ids[1:]:
            self.assertEqual(actual[identifier], originals[identifier])

    def test_unauthorized_removal_or_retention_is_rejected_even_with_valid_counts(self):
        for excluded in (set(), self.excluded | {self.annotation_ids[1]}):
            with self.subTest(excluded=excluded):
                self.make_bundle(excluded)
                with self.assertRaisesRegex(ValueError, "retention/quarantine"):
                    self.load()

    def test_clean_original_with_correct_annotation_remains_classifier_supervision(self):
        train, _ = split_records(load_records(self.original), 0.2, 42)
        clean_id = next(r["id"] for r in train if r["label"] == 0)
        record = next(r for r in self.raw if r["ID"] == clean_id)
        record["Output"]["Fixed Sentence"] = "(Câu này đúng.)"
        self.original.write_text(json.dumps(self.raw), encoding="utf-8")
        self.make_bundle()
        actual, _ = self.load()
        self.assertEqual(next(r["target"] for r in actual if r["id"] == clean_id), "(Câu này đúng.)")

    def test_classifier_reference_counts_must_equal_recomputed_eligible_originals(self):
        self.bundle["provenance"]["augmentation_reference_counts"]["2"] = 4
        with self.assertRaisesRegex(ValueError, "reference/original counts"):
            self.load()

    def test_original_mutation_or_relabel_is_rejected(self):
        for field, value in (("Input", "changed organizer source"),
                             ("Fixed Sentence", "changed organizer correction"),
                             ("Error_Type_Id", 1)):
            with self.subTest(field=field):
                self.make_bundle()
                record = next(r for r in self.bundle["train"] if r["ID"] == self.annotation_ids[1])
                if field == "Input":
                    record[field] = value
                else:
                    record["Output"][field] = value
                with self.assertRaises(ValueError):
                    self.load()

    def test_raw_null_clean_targets_cannot_be_rewritten_as_explicit_identity(self):
        for partition in ("train", "validation"):
            with self.subTest(partition=partition):
                self.make_bundle()
                record = next(r for r in self.bundle[partition] if r["Output"]["Error_Type_Id"] == 0)
                record["Output"]["Fixed Sentence"] = record["Input"]
                with self.assertRaisesRegex(ValueError, "raw originals/holdout"):
                    self.load()

    def test_forged_original_partition_and_quarantine_ids_are_rejected(self):
        for field in ("original_train_ids", "quarantined_ids"):
            with self.subTest(field=field):
                self.make_bundle()
                self.bundle["provenance"][field].append("not-an-original")
                with self.assertRaises(ValueError):
                    self.load()


if __name__ == "__main__":
    unittest.main()
