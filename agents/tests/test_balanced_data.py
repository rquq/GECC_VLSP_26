"""Capped augmentation must preserve originals without leaking into validation."""
import json
from pathlib import Path
import tempfile
import unittest

from model.data import load_training_split


def row(identifier, label, source, target):
    return {"ID": identifier, "Input": source,
            "Output": {"Error_Type_Id": label, "Fixed Sentence": target}}


def bundle(original_counts=(4, 1, 2, 3, 1)):
    originals = [row(label * 10 + index, label, f"original {label} {index}",
                     None if label == 0 else f"correction {label} {index} ...")
                 for label, count in enumerate(original_counts) for index in range(count)]
    largest = max(original_counts)
    targets = {str(label): count if label == 0 else min(largest, 2 * count)
               for label, count in enumerate(original_counts)}
    synthetics = [row(f"syn:{label}:{index}", label, f"synthetic {label} {index}",
                      f"synthetic correction {label} {index}")
                  for label in range(1, 5)
                  for index in range(targets[str(label)] - original_counts[label])]
    validation = [row(1000, 0, "Original clean sentence.", None),
                  row(1001, 1, "Original incorrect sentence.", "(Annotation) ...")]
    provenance = {
        "balance_policy": "original_capped_2x", "balance_targets_by_class": targets,
        "original_train_ids": [r["ID"] for r in originals] + [999],
        "retained_original_train_ids": [r["ID"] for r in originals],
        "selected_synthetic_ids": [r["ID"] for r in synthetics],
        "train_rows": [{"id": r["ID"], "kind": "original", "original_id": r["ID"]}
                       for r in originals] +
                      [{"id": r["ID"], "kind": "synthetic", "parent_ids": [r["Output"]["Error_Type_Id"] * 10]}
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
        self.assertEqual([r["id"] for r in validation], [1000, 1001])
        self.assertEqual(validation[0]["target"], "Original clean sentence.")
        self.assertEqual(validation[1]["target"], "(Annotation) ...")
        self.assertEqual({r["id"] for r in train}, {r["ID"] for r in self.train})
        self.assertEqual(next(r["target"] for r in train if r["id"] == 10), "correction 1 0 ...")
        counts = {str(label): sum(r["label"] == label for r in train) for label in range(5)}
        self.assertEqual(counts, {"0": 4, "1": 2, "2": 4, "3": 4, "4": 2})
        originals = self.provenance["retained_original_train_ids"]
        self.assertTrue(all(sum(r["label"] == label for r in train) <=
                            2 * sum(r["label"] == label and r["id"] in originals for r in train)
                            for label in range(1, 5)))
        self.assertEqual(metadata["provenance"], self.provenance)
        self.assertEqual(metadata["format"], "vgec-balanced-v3")

    def test_majority_can_be_an_error_class_and_class_zero_is_not_augmented(self):
        self.reset_bundle((2, 5, 1, 3, 1))
        train, _, _ = self.load()
        self.assertEqual([sum(r["label"] == label for r in train) for label in range(5)],
                         [2, 5, 2, 5, 2])

    def test_seed_and_fraction_mismatch_cannot_resplit_augmented_rows(self):
        with self.assertRaisesRegex(ValueError, "Training seed must be 42"):
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
        with self.assertRaisesRegex(ValueError, "Training seed must be 42"):
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
        self.train[0]["ID"] = 1000
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

    def test_synthetic_class_zero_rejected(self):
        self.add_synthetic(0)
        with self.assertRaises(ValueError):
            self.load()

    def test_all_five_classes_require_retained_originals(self):
        self.train[:] = [r for r in self.train if r["ID"] != 40]
        self.provenance["retained_original_train_ids"].remove(40)
        self.provenance["train_rows"][:] = [r for r in self.provenance["train_rows"] if r["id"] != 40]
        with self.assertRaises(ValueError):
            self.load()

    def test_policy_and_declared_targets_must_match_actual_original_counts(self):
        self.provenance["balance_policy"] = "equal_classes"
        with self.assertRaises(ValueError):
            self.load()
        for targets in ({"0": 4, "1": 4, "2": 4, "3": 4, "4": 4},
                        {"0": 4, "1": 2, "2": 4, "3": 4},
                        {"0": 4, "1": 2.0, "2": 4, "3": 4, "4": 2}):
            with self.subTest(targets=targets):
                self.reset_bundle()
                self.provenance["balance_targets_by_class"] = targets
                with self.assertRaises(ValueError):
                    self.load()

    def test_partition_id_lists_require_arrays_and_unique_ids(self):
        for field in ("original_train_ids", "retained_original_train_ids", "selected_synthetic_ids"):
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
        self.provenance["original_train_ids"].remove(10)
        with self.assertRaises(ValueError):
            self.load()
        self.reset_bundle()
        self.provenance["original_train_ids"].append(1001)
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
        for parents in ([1001], ["unknown"], "not an array", [True]):
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


if __name__ == "__main__":
    unittest.main()
