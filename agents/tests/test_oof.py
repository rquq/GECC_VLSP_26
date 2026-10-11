"""Consumer-visible leakage boundaries for classifier-conditioned correction."""
import unittest

from model.classification import _connected_folds
from model.data import _id_key, _source_key


class OutOfFoldGroupingTests(unittest.TestCase):
    def test_synthetic_rows_and_omitted_original_parents_never_train_their_hint_model(self):
        originals = [{"id": i, "source": f"source {i}", "label": i % 5, "target": f"target {i}"}
                     for i in range(20)]
        # Original0 is excluded from correction eligibility, but remains classifier-eligible.
        correction = originals[1:] + [{"id": "synthetic", "source": "new error", "label": 1, "target": "fixed"}]
        provenance = {"train_rows": [{"id": "synthetic", "kind": "synthetic", "parent_ids": [0, 1]}]}
        mixed = originals + correction[-1:]
        plans = _connected_folds(originals, correction, provenance, 3, 42, mixed)
        predictions = [row["id"] for _, _, rows in plans for row in rows]
        self.assertCountEqual(predictions, [row["id"] for row in correction])
        for train, heldout, predict in plans:
            train_ids = {_id_key(row["id"]) for row in train}
            self.assertFalse(train_ids & {_id_key(row["id"]) for row in heldout})
            self.assertFalse({_source_key(row["source"]) for row in train} &
                             {_source_key(row["source"]) for row in predict})
            if not any(row["id"] == "synthetic" for row in predict):
                self.assertIn("synthetic", [row["id"] for row in train])
            if any(row["id"] == "synthetic" for row in predict):
                self.assertNotIn(_id_key(0), train_ids)
                self.assertNotIn(_id_key(1), train_ids)
                self.assertIn(1, [row["id"] for row in predict])

    def test_normalized_source_duplicates_remain_one_fold_even_with_typed_ids(self):
        originals = [{"id": i, "source": f"source {i}", "label": i % 5, "target": "fixed"}
                     for i in range(20)]
        duplicate = {"id": "0", "source": "  SOURCE   0 ", "label": 0, "target": "fixed"}
        originals.append(duplicate)
        plans = _connected_folds(originals, originals, {}, 3, 42, originals)
        assignments = {_id_key(row["id"]): fold for fold, (_, _, rows) in enumerate(plans) for row in rows}
        self.assertEqual(assignments[_id_key(0)], assignments[_id_key("0")])
        for train, _, predict in plans:
            self.assertFalse({_source_key(row["source"]) for row in train} &
                             {_source_key(row["source"]) for row in predict})

    def test_clean_error_and_template_variants_never_train_each_others_hints(self):
        originals = [{"id": i, "source": f"original {i}", "label": i % 5, "target": "fixed"}
                     for i in range(40)]
        synthetic = [
            {"id": "clean", "source": "legal construction", "label": 0, "target": "legal construction"},
            {"id": "error", "source": "broken construction", "label": 2, "target": "legal construction"},
            {"id": "variant", "source": "related broken construction", "label": 2, "target": "related legal construction"},
        ]
        metadata = [{"id": r["id"], "authored_provenance": {
            "family_id": "pair" if r["id"] != "variant" else "another-pair",
            "template_id": "phrase-family"}} for r in synthetic]
        plans = _connected_folds(originals, originals + synthetic,
                                 {"train_rows": metadata}, 3, 42, originals + synthetic)
        for train, _, predicted in plans:
            if any(r["id"] == "error" for r in predicted):
                self.assertTrue({"clean", "error", "variant"}.isdisjoint(r["id"] for r in train))
                self.assertTrue({"clean", "error", "variant"}.issubset(r["id"] for r in predicted))


class MixedClassifierFoldTests(unittest.TestCase):
    def test_classifier_only_synthetic_parent_components_and_fold_caps(self):
        from collections import Counter

        originals = [{"id": i, "source": f"original {i}", "label": i % 5, "target": "fixed"}
                     for i in range(40)]
        # Classifier-only row joins two correction source components.
        synthetic = [{"id": f"syn:{i}", "source": f"synthetic {i}", "label": 1, "target": "fixed"}
                     for i in range(8)]
        metadata = [{"id": r["id"], "parent_ids": [1, 6] if i == 0 else []}
                    for i, r in enumerate(synthetic)]
        mixed = originals + synthetic
        plans = _connected_folds(originals, originals, {"train_rows": metadata}, 3, 42, mixed)
        assignments = {r["id"]: fold for fold, (_, _, rows) in enumerate(plans) for r in rows}
        self.assertEqual(assignments[1], assignments[6])
        for fit, _, predicted in plans:
            keys = {r["id"] for r in fit}
            if any(r["id"] in (1, 6) for r in predicted):
                self.assertNotIn("syn:0", keys)
            counts = Counter(r["label"] for r in fit if type(r["id"]) is int)
            added = Counter(r["label"] for r in fit if type(r["id"]) is str)
            self.assertTrue(all(added[label] <= counts[label] for label in range(5)))

