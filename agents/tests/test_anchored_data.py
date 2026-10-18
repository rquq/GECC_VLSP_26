"""Sampling invariants use artificial strings, never organizer corpus fixtures."""

import copy
import unittest
from collections import Counter

from agents.tests.test_grouped_validation import row, grouped_packet


class AnchoredDataTests(unittest.TestCase):
    def test_originals_retained_and_rare_original_replay_is_bounded(self):
        from model.anchored_data import curate

        original = [
            row(1000 + k, k, f"source{k} fixture.", f"target{k} fixture.")
            for k in range(5)
        ]
        train, report = curate(original, "classification")
        self.assertEqual(train[:5], original)
        self.assertEqual(report["original_canonical_rows"], 5)
        self.assertEqual(
            Counter(r["Output"]["Error_Type_Id"] for r in train),
            {0: 1, 1: 2, 2: 3, 3: 3, 4: 3},
        )
        self.assertEqual(len({r["ID"] for r in train}), len(train))

    def test_notes_and_abbreviated_correction_targets_are_never_repeated(self):
        from model.anchored_data import curate

        original = [
            row(103, 2, "note source.", "(annotation)"),
            row(1001, 3, "error source.", "prefix ..."),
        ]
        train, _ = curate(original, "correction")
        self.assertEqual(train, original)
        classifier, _ = curate(original, "classification")
        self.assertEqual(sum(r["ID"] == 103 for r in classifier), 1)

    def test_synthetic_repetition_shortcuts_and_duplicate_inputs_are_excluded(self):
        from model.anchored_data import curate, POLICY

        original = [row(1000, 0, "clean parent fixture.")]
        pool = [
            row(
                f"candidate:{i}",
                2,
                f"synthetic{i} fixture.",
                "clean parent fixture.",
                [1000],
                template_rule="same-rule",
            )
            for i in range(30)
        ]
        pool.append(
            row(
                "quoted",
                2,
                "quoted fixture.",
                "clean parent fixture.",
                [1000],
                template_rule="quoted_original_clause_edit_ascii",
            )
        )
        pool.append(copy.deepcopy(pool[0]))
        pool[-1]["ID"] = "duplicate"
        train, report = curate(original + pool, "classification")
        self.assertLessEqual(len(train) - 1, POLICY["parent_cap"])
        self.assertFalse(any(r["ID"] == "quoted" for r in train))
        self.assertEqual(len({r["Input"] for r in train}), len(train))
        self.assertGreater(report["withheld_synthetic_rows"], 0)

    def test_external_parent_is_preserved_when_selected_child_needs_it(self):
        from model.anchored_data import curate

        parent = row(
            "ud:fixture",
            0,
            "external parent fixture.",
            origin="human_annotated_ud_vietnamese_vtb_train",
        )
        child = row(
            "candidate",
            4,
            "external parent fixture.[",
            "external parent fixture.",
            ["ud:fixture"],
            template_rule="unmatched_bracket_opening",
        )
        train, _ = curate([parent, child], "classification")
        self.assertEqual({r["ID"] for r in train}, {parent["ID"], child["ID"]})

    def test_unknown_parent_does_not_enter_fit(self):
        from model.anchored_data import curate

        child = row(
            "candidate",
            4,
            "error fixture.[",
            "target fixture.",
            [9999],
            template_rule="unmatched_bracket_opening",
        )
        train, _ = curate([child], "classification")
        self.assertFalse(train)

    def test_roundtrip_preserves_gold_and_rejects_resealed_injection(self):
        from model.anchored_data import seal, load_anchored
        from model.quality_data import seal as quality_seal
        from model.retained_rules import digest

        source, _ = grouped_packet()
        base = quality_seal(source)
        bundle = seal(base)
        train, val, info = load_anchored(bundle, 0.2, 42)
        self.assertTrue(train)
        self.assertEqual([r["id"] for r in val], [r["ID"] for r in base["validation"]])
        self.assertTrue(info["leakage_audit"]["passed"])
        broken = copy.deepcopy(bundle)
        broken["train"].append(source["validation"][0])
        broken["content_sha256"] = digest(
            {k: v for k, v in broken.items() if k != "content_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "deterministic"):
            load_anchored(broken, 0.2, 42)

    def test_policy_and_source_changes_fail_even_when_outer_hash_recomputed(self):
        from model.anchored_data import seal, load_anchored
        from model.quality_data import seal as quality_seal
        from model.retained_rules import digest

        source, _ = grouped_packet()
        bundle = seal(quality_seal(source))
        bundle["policy"]["parent_cap"] = 99
        bundle["content_sha256"] = digest(
            {k: v for k, v in bundle.items() if k != "content_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "protocol"):
            load_anchored(bundle, 0.2, 42)
