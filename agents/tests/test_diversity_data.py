"""Diversity gates use artificial strings and fixed evidence, never corpus-derived fixtures."""

import copy
import unittest


class DiversityDataTests(unittest.TestCase):
    def test_reviewer_qualification_requires_each_subtype_and_fixed_controls(self):
        from model.diversity_data import qualification

        events = [
            {"task": {"expected_label": k}, "parsed": {"label": k, "certain": True}}
            for k in range(5)
            for _ in range(2)
        ]
        self.assertTrue(qualification(events)["passed"])
        bad = copy.deepcopy(events)
        for e in bad:
            if e["task"]["expected_label"] == 2:
                e["parsed"]["label"] = 0
        self.assertFalse(qualification(bad)["passed"])
        self.assertFalse(qualification(events[:5])["passed"])

    def test_pair_gates_reject_semantic_rewrites_and_subtype_mismatch(self):
        from model.diversity_data import pair_gate

        source = "alpha gamma beta delta."
        target = "alpha beta gamma delta."
        review = {"label": 2, "corrected": target, "certain": True}
        self.assertTrue(pair_gate(source, target, 2, review)[0])
        for label in (0, 3):
            wrong = {**review, "label": label}
            self.assertFalse(pair_gate(source, target, 2, wrong)[0])
        self.assertFalse(pair_gate("alpha omega beta delta.", target, 2, review)[0])
        self.assertFalse(
            pair_gate(source, target, 2, {**review, "corrected": source})[0]
        )
        self.assertFalse(pair_gate(source, target, 2, {**review, "certain": False})[0])
        self.assertFalse(
            pair_gate(
                "alpha beta gamma 12.",
                "alpha beta gamma 21.",
                3,
                {"label": 3, "corrected": "alpha beta gamma 21.", "certain": True},
            )[0]
        )

    def test_parent_boundary_blocks_reordered_and_near_validation_text(self):
        from model.diversity_data import blocked_text

        v = [
            {
                "ID": 99,
                "Input": "one two three four five six seven eight nine ten.",
                "Output": {"Error_Type_Id": 0, "Fixed Sentence": None},
            }
        ]
        self.assertTrue(
            blocked_text("ten nine eight seven six five four three two one.", v)
        )
        self.assertFalse(blocked_text("entirely different fixture wording.", v))

    def test_dependency_degree_and_real_paired_punctuation_are_reversible(self):
        from model.diversity_data import extra_candidates
        from model.retained_rules import restore

        t = {
            "id": "fixture",
            "text": "alpha rất beta “gamma”.",
            "tokens": [
                {
                    "id": 1,
                    "form": "alpha",
                    "upos": "NOUN",
                    "head": 3,
                    "relation": "nsubj",
                    "start": 0,
                    "end": 5,
                },
                {
                    "id": 2,
                    "form": "rất",
                    "upos": "ADV",
                    "head": 3,
                    "relation": "advmod",
                    "start": 6,
                    "end": 9,
                },
                {
                    "id": 3,
                    "form": "beta",
                    "upos": "ADJ",
                    "head": 0,
                    "relation": "root",
                    "start": 10,
                    "end": 14,
                },
            ],
        }
        values = list(extra_candidates(t))
        self.assertTrue(any(v["label"] == 2 for v in values))
        self.assertTrue(any(v["label"] == 4 for v in values))
        for v in values:
            self.assertEqual(restore(v), t["text"])

    def test_diverse_loader_preserves_holdout_and_rejects_added_unverified_text(self):
        from agents.tests.test_grouped_validation import grouped_packet
        from model.quality_data import seal as quality_seal
        from model.diversity_data import seal_diverse, load_diverse
        from model.retained_rules import digest

        base, _ = grouped_packet()
        base = quality_seal(base)
        bundle = seal_diverse(base, "")
        train, validation, info = load_diverse(bundle, 0.2, 42)
        self.assertTrue(train)
        self.assertEqual(
            [r["id"] for r in validation], [r["ID"] for r in base["validation"]]
        )
        broken = copy.deepcopy(bundle)
        broken["train"].append(base["validation"][0])
        broken["content_sha256"] = digest(
            {k: v for k, v in broken.items() if k != "content_sha256"}
        )
        with self.assertRaises(ValueError):
            load_diverse(broken, 0.2, 42)

    def test_observed_prefix_rules_use_human_labels_and_reject_conflicting_taxonomy(
        self,
    ):
        from model.diversity_data import observed_prefixes

        def r(i, label, s, t):
            return {
                "ID": i,
                "Input": s,
                "Output": {"Error_Type_Id": label, "Fixed Sentence": t},
            }

        rows = [r(1, 3, "Bằng alpha beta gamma.", "Alpha beta gamma.")]
        self.assertEqual(observed_prefixes(rows), {"Bằng": [1]})
        rows.append(r(2, 1, "Bằng omega psi gamma.", "Omega psi gamma."))
        self.assertEqual(observed_prefixes(rows), {})

    def test_additive_exposure_policy_preserves_baseline_training_exactly(self):
        from agents.tests.test_grouped_validation import grouped_packet
        from model.quality_data import seal
        from model.diversity_data import derive_diverse, ADDITIVE_POLICY

        base, _ = grouped_packet()
        base = seal(base)
        # A pure sampler check; the source loader separately enforces its sealed contract.
        row = copy.deepcopy(
            next(r for r in base["train"] if r["Output"]["Error_Type_Id"] == 1)
        )
        row["Provenance"] = {
            "resampling": {
                "source_id": row["ID"],
                "draw": 2,
                "is_new_unique_example": False,
            }
        }
        row["ID"] = "existing-repeat"
        base["train"].append(row)
        train, report = derive_diverse(base, "", policy=ADDITIVE_POLICY)
        self.assertEqual(train[: len(base["train"])], base["train"])
        self.assertEqual(
            report["exposure_strategy"], "preserve_baseline_append_new_once"
        )
