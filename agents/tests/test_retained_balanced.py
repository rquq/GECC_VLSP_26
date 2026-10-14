"""Retention/balance contracts use toy strings, never official corpus text."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class RetainedBalanceTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("tools.build_retained_balanced"))
        from tools import build_retained_balanced

        return build_retained_balanced

    def tree(self):
        return (
            "# sent_id = fixture\n# text =FixtureSubject thấy các fixturehead modifier alpha beta .\n"
            "1\tFixtureSubject\tsubject\tNOUN\tN\t_\t2\tnsubj\t_\t_\n"
            "2\tthấy\tthấy\tVERB\tV\t_\t0\troot\t_\t_\n"
            "3\tcác\tcác\tDET\tDet\t_\t4\tdet\t_\t_\n"
            "4\tfixturehead\thead\tNOUN\tN\t_\t2\tobj\t_\t_\n"
            "5\tmodifier\tmodifier\tADJ\tA\t_\t4\tamod\t_\t_\n"
            "6\talpha\talpha\tADJ\tA\t_\t4\tamod\t_\t_\n"
            "7\tbeta\tbeta\tADJ\tA\t_\t4\tamod\t_\t_\n"
            "8\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_\n"
        )

    def row(self, i, label, source=None):
        return {
            "ID": i,
            "Input": source or f"fixture {i}",
            "Output": {"Error_Type_Id": label, "Fixed Sentence": f"target {i}"},
        }

    def test_tree_parser_preserves_exact_text_and_token_offsets(self):
        m = self.module()
        p = m.parse_treebank(self.tree())[0]
        self.assertEqual(
            p["text"], "FixtureSubject thấy các fixturehead modifier alpha beta ."
        )
        for t in p["tokens"]:
            self.assertEqual(p["text"][t["start"] : t["end"]], t["form"])
        self.assertTrue(m.eligible_tree(p))
        bad = copy.deepcopy(p)
        bad["tokens"][0]["relation"] = "obl"
        self.assertFalse(m.eligible_tree(bad))

    def test_generated_pairs_are_reversible_and_vocabulary_safe(self):
        m = self.module()
        p = m.parse_treebank(self.tree())[0]
        candidates = m.tree_candidates(p)
        self.assertTrue(any(c["label"] == 1 for c in candidates))
        self.assertTrue(any(c["label"] == 2 for c in candidates))
        self.assertTrue(any(c["label"] == 4 for c in candidates))
        for c in candidates:
            m.validate_edit(c)
            self.assertEqual(m.restore(c), c["target"])
        bad = copy.deepcopy(candidates[0])
        bad["source"] += " extra"
        with self.assertRaises(ValueError):
            m.validate_edit(bad)

    def test_selection_does_not_relabel_originals_or_fill_with_duplicate_sources(self):
        m = self.module()
        base = [self.row(1, 0), self.row(2, 1)]
        p = m.parse_treebank(self.tree())[0]
        pool = m.tree_candidates(p)
        picked, rejected = m.select_candidates(base, pool + pool, 3)
        self.assertEqual(len({m.key(r["Input"]) for r in picked}), len(picked))
        self.assertTrue(rejected)
        self.assertEqual(base[1]["Output"]["Error_Type_Id"], 1)

    def test_eight_previously_excluded_rows_are_restored_unchanged_for_classification(
        self,
    ):
        m = self.module()
        originals = [self.row(i, 1) for i in sorted(m.PREVIOUSLY_EXCLUDED)]
        baseline = [self.row(1, 0)]
        result = m.classifier_base(originals, baseline)
        self.assertEqual(result[1:], originals)
        self.assertEqual(result[:1], baseline)
        self.assertEqual(m.corrector_originals(originals, []), ([], []))

    def test_balanced_view_reports_repeats_and_never_renames_canonical_rows(self):
        m = self.module()
        rows = [self.row(i, i % 5) for i in range(10)] + [self.row(10, 0)]
        balanced, info = m.balanced_view(rows, target=3, max_repeat=3)
        self.assertEqual(info["exposure_counts"], {str(i): 3 for i in range(5)})
        self.assertEqual(
            info["canonical_counts"], {"0": 3, "1": 2, "2": 2, "3": 2, "4": 2}
        )
        self.assertEqual(balanced[: len(rows)], rows)
        copies = [r for r in balanced if "resampling" in r.get("Provenance", {})]
        self.assertTrue(
            all(
                r["Provenance"]["resampling"]["is_new_unique_example"] is False
                for r in copies
            )
        )
        self.assertEqual(len({r["ID"] for r in balanced}), len(balanced))

    def test_balanced_view_does_not_hide_repeat_limit_shortfall(self):
        m = self.module()
        rows = [self.row(i, 0) for i in range(10)] + [self.row(10, 1)]
        _, info = m.balanced_view(rows, target=10, max_repeat=2)
        self.assertEqual(info["exposure_counts"]["1"], 2)
        self.assertEqual(info["shortfalls"]["1"], 8)

    def test_note_rows_are_kept_once_but_never_oversampled(self):
        m = self.module()
        rows = [self.row(104, 1), self.row(1, 1)]
        train, _ = m.balanced_view(rows, target=4, max_repeat=4)
        self.assertEqual(sum(r["ID"] == 104 for r in train), 1)
        self.assertTrue(
            all(
                r.get("Provenance", {}).get("resampling", {}).get("source_id") != 104
                for r in train
            )
        )

    def test_loader_rejects_resealed_synthetic_label_tampering(self):
        m = self.module()
        from model.retained_balanced import seal, validate
        from augmentation.review import digest

        tree = m.parse_treebank(self.tree())[0]
        external = [
            {
                "ID": "ud:fixture",
                "Input": tree["text"],
                "Output": {"Error_Type_Id": 0, "Fixed Sentence": tree["text"]},
            }
        ]
        row = m.make_row(m.tree_candidates(tree)[0])
        canonical = external + [row]
        raw = seal(
            "classification",
            [],
            [],
            canonical,
            canonical,
            external,
            {"external_parent_trees": [tree]},
        )
        validate(raw)
        for section in ("unique", "train"):
            raw[section][-1]["Output"]["Error_Type_Id"] = 3
        raw["content_sha256"] = digest(
            {k: v for k, v in raw.items() if k != "content_sha256"}
        )
        with self.assertRaises(ValueError):
            validate(raw)

    def test_loader_checks_all_rows_and_rejects_changed_resampled_labels(self):
        self.module()
        from model.retained_balanced import seal, validate

        original = [self.row(1, 0), self.row(2, 1)]
        raw = seal(
            stage="classification",
            original=original,
            baseline=original,
            unique=original,
            train=original,
            external=[],
            metadata={},
        )
        validate(raw)
        bad = copy.deepcopy(raw)
        bad["train"][1]["Output"]["Error_Type_Id"] = 2
        with self.assertRaises(ValueError):
            validate(bad)

    def test_loader_rejects_resealed_original_and_repeat_tampering(self):
        from model.retained_balanced import seal, validate
        from augmentation.review import digest

        m = self.module()
        original = [self.row(1, 0), self.row(2, 1)]
        train, _ = m.balanced_view(original, target=2, max_repeat=2)
        raw = seal("classification", original, original, original, train, [], {})
        validate(raw)
        for section, index, field in [("unique", 0, "Input"), ("train", 2, "Input")]:
            bad = copy.deepcopy(raw)
            bad[section][index][field] = "tampered fixture"
            bad["content_sha256"] = digest(
                {k: v for k, v in bad.items() if k != "content_sha256"}
            )
            with self.assertRaises(ValueError):
                validate(bad)

    def test_public_loaders_dispatch_new_format_without_creating_validation(self):
        from model.retained_balanced import seal
        from model.data import load_training_split, load_classifier_split
        import hashlib

        original = [self.row(1, 0), self.row(2, 1)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            source = path / "original.json"
            bundle = path / "bundle.json"
            source.write_text(json.dumps(original))
            raw = seal(
                "classification",
                original,
                original,
                original,
                original,
                [],
                {
                    "original_file_sha256": hashlib.sha256(
                        source.read_bytes()
                    ).hexdigest()
                },
            )
            bundle.write_text(json.dumps(raw))
            train, validation, info = load_training_split(bundle, 0, 42)
            self.assertEqual(len(train), 2)
            self.assertEqual(validation, [])
            classifier, ci = load_classifier_split(source, bundle, 0, 42)
            self.assertEqual(classifier, train)
            self.assertEqual(ci["provenance"]["quarantined_ids"], [])
            self.assertEqual(ci["provenance"]["selected_synthetic_ids"], [])
            with self.assertRaises(ValueError):
                load_training_split(bundle, 0.2, 42)


if __name__ == "__main__":
    unittest.main()
