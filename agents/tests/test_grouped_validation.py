"""Partition guarantees use artificial strings, not competition corpus text."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


def row(i, label, text, target=None, parents=(), **provenance):
    result = {
        "ID": i,
        "Input": text,
        "Output": {"Error_Type_Id": label, "Fixed Sentence": target or text},
    }
    if parents or provenance:
        result["Provenance"] = {"parent_ids": list(parents), **provenance}
    return result


def source_packet():
    """A valid retained-v2 packet made exclusively of artificial one-word strings."""
    from model.retained_balanced import seal, FORMAT_V2

    originals = [
        row(
            1000 + label * 10 + i,
            label,
            f"input{label}fixture{i}",
            f"target{label}fixture{i}" if label else None,
        )
        for label in range(5)
        for i in range(10)
    ]
    duplicate = copy.deepcopy(originals[0])
    duplicate["ID"] = 2000
    originals.append(duplicate)
    child = row("child", 2, "childfixture", originals[0]["Input"], [1000])
    baseline = originals + [child]
    sha = hashlib.sha256(json.dumps(originals).encode()).hexdigest()
    return {
        stage: seal(
            stage,
            originals,
            baseline,
            baseline,
            baseline,
            [],
            {"original_file_sha256": sha},
            FORMAT_V2,
        )
        for stage in ("classification", "correction")
    }


def grouped_packet():
    from model.grouped_validation import derive, seal, select_validation

    packet = source_packet()
    ids, stats = select_validation(packet)
    training, validation, *_ = derive(packet, ids)
    return (
        seal("classification", packet, ids, training["classification"], validation, {}),
        stats,
    )


class GroupedValidationTests(unittest.TestCase):
    def test_parent_clean_counterpart_and_template_chain_form_one_group(self):
        from model.grouped_validation import components

        a = row(1, 0, "fixture alpha beta gamma")
        b = row(
            "derived", 2, "fixture beta alpha gamma", a["Input"], [1], family_id="f"
        )
        c = row(
            "linked",
            3,
            "fixture delta",
            "fixture epsilon",
            family_id="f",
            template_id="t",
        )
        d = row("template", 4, "fixture zeta", "fixture theta", template_id="t")
        graph = components([a, b, c, d])
        self.assertEqual(len(graph), 1)

    def test_unicode_punctuation_token_bag_and_near_duplicates_are_grouped(self):
        from model.grouped_validation import components

        first = "one two three four five six seven eight nine ten eleven twelve"
        changed = "one two three four five six seven eight nine ten eleven changed"
        rows = [
            row(1, 1, first),
            row(2, 2, changed),
            row(3, 0, first.upper() + "."),
            row(4, 2, " ".join(reversed(first.split()))),
        ]
        self.assertEqual(len(components(rows)), 1)

    def test_typed_ids_and_unrelated_rows_do_not_collapse(self):
        from model.grouped_validation import components

        groups = components([row(1, 0, "fixturealpha"), row("1", 1, "fixturebeta")])
        self.assertEqual(len(groups), 2)

    def test_near_duplicate_grouping_without_any_equal_ten_token_span(self):
        from model.grouped_validation import components, near, tokens

        first = "one two three four five six seven eight nine ten"
        second = "one two three four changed six seven eight nine ten"
        self.assertTrue(near(tokens(first), tokens(second)))
        self.assertEqual(len(components([row(1, 1, first), row(2, 1, second)])), 1)

    def test_missing_parent_is_rejected(self):
        from model.grouped_validation import components

        with self.assertRaises(ValueError):
            components([row("x", 2, "x", "y", [999])])

    def test_audit_rejects_cross_stage_clean_target_overlap(self):
        from model.grouped_validation import audit

        held = [row(1, 2, "fixture incorrect phrase", "fixture correct phrase")]
        training = {
            "classification": [row("a", 0, "unrelated fixture")],
            "correction": [row("b", 0, "fixture correct phrase")],
        }
        with self.assertRaises(ValueError):
            audit(training, held)

    def test_audit_detects_a_parent_even_when_text_is_unrelated(self):
        from model.grouped_validation import audit

        held = [row(1, 2, "held input", "held target")]
        with self.assertRaises(ValueError):
            audit(
                {
                    "classification": [
                        row("x", 2, "different input", "different target", [1])
                    ]
                },
                held,
            )

    def test_audit_rejects_synthetic_validation(self):
        from model.grouped_validation import audit

        with self.assertRaises(ValueError):
            audit({"classification": [row(1, 0, "train")]}, [row("fake", 2, "test")])

    def test_held_parent_descendant_and_duplicate_are_all_withheld(self):
        from model.grouped_validation import derive

        packet = source_packet()
        training, validation, canonical, withheld, _, report = derive(packet, [1000])
        self.assertEqual(validation, [packet["classification"]["original_snapshot"][0]])
        for stage in training:
            ids = {r["ID"] for r in training[stage]}
            self.assertTrue({1000, 2000, "child"}.isdisjoint(ids))
            self.assertEqual({r["ID"] for r in withheld[stage]}, {2000, "child"})
            self.assertEqual(training[stage][: len(canonical[stage])], canonical[stage])
        self.assertTrue(report["passed"])

    def test_partial_original_family_and_synthetic_validation_are_rejected(self):
        from model.grouped_validation import derive

        packet = source_packet()
        packet["classification"]["original_snapshot"][1]["Provenance"] = {
            "family_id": "shared"
        }
        packet["classification"]["original_snapshot"][2]["Provenance"] = {
            "family_id": "shared"
        }
        with self.assertRaises(ValueError):
            derive(packet, [1001])
        with self.assertRaises(ValueError):
            derive(source_packet(), ["child"])

    def test_selection_is_repeatable_and_never_resamples_validation(self):
        from model.grouped_validation import select_validation, derive

        packet = source_packet()
        first, stats = select_validation(packet)
        self.assertEqual(select_validation(packet), (first, stats))
        self.assertEqual(stats["actual_counts"], {label: 2 for label in range(5)})
        training, validation, canonical, _, _, _ = derive(packet, first)
        self.assertEqual(len(validation), 10)
        for stage in training:
            self.assertTrue(
                all(
                    type(r["ID"]) is not int
                    for r in training[stage][len(canonical[stage]) :]
                )
            )
        self.assertTrue(
            all(type(r["ID"]) is int and not r.get("Provenance") for r in validation)
        )

    def test_loader_rejects_resealed_training_contamination_and_protocol_changes(self):
        from model.grouped_validation import load_grouped
        from model.retained_rules import digest

        raw, _ = grouped_packet()
        for fraction, seed in ((0, 42), (0.25, 42), (0.2, 43)):
            with self.assertRaises(ValueError):
                load_grouped(raw, fraction, seed)
        dirty = copy.deepcopy(raw)
        dirty["train"].append(dirty["validation"][0])
        dirty["content_sha256"] = digest(
            {k: v for k, v in dirty.items() if k != "content_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "Derived"):
            load_grouped(dirty, 0.2, 42)

    def test_public_loaders_use_the_shared_holdout_and_bind_the_original_file(self):
        from model.data import load_classifier_split, load_training_split
        from model.classification import _organizer_partition

        raw, _ = grouped_packet()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bundle.json"
            original = Path(temp) / "original.json"
            path.write_text(json.dumps(raw))
            original.write_text(
                json.dumps(raw["source_packet"]["classification"]["original_snapshot"])
            )
            train, validation, info = load_training_split(path, 0.2, 42)
            classifier_train, classifier_info = load_classifier_split(
                original, path, 0.2, 42
            )
            original_train, classifier_validation = _organizer_partition(
                original, 0.2, 42, classifier_info
            )
            self.assertEqual(classifier_train, train)
            self.assertEqual(classifier_validation, validation)
            self.assertTrue(
                {r["id"] for r in original_train}.isdisjoint(
                    {r["id"] for r in validation}
                )
            )
            if 1000 in raw["validation_original_ids"]:
                self.assertNotIn(2000, {r["id"] for r in original_train})
            original.write_text(original.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "snapshot/hash"):
                load_classifier_split(original, path, 0.2, 42)

    def test_private_package_reopens_and_rejects_changed_zip_payload(self):
        import zipfile
        from tools.package_grouped_validation import package, verify_archive
        from model.data import load_training_split

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            packet = source_packet()
            for stage, name in [
                ("classification", "VGEC_classifier.json"),
                ("correction", "VGEC_balanced.json"),
            ]:
                (source / name).write_text(json.dumps(packet[stage]))
            (source / "VGEC 2026_v1.json").write_text(
                json.dumps(packet["classification"]["original_snapshot"])
            )
            public = root / "public.json"
            public.write_text('[{"ID": 9, "Input": "inferencefixture"}]')
            output = root / "package"
            result = package(source, public, output)
            a = load_training_split(output / "VGEC_classifier.json", 0.2, 42)
            b = load_training_split(output / "VGEC_balanced.json", 0.2, 42)
            self.assertEqual(a[1], b[1])
            self.assertEqual(result["validation"]["rows"], 10)
            files = json.loads((output / "file_manifest.json").read_text())["files"]
            damaged = root / "damaged.zip"
            with zipfile.ZipFile(result["archive"]) as good, zipfile.ZipFile(
                damaged, "w"
            ) as bad:
                for name in good.namelist():
                    bad.writestr(
                        name,
                        b"changed" if name == "validation.json" else good.read(name),
                    )
            with self.assertRaisesRegex(ValueError, "payload differs"):
                verify_archive(damaged, files)


if __name__ == "__main__":
    unittest.main()
