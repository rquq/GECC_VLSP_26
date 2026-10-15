"""Voting and artifact contracts use artificial fixtures only."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from model.runtime import write_json


def prediction(i, values):
    return {
        "id": i,
        "label": max(range(5), key=values.__getitem__),
        "probabilities": values,
    }


class VotingTests(unittest.TestCase):
    def test_soft_vote_can_override_hard_majority_and_preserves_probabilities(self):
        from model.classifier_ensemble import vote

        rows = [
            [prediction(1, [0.30, 0.29, 0.2, 0.11, 0.1])],
            [prediction(1, [0.30, 0.29, 0.2, 0.11, 0.1])],
            [prediction(1, [0, 1, 0, 0, 0])],
        ]
        result = vote(rows)[0]
        self.assertEqual(result["label"], 1)
        self.assertGreater(
            result["mean_probabilities"][1], result["mean_probabilities"][0]
        )
        self.assertEqual(result["votes"], [2, 1, 0, 0, 0])
        self.assertEqual(max(range(5), key=result["probabilities"].__getitem__), 1)
        self.assertEqual(result["probabilities"], result["mean_probabilities"])
        self.assertAlmostEqual(result["probabilities"][1], 1.58 / 3)
        self.assertAlmostEqual(sum(result["probabilities"]), 1)

    def test_three_way_tie_uses_mean_scores_then_lowest_label(self):
        from model.classifier_ensemble import vote

        rows = [
            [prediction(1, [0.51, 0.49, 0, 0, 0])],
            [prediction(1, [0, 0.6, 0.4, 0, 0])],
            [prediction(1, [0, 0, 1, 0, 0])],
        ]
        self.assertEqual(vote(rows)[0]["label"], 2)
        exact = [
            [prediction(1, [1, 0, 0, 0, 0])],
            [prediction(1, [0, 1, 0, 0, 0])],
            [prediction(1, [0, 0, 1, 0, 0])],
        ]
        self.assertEqual(vote(exact)[0]["label"], 0)

    def test_alignment_uses_typed_ids_and_preserves_first_member_order(self):
        from model.classifier_ensemble import vote

        one = [prediction(1, [1, 0, 0, 0, 0]), prediction("1", [0, 1, 0, 0, 0])]
        self.assertEqual(
            [p["id"] for p in vote([one, list(reversed(one)), one])], [1, "1"]
        )
        for bad in ([one, one], [one, one, one[:1]], [one, one, one + [one[0]]]):
            with self.assertRaises(ValueError):
                vote(bad)

    def test_invalid_probabilities_or_label_fail_closed(self):
        from model.classifier_ensemble import vote

        valid = [prediction(1, [1, 0, 0, 0, 0])]
        for values, label in [
            ([float("nan"), 0, 0, 0, 0], 0),
            ([1, 1, 0, 0, 0], 0),
            ([1, 0, 0, 0, 0], 2),
            ([True, 0, 0, 0, 0], 0),
        ]:
            bad = [{"id": 1, "label": label, "probabilities": values}]
            with self.assertRaises(ValueError):
                vote([valid, valid, bad])

    def test_recipe_rejects_hard_voting_duplicate_bases_and_non42_seed(self):
        from model.runtime import load_config

        baseline = json.loads(
            (
                Path(__file__).resolve().parents[2] / "configs/validation_baseline.json"
            ).read_text()
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for change in ("hard", "duplicate", "seed", "missing_model"):
                config = copy.deepcopy(baseline)
                if change == "hard":
                    config["ensemble"]["method"] = "hard"
                elif change == "duplicate":
                    config["ensemble"]["members"][2]["model"] = config["model"]
                elif change == "seed":
                    config["run"]["seed"] = 43
                else:
                    del config["ensemble"]["members"][2]["model"]
                write_json(path, config)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_config(path)


class EnsembleArtifactTests(unittest.TestCase):
    def fixture(self, root):
        from agents.tests.test_run_reuse import classifier_fixture

        rows = [
            {"id": i, "source": f"fixture{i}", "label": i, "target": f"fixture{i}"}
            for i in range(5)
        ]
        entries = []
        for name in ("qwen7b", "sailor7b", "mistral7b"):
            stage = root / "members" / name
            config = classifier_fixture(stage, hints=False)
            config.pop("ensemble", None)
            config["run"]["seed"] = 42
            config["model"]["name"] = f"local/{name}"
            entries.append({"name": name, "model": copy.deepcopy(config["model"])})
            saved_path = stage / "classifier/classifier_config.json"
            saved = json.loads(saved_path.read_text())
            saved["model"] = config["model"]
            write_json(saved_path, saved)
            adapter_path = stage / "classifier/adapter_config.json"
            adapter = json.loads(adapter_path.read_text())
            adapter["base_model_name_or_path"] = config["model"]["name"]
            write_json(adapter_path, adapter)
            write_json(stage / "config.json", config)
            manifest = json.loads((stage / "run_manifest.json").read_text())
            manifest.update(
                seed=42,
                data_sha256="same-original",
                train_count=5,
                validation_count=5,
            )
            write_json(stage / "run_manifest.json", manifest)
            write_json(stage / "train.json", rows)
            write_json(stage / "validation.json", rows)
            write_json(
                stage / "predictions.json",
                [
                    {
                        "id": r["id"],
                        "predicted_class": r["label"],
                        "probabilities": [int(i == r["label"]) for i in range(5)],
                    }
                    for r in rows
                ],
            )
        root_config = copy.deepcopy(config)
        root_config["run"]["seed"] = 42
        root_config["model"] = entries[0]["model"]
        root_config["ensemble"] = {"members": entries}
        return root_config

    def test_bundle_copies_portably_and_is_discovered_as_one_classifier(self):
        from model.classifier_ensemble import build, members
        from model.run_reuse import (
            copy_classifier_run,
            resolve_run,
            validate_classifier_run,
        )

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "source/classification"
            config = self.fixture(root)
            report = build(root, config)
            self.assertEqual(report["classification"]["macro_f1"], 1)
            self.assertEqual(
                [
                    r["name"]
                    for r in validate_classifier_run(root, require_hints=False)[
                        "ensemble"
                    ]["members"]
                ],
                ["qwen7b", "sailor7b", "mistral7b"],
            )
            destination = Path(folder) / "mount/uploaded/classification"
            copy_classifier_run(root, destination)
            import shutil

            shutil.rmtree(root.parent)
            self.assertEqual(resolve_run(Path(folder) / "mount"), destination)
            validate_classifier_run(destination, require_hints=False)
            self.assertEqual(len(members(destination / "classifier")), 3)
            with self.assertRaisesRegex(ValueError, "without OOF"):
                validate_classifier_run(destination, require_hints=True)

    def test_mismatched_member_population_and_altered_adapter_are_rejected(self):
        from model.classifier_ensemble import build, members

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self.fixture(root)
            path = root / "members/mistral7b/train.json"
            original = path.read_bytes()
            rows = json.loads(original)
            rows[0]["source"] = "changed"
            write_json(path, rows)
            with self.assertRaisesRegex(ValueError, "differ in data"):
                build(root, config)
            path.write_bytes(original)
            build(root, config)
            weights = root / "members/mistral7b/classifier/adapter_model.safetensors"
            weights.write_bytes(weights.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                members(root / "classifier")

    def test_changed_root_votes_are_rejected_before_reuse(self):
        from model.classifier_ensemble import build, validate

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self.fixture(root)
            build(root, config)
            path = root / "predictions.json"
            rows = json.loads(path.read_text())
            rows[0]["predicted_class"] = 4
            write_json(path, rows)
            with self.assertRaisesRegex(ValueError, "votes differ"):
                validate(root)


if __name__ == "__main__":
    unittest.main()
