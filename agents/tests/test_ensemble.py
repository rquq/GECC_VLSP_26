"""Behavior contracts for typed-ID probability ensembles and saved evaluation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from model.ensemble import _load_config, average_predictions, classify, evaluate


def prediction(identifier, probabilities):
    return {"id": identifier, "label": max(range(5), key=probabilities.__getitem__),
            "probabilities": probabilities}


def vector(label):
    return [float(index == label) for index in range(5)]


class AveragePredictionsTests(unittest.TestCase):
    def test_probability_average_disagrees_with_majority_vote(self):
        records = [{"id": "unseen", "source": "new source"}]
        members = [
            [prediction("unseen", [0.36, 0.34, 0.1, 0.1, 0.1])],
            [prediction("unseen", [0.36, 0.34, 0.1, 0.1, 0.1])],
            [prediction("unseen", [0.0, 1.0, 0.0, 0.0, 0.0])],
        ]
        before = copy.deepcopy((records, members))
        averaged = average_predictions(records, members)
        self.assertEqual([member[0]["label"] for member in members], [0, 0, 1])
        self.assertEqual(averaged[0]["label"], 1)
        for actual, expected in zip(averaged[0]["probabilities"], [0.24, 0.56, 1 / 15, 1 / 15, 1 / 15]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual((records, members), before)
        # Nesting a two-model mean would incorrectly give the third 1/2 weight.
        nested = average_predictions(records, [average_predictions(records, members[:2]), members[2]])
        self.assertNotEqual(averaged[0]["probabilities"], nested[0]["probabilities"])

    def test_unordered_exact_typed_ids_align_to_record_order(self):
        records = [{"id": 1, "source": "integer"}, {"id": "1", "source": "string"}]
        members = [
            [prediction("1", vector(4)), prediction(1, vector(2))],
            [prediction(1, [0, 0, 0.8, 0.2, 0]), prediction("1", [0, 0, 0, 0.2, 0.8])],
        ]
        averaged = average_predictions(records, members)
        self.assertEqual([row["id"] for row in averaged], [1, "1"])
        self.assertEqual([row["label"] for row in averaged], [2, 4])
        self.assertEqual(averaged[0]["probabilities"], [0, 0, 0.9, 0.1, 0])
        self.assertEqual(averaged[1]["probabilities"], [0, 0, 0, 0.1, 0.9])

    def test_supplied_labels_do_not_override_probabilities_and_ties_are_stable(self):
        records = [{"id": 7, "source": "source"}]
        result = average_predictions(records, [[{"id": 7, "label": 4,
                                                   "probabilities": [0.5, 0.5, 0, 0, 0]}]])
        self.assertEqual(result[0]["label"], 0)

    def test_duplicate_missing_extra_and_wrong_typed_member_ids_fail(self):
        records = [{"id": 1, "source": "source"}]
        row = prediction(1, vector(0))
        for member in ([], [row, row], [prediction("1", vector(0))],
                       [row, prediction(2, vector(0))], [{"probabilities": vector(0)}],
                       [prediction(True, vector(0))], [prediction(1.0, vector(0))]):
            with self.subTest(member=member), self.assertRaises(ValueError):
                average_predictions(records, [[row], member])
        for malformed in ([records[0], records[0]], [{"id": True, "source": "source"}],
                          [{"id": 1, "source": None}]):
            with self.subTest(records=malformed), self.assertRaises(ValueError):
                average_predictions(malformed, [[row]])
        with self.assertRaises(ValueError):
            average_predictions(records, [])

    def test_invalid_vectors_fail_without_normalization_or_clipping(self):
        records = [{"id": 1, "source": "source"}]
        vectors = [None, [], [1, 0, 0, 0], [1, 0, 0, 0, 0, 0],
                   [0.2, 0.2, 0.2, 0.2, 0.1], [0, 0, 0, 0, 0],
                   [-0.1, 1.1, 0, 0, 0], [float("nan"), 1, 0, 0, 0],
                   [float("inf"), 0, 0, 0, 0], [float("-inf"), 1, 0, 0, 0],
                   ["1", 0, 0, 0, 0], [True, 0, 0, 0, 0],
                   {"0": 1, "1": 0, "2": 0, "3": 0, "4": 0}]
        for probabilities in vectors:
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                average_predictions(records, [[{"id": 1, "probabilities": probabilities}]])

    def test_small_float_error_is_preserved_not_renormalized(self):
        records = [{"id": 1, "source": "source"}]
        probabilities = [0.5, 0.4999999, 0, 0, 0]
        self.assertEqual(average_predictions(records, [[prediction(1, probabilities)]])[0]["probabilities"],
                         probabilities)
        self.assertEqual(average_predictions([], [[], []]), [])


class SavedEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.records = [{"id": identifier, "source": f"source {index}", "label": index,
                         "target": f"target {index}"}
                        for index, identifier in enumerate([1, "1", 2, "2", 3])]
        self.qwen_vectors = [vector(1), vector(1), [0, 0, 0.4, 0.35, 0.25], vector(3), vector(4)]
        self.bamibert_vectors = [vector(0), vector(1), vector(2), vector(3), vector(4)]
        self.xlmr_vectors = [vector(0), vector(1), vector(3), vector(3), vector(4)]
        self.folders = [self._run(name, probabilities) for name, probabilities in
                        (("qwen", self.qwen_vectors), ("bamibert", self.bamibert_vectors),
                         ("xlmr", self.xlmr_vectors))]

    def _write(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def _run(self, name, probabilities):
        folder = self.root / name
        folder.mkdir()
        (folder / "classifier").mkdir()
        self._write(folder / "train.json", self.records)
        self._write(folder / "validation.json", self.records)
        # Deliberately different model configs: heterogeneous data matching must
        # not accidentally become a matched-architecture/config restriction.
        self._write(folder / "run_manifest.json", {"classifier_path": "classifier",
                                                   "config": {"model": name}})
        rows = [{"id": record["id"], "source": record["source"],
                 "target_class": record["label"], "predicted_class": max(range(5), key=p.__getitem__),
                 "probabilities": p} for record, p in zip(self.records, probabilities)]
        self._write(folder / "predictions.json", list(reversed(rows)))
        return folder

    def test_saved_artifacts_produce_exact_three_combinations_and_real_metrics(self):
        output = self.root / "results"
        summary = evaluate(*self.folders, output)
        self.assertEqual(set(summary["ensembles"]), {"qwen_bamibert", "qwen_xlmr", "all_three"})
        self.assertEqual(summary["best_individual"], "bamibert")
        self.assertEqual(summary["individuals"]["bamibert"]["classification"]["macro_f1"], 1.0)
        self.assertEqual(summary, json.loads((output / "summary.json").read_text()))
        self.assertEqual(summary["matched_data"], {"ordered_canonical_train_equal": True,
                                                    "ordered_canonical_validation_equal": True,
                                                    "train_count": 5, "validation_count": 5})
        for name, member_indexes in {"qwen_bamibert": [0, 1], "qwen_xlmr": [0, 2],
                                     "all_three": [0, 1, 2]}.items():
            folder = output / name
            config = _load_config(folder)
            self.assertEqual(len(config["members"]), len(member_indexes))
            self.assertTrue(all(Path(member["checkpoint"]).is_absolute() for member in config["members"]))
            self.assertTrue(all(member["weight"] == 1 / len(member_indexes) for member in config["members"]))
            exported = json.loads((folder / "predictions.json").read_text())
            self.assertEqual([row["id"] for row in exported], [row["id"] for row in self.records])
            vectors = [self.qwen_vectors, self.bamibert_vectors, self.xlmr_vectors]
            expected = average_predictions(self.records, [
                [prediction(row["id"], p) for row, p in zip(self.records, vectors[index])]
                for index in member_indexes])
            self.assertEqual([row["probabilities"] for row in exported],
                             [row["probabilities"] for row in expected])
            metrics = json.loads((folder / "metrics.json").read_text())
            self.assertEqual(set(metrics["classification"]["per_class"]), {"0", "1", "2", "3", "4"})
            self.assertEqual(metrics["classification"], summary["ensembles"][name]["classification"])
        comparison = json.loads((output / "all_three" / "comparison.json").read_text())
        self.assertEqual(comparison["vs_qwen"]["fix_ids"], [1])
        self.assertEqual(comparison["vs_qwen"]["regression_ids"], [])
        self.assertEqual(comparison["vs_qwen"]["changed_ids"], [1])
        self.assertEqual(comparison["vs_best_individual"]["baseline"], "bamibert")
        self.assertEqual(comparison["vs_best_individual"]["changed_ids"], [])
        self.assertGreater(comparison["probability_disagreement"]["qwen__bamibert"]["mean_total_variation"], 0)
        # A regression is distinct from a wrong-to-wrong change or a fix.
        pair = json.loads((output / "qwen_xlmr" / "comparison.json").read_text())
        self.assertEqual(pair["vs_qwen"]["fix_ids"], [1])
        self.assertEqual(pair["vs_qwen"]["regression_ids"], [2])
        self.assertEqual(pair["vs_qwen"]["changed_ids"], [1, 2])

    def test_ordered_canonical_train_and_validation_must_match(self):
        for split in ("train", "validation"):
            path = self.folders[1] / f"{split}.json"
            for changed in (list(reversed(self.records)),
                            [{**self.records[0], "id": "different"}, *self.records[1:]],
                            [{**self.records[0], "source": "changed"}, *self.records[1:]],
                            [{**self.records[0], "target": "changed"}, *self.records[1:]]):
                self._write(path, changed)
                with self.subTest(split=split, changed=changed), self.assertRaises(ValueError):
                    evaluate(*self.folders, self.root / "invalid")
            self._write(path, self.records)

    def test_saved_prediction_mismatches_fail(self):
        path = self.folders[1] / "predictions.json"
        original = json.loads(path.read_text())
        mutations = [original[:-1], original + [original[0]],
                     [{**original[0], "source": "wrong"}, *original[1:]],
                     [{**original[0], "target_class": 1}, *original[1:]],
                     [{**original[0], "predicted_class": 0}, *original[1:]],
                     [{**original[0], "probabilities": [0, 0, 0, 0, 0]}, *original[1:]]]
        for rows in mutations:
            self._write(path, rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluate(*self.folders, self.root / "invalid")

    def test_serving_rejects_weighted_nested_and_relative_configs_before_loading(self):
        folder = self.root / "ensemble"
        folder.mkdir()
        valid = {"labels": list(range(5)), "aggregation": "equal_probability_mean", "members": [
            {"kind": "qwen", "checkpoint": str(self.folders[0] / "classifier"), "weight": 0.5},
            {"kind": "encoder", "checkpoint": str(self.folders[1] / "classifier"), "weight": 0.5}]}
        self._write(folder / "ensemble_config.json", valid)
        self.assertEqual(_load_config(folder), valid)
        invalid = []
        for field, value in (("kind", "ensemble"), ("checkpoint", "relative/path"), ("weight", 0.7),
                             ("weight", float("nan")), ("weight", True)):
            config = copy.deepcopy(valid)
            config["members"][0][field] = value
            invalid.append(config)
        invalid.extend([{**valid, "labels": [False, 1, 2, 3, 4]},
                        {**valid, "labels": [4, 3, 2, 1, 0]},
                        {**valid, "aggregation": "vote"},
                        {**valid, "members": [valid["members"][0]]},
                        {**valid, "members": [valid["members"][0], valid["members"][0]]}])
        for config in invalid:
            self._write(folder / "ensemble_config.json", config)
            with self.subTest(config=config), self.assertRaises(ValueError):
                classify(folder, [{"id": "unseen", "source": "not in validation"}], device="cpu")


if __name__ == "__main__":
    unittest.main()
