"""Independent correction keeps real validation predictions and strict provenance."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model.run_reuse import validation_predictions, validate_reuse_data
from tools.propose_training_repairs import trailing_reference_candidate
from tools.experiment_log import record


class IndependentCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [{"id": 1, "source": "source one", "label": 2, "target": "target one"},
                     {"id": "1", "source": "source two", "label": 0, "target": "source two"}]
        self.predictions = [{"id": 1, "source": "source one", "predicted_class": 1,
                             "probabilities": [0, 1, 0, 0, 0]},
                            {"id": "1", "source": "source two", "predicted_class": 3,
                             "probabilities": [0, 0, 0, 1, 0]}]
        self.write("validation.json", self.rows)
        self.write("predictions.json", self.predictions)
        self.write("run_manifest.json", {})

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def test_scoring_uses_actual_predictions_even_when_they_disagree_with_gold(self):
        from model.data import _id_key
        values = validation_predictions(self.root, self.rows)
        self.assertEqual(values[_id_key(1)]["label"], 1)
        self.assertEqual(values[_id_key("1")]["label"], 3)

    def test_changed_source_and_nonfinite_probabilities_are_rejected(self):
        self.predictions[0]["source"] = "different source"
        self.write("predictions.json", self.predictions)
        with self.assertRaisesRegex(ValueError, "source differs"):
            validation_predictions(self.root, self.rows)
        self.predictions[0]["source"] = "source one"
        self.predictions[0]["probabilities"][0] = float("nan")
        self.write("predictions.json", self.predictions)
        with self.assertRaisesRegex(ValueError, "probabilities"):
            validation_predictions(self.root, self.rows)

    def test_no_hint_reuse_accepts_no_oof_artifact_but_checks_holdout(self):
        config = {"run": {"seed": 42}, "data": {"validation_fraction": .2}, "inference": {"hint_mode": "none"}}
        with patch("model.data.load_training_split", return_value=(self.rows, self.rows, {})):
            report = validate_reuse_data(self.root, config, self.root / "bundle.json")
        self.assertFalse(report["hints_enabled"])
        self.assertFalse((self.root / "hints.json").exists())
        with patch("model.data.load_training_split", return_value=(self.rows, [], {})), \
                self.assertRaisesRegex(ValueError, "validation split differs"):
            validate_reuse_data(self.root, config, self.root / "bundle.json")

    def test_runtime_record_is_copied_into_the_run_and_local_agent_ledger(self):
        with patch("tools.experiment_log.PROJECT_ROOT", self.root / "repo"):
            record(self.root / "run", "started", commit="abc", name="independent")
            record(self.root / "run", "failed", error_type="ValueError")
        local = (self.root / "repo/agents/experiments/local.jsonl").read_text()
        self.assertEqual(local, (self.root / "run/experiment_events.jsonl").read_text())
        self.assertEqual([json.loads(line)["status"] for line in local.splitlines()], ["started", "failed"])


class ConservativeRepairTests(unittest.TestCase):
    source = "This long complete sentence contains a very specific anchor and additional suffix."
    target = "This long complete sentence now contains a very specific anchor..."

    def test_completion_copies_source_suffix_and_only_reference_supplies_the_edit(self):
        result = trailing_reference_candidate(self.source, self.target)
        self.assertEqual(result["target"], "This long complete sentence now contains a very specific anchor and additional suffix.")
        self.assertFalse(result["evidence"]["new_wording_generated"])
        self.assertFalse(result["evidence"]["grammatical_adjudication"])

    def test_ambiguous_or_missing_context_remains_unrepaired(self):
        for source, target in ((self.source + " a very specific anchor", self.target),
                               (self.source + "...", self.target),
                               (self.source, "This...specific anchor..."),
                               (self.source, "This long complete sentence contains a very specific anchor...")):
            self.assertIsNone(trailing_reference_candidate(source, target))


if __name__ == "__main__":
    unittest.main()
