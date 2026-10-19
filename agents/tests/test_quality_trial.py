"""Frozen private predictions are input-bound inference, never fitting labels."""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from model.prediction_export import read_submission_archive, write_submission, SubmissionExportError
from tools.run_quality_trial import export_frozen_corrector, validate


class ClassifierTrialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = [{"id": 1, "source": "unseen alpha", "label": None, "target": None},
                        {"id": "1", "source": "unseen beta", "label": None, "target": None}]
        self.rows = [{"ID": 1, "Output": {"Error_Type_Id": 0, "Fixed Sentence": ["kept alpha"]}},
                     {"ID": "1", "Output": {"Error_Type_Id": 3, "Fixed Sentence": ["fixed beta"]}}]
        self.control = write_submission(self.root / "results.json", self.rows)

    def test_private_ids_remain_typed_and_ordered(self):
        values = read_submission_archive(self.control, self.records)
        self.assertEqual([type(r["id"]) for r in values], [int, str])
        with self.assertRaisesRegex(SubmissionExportError, "order"):
            read_submission_archive(self.control, self.records[::-1])

    def test_extra_members_or_missing_ids_do_not_become_a_control(self):
        with zipfile.ZipFile(self.control, "a") as z:
            z.writestr("metadata", "unexpected")
        with self.assertRaisesRegex(SubmissionExportError, "only"):
            read_submission_archive(self.control, self.records)
        write_submission(self.root / "results.json", self.rows[:1])
        with self.assertRaisesRegex(ValueError, "IDs"):
            read_submission_archive(self.control, self.records)

    def test_challenger_changes_labels_and_preserves_every_correction(self):
        original = self.control.read_bytes()
        output = self.root / "challenger"
        output.mkdir()
        classes = [{"id": "1", "label": 2}, {"id": 1, "label": 4}]
        export_frozen_corrector(self.records, classes, self.control, output)
        rows = json.loads((output / "results.json").read_text())
        self.assertEqual([r["Output"]["Error_Type_Id"] for r in rows], [4, 2])
        self.assertEqual([r["Output"]["Fixed Sentence"] for r in rows],
                         [r["Output"]["Fixed Sentence"] for r in self.rows])
        self.assertEqual(self.control.read_bytes(), original)
        with zipfile.ZipFile(output / "results.zip") as z:
            self.assertIsNone(z.testzip())

    def test_unknown_or_duplicate_classifier_ids_fail(self):
        output = self.root / "rejected"
        output.mkdir()
        for values in [[{"id": 2, "label": 0}, {"id": "1", "label": 0}],
                       [{"id": 1, "label": 0}, {"id": 1, "label": 0}]]:
            with self.assertRaisesRegex(ValueError, "IDs"):
                export_frozen_corrector(self.records, values, self.control, output)
        self.assertFalse((output / "results.zip").exists())

    def test_committed_trial_has_two_five_epoch_fits_and_quality_hashes(self):
        project = Path(__file__).resolve().parents[2]
        plan = json.loads((project / "configs/quality_trial_plan.json").read_text())
        recipe = validate(plan)
        quality = json.loads((project / "configs/quality_experiment.json").read_text())
        self.assertEqual(recipe["mode"], "both")
        self.assertEqual(recipe["data_sha256"], quality["data_sha256"])
        self.assertEqual(recipe["submission_sha256"], plan["inference_sha256"])

    def test_fewer_than_five_epochs_or_classifier_hints_are_rejected(self):
        project = Path(__file__).resolve().parents[2]
        plan = json.loads((project / "configs/quality_trial_plan.json").read_text())
        recipe = json.loads((project / plan["training_recipe"]).read_text())
        config = json.loads((project / recipe["correction_config"]).read_text())
        recipe["correction_config"] = str(self.root / "correction.json")
        plan["training_recipe"] = str(self.root / "recipe.json")
        (self.root / "recipe.json").write_text(json.dumps(recipe))
        for field, bad in [("epochs", 4), ("hints", "classifier")]:
            value = json.loads(json.dumps(config))
            if field == "epochs":
                value["training"]["args"]["num_train_epochs"] = bad
            else:
                value["inference"]["hint_mode"] = bad
            (self.root / "correction.json").write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                validate(plan)


if __name__ == "__main__":
    unittest.main()
