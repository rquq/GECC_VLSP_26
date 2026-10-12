"""Consumer-visible stage modes, saved population checks, and portable handoffs."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

from model.classification import _architecture, _preprocessing
from model.data import LABELS
from model.run_reuse import (copy_classifier_run, resolve_adapter, resolve_run,
                             validate_classifier_run, validate_reuse_data, write_handoff)
from model.runtime import write_json
from model.two_phase import main, preflight

PROJECT = Path(__file__).resolve().parents[2]


def classifier_fixture(root, *, hints=True):
    checkpoint = root / "classifier"
    checkpoint.mkdir(parents=True)
    config = json.loads((PROJECT / "configs/baseline.json").read_text())
    config["run"].update(device="cuda", world_size=2)
    config["model"].update(name="local/base", revision="pinned")
    lora = {**config["lora"], "task_type": "SEQ_CLS", "revision": "pinned",
            "base_model_name_or_path": "local/base"}
    saved = {"model": config["model"], "num_labels": 5, "labels": list(LABELS), "architecture": _architecture()}
    for name, value in {"adapter_config.json": lora, "classifier_config.json": saved,
                        "preprocessing.json": _preprocessing(config), "tokenizer_config.json": {}}.items():
        write_json(checkpoint / name, value)
    tensors = {"base.lora_A.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
               "base.score.weight": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}
    header = json.dumps(tensors).encode()
    (checkpoint / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))
    artifact = {"seed": 42, "validation_fraction": .2, "correction_data_sha256": "population", "folds": []}
    manifest = {"task": "classification", "seed": 42, "validation_fraction": .2,
                "saved_classifier_reloaded_for_evaluation": True, "classifier_data_sha256": "mixed",
                "hints_enabled": hints, "train_count": 2, "correction_data_sha256": "population", "oof_folds": []}
    for name, value in {"config.json": config, "run_manifest.json": manifest, "train.json": [],
                        "validation.json": [], "metrics.json": {}}.items():
        write_json(root / name, value)
    if hints:
        write_json(root / "hints.json", artifact)
    return config


class PortableRunTests(unittest.TestCase):
    def test_legacy_direct_manifest_and_kaggle_mount(self):
        with tempfile.TemporaryDirectory() as folder:
            mount = Path(folder)
            root = mount / "private-version" / "saved-output"
            classifier_fixture(root / "classification")
            stage = root / "classification"
            self.assertEqual(resolve_run(root), stage)
            self.assertEqual(resolve_run(stage), stage)
            self.assertEqual(resolve_run(mount), stage)
            write_handoff(root)
            self.assertEqual(resolve_run(mount), stage)
            self.assertEqual(resolve_run(root / "run_handoff.json"), stage)
            with self.assertRaisesRegex(ValueError, "no completed correction"):
                resolve_run(root, "correction")

    def test_ambiguity_and_escaping_paths_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("one", "two"):
                classifier_fixture(root / name / "classification")
                write_handoff(root / name)
            with self.assertRaisesRegex(ValueError, "Ambiguous prior run"):
                resolve_run(root)
            write_json(root / "one/run_handoff.json", {
                "format": "vgec-run-handoff-v1", "completed_stages": ["classification"],
                "stages": {"classification": "../two/classification"}})
            with self.assertRaisesRegex(ValueError, "escapes"):
                resolve_run(root / "one")

    def test_second_export_survives_removing_old_mount(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "old-mount/classification"
            classifier_fixture(original)
            second = root / "second"
            copy_classifier_run(original, second / "classification")
            write_handoff(second)
            third = root / "third"
            copy_classifier_run(resolve_run(second), third / "classification")
            write_handoff(third)
            shutil.rmtree(root / "old-mount")
            shutil.rmtree(second)
            stage = resolve_run(third)
            validate_classifier_run(stage)
            self.assertTrue((stage / "hints.json").is_file())
            self.assertEqual(json.loads((stage / "run_manifest.json").read_text())["hints_path"], "hints.json")

    def test_saved_score_head_and_oof_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            stage = Path(folder) / "classification"
            classifier_fixture(stage, hints=False)
            validate_classifier_run(stage, require_hints=False)
            with self.assertRaisesRegex(FileNotFoundError, "oof.enabled=true"):
                validate_classifier_run(stage)
            (stage / "classifier/adapter_model.safetensors").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "score head"):
                validate_classifier_run(stage, require_hints=False)

    def test_peft_automatic_heads_preserve_required_module_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            stage = Path(folder) / "classification"
            classifier_fixture(stage)
            path = stage / "classifier/adapter_config.json"
            adapter = json.loads(path.read_text())
            adapter["modules_to_save"] = ["score", "classifier", "score"]
            write_json(path, adapter)
            validate_classifier_run(stage)
            for invalid in (["classifier"], ["score", "unexpected_layer"]):
                adapter["modules_to_save"] = invalid
                write_json(path, adapter)
                with self.assertRaises(ValueError):
                    validate_classifier_run(stage)

    def test_resolve_correction_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            adapter = root / "correction/adapter"
            adapter.mkdir(parents=True)
            write_json(adapter / "adapter_config.json", {"task_type": "CAUSAL_LM"})
            write_json(root / "correction/run_manifest.json", {})
            write_handoff(root, correction=True)
            self.assertEqual(resolve_adapter(adapter), adapter)
            self.assertEqual(resolve_adapter(root), adapter)


class ModeAndPopulationTests(unittest.TestCase):
    def test_task2_requires_completed_classifier_before_data_resolution(self):
        with patch("sys.argv", ["two_phase", "--mode", "task2"]), patch("model.two_phase.resolve_data") as data:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()
            data.assert_not_called()

    def test_task1_reuse_needs_no_data_or_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            stage = Path(folder) / "classification"
            config = classifier_fixture(stage, hints=False)
            with patch("model.two_phase._stage_placement") as placement, contextlib.redirect_stdout(io.StringIO()):
                report = preflight(config, mode="task1", classifier_run=stage)
            placement.assert_not_called()
            self.assertTrue(report["classifier_reused"])

    def test_preflight_only_never_mutates_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            classifier_fixture(root / "prior/classification", hints=False)
            write_handoff(root / "prior")
            output = root / "new-output"
            with patch("sys.argv", ["two_phase", "--mode", "task1", "--prior-run", str(root / "prior"),
                                     "--preflight-only", "--output-dir", str(output)]), \
                    patch("model.two_phase.subprocess.run") as launch, contextlib.redirect_stdout(io.StringIO()):
                main()
            launch.assert_not_called()
            self.assertFalse(output.exists())

    def test_population_mismatch_is_actionable_and_never_regenerates_hints(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stage = root / "classification"
            classifier_fixture(stage)
            data = root / "changed.json"
            data.write_text("changed population")
            correction = {"run": {"seed": 42}, "data": {"validation_fraction": .2}}
            with patch("model.data.load_training_split", return_value=([], [], {})), \
                    self.assertRaisesRegex(ValueError, "fresh Task1.*cannot safely regenerate"):
                validate_reuse_data(stage, correction, data)


if __name__ == "__main__":
    unittest.main()
