import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


class FocusedRunnerTests(unittest.TestCase):
    def test_worker_accepts_saved_normalized_validation_and_raw_private_input(self):
        from tools.focused_worker import read_records

        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "rows.json"
            normalized = [
                {"id": 1, "source": "fixture", "label": 0, "target": "fixture"}
            ]
            p.write_text(json.dumps(normalized))
            self.assertEqual(read_records(p), normalized)
            p.write_text(json.dumps([{"ID": "1", "Input": "private fixture"}]))
            self.assertEqual(
                read_records(p)[0],
                {"id": "1", "source": "private fixture", "label": None, "target": None},
            )

    def test_root_accepts_numeric_decoder_progress_without_corpus_text(self):
        import time
        from unittest.mock import MagicMock
        from model.tracking import TrackingRun

        tracking = TrackingRun()
        tracking.run = MagicMock()
        tracking.started_at = time.monotonic()
        tracking.progress(
            {
                "stage": "baseline-decoders",
                "prefix": "decoding",
                "metrics": {
                    "completed_rows": 3,
                    "requested_rows": 78,
                    "source": "PRIVATE_SENTENCE",
                },
            }
        )
        tracking.run.log.assert_called_once()
        values = tracking.run.log.call_args.args[0]
        self.assertEqual(values["baseline-decoders/decoding/completed_rows"], 3)
        self.assertNotIn("PRIVATE_SENTENCE", str(values))

    def package(self, root):
        files = {"baseline/weights": b"fixture weights", "private_test.json": b"[]"}
        m = {
            "format": "vgec-focused-package-v1",
            "file_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
        }
        raw = json.dumps(m).encode()
        archive = root / "package.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("nested/focused_package_manifest.json", raw)
            for k, v in files.items():
                z.writestr("nested/" + k, v)
        return hashlib.sha256(raw).hexdigest()

    def test_nested_archive_is_resolved_and_all_weights_are_verified(self):
        from tools.run_focused import materialize_package

        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            expected = self.package(root)
            package, _ = materialize_package(
                root, root / "out", "package.zip", expected
            )
            self.assertEqual(
                (package / "baseline/weights").read_bytes(), b"fixture weights"
            )
            (package / "baseline/weights").write_bytes(b"changed")
            (root / "package.zip").unlink()
            with self.assertRaisesRegex(ValueError, "hash"):
                materialize_package(root, root / "other", "package.zip", expected)

    def test_duplicate_or_changed_manifest_is_rejected(self):
        from tools.run_focused import materialize_package

        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            expected = self.package(root)
            with self.assertRaisesRegex(ValueError, "manifest"):
                materialize_package(root, root / "out", "package.zip", "0" * 64)
            with zipfile.ZipFile(root / "package.zip", "a") as z:
                z.writestr("other/focused_package_manifest.json", "{}")
            with self.assertRaisesRegex(ValueError, "one"):
                materialize_package(root, root / "other", "package.zip", expected)

    def test_zip_escape_is_rejected_without_writing_outside_destination(self):
        from tools.run_focused import materialize_package

        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            m = {
                "format": "vgec-focused-package-v1",
                "file_sha256": {"../escape": "0" * 64},
            }
            raw = json.dumps(m).encode()
            with zipfile.ZipFile(root / "bad.zip", "w") as z:
                z.writestr("focused_package_manifest.json", raw)
            with self.assertRaisesRegex(ValueError, "path"):
                materialize_package(
                    root, root / "out", "bad.zip", hashlib.sha256(raw).hexdigest()
                )
            self.assertFalse((root / "escape").exists())

    def test_decisions_keep_fixed_binary_probe_separate_and_align_typed_ids(self):
        from tools.run_focused import decision_diagnostics

        gold = [
            {"id": 1, "source": "fixture", "label": 1, "target": "fixed"},
            {"id": "1", "source": "other", "label": 0, "target": "other"},
        ]
        rows = [
            {
                "id": "1",
                "predicted_class": 0,
                "probabilities": [0.9, 0.025, 0.025, 0.025, 0.025],
            },
            {"id": 1, "predicted_class": 0, "probabilities": [0.4, 0.3, 0.1, 0.1, 0.1]},
        ]
        d = decision_diagnostics(gold, rows)
        self.assertEqual(d["argmax"]["micro_f1"], 0.5)
        self.assertEqual(d["binary_first_05"]["micro_f1"], 1)
        self.assertEqual(d["export_rule"], "argmax")
        with self.assertRaisesRegex(ValueError, "IDs"):
            decision_diagnostics(gold, [rows[0], rows[0]])

    def test_failed_decoder_never_wins_and_native_greedy_is_mandatory(self):
        from tools.run_focused import select_corrector

        good = {
            "status": "completed",
            "strategy": {"name": "greedy_reference"},
            "metrics": {
                "corrections": {
                    "all": {"gleu_plus": 0.5},
                    "error_only": {"gleu_plus": 0.1},
                },
                "clean_preservation": {"exact_source_preservation": 1},
            },
        }
        failed = {"status": "failed", "strategy": {"name": "beam4"}}
        self.assertEqual(
            select_corrector({"baseline": {"full": [good, failed]}})[0], "baseline"
        )
        with self.assertRaisesRegex(ValueError, "greedy"):
            select_corrector({"baseline": {"full": [failed]}})

    def test_timeout_respects_whole_run_and_fit_stage_budget(self):
        from tools.run_focused import stage_timeout

        plan = {"max_wall_seconds": 100, "max_stage_seconds": 40}
        self.assertEqual(stage_timeout(plan, 90), 10)
        self.assertEqual(stage_timeout(plan, 10, "selected-private-corrector"), 90)
        with self.assertRaises(TimeoutError):
            stage_timeout(plan, 101)

    def test_selected_models_relocate_and_reject_changed_decoder_or_dependency(self):
        import shutil
        from tools.run_focused import resolve_selected
        from model.runtime import write_json

        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "original"
            models = root / "selected_models"
            models.mkdir(parents=True)
            files = {
                "classification/classifier/classifier_config.json": "{}",
                "correction/adapter/inference.json": "{}",
            }
            for name, value in files.items():
                p = models / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(value)
            strategy = {"name": "beam2", "generation": {"num_beams": 2}}
            write_json(
                models / "file_manifest.json",
                {
                    "format": "vgec-selected-models-v1",
                    "strategy": strategy,
                    "file_sha256": {
                        k: hashlib.sha256(v.encode()).hexdigest()
                        for k, v in files.items()
                    },
                },
            )
            selected = {
                "decision_rule": "argmax",
                "classifier_path": "selected_models/classification/classifier",
                "adapter": "selected_models/correction/adapter",
                "strategy": strategy,
                "file_manifest": "selected_models/file_manifest.json",
                "file_manifest_sha256": hashlib.sha256(
                    (models / "file_manifest.json").read_bytes()
                ).hexdigest(),
            }
            write_json(root / "selected.json", selected)
            moved = Path(t) / "moved"
            shutil.move(root, moved)
            _, paths = resolve_selected(moved / "selected.json")
            self.assertTrue(paths["adapter"].is_relative_to(moved))
            selected["strategy"] = {"name": "greedy_reference", "generation": {}}
            write_json(moved / "selected.json", selected)
            with self.assertRaisesRegex(ValueError, "strategy"):
                resolve_selected(moved / "selected.json")
            selected["strategy"] = strategy
            write_json(moved / "selected.json", selected)
            (moved / "selected_models/correction/adapter/inference.json").write_text(
                "changed"
            )
            with self.assertRaisesRegex(ValueError, "dependency hash"):
                resolve_selected(moved / "selected.json")

    def test_export_rejects_duplicate_class_ids_and_keeps_typed_id_order(self):
        from tools.run_focused import merge_export

        records = [
            {"id": 1, "source": "fixture", "label": None, "target": None},
            {"id": "1", "source": "other", "label": None, "target": None},
        ]
        classes = [{"id": 1, "label": 0}, {"id": "1", "label": 2}]
        corrections = [
            {"id": "1", "target": "other fixed"},
            {"id": 1, "target": "fixture"},
        ]
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            merge_export(records, classes, corrections, root)
            rows = json.loads((root / "results.json").read_text())
            self.assertEqual([r["ID"] for r in rows], [1, "1"])
            with self.assertRaisesRegex(ValueError, "IDs"):
                merge_export(
                    records, [classes[0], classes[0]], corrections, root / "invalid"
                )
