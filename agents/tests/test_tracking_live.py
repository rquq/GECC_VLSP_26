"""A root run remains live through child progress; metrics must stay corpus-safe."""

import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from model.tracking import TrackingRun


class LiveTrackingTests(unittest.TestCase):
    def config(self):
        return {
            "run": {"seed": 42},
            "tracking": {"enabled": True, "mode": "offline", "log_models": False},
        }

    def test_root_literal_name_progress_heartbeat_and_finish_scope(self):
        sdk = MagicMock()
        run = sdk.init.return_value
        run.id = "rootid"
        run.url = None
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            "sys.modules", {"wandb": sdk}
        ):
            with TrackingRun() as tracking:
                tracking.start(
                    self.config(),
                    Path(folder) / "run",
                    "pipeline",
                    group="unique-experiment",
                )
                self.assertEqual(sdk.init.call_args.kwargs["name"], "run")
                self.assertTrue(
                    all(
                        "*" not in call.args[0][:-1]
                        for call in run.define_metric.call_args_list
                    )
                )
                tracking.progress(
                    {
                        "stage": "classifier-sailor7b",
                        "prefix": "train",
                        "metrics": {
                            "loss": 0.5,
                            "epoch": 1,
                            "source": "private",
                            "password": "secret",
                        },
                        "optimizer_step": 10,
                    }
                )
                tracking.heartbeat(stage="classifier-sailor7b")
                run.finish.assert_not_called()
                payloads = [call.args[0] for call in run.log.call_args_list]
                self.assertTrue(
                    any(
                        p.get("classifier-sailor7b/train/loss") == 0.5 for p in payloads
                    )
                )
                self.assertTrue(any("pipeline/elapsed_seconds" in p for p in payloads))
                self.assertNotIn("private", str(payloads))
                self.assertNotIn("secret", str(payloads))
                status = json.loads(
                    (Path(folder) / "run/pipeline_progress.json").read_text()
                )
                self.assertEqual(status["current_stage"], "classifier-sailor7b")
            run.finish.assert_called_once_with(exit_code=0)

    def test_progress_is_emitted_even_when_stage_wandb_is_disabled(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ,
            {"VGEC_PIPELINE_TRACKING": "1", "VGEC_STAGE_NAME": "classifier-mistral7b"},
        ):
            with TrackingRun() as tracking, contextlib.redirect_stdout(
                io.StringIO()
            ) as output:
                tracking.start(
                    {"tracking": {"enabled": False}}, folder, "classification"
                )
                tracking.log({"loss": 0.4, "epoch": 2}, optimizer_step=5)
            from model.tracking import PROGRESS_EVENT

            events = [
                json.loads(line[len(PROGRESS_EVENT) :])
                for line in output.getvalue().splitlines()
                if line.startswith(PROGRESS_EVENT)
            ]
            self.assertEqual(events[0]["stage"], "classifier-mistral7b")
            self.assertEqual(events[0]["metrics"]["loss"], 0.4)

    def test_exception_finishes_root_as_failed_without_hiding_original(self):
        sdk = MagicMock()
        sdk.init.return_value.id = "rootid"
        sdk.init.return_value.url = None
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            "sys.modules", {"wandb": sdk}
        ):
            with self.assertRaisesRegex(RuntimeError, "original"):
                with TrackingRun() as tracking:
                    tracking.start(
                        self.config(), folder, "pipeline", group="unique-experiment"
                    )
                    tracking.heartbeat(stage="corrector")
                    raise RuntimeError("original")
            sdk.init.return_value.finish.assert_called_once_with(exit_code=1)
            identity = json.loads((Path(folder) / "wandb_run.json").read_text())
            self.assertEqual(identity["status"], "failed")


if __name__ == "__main__":
    unittest.main()
