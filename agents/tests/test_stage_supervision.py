"""Real short-lived subprocesses exercise quiet heartbeats and interruption cleanup."""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from model.two_phase import _run_stage


class StageSupervisionTests(unittest.TestCase):
    def test_quiet_child_keeps_heartbeat_and_forwards_safe_progress(self):
        from model.tracking import PROGRESS_EVENT

        tracking = MagicMock()
        script = (
            "import os,time;print("
            + repr(
                "progress bar: "
                + PROGRESS_EVENT
                + '{"stage":"classifier-seed-43","prefix":"train","metrics":{"loss":0.4}}'
            )
            + ",flush=True);assert os.environ['PYTHONHASHSEED']=='43';time.sleep(.2)"
        )
        with patch(
            "model.two_phase.HEARTBEAT_SECONDS", 0.04
        ), contextlib.redirect_stdout(io.StringIO()):
            _run_stage(
                [sys.executable, "-u", "-c", script],
                {
                    **os.environ,
                    "VGEC_MODEL_SEED": "43",
                    "VGEC_STAGE_NAME": "classifier-seed-43",
                },
                tracking,
            )
        self.assertGreaterEqual(tracking.heartbeat.call_count, 3)
        tracking.progress.assert_called_once()
        self.assertEqual(tracking.progress.call_args.args[0]["metrics"]["loss"], 0.4)

    def test_nonzero_exit_propagates_and_interrupt_reaps_child_group(self):
        tracking = MagicMock()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(
            subprocess.CalledProcessError
        ):
            _run_stage(
                [sys.executable, "-c", "raise SystemExit(3)"],
                os.environ.copy(),
                tracking,
            )
        from model.tracking import PROGRESS_EVENT

        tracking.progress.side_effect = KeyboardInterrupt
        script = (
            "import time;print("
            + repr(
                PROGRESS_EVENT
                + '{"stage":"corrector","prefix":"train","metrics":{"loss":0.4}}'
            )
            + ",flush=True);time.sleep(30)"
        )
        original = os.killpg
        with patch(
            "model.two_phase.os.killpg", wraps=original
        ) as kill, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                _run_stage(
                    [sys.executable, "-u", "-c", script], os.environ.copy(), tracking
                )
            kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
