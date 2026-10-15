"""GPU release gates use mocked device snapshots, never kill unrelated processes."""

from pathlib import Path
import unittest
from unittest.mock import patch


class GPUReleaseTests(unittest.TestCase):
    def test_delayed_release_is_waited_for_before_next_model(self):
        from model.gpu_lifecycle import wait_for_release

        before = {"0": {"free_bytes": 5000, "total_bytes": 9000}}
        with patch(
            "model.gpu_lifecycle.snapshot",
            side_effect=[{"0": {"free_bytes": 3000, "total_bytes": 9000}}, before],
        ), patch("model.gpu_lifecycle.time.sleep") as sleep:
            self.assertEqual(wait_for_release(before, tolerance_bytes=10), before)
            sleep.assert_called_once()

    def test_unreleased_or_missing_gpu_fails_closed(self):
        from model.gpu_lifecycle import wait_for_release

        before = {"0": {"free_bytes": 5000, "total_bytes": 9000}}
        with patch(
            "model.gpu_lifecycle.snapshot",
            return_value={"0": {"free_bytes": 3000, "total_bytes": 9000}},
        ):
            with self.assertRaisesRegex(RuntimeError, "not released"):
                wait_for_release(before, timeout=0, tolerance_bytes=10)
        with patch("model.gpu_lifecycle.snapshot", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "GPU set changed"):
                wait_for_release(before, timeout=0)

    def test_snapshot_parses_nvidia_mib_without_initializing_torch_cuda(self):
        from model.gpu_lifecycle import snapshot
        from types import SimpleNamespace

        with patch(
            "model.gpu_lifecycle.subprocess.run",
            return_value=SimpleNamespace(stdout="0, 15000, 15360\n1, 14000, 15360\n"),
        ):
            result = snapshot()
        self.assertEqual(result["0"]["free_bytes"], 15000 * 1024**2)
        self.assertEqual(result["1"]["total_bytes"], 15360 * 1024**2)


if __name__ == "__main__":
    unittest.main()
