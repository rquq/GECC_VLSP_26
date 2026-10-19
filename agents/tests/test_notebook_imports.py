"""Kaggle loads the launcher by filename outside the cloned checkout."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]


class NotebookImportTests(unittest.TestCase):
    def test_quality_trial_exposes_four_candidates_without_task1_paths(self):
        from tools import notebook
        import types
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "pipeline_manifest.json").write_text(json.dumps({"mode": "quality_trial"}))
            candidates = {name: {"path": name + "/results.zip", "scope": name}
                          for name in ["quality_control", "classifier_challenger", "corrector_challenger", "both_challengers"]}
            (output / "submission_candidates.json").write_text(json.dumps(candidates))
            class Finished:
                stdout = ["VGEC_RUN_DIR:" + str(output) + "\n"]
                args = []
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def wait(self): return 0
            shown = []
            ipython = types.ModuleType("IPython.display")
            ipython.FileLink = str
            ipython.display = shown.append
            with patch.object(notebook.subprocess, "Popen", return_value=Finished()), \
                 patch.dict(sys.modules, {"IPython.display": ipython}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(notebook.launch(PROJECT, sys.executable), output)
            self.assertEqual(shown[:4], [str(output / c["path"]) for c in candidates.values()])

    def test_file_loaded_launcher_works_outside_checkout_without_pythonpath(self):
        script = '''
import importlib.util, json, sys, types
from pathlib import Path
from unittest.mock import patch
project, output = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("gecc_notebook", project / "tools/notebook.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
assert str(project) not in sys.path
(output / "classification").mkdir(parents=True)
(output / "pipeline_manifest.json").write_text(json.dumps({"mode": "task1"}))
(output / "classification/metrics.json").write_text(json.dumps({"classification": None}))
class Finished:
    stdout = ["VGEC_RUN_DIR:" + str(output) + "\\n"]
    args = []
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def wait(self): return 0
ipython = types.ModuleType("IPython.display")
ipython.FileLink = str
ipython.display = lambda *args: None
with patch.dict(sys.modules, {"IPython.display": ipython}), patch.object(launcher.subprocess, "Popen", return_value=Finished()) as start:
    assert launcher.launch(project, sys.executable) == output
    assert start.call_args.kwargs["env"]["PYTHONHASHSEED"] == "42"
    assert start.call_args.kwargs["cwd"] == project
'''
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(
                [sys.executable, "-I", "-c", script, str(PROJECT), str(Path(folder) / "output")],
                cwd=folder, text=True, capture_output=True,
                env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_file_loaded_launcher_cancellation_imports_cleanup_helper(self):
        spec = importlib.util.spec_from_file_location("gecc_notebook", PROJECT / "tools/notebook.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)

        class Interrupted:
            pid = 12345
            args = []
            @property
            def stdout(self):
                raise KeyboardInterrupt
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def wait(self, timeout=None): return 0

        with patch.object(launcher.subprocess, "Popen", return_value=Interrupted()), \
             patch("tools.process_groups.descendant_groups", return_value={12346}), \
             patch("tools.process_groups.signal_groups") as signal_groups, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                launcher.launch(PROJECT, sys.executable)
        self.assertEqual(signal_groups.call_args_list[0].args[0], {12345, 12346})
        self.assertEqual(signal_groups.call_args_list[-1].args[0], {12346})


if __name__ == "__main__":
    unittest.main()
