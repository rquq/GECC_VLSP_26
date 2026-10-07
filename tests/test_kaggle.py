"""Exercise the notebook bootstrap with real, local Git repositories."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from tools import kaggle_run

from tools.kaggle_run import DATA_FILENAME, discover_data

PROJECT = Path(__file__).resolve().parents[1]


class NotebookSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.origin = self.folder / "origin"
        self.clone = self.folder / "clone"
        self.origin.mkdir()
        self.git(self.origin, "init", "-b", "main")
        self.git(self.origin, "config", "user.name", "Notebook test")
        self.git(self.origin, "config", "user.email", "notebook@example.invalid")
        self.write_project("main contents")
        self.git(self.origin, "add", ".")
        self.git(self.origin, "commit", "-m", "Main fixture")
        self.git(self.origin, "switch", "-c", "test/baseline")
        self.write_project("baseline contents")
        self.git(self.origin, "commit", "-am", "Baseline fixture")
        notebook = json.loads((PROJECT / "notebooks/kaggle_smoke.ipynb").read_text())
        self.source = "".join(notebook["cells"][2]["source"])
        self.namespace = {"Path": Path, "subprocess": subprocess, "json": json,
                          "sys": __import__("sys"), "urllib": __import__("urllib.request"),
                          "REPO_URL": self.origin.as_uri(), "BRANCH": "test/baseline",
                          "PROJECT_SUBDIR": ".", "CLONE_DIR": self.clone}

    def git(self, directory, *args):
        return subprocess.check_output(["git", "-C", str(directory), *args],
                                       stderr=subprocess.PIPE, text=True).strip()

    def write_project(self, marker):
        (self.origin / "README.md").write_text(marker)
        for name in ["requirements.txt", "vgec/train.py", "tools/kaggle_run.py",
                     "configs/smoke.json", "configs/baseline.json"]:
            path = self.origin / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("" if path.suffix != ".json" else "{}")

    def run_cell(self):
        # The original cell mixes dependency installation with cloning. Keep
        # Git real while avoiding network checks/pip in this focused test.
        real_run = subprocess.run
        def run(command, *args, **kwargs):
            if "pip" in command:
                return subprocess.CompletedProcess(command, 0)
            return real_run(command, *args, **kwargs)
        with patch("urllib.request.urlopen") as internet, patch("subprocess.run", side_effect=run):
            internet.return_value.__enter__.return_value.status = 200
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(self.source, "kaggle-source-cell", "exec"), self.namespace)

    def test_clones_requested_branch_at_repository_root(self):
        self.run_cell()
        self.assertEqual((self.namespace["PROJECT_DIR"] / "README.md").read_text(), "baseline contents")
        self.assertEqual(self.namespace.get("SOURCE_INFO", {}).get("commit"),
                         self.git(self.origin, "rev-parse", "HEAD"))

    def test_rerun_reuses_clone_and_fetches_branch_updates(self):
        self.run_cell()
        self.write_project("updated baseline")
        self.git(self.origin, "commit", "-am", "Update baseline")
        self.run_cell()
        self.assertEqual((self.clone / "README.md").read_text(), "updated baseline")
        self.assertEqual(self.namespace["SOURCE_INFO"]["commit"], self.git(self.origin, "rev-parse", "HEAD"))

    def test_dirty_clone_is_preserved_and_rejected(self):
        self.run_cell()
        (self.clone / "README.md").write_text("local changes")
        with self.assertRaisesRegex(RuntimeError, "local changes"):
            self.run_cell()
        self.assertEqual((self.clone / "README.md").read_text(), "local changes")

    def test_another_repository_is_rejected_without_changing_clone(self):
        self.run_cell()
        self.namespace["REPO_URL"] = str(self.folder / "another-repository")
        with self.assertRaisesRegex(RuntimeError, "different repository"):
            self.run_cell()
        self.assertEqual(self.git(self.clone, "remote", "get-url", "origin"), self.origin.as_uri())

    def test_missing_branch_has_actionable_error(self):
        self.namespace["BRANCH"] = "not-published"
        with self.assertRaisesRegex(RuntimeError, "[Bb]ranch.*does not exist"):
            self.run_cell()
        self.assertFalse(self.clone.exists())


    def test_run_provenance_records_real_git_identity(self):
        self.run_cell()
        source = kaggle_run.source_identity(self.clone)
        self.assertEqual(source["repo_url"], self.origin.as_uri())
        self.assertEqual(source["branch"], "test/baseline")
        self.assertEqual(source["commit"], self.git(self.origin, "rev-parse", "HEAD"))
        self.assertFalse(source["dirty"])
        (self.clone / "README.md").write_text("changed locally")
        self.assertTrue(kaggle_run.source_identity(self.clone)["dirty"])


class KaggleDataTests(unittest.TestCase):
    def test_mounted_input_wins_over_repository_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / "repo"
            mounted = root / "input" / "private-data" / DATA_FILENAME
            fallback = repo / "dataset" / DATA_FILENAME
            for path in [mounted, fallback]:
                path.parent.mkdir(parents=True)
                path.write_text("[]")
            self.assertEqual(discover_data(None, repo, root / "input"), mounted.resolve())

    def test_ambiguous_inputs_require_explicit_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files = [root / name / DATA_FILENAME for name in ["one", "two"]]
            for path in files:
                path.parent.mkdir()
                path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "Multiple Kaggle datasets"):
                discover_data(None, root / "repo", root)
            self.assertEqual(discover_data(files[1], root / "repo", root), files[1].resolve())


class KagglePreflightTests(unittest.TestCase):
    def test_preflight_validates_data_without_training_or_creating_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "training data.json"
            data.write_text(json.dumps([
                {"ID": label * 10 + i, "Input": f"source {label} {i}",
                 "Output": {"Error_Type_Id": label, "Fixed Sentence": f"target {label} {i}"}}
                for label in range(5) for i in range(5)]))
            output = root / "new-run"
            hardware = SimpleNamespace(cuda=SimpleNamespace(
                is_available=lambda: True, device_count=lambda: 2,
                get_device_name=lambda index: "Tesla T4"))
            real_run = subprocess.run
            def run(command, *args, **kwargs):
                if "torch.distributed.run" in command:
                    self.fail("Preflight started training")
                return real_run(command, *args, **kwargs)
            with patch("sys.argv", ["kaggle_run.py", "--preflight-only", "--data", str(data),
                                    "--output-dir", str(output)]), \
                 patch.dict("sys.modules", {"torch": hardware}), \
                 patch("importlib.metadata.version", return_value="test-version"), \
                 patch("subprocess.run", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                kaggle_run.main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
