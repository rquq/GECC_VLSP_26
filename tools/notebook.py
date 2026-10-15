"""Shared notebook execution: credentials, subprocess logging, and result links."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def training_python(project):
    """Keep the environment path intact rather than resolve the Python symlink."""
    explicit = os.environ.get("GECC_TRAINING_PYTHON")
    if explicit:
        return str(Path(explicit).expanduser().absolute())
    roots = [
        root / name for root in (project, project.parent) for name in (".venv", "venv")
    ]
    roots += [
        Path(os.environ[name])
        for name in ("VIRTUAL_ENV", "CONDA_PREFIX")
        if os.environ.get(name)
    ]
    return str(
        next(
            (root / "bin/python" for root in roots if (root / "bin/python").is_file()),
            Path(sys.executable),
        )
    )


def kaggle_credentials():
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    for name in ("WANDB_API_KEY", "HF_TOKEN"):
        if not os.environ.get(name):
            try:
                value = secrets.get_secret(name)
            except Exception:
                continue
            if value:
                os.environ[name] = value


def setup_kaggle(project):
    """Install training pins outside the kernel while preserving CUDA PyTorch."""
    from importlib.metadata import version
    import tempfile

    installed_torch = version("torch")
    environment = Path(tempfile.mkdtemp(prefix="gecc-training-env-"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "venv",
            "--without-pip",
            "--system-site-packages",
            str(environment),
        ],
        check=True,
    )
    python = str(environment / "bin/python")
    constraints = environment / "constraints.txt"
    constraints.write_text(f"torch=={installed_torch}\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            python,
            "install",
            "-r",
            str(Path(project) / "requirements-kaggle.txt"),
            "--constraint",
            str(constraints),
        ],
        cwd=project,
        check=True,
    )
    subprocess.run(
        [
            python,
            "-c",
            "import numpy, scipy, sklearn, torch, transformers, peft, accelerate, bitsandbytes, lmformatenforcer, wandb; "
            "from transformers import AutoModelForCausalLM; from sklearn.metrics import roc_curve; "
            f"assert torch.__version__ == {installed_torch!r}; "
            "print('Training imports passed; torch', torch.__version__, 'CUDA', torch.version.cuda)",
        ],
        check=True,
    )
    kaggle_credentials()
    print("Training environment ready; no kernel restart needed.")
    return python


def redact(text):
    for name in ("WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(name):
            text = text.replace(os.environ[name], "[REDACTED]")
    return text


def launch(project, python):
    """Run the committed recipe; interrupts stop the entire trainer process group."""
    project = Path(project)
    run_dir = None
    log_path = None
    from model.reproducibility import process_environment

    with subprocess.Popen(
        [python, "-u", "-m", "tools.run_experiment"],
        cwd=project,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=process_environment(42),
    ) as process:
        try:
            for line in process.stdout:
                if line.startswith("VGEC_RUN_DIR:"):
                    run_dir = Path(line.partition(":")[2].strip())
                    log_path = run_dir.parent / "console.log"
                line = redact(line)
                print(line, end="", flush=True)
                if log_path:
                    with log_path.open("a", encoding="utf-8") as log:
                        log.write(line)
            returncode = process.wait()
        except BaseException:
            from .process_groups import descendant_groups, signal_groups

            try:
                groups = descendant_groups(process.pid) | {process.pid}
            except (OSError, subprocess.SubprocessError):
                groups = {process.pid}
            try:
                signal_groups(groups, signal.SIGTERM)
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                signal_groups(groups, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            finally:
                # The launcher can exit before a separately grouped trainer.
                signal_groups(groups - {process.pid}, signal.SIGKILL)
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, process.args)
    if run_dir is None or not (run_dir / "pipeline_manifest.json").is_file():
        raise RuntimeError("No completed pipeline produced; inspect the training log")
    from IPython.display import FileLink, display

    manifest = json.loads((run_dir / "pipeline_manifest.json").read_text())
    metrics = json.loads((run_dir / "classification/metrics.json").read_text())[
        "classification"
    ]
    report = {
        "classification": (
            {key: metrics[key] for key in ("macro_f1", "micro_f1")} if metrics else None
        ),
        "evaluation_status": "evaluated" if metrics else "unavailable_full_training",
    }
    if manifest["mode"] == "task1":
        print(json.dumps(report, indent=2))
        display(FileLink(str(run_dir / "classification/metrics.json")))
    else:
        if not (run_dir / "results.zip").is_file():
            raise RuntimeError(
                "No valid submission produced; inspect results.evaluation.json"
            )
        report["corrections"] = json.loads(
            (run_dir / "correction/metrics.json").read_text()
        )["corrections"]
        print(json.dumps(report, indent=2))
        display(FileLink(str(run_dir / "results.zip")))
    return run_dir
