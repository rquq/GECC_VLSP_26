"""Launch independent single-GPU QLoRA experiments and package their artifacts."""

import argparse
import json
import importlib.metadata
import os
import shlex
from pathlib import Path
import shutil
import subprocess
import sys


DATA_FILENAME = "VGEC 2026_v1.json"


def discover_data(explicit, project_root, input_root=Path("/kaggle/input"), filename=DATA_FILENAME):
    """Prefer an explicit path, then an unambiguous input, then repo data."""
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DATA_PATH does not exist: {path}")
        return path
    matches = sorted(path.resolve() for path in input_root.rglob(filename) if path.is_file()) if input_root.exists() else []
    if len(matches) > 1:
        raise ValueError("Multiple Kaggle datasets match; set DATA_PATH explicitly: " + ", ".join(map(str, matches)))
    if matches:
        return matches[0]
    fallback = project_root / "dataset" / filename
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"Attach a dataset containing {filename!r} or set data.path explicitly.")


def source_identity(project_root):
    """Read the exact checkout used for training, without trusting parent defaults."""
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=project_root,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    try:
        root = Path(git("rev-parse", "--show-toplevel"))
    except subprocess.CalledProcessError:
        return {"repo_url": None, "branch": None, "commit": None,
                "dirty": None, "project_subdir": None}
    try:
        repo_url = git("remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        repo_url = None
    # Do not retain URL-embedded credentials in public run artifacts.
    from urllib.parse import urlsplit, urlunsplit
    if repo_url and urlsplit(repo_url).username is not None:
        parts = urlsplit(repo_url)
        repo_url = urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))
    return {"repo_url": repo_url, "branch": git("branch", "--show-current") or None,
            "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "project_subdir": str(Path(project_root).resolve().relative_to(root.resolve()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--config", type=Path, help="Configuration file; overrides the mode's default.")
    parser.add_argument("--data", type=Path, help="Explicit data path; otherwise discover Kaggle input.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true",
                        help="Check source, data, packages, and GPUs without starting training.")
    parser.add_argument("--expected-commit", help="Require this exact clean Git commit, as verified by the notebook.")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    # Running a script inside tools/ otherwise omits the package root.
    sys.path.insert(0, str(project_root))
    from model.config import load_config, resolve_output
    config = args.config.resolve() if args.config else project_root / "configs" / ("smoke.json" if args.mode == "smoke" else "baseline.json")
    settings = load_config(config)
    data_path = discover_data(args.data or settings["data"]["path"], project_root,
                              Path(settings["data"]["input_root"]), settings["data"]["filename"])
    source = source_identity(project_root)
    if args.expected_commit and (source["commit"] != args.expected_commit or source["dirty"]):
        raise RuntimeError("Source changed after notebook verification; rerun the source cell with a clean clone.")
    output_dir = resolve_output(settings, args.output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Choose a new output directory for a reproducible run.")

    import torch

    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    print(f"Visible CUDA devices ({count}): {names}", flush=True)
    device = settings["run"]["device"]
    if device == "cuda" and (not torch.cuda.is_available() or count < 1):
        raise RuntimeError("Enable a CUDA GPU or configure run.device=cpu and disable quantization.")
    seeds = settings["run"]["seeds"]
    import transformers
    import peft
    import accelerate
    import bitsandbytes
    from bitsandbytes.cextension import lib
    if settings["model"]["load_in_4bit"] and not getattr(lib, "compiled_with_cuda", False):
        raise RuntimeError("bitsandbytes must have its CUDA backend available for 4-bit QLoRA.")
    capability = torch.cuda.get_device_capability(0) if device == "cuda" else None
    if settings["model"]["load_in_4bit"] and (device != "cuda" or capability < (6, 0)):
        raise RuntimeError("bitsandbytes 4-bit quantization requires a Pascal-or-newer CUDA GPU.")
    command = [sys.executable, "-m", "model.experiment", "--config", str(config), "--data", str(data_path), "--output-dir", str(output_dir)]
    from model.data import load_records, split_records
    from model.train import balanced_limit
    records = load_records(data_path)
    splits = []
    for seed in seeds:
        train, validation = split_records(records, settings["data"]["validation_fraction"], seed)
        train = balanced_limit(train, settings["data"]["train_limit"], seed)
        validation = balanced_limit(validation, settings["data"]["validation_limit"], seed + 1)
        splits.append({"seed": seed, "train_rows": len(train), "validation_rows": len(validation)})
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")}
    dtype = settings["model"]["compute_dtype"]
    if dtype == "auto":
        dtype = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device == "cuda" else "float32"
    print(json.dumps({"source": source, "data_path": str(data_path), "rows": len(records),
                      "splits": splits, "gpu_capability": capability,
                      "compute_dtype": dtype,
                      "versions": versions, "config": str(config)}, ensure_ascii=False, indent=2), flush=True)
    print("Experiment command:", shlex.join(command), flush=True)
    if args.preflight_only:
        print("Preflight passed; no training started and no output directory created.", flush=True)
        return
    environment = os.environ.copy()
    # Preserve an existing device selection while exposing only its first GPU.
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    environment["CUDA_VISIBLE_DEVICES"] = visible.split(",")[0].strip() if visible else "0"
    subprocess.run(command, cwd=project_root, env=environment, check=True)
    for seed in seeds:
        seed_dir = output_dir / f"seed-{seed}"
        for filename in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json", "loading.json", "inference.json"):
            path = seed_dir / "adapter" / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Training did not produce the required saved artifact: {path}")
        for filename in ("baseline_predictions.json", "baseline_metrics.json", "metrics.json", "predictions.json", "identity_metrics.json", "epoch_history.json", "run_manifest.json"):
            path = seed_dir / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Experiment did not produce the required artifact: {path}")
        manifest_path = seed_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source"] = source
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source_manifest = {**source, "training_command": command}
    (output_dir / "source_manifest.json").write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    comparison_path = output_dir / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["source"] = source
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\ncomparison.json:\n" + json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    if settings["run"]["archive"]:
        archive = shutil.make_archive(str(output_dir) + "-artifacts", "zip", root_dir=output_dir.parent, base_dir=output_dir.name)
        print(f"Saved local run artifacts: {archive}", flush=True)
    print(f"Completed run: {output_dir}", flush=True)
    for seed in seeds:
        print(f"Saved adapter: {output_dir / f'seed-{seed}' / 'adapter'}", flush=True)
    print("Generated predictions are a local proposal, not an official competition submission.", flush=True)


if __name__ == "__main__":
    main()
