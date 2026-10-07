"""Launch the baseline on Kaggle's two GPUs and package its local artifacts."""

import argparse
import json
import importlib.metadata
import shlex
from pathlib import Path
import shutil
import subprocess
import sys


DATA_FILENAME = "VGEC 2026_v1.json"


def discover_data(explicit, project_root, input_root=Path("/kaggle/input")):
    """Prefer an explicit path, then an unambiguous Kaggle input, then repo data."""
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DATA_PATH does not exist: {path}")
        return path
    matches = sorted(path.resolve() for path in input_root.rglob(DATA_FILENAME) if path.is_file()) if input_root.exists() else []
    if len(matches) > 1:
        raise ValueError("Multiple Kaggle datasets match; set DATA_PATH explicitly: " + ", ".join(map(str, matches)))
    if matches:
        return matches[0]
    fallback = project_root / "dataset" / DATA_FILENAME
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"Attach a dataset containing {DATA_FILENAME!r} or set DATA_PATH explicitly.")


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
    parser.add_argument("--data", type=Path, help="Explicit data path; otherwise discover Kaggle input.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-gpus", type=int, choices=(1, 2), default=2,
                        help="Kaggle uses two GPUs; one is available for local diagnostics.")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Check source, data, packages, and GPUs without starting training.")
    parser.add_argument("--expected-commit", help="Require this exact clean Git commit, as verified by the notebook.")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    # Running a script inside tools/ otherwise omits the package root.
    sys.path.insert(0, str(project_root))
    data_path = discover_data(args.data, project_root)
    source = source_identity(project_root)
    if args.expected_commit and (source["commit"] != args.expected_commit or source["dirty"]):
        raise RuntimeError("Source changed after notebook verification; rerun the source cell with a clean clone.")
    output_dir = (args.output_dir or Path(f"/kaggle/working/vgec-{args.mode}")).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Choose a new output directory for a reproducible run.")

    import torch

    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    print(f"Visible CUDA devices ({count}): {names}", flush=True)
    if not torch.cuda.is_available() or count < args.num_gpus:
        raise RuntimeError(f"Requested {args.num_gpus} CUDA devices, found {count}. Select Kaggle GPU T4 x2, or use --num-gpus 1 for a local GPU diagnostic.")
    if not all("T4" in name for name in names[:args.num_gpus]):
        print("Warning: selected GPUs are not all T4; actual devices are listed above.", flush=True)
    config = project_root / "configs" / ("smoke.json" if args.mode == "smoke" else "baseline.json")
    settings = json.loads(config.read_text(encoding="utf-8"))
    if settings.get("bf16", False) or not settings.get("fp16", False):
        raise ValueError("Kaggle T4 configuration must set fp16=true and bf16=false.")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.num_gpus}", "-m", "vgec.train", "--config", str(config), "--data", str(data_path), "--output-dir", str(output_dir)]
    from vgec.data import load_records, split_records
    records = load_records(data_path)
    train, validation = split_records(records, settings["validation_fraction"], settings["seed"])
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate")}
    print(json.dumps({"source": source, "data_path": str(data_path), "rows": len(records),
                      "train_rows": len(train), "validation_rows": len(validation),
                      "versions": versions, "config": str(config)}, ensure_ascii=False, indent=2), flush=True)
    print("Training command:", shlex.join(command), flush=True)
    if args.preflight_only:
        print("Preflight passed; no training started and no output directory created.", flush=True)
        return
    subprocess.run(command, cwd=project_root, check=True)
    for filename in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
        path = output_dir / "adapter" / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Training did not produce the required saved artifact: {path}")
    source_manifest = {**source, "training_command": command}
    (output_dir / "source_manifest.json").write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"] = source
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for filename in ("metrics.json", "identity_metrics.json", "run_manifest.json"):
        path = output_dir / filename
        print(f"\n{filename}:\n" + json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2), flush=True)
    archive = shutil.make_archive(str(output_dir) + "-artifacts", "zip", root_dir=output_dir.parent, base_dir=output_dir.name)
    print(f"Saved local run artifacts: {archive}", flush=True)
    print("Generated predictions are a local proposal, not an official competition submission.", flush=True)


if __name__ == "__main__":
    main()
