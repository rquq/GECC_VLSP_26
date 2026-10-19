"""Run the committed experiment recipe; notebooks contain no experiment switches."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import zipfile

from model.data import load_records
from model.runtime import PROJECT_ROOT, load_config, parameter_limit, resolve_output, write_json
from tools.runtime import discover_data, source_identity


def model_size(config):
    """Count unquantized architecture plus adapter tensors, not packed NF4 storage."""
    limit = parameter_limit(config)
    from accelerate import init_empty_weights
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
    )

    settings = config["model"]
    architecture = AutoConfig.from_pretrained(
        settings["name"], revision=settings["revision"]
    )
    factory = (
        AutoModelForSequenceClassification
        if config["task"] == "classification"
        else AutoModelForCausalLM
    )
    factory._model_mapping[
        type(architecture)
    ]  # Resolve lazy imports before entering meta-device context.
    if config["task"] == "classification":
        architecture.num_labels = 5
    with init_empty_weights(include_buffers=True):
        model = factory.from_config(architecture)
    model.tie_weights()
    base = sum(parameter.numel() for parameter in model.parameters())
    lora = copy.deepcopy(config["lora"])
    lora.pop("task_type", None)
    task_type = (
        TaskType.SEQ_CLS if config["task"] == "classification" else TaskType.CAUSAL_LM
    )
    with init_empty_weights(include_buffers=True):
        model = get_peft_model(model, LoraConfig(**lora, task_type=task_type))
    expanded = sum(parameter.numel() for parameter in model.parameters())
    adapter = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )
    saved = expanded - base - adapter
    # The classifier may add a dedicated pad token. Reserve its embedding too.
    padding = architecture.hidden_size if config["task"] == "classification" else 0
    total = base + adapter + saved + padding
    if limit is not None and total > limit:
        raise ValueError(
            f"{config['task']} model plus adapter exceeds the constrained-track 8B limit: {total:,}"
        )
    result = {
        "base": base,
        "adapter": adapter,
        "saved_module_copies": saved,
        "padding_allowance": padding,
        "total_upper_bound": total,
        "competition_id": config.get("competition_id", 1038),
        "limit": limit,
    }
    if config.get("ensemble"):
        sizes = {}
        from model.runtime import classifier_member_config
        for entry in config["ensemble"]["members"]:
            child = classifier_member_config(config, entry)
            sizes[entry["name"]] = model_size(child)
        result["members"] = sizes
        result["total_member_parameters"] = sum(
            v["total_upper_bound"] for v in sizes.values()
        )
        result["eligibility_scope"] = (
            "Each separate classifier component <=8B; bases are different and not shared."
            if limit is not None else
            "Track1039 components; native parameter counts retained, no numeric 8B gate."
        )
    return result


def resolve_inputs(recipe, input_root, destination):
    """Accept the canonical private ZIP or its unambiguously extracted files."""
    filenames = {
        "original": "VGEC 2026_v1.json",
        "classification": "VGEC_classifier.json",
        "correction": "VGEC_balanced.json",
    }
    if recipe["mode"] != "task1":
        filenames["submission"] = recipe["submission_filename"]
    archives = sorted(Path(input_root).rglob(recipe["data_archive"]))
    if len(archives) > 1:
        raise ValueError(
            "Multiple canonical data archives found; attach just one dataset version"
        )
    paths = {}
    if archives:
        destination.mkdir(mode=0o700)
        with zipfile.ZipFile(archives[0]) as archive:
            for role, filename in filenames.items():
                matches = [
                    entry
                    for entry in archive.infolist()
                    if not entry.is_dir() and Path(entry.filename).name == filename
                ]
                if len(matches) > 1:
                    raise ValueError(f"Duplicate {filename} in data archive")
                if matches:
                    # Write only explicit basenames, never extract arbitrary archive paths.
                    path = destination / filename
                    path.write_bytes(archive.read(matches[0]))
                    path.chmod(0o600)
                    paths[role] = path
    for role, filename in filenames.items():
        if role not in paths:
            paths[role] = discover_data(None, PROJECT_ROOT, Path(input_root), filename)
    for role, expected in recipe["data_sha256"].items():
        if role not in ("original", "classification", "correction"):
            raise ValueError(f"Unknown dataset hash role: {role}")
        actual = hashlib.sha256(paths[role].read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"{role} dataset differs from committed recipe: {actual}; "
                "use the correct private data version or commit its new hash"
            )
    for role, path in paths.items():
        if role in ("original", "submission") and not load_records(
            path, require_labels=role == "original"
        ):
            raise ValueError(f"Empty {role} dataset")
    if recipe.get("submission_sha256") is not None:
        if (
            "submission" not in paths
            or hashlib.sha256(paths["submission"].read_bytes()).hexdigest()
            != recipe["submission_sha256"]
        ):
            raise ValueError(
                "Submission input differs from committed private-test hash"
            )
    return paths


def resolve_saved_classifier(value, input_root, destination, archive_name):
    """Find one uploaded handoff or stream it safely from the canonical private ZIP."""
    relative = Path(value).expanduser()
    if relative.is_absolute():
        if not relative.is_dir():
            raise FileNotFoundError(
                f"Saved classifier directory does not exist: {relative}"
            )
        return relative
    if not relative.parts or ".." in relative.parts:
        raise ValueError("Saved classifier path must stay inside the input mount")
    direct = Path(input_root) / relative
    if direct.is_dir():
        return direct
    candidates = [
        p
        for p in Path(input_root).rglob(relative.name)
        if p.is_dir() and p.parts[-len(relative.parts) :] == relative.parts
    ]
    if len(candidates) > 1:
        raise ValueError(
            "Multiple saved classifier handoffs found; attach just one version"
        )
    if candidates:
        return candidates[0]
    archives = list(Path(input_root).rglob(archive_name))
    if len(archives) != 1:
        raise FileNotFoundError(
            "Saved classifier is absent; attach the complete private handoff ZIP"
        )
    prefix = relative.as_posix() + "/"
    with zipfile.ZipFile(archives[0]) as archive:
        entries = [
            e
            for e in archive.infolist()
            if not e.is_dir() and e.filename.startswith(prefix)
        ]
        if not entries:
            raise FileNotFoundError(
                "Canonical data ZIP does not contain its saved classifier handoff"
            )
        seen = set()
        for entry in entries:
            tail = Path(entry.filename[len(prefix) :])
            if (
                not tail.parts
                or tail.is_absolute()
                or ".." in tail.parts
                or tail in seen
            ):
                raise ValueError("Invalid or duplicate saved classifier archive member")
            seen.add(tail)
            path = destination / tail
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with archive.open(entry) as source, path.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            path.chmod(0o600)
    return destination


def prepare_run(recipe_path, input_root, output_root, *, world_size, device="cuda"):
    """Resolve immutable inputs/configs before any model loads or training begins."""
    recipe_path = Path(recipe_path)
    recipe = json.loads(recipe_path.read_text())
    expected = {
        "name",
        "mode",
        "classification_config",
        "correction_config",
        "classifier_run",
        "initial_adapter",
        "data_archive",
        "submission_filename",
        "data_sha256",
    }
    if set(recipe) - {
        "classifier_resume_sha256",
        "submission_sha256",
    } != expected or recipe["mode"] not in (
        "both",
        "task1",
        "task2",
    ):
        raise ValueError(
            "Recipe must contain exactly the documented fields; mode is both, task1, or task2"
        )
    for hash_field in ("classifier_resume_sha256", "submission_sha256"):
        if hash_field not in recipe:
            continue
        digest = recipe[hash_field]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"{hash_field} must pin a lowercase SHA256 digest")
    if recipe["mode"] == "task2" and not recipe["classifier_run"]:
        raise ValueError("task2 recipe requires a completed classifier_run")
    for key in ("data_archive", "submission_filename"):
        if not isinstance(recipe[key], str) or Path(recipe[key]).name != recipe[key]:
            raise ValueError(f"{key} must be a basename")
    if set(recipe["data_sha256"]) != {"original", "classification", "correction"}:
        raise ValueError("Recipe must pin all three training dataset hashes")
    work = resolve_output(
        {"run": {"name": recipe["name"], "output_root": str(output_root)}}
    )
    work.mkdir(parents=True, mode=0o700)
    inputs = resolve_inputs(recipe, Path(input_root), work / "inputs")
    configs, sizes = {}, {}
    for task, field in (
        ("classification", "classification_config"),
        ("correction", "correction_config"),
    ):
        config = load_config(PROJECT_ROOT / recipe[field], task)
        config["run"].update(
            device=device, world_size=world_size, output_root=str(work)
        )
        config["data"].update(
            input_root=str(input_root),
            path=str(inputs["original" if task == "classification" else "correction"]),
        )
        if task == "classification":
            config["data"].update(
                augmentation_path=str(inputs["classification"]),
                correction_path=str(inputs["correction"]),
            )
        sizes[task] = model_size(config)
        tracking = config.get("tracking", {})
        if tracking.get("enabled"):
            tracking["mode"] = os.environ.get("WANDB_MODE") or (
                tracking.get("mode", "online")
                if os.environ.get("WANDB_API_KEY")
                else "offline"
            )
        path = work / f"{task}_config.json"
        write_json(path, config)
        load_config(path, task)
        configs[task] = path
    command = [
        sys.executable,
        "-m",
        "model.two_phase",
        "--mode",
        recipe["mode"],
        "--classification-config",
        str(configs["classification"]),
        "--correction-config",
        str(configs["correction"]),
        "--output-dir",
        str(work / "run"),
    ]
    if recipe.get("classifier_resume_sha256"):
        if recipe["classifier_run"] or recipe["mode"] == "task2":
            raise ValueError(
                "Partial classifier resume requires a fresh ensemble in both/task1 mode"
            )
        from model.classifier_resume import materialize

        materialize(
            input_root,
            recipe["data_archive"],
            recipe["classifier_resume_sha256"],
            work / "run",
            source_manifest=PROJECT_ROOT / "configs/classifier_resume_source.json",
        )
    if recipe["mode"] != "task1":
        command.extend(["--submission-data", str(inputs["submission"])])
    for field, flag in (
        ("classifier_run", "--classifier-run"),
        ("initial_adapter", "--initial-adapter"),
    ):
        if recipe[field]:
            if field == "classifier_run":
                value = resolve_saved_classifier(
                    recipe[field],
                    Path(input_root),
                    work / "saved-classifier",
                    recipe["data_archive"],
                )
                command.extend(["--classification-data", str(inputs["original"])])
            else:
                value = Path(recipe[field]).expanduser()
                value = value if value.is_absolute() else Path(input_root) / value
            command.extend([flag, str(value)])
    write_json(
        work / "experiment_manifest.json",
        {
            "recipe": recipe,
            "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
            "source": source_identity(PROJECT_ROOT),
            "world_size": world_size,
            "device": device,
            "model_parameters": sizes,
            "inputs": {
                role: {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for role, path in inputs.items()
            },
            "command": command,
        },
    )
    return work, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe", default=str(PROJECT_ROOT / "configs/experiment.json")
    )
    parser.add_argument("--input-root")
    parser.add_argument("--output-root")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    source = source_identity(PROJECT_ROOT)
    if source["commit"] is None or source["dirty"]:
        raise ValueError(
            "Notebook experiments require a clean Git checkout; commit code/config changes before running"
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This committed experiment requires CUDA; select Kaggle GPU T4 x2"
        )
    recipe = json.loads(Path(args.recipe).read_text())
    if recipe.get("format") == "vgec-quality-trial-v1":
        from tools.run_quality_trial import run_from_args
        run_from_args(args, source)
        return
    if recipe.get("format") == "vgec-focused-v1":
        from tools.run_focused import run_from_args
        run_from_args(args, source)
        return
    if recipe.get("format") == "vgec-tuning-v1":
        from tools.run_tuning import run_from_args

        run_from_args(args, source)
        return
    stage = load_config(
        PROJECT_ROOT / recipe["classification_config"], "classification"
    )
    world_size = stage["run"].get("world_size", 1)
    kaggle = Path("/kaggle/working").is_dir()
    input_root = (
        Path(args.input_root)
        if args.input_root
        else (Path("/kaggle/input") if kaggle else PROJECT_ROOT / "dataset")
    )
    output_root = (
        Path(args.output_root)
        if args.output_root
        else (Path("/kaggle/working/runs") if kaggle else PROJECT_ROOT / "runs")
    )
    work, command = prepare_run(
        args.recipe, input_root, output_root, world_size=world_size
    )
    print(f"VGEC_RUN_DIR:{work / 'run'}", flush=True)
    from tools.experiment_log import record

    manifest = json.loads((work / "experiment_manifest.json").read_text())
    identity = {
        "name": manifest["recipe"]["name"],
        "mode": manifest["recipe"]["mode"],
        "commit": source["commit"],
        "recipe_sha256": manifest["recipe_sha256"],
        "input_sha256": {
            key: item["sha256"] for key, item in manifest["inputs"].items()
        },
        "preflight_only": args.preflight_only,
    }
    record(work, "started", **identity)
    from model.reproducibility import process_environment

    environment = process_environment(42)
    try:
        subprocess.run(
            command + ["--preflight-only"],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )
        if not args.preflight_only:
            subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
            required = (
                "classification/metrics.json"
                if identity["mode"] == "task1"
                else "results.zip"
            )
            if not (work / "run" / required).is_file():
                raise RuntimeError(
                    f"Training ended without required artifact {required}; inspect the run diagnostics"
                )
            from model.tracking import result_metrics

            report = json.loads((work / "run/pipeline_manifest.json").read_text())
            identity["classification"] = result_metrics(
                report["classification_metrics"]
            )
            if "correction_metrics" in report:
                identity["correction"] = result_metrics(report["correction_metrics"])
        record(
            work, "preflight_passed" if args.preflight_only else "completed", **identity
        )
    except BaseException as error:
        record(work, "failed", error_type=type(error).__name__, **identity)
        raise


if __name__ == "__main__":
    main()
