"""Shared configuration, paths, artifacts, and selection for both task stages."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def native_bf16_supported():
    """Do not mistake PyTorch's BF16 emulation on T4 for native support."""
    import torch

    return torch.cuda.is_bf16_supported(including_emulation=False)


class CompleteEpochMixin:
    """Consume rank-local final batches with Transformers 4.x accumulation."""

    def set_initial_training_values(self, args, dataloader, total_train_batch_size):
        values = super().set_initial_training_values(
            args, dataloader, total_train_batch_size
        )
        epochs, updates, examples, samples, epoch_based, batches, steps = values
        if epoch_based and batches is not None:
            updates = max(math.ceil(batches / args.gradient_accumulation_steps), 1)
            steps = math.ceil(args.num_train_epochs * updates)
        return epochs, updates, examples, samples, epoch_based, batches, steps

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        # Trainer 4.51 computes its last window from global example count,
        # not rank-local microbatches. Iterator exhaustion gives the true tail.
        return super().get_batch_samples(
            epoch_iterator, self.args.gradient_accumulation_steps, device
        )


def load_config(path, expected_task=None):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    task = config.get("task")
    if task not in ("classification", "correction") or (
        expected_task is not None and task != expected_task
    ):
        raise ValueError(
            f"Expected {expected_task or 'classification/correction'} task configuration"
        )
    for section in ("run", "data", "model", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    allowed_seeds = (42, 43, 44) if task == "classification" else (42,)
    if (
        type(config["run"].get("seed")) is not int
        or config["run"]["seed"] not in allowed_seeds
    ):
        raise ValueError(f"run.seed must be one of {allowed_seeds}")
    split_seed(config)
    if config.get("ensemble") is not None:
        spec = config["ensemble"]
        from .classifier_ensemble import calibration_settings
        calibration_settings(spec.get('calibration') if isinstance(spec, dict) else None)
        entries = spec.get("members", []) if isinstance(spec, dict) else []
        expected = ["qwen7b", "sailor7b", "mistral7b"]
        if (
            task != "classification"
            or len(entries) != 3
            or any(not isinstance(entry, dict) for entry in entries)
            or [entry.get("name") for entry in entries] != expected
            or spec.get("method", "soft") != "soft"
            or config["run"]["seed"] != 42
            or config.get("oof", {}).get("enabled", False)
        ):
            raise ValueError(
                "Soft voting requires Qwen/Sailor/Mistral members, seed 42 and no OOF hints"
            )
        models = [entry.get("model") for entry in entries]
        if (
            any(
                not isinstance(model, dict)
                or not all(
                    isinstance(model.get(key), str) and model[key]
                    for key in ("name", "revision")
                )
                for model in models
            )
            or len({model["name"] for model in models}) != 3
            or models[0] != config["model"]
        ):
            raise ValueError(
                "Soft voting requires three distinct pinned models and the first model as root"
            )
    world_size = config["run"].get("world_size", 1)
    if type(world_size) is not int or world_size not in (1, 2):
        raise ValueError("run.world_size must be 1 or 2")
    if config["model"].get("device_map") == "balanced" and world_size != 1:
        raise ValueError(
            "Balanced placement uses one process across two GPUs; world_size must be 1"
        )
    batch = config["training"]["args"]["per_device_train_batch_size"]
    effective = config["training"]["effective_batch_size"]
    if (
        type(batch) is not int
        or type(effective) is not int
        or batch <= 0
        or effective < batch
        or effective % batch
    ):
        raise ValueError(
            "effective_batch_size must be a positive multiple of the microbatch"
        )
    if effective < batch * world_size or effective % (batch * world_size):
        raise ValueError(
            "effective_batch_size must be a positive multiple of microbatch * world_size"
        )
    return config


def split_seed(config):
    value = config["data"].get("split_seed", 42)
    if type(value) is not int or value != 42:
        raise ValueError("data.split_seed must stay 42 for every model seed")
    return value


def resolve_data(config, explicit=None):
    from tools.runtime import discover_data

    path = explicit or config["data"].get("path")
    if path is not None:
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    return discover_data(
        path,
        PROJECT_ROOT,
        Path(config["data"]["input_root"]),
        config["data"]["filename"],
    )


def resolve_submission_data(value="auto", *, input_root=Path("/kaggle/input")):
    """Resolve explicit inference input or an optional, unambiguous public test.

    None opts out of public-test discovery. Missing automatic input returns None;
    ambiguous matches and missing explicit paths remain errors.
    """
    if value is None:
        return None
    from tools.runtime import discover_data

    if value != "auto":
        return discover_data(value, PROJECT_ROOT, Path(input_root), "public_test.json")
    try:
        return discover_data(None, PROJECT_ROOT, Path(input_root), "public_test.json")
    except FileNotFoundError:
        return None


def stage_output(config, explicit=None):
    output = resolve_output(config, explicit)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    return output


def resolve_output(config, explicit=None):
    """Resolve CLI paths from cwd; default runs live under the project root."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    run = config["run"]
    root = Path(run["output_root"]).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (root / f"{run['name']}-{timestamp}").resolve()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def balanced_limit(records, limit, seed):
    if limit is None:
        return records
    import random

    groups = {label: [] for label in range(5)}
    for record in records:
        groups[record["label"]].append(record)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < limit and any(groups.values()):
        for group in groups.values():
            if group and len(selected) < limit:
                selected.append(group.pop())
    return selected


def selection_key(metrics, metric_paths):
    """Maximize explicit dotted metric paths; null values count as zero."""
    values = []
    for path in metric_paths:
        value = metrics
        for field in path.split("."):
            value = value[field]
        if value is None:
            value = 0.0
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"Selection metric {path!r} must be a finite number or null"
            )
        values.append(value)
    return tuple(values)


def unscored_final_report():
    """Explicitly unavailable scores for a fit with no independent validation."""
    from .metrics import COMBINED_SCORE_METADATA

    return {
        "count": 0,
        "evaluation_status": "unavailable_full_training",
        "classification": None,
        "corrections": {},
        "clean_preservation": {},
        "combined_score": None,
        "combined_score_metadata": dict(COMBINED_SCORE_METADATA),
    }


def promote_trainable_parameters(model):
    """AMP scaling requires FP32 trainable tensors, including PEFT saved heads."""
    import torch

    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            parameter.data = parameter.data.float()
