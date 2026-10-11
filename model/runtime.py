"""Shared configuration, paths, artifacts, and selection for both task stages."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CompleteEpochMixin:
    """Consume rank-local final batches with Transformers 4.x accumulation."""

    def set_initial_training_values(self, args, dataloader, total_train_batch_size):
        values = super().set_initial_training_values(args, dataloader, total_train_batch_size)
        epochs, updates, examples, samples, epoch_based, batches, steps = values
        if epoch_based and batches is not None:
            updates = max(math.ceil(batches / args.gradient_accumulation_steps), 1)
            steps = math.ceil(args.num_train_epochs * updates)
        return epochs, updates, examples, samples, epoch_based, batches, steps

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        # Trainer 4.51 computes its last window from global example count,
        # not rank-local microbatches. Iterator exhaustion gives the true tail.
        return super().get_batch_samples(
            epoch_iterator, self.args.gradient_accumulation_steps, device)


def load_config(path, expected_task=None):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    task = config.get("task")
    if task not in ("classification", "correction") or (expected_task is not None and task != expected_task):
        raise ValueError(f"Expected {expected_task or 'classification/correction'} task configuration")
    for section in ("run", "data", "model", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    if type(config["run"].get("seed")) is not int or config["run"]["seed"] != 42:
        raise ValueError("run.seed must be 42")
    world_size = config["run"].get("world_size", 1)
    if type(world_size) is not int or world_size not in (1, 2):
        raise ValueError("run.world_size must be 1 or 2")
    batch = config["training"]["args"]["per_device_train_batch_size"]
    effective = config["training"]["effective_batch_size"]
    if type(batch) is not int or type(effective) is not int or batch <= 0 or effective < batch or effective % batch:
        raise ValueError("effective_batch_size must be a positive multiple of the microbatch")
    if effective < batch * world_size or effective % (batch * world_size):
        raise ValueError("effective_batch_size must be a positive multiple of microbatch * world_size")
    return config


def resolve_data(config, explicit=None):
    from tools.runtime import discover_data
    path = explicit or config["data"].get("path")
    if path is not None:
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    return discover_data(path, PROJECT_ROOT, Path(config["data"]["input_root"]), config["data"]["filename"])


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
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Selection metric {path!r} must be a finite number or null")
        values.append(value)
    return tuple(values)
