"""Shared paths, artifact writing, and selection helpers for separate tasks."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
