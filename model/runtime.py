"""Baseline configuration, paths, and artifact writing."""
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("task") != "classification":
        raise ValueError("Expected classification task configuration")
    for section in ("run", "data", "model", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    if type(config["run"].get("seed")) is not int or config["run"]["seed"] != 42:
        raise ValueError("run.seed must be 42")
    batch = config["training"]["args"]["per_device_train_batch_size"]
    effective = config["training"]["effective_batch_size"]
    if type(batch) is not int or type(effective) is not int or batch <= 0 or effective < batch or effective % batch:
        raise ValueError("effective_batch_size must be a positive multiple of the microbatch")
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
