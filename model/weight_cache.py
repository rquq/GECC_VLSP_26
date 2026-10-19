"""Bound temporary Kaggle base-weight storage without touching user caches."""

import json
import os
from pathlib import Path
import re
import shutil

MARKER = ".vgec-owned-cache"


def prepare_weight_cache(settings):
    """Keep one base in the controller-owned cache; saved adapters live elsewhere."""
    value = os.environ.get("VGEC_WEIGHT_CACHE_ROOT")
    if value is None:
        return
    root = Path(value)
    if (not root.is_absolute() or root.is_symlink()
            or root.resolve() != root or not (root / MARKER).is_file()
            or (root / MARKER).read_text() != "vgec-temporary-base-cache-v1\n"
            or os.environ.get("HF_HUB_CACHE") != str(root)):
        raise ValueError("Weight cleanup requires the controller-owned temporary HF cache")
    name = settings["name"]
    if Path(name).is_dir():
        return  # Tiny/local fixtures are never cleanup targets.
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*", name):
        raise ValueError("Invalid base repository for temporary cache")
    keep = "models--" + name.replace("/", "--")
    removed = 0
    for directory in root.glob("models--*"):
        if directory.name == keep:
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Unexpected entry in owned weight cache")
        shutil.rmtree(directory)
        removed += 1
    print(json.dumps({"temporary_weight_cache": "one_base_at_a_time",
                      "evicted_base_directories": removed}), flush=True)
