"""Bound temporary Kaggle base-weight storage without touching user caches."""

import json
import os
from pathlib import Path
import re
import shutil

MARKER = ".vgec-owned-cache"


def temporary_cache_environment(resources):
    """Keep task-owned base downloads outside saved notebook output."""
    import tempfile
    parent = Path("/kaggle/temp")
    cache = Path(resources.enter_context(tempfile.TemporaryDirectory(
        prefix="vgec-track1039-base-cache-",
        dir=parent if parent.is_dir() else None))).resolve()
    (cache / MARKER).write_text("vgec-temporary-base-cache-v1\n")
    return {"HF_HOME": str(cache), "HF_HUB_CACHE": str(cache),
            "HF_XET_CACHE": str(cache / "xet"),
            "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0", "HF_XET_SHARD_CACHE_SIZE_LIMIT": "0",
            "HF_HUB_DISABLE_SHARED_BLOBS": "1", "VGEC_WEIGHT_CACHE_ROOT": str(cache)}


def prepare_weight_cache(settings, environment=None):
    """Keep one base in the controller-owned cache; saved adapters live elsewhere."""
    environment = os.environ if environment is None else environment
    value = environment.get("VGEC_WEIGHT_CACHE_ROOT")
    if value is None:
        return
    root = Path(value)
    if (not root.is_absolute() or root.is_symlink()
            or root.resolve() != root or not (root / MARKER).is_file()
            or (root / MARKER).read_text() != "vgec-temporary-base-cache-v1\n"
            or environment.get("HF_HUB_CACHE") != str(root)):
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
    # Every model worker has exited before the parent calls this. Xet caches
    # belong to the same marked temporary root; retain them for same-base reuse.
    xet = root / "xet"
    if removed and (xet.exists() or xet.is_symlink()):
        if xet.is_symlink() or not xet.is_dir() or environment.get("HF_XET_CACHE") != str(xet):
            raise ValueError("Unexpected Xet entry in owned weight cache")
        shutil.rmtree(xet)
    report = {"temporary_weight_cache": "one_base_at_a_time", "model": name,
              "evicted_base_directories": removed, "disk_free_bytes": shutil.disk_usage(root).free}
    print(json.dumps(report), flush=True)
    return report
