"""Observe CUDA release without creating CUDA contexts in the coordinator."""

import subprocess
import time


def snapshot():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "Cannot inspect GPU memory before a model transition; nvidia-smi failed"
        ) from error
    devices = {}
    for line in result.stdout.splitlines():
        index, free, total = [v.strip() for v in line.split(",")]
        devices[index] = {
            "free_bytes": int(free) * 1024**2,
            "total_bytes": int(total) * 1024**2,
        }
    if not devices:
        raise RuntimeError("No CUDA devices reported by nvidia-smi")
    return devices


def wait_for_release(before, timeout=30, tolerance_bytes=256 * 1024**2, on_wait=None):
    started = time.monotonic()
    while True:
        current = snapshot()
        if set(current) != set(before):
            raise RuntimeError("GPU set changed during model transition")
        if all(
            current[k]["free_bytes"] + tolerance_bytes >= v["free_bytes"]
            for k, v in before.items()
        ):
            return current
        if time.monotonic() - started >= timeout:
            raise RuntimeError(
                "GPU memory not released after child exit; refusing to load another model: "
                + str(current)
            )
        if on_wait:
            on_wait()
        time.sleep(1)
