"""Explicit training RNGs and environment set before each child interpreter starts."""

import os
import random


def process_environment(seed, environment=None):
    if type(seed) is not int or seed not in (42, 43, 44):
        raise ValueError("Model seed must be 42, 43 or 44")
    result = dict(os.environ if environment is None else environment)
    result.update(
        PYTHONHASHSEED=str(seed),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        TOKENIZERS_PARALLELISM="false",
    )
    return result


def seed_everything(seed):
    """Fail on known nondeterministic kernels; do not silently downgrade policy."""
    environment = process_environment(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = environment["CUBLAS_WORKSPACE_CONFIG"]
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    return {
        "seed": seed,
        "python_random": seed,
        "numpy_global": seed,
        "torch_cpu_cuda": seed,
        "trainer_sampler_seed": seed,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "hash_seed_matches_process_start": os.environ.get("PYTHONHASHSEED")
        == str(seed),
        "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "strict_determinism": True,
        "attention": "math SDPA / eager",
        "cudnn_benchmark": False,
        "tf32": False,
        "scope": "Replay within the same software/hardware environment; not cross-platform identity.",
    }
