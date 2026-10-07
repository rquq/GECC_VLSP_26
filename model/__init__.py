"""Vietnamese grammatical error correction baseline utilities."""

import os

# Training and inference use PyTorch only. Keep optional image packages out.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
