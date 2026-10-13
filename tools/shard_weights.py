"""Stream unchanged safetensors into smaller shards for low-RAM local loading."""

import json
from pathlib import Path
import shutil
import struct


def shard_checkpoint(source, destination, max_bytes=2 * 1024**3):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError("Sharded checkpoint destination must be new")
    weight = source / "model.safetensors"
    with weight.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > 64 * 1024**2:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(stream.read(length))
    start = 8 + length
    groups, current, size = [], [], 0
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        begin, end = tensor["data_offsets"]
        count = end - begin
        if not 0 <= begin < end <= weight.stat().st_size - start or count > max_bytes:
            raise ValueError("Tensor offsets or single-tensor shard limit invalid")
        if current and size + count > max_bytes:
            groups.append(current)
            current, size = [], 0
        current.append((name, tensor))
        size += count
    if current:
        groups.append(current)
    destination.mkdir(parents=True)
    for path in source.glob("*.json"):
        if "safetensors" not in path.name:
            shutil.copy2(path, destination / path.name)
    index, total = {}, 0
    with weight.open("rb") as original:
        for number, group in enumerate(groups, 1):
            name = f"model-{number:05d}-of-{len(groups):05d}.safetensors"
            offsets, offset = {
                "__metadata__": header.get("__metadata__", {"format": "pt"})
            }, 0
            for key, tensor in group:
                count = tensor["data_offsets"][1] - tensor["data_offsets"][0]
                offsets[key] = {**tensor, "data_offsets": [offset, offset + count]}
                offset += count
                index[key] = name
            encoded = json.dumps(offsets, separators=(",", ":")).encode()
            encoded += b" " * ((-len(encoded)) % 8)
            with (destination / name).open("wb") as output:
                output.write(struct.pack("<Q", len(encoded)))
                output.write(encoded)
                for key, tensor in group:
                    begin, end = tensor["data_offsets"]
                    original.seek(start + begin)
                    remaining = end - begin
                    while remaining:
                        data = original.read(min(1024**2, remaining))
                        if not data:
                            raise ValueError("Truncated original tensor payload")
                        output.write(data)
                        remaining -= len(data)
            total += offset
    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": index}, indent=2)
        + "\n"
    )
    return dict(
        shards=len(groups),
        tensor_bytes=total,
        tensor_count=len(index),
        altered_tensor_values=False,
    )
