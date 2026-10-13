import json
from pathlib import Path
import tempfile
import unittest


class StreamingShardTests(unittest.TestCase):
    def test_shards_preserve_every_tensor_exactly(self):
        import torch
        from safetensors.torch import save_file, load_file
        from tools.shard_weights import shard_checkpoint

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            weights = {
                "a": torch.arange(9, dtype=torch.float32),
                "b": torch.arange(8, dtype=torch.float16),
                "c": torch.arange(7, dtype=torch.float32),
            }
            save_file(weights, source / "model.safetensors", metadata={"format": "pt"})
            (source / "config.json").write_text('{"fixture":true}')
            output = root / "shards"
            layout = shard_checkpoint(source, output, max_bytes=40)
            self.assertEqual(layout["shards"], 3)
            self.assertFalse(layout["altered_tensor_values"])
            index = json.loads((output / "model.safetensors.index.json").read_text())[
                "weight_map"
            ]
            restored = {}
            for filename in set(index.values()):
                restored.update(load_file(output / filename))
            self.assertEqual(set(restored), set(weights))
            for key in weights:
                self.assertTrue(torch.equal(weights[key], restored[key]))
            self.assertEqual(
                (output / "config.json").read_text(),
                (source / "config.json").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
