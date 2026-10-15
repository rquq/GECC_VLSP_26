"""Choose the native fast backend for remote Mistral/Llama tokenizer metadata."""

from unittest import TestCase
from unittest.mock import MagicMock, patch


class TokenizerBackendTests(TestCase):
    def test_remote_mistral_materializes_assets_before_fast_load(self):
        import tempfile
        from pathlib import Path
        from model.classification import _load_tokenizer

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ):
                (root / name).write_text("{}")
            with patch(
                "transformers.models.auto.tokenization_auto.get_tokenizer_config",
                return_value={"tokenizer_class": "LlamaTokenizer"},
            ), patch(
                "huggingface_hub.hf_hub_download",
                side_effect=lambda repo, filename, **kwargs: str(root / filename),
            ) as download, patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()
            ) as load:
                _load_tokenizer("remote/mistral-fixture", revision="pinned")
            self.assertEqual(load.call_args.args[0], str(root))
            self.assertTrue(load.call_args.kwargs["use_fast"])
            self.assertTrue(load.call_args.kwargs["local_files_only"])
            self.assertFalse(load.call_args.kwargs["from_slow"])
            self.assertIsNone(load.call_args.kwargs["add_prefix_space"])
            self.assertEqual(download.call_count, 3)
            self.assertTrue(
                all(c.kwargs["revision"] == "pinned" for c in download.call_args_list)
            )

    def test_remote_qwen_retains_existing_backend_selection(self):
        from model.classification import _load_tokenizer

        with patch(
            "transformers.models.auto.tokenization_auto.get_tokenizer_config",
            return_value={"tokenizer_class": "Qwen2Tokenizer"},
        ), patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()
        ) as load:
            _load_tokenizer("remote/qwen-fixture", revision="pinned")
        self.assertFalse(load.call_args.kwargs["use_fast"])

    def test_remote_missing_backend_resolution_never_falls_back_to_slow(self):
        import json
        from pathlib import Path
        import tempfile
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers.tokenization_utils_base import cached_file
        from model.classification import _load_tokenizer

        with tempfile.TemporaryDirectory() as folder:
            assets = Path(folder)
            backend = Tokenizer(
                WordLevel(
                    {"<unk>": 0, "<s>": 1, "</s>": 2, "xin": 3}, unk_token="<unk>"
                )
            )
            backend.save(str(assets / "tokenizer.json"))
            metadata = {
                "tokenizer_class": "LlamaTokenizer",
                "legacy": False,
                "add_prefix_space": True,
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "add_bos_token": False,
                "add_eos_token": False,
            }
            (assets / "tokenizer_config.json").write_text(json.dumps(metadata))
            (assets / "special_tokens_map.json").write_text(
                json.dumps(
                    {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}
                )
            )

            def incomplete_remote_lookup(path, filename, **kwargs):
                if Path(path).is_dir():
                    return cached_file(path, filename, **kwargs)
                if filename in ("tokenizer.json", "tokenizer.model"):
                    return None
                candidate = assets / filename
                return str(candidate) if candidate.is_file() else None

            def download(repo_id, filename, **kwargs):
                self.assertEqual(repo_id, "remote/mistral-fixture")
                return str(assets / filename)

            with patch(
                "transformers.models.auto.tokenization_auto.get_tokenizer_config",
                return_value=metadata,
            ), patch(
                "transformers.tokenization_utils_base.cached_file",
                side_effect=incomplete_remote_lookup,
            ), patch(
                "huggingface_hub.hf_hub_download", side_effect=download
            ):
                tokenizer = _load_tokenizer("remote/mistral-fixture", revision="pinned")
            self.assertTrue(tokenizer.is_fast)
            self.assertEqual(
                tokenizer("xin", add_special_tokens=False)["input_ids"], [3]
            )
