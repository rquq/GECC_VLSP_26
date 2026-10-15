"""Choose the native fast backend for remote Mistral/Llama tokenizer metadata."""

from unittest import TestCase
from unittest.mock import MagicMock, patch


class TokenizerBackendTests(TestCase):
    def test_remote_mistral_uses_fast_and_preserves_revision(self):
        from model.classification import _load_tokenizer

        with patch(
            "transformers.models.auto.tokenization_auto.get_tokenizer_config",
            return_value={"tokenizer_class": "LlamaTokenizer"},
        ), patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()
        ) as load:
            _load_tokenizer("remote/mistral-fixture", revision="pinned")
        self.assertTrue(load.call_args.kwargs["use_fast"])
        self.assertEqual(load.call_args.kwargs["revision"], "pinned")

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
