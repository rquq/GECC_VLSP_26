import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock

from model.correction import _parse_correction, correct, encode_record, validate_hints


class CorrectionGenerationTests(unittest.TestCase):
    def test_encoding_uses_predicted_hint_and_supervises_only_completion(self):
        record = {"id": 7, "source": "Câu gốc.", "label": 1, "target": "Câu đích riêng biệt."}
        inference = {"prompt": "correct", "hint_descriptions": [f"description {i}" for i in range(5)]}
        tokenizer = MagicMock(eos_token_id=99)
        tokenizer.apply_chat_template.return_value = [10, 11, 12]
        tokenizer.encode.return_value = [20, 21]

        encoded = encode_record(tokenizer, record, {"label": 3}, 20, inference)

        messages = tokenizer.apply_chat_template.call_args.args[0]
        prompt = "\n".join(message["content"] for message in messages)
        self.assertIn("Gợi ý nhãn dự đoán: 3", prompt)
        self.assertIn("description 3", prompt)
        self.assertIn(record["source"], prompt)
        self.assertNotIn("description 1", prompt)
        self.assertNotIn(record["target"], prompt)
        tokenizer.encode.assert_called_once_with(
            json.dumps({"corrected_sentence": record["target"]}, ensure_ascii=False), add_special_tokens=False)
        self.assertEqual(encoded["input_ids"], [10, 11, 12, 20, 21, 99])
        self.assertEqual(encoded["labels"], [-100, -100, -100, 20, 21, 99])
        self.assertEqual(encoded["attention_mask"], [1] * 6)

    def test_invalid_generation_has_no_success_fallback(self):
        for text in ('[]', 'null', '1', '"text"',
                     '{"corrected_sentence":true}', '{"corrected_sentence":""}',
                     '{"corrected_sentence":"a","explanation":"b"}',
                     "explanation, not JSON"):
            with self.subTest(text=text):
                target, error = _parse_correction(text)
                self.assertEqual(target, "")
                self.assertIsNotNone(error)

    def test_clean_hints_are_generated_and_invalid_parse_is_retained(self):
        import torch

        records = [{"id": i, "source": "Câu đúng.", "label": 0, "target": "Câu đúng."}
                   for i in (1, 2)]
        hints = [{"id": row["id"], "label": 0, "probabilities": [1., 0., 0., 0., 0.]}
                 for row in records]
        tokenizer = MagicMock(pad_token_id=0, eos_token_id=3)
        tokenizer.apply_chat_template.return_value = [1, 2]
        tokenizer.decode.side_effect = ['{"corrected_sentence":"Câu được sửa."}', "not JSON"]
        model = MagicMock()
        model.get_input_embeddings.return_value = SimpleNamespace(weight=torch.empty(0))
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        inference = {"prompt": "correct", "hint_descriptions": ["clean"] * 5,
                     "generation": {"max_new_tokens": 8}, "max_length": 8}

        predictions = correct(model, tokenizer, records, hints, inference)

        self.assertEqual(model.generate.call_count, 2)
        self.assertEqual([row["label"] for row in predictions], [0, 0])
        self.assertEqual(predictions[0]["target"], "Câu được sửa.")
        self.assertIsNone(predictions[0]["parse_error"])
        self.assertEqual(predictions[1]["target"], "")
        self.assertEqual(predictions[1]["raw_generation"], "not JSON")
        self.assertIsNotNone(predictions[1]["parse_error"])


class HintContractTests(unittest.TestCase):
    def test_quarantined_original_partition_and_contrast_family_leakage(self):
        import copy
        import hashlib

        first = {"id": 1, "source": "first source", "label": 1, "target": "first fixed"}
        second = {"id": 2, "source": "second source", "label": 0, "target": "second source"}
        clean = {"id": "clean", "source": "legal construction", "label": 0, "target": "legal construction"}
        error = {"id": "error", "source": "broken construction", "label": 2, "target": "legal construction"}
        validation = [{"id": 9, "source": "holdout", "label": 0, "target": "holdout"}]
        correction = [first, second, error]
        classifier = [first, second, clean, error]

        def hint(row, fold=None):
            result = {"id": row["id"], "label": 0, "probabilities": [1., 0., 0., 0., 0.]}
            if fold is not None:
                result["fold"] = fold
            return result

        for field in ("family_id", "template_id"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "data.json"
                path.write_text("fixture", encoding="utf-8")
                metadata = [{"id": row["id"], "authored_provenance": {field: "shared"}}
                            for row in (clean, error)]
                provenance = {"provenance": {"original_train_ids": [1, 2, 3],
                                             "train_rows": metadata[1:]}}
                artifact = {
                    "seed": 42, "validation_fraction": .2,
                    "correction_data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "classifier_provenance": {
                        "original_train_ids": [1, 2, 3], "retained_original_train_ids": [1, 2],
                        "quarantined_ids": [3], "selected_synthetic_ids": ["clean", "error"],
                        "train_rows": metadata},
                    "train": [hint(first, 0), hint(error, 0), hint(second, 1)],
                    "validation": [hint(validation[0])],
                    "folds": [{"fold": 0, "train_ids": [2], "predict_ids": [1, "error"]},
                              {"fold": 1, "train_ids": [1, "clean", "error"], "predict_ids": [2]}],
                }
                indexed, _ = validate_hints(artifact, correction, validation, provenance,
                                            classifier, path, 42, .2)
                self.assertEqual({key[1] for key in indexed}, {1, 2, "error"})
                leaked = copy.deepcopy(artifact)
                leaked["folds"][0]["train_ids"].append("clean")
                with self.assertRaisesRegex(ValueError, "contrast/template family"):
                    validate_hints(leaked, correction, validation, provenance, classifier, path, 42, .2)
                missing_quarantine = copy.deepcopy(artifact)
                missing_quarantine["classifier_provenance"]["quarantined_ids"] = []
                with self.assertRaisesRegex(ValueError, "declared mixed partition"):
                    validate_hints(missing_quarantine, correction, validation, provenance,
                                   classifier, path, 42, .2)


if __name__ == "__main__":
    unittest.main()
