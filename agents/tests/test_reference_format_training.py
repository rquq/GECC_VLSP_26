import copy
import json
import unittest
from unittest.mock import MagicMock


class ReferenceFormatTrainingTests(unittest.TestCase):
    def test_only_training_abbreviations_change_instruction_and_never_target(self):
        from model.correction import encode_training_record

        inference = {
            "prompt": "complete sentence instruction",
            "output_format": "sentence",
            "hint_mode": "none",
            "user_role": "user",
        }
        config = {"training": {"reference_format_conditioning": "separate_abbreviated"}}
        record = {
            "id": 1,
            "source": "source fixture.",
            "label": 2,
            "target": "reference ...",
        }
        before = copy.deepcopy((inference, record))
        tokenizer = MagicMock(eos_token_id=99)
        tokenizer.apply_chat_template.return_value = [1, 2]
        tokenizer.encode.return_value = [3, 4]
        encoded = encode_training_record(
            tokenizer, record, {"label": 0}, 30, inference, config
        )
        messages = tokenizer.apply_chat_template.call_args.args[0]
        self.assertNotIn(inference["prompt"], str(messages))
        self.assertNotIn(record["target"], str(messages))
        answer = json.loads(tokenizer.encode.call_args.args[0])
        self.assertEqual(answer, {"corrected_sentence": record["target"]})
        self.assertEqual((inference, record), before)
        self.assertEqual(encoded["labels"], [-100, -100, 3, 4, 99])

    def test_complete_targets_and_inference_keep_the_standard_prompt(self):
        from model.correction import training_record_inference

        inference = {
            "prompt": "full instruction",
            "output_format": "sentence",
            "hint_mode": "none",
        }
        config = {"training": {"reference_format_conditioning": "separate_abbreviated"}}
        r = {"target": "complete fixture."}
        self.assertIs(training_record_inference(r, inference, config), inference)
        self.assertIs(
            training_record_inference(
                {"target": "fragment ..."}, inference, {"training": {}}
            ),
            inference,
        )

    def test_unknown_policy_or_edits_output_fails(self):
        from model.correction import training_record_inference

        for policy, inference in [
            ("unknown", {"output_format": "sentence", "hint_mode": "none"}),
            ("separate_abbreviated", {"output_format": "edits", "hint_mode": "none"}),
        ]:
            with self.assertRaises(ValueError):
                training_record_inference(
                    {"target": "fragment ..."},
                    inference,
                    {"training": {"reference_format_conditioning": policy}},
                )
