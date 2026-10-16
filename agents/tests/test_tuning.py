import copy
import unittest


class TuningTests(unittest.TestCase):
    def test_uniform_reproduces_soft_vote_and_alignment(self):
        from model.tuning import calibrated_vote

        rows = [
            [{"id": 1, "label": 0, "probabilities": [0.4, 0.3, 0.1, 0.1, 0.1]}],
            [{"id": 1, "label": 1, "probabilities": [0.2, 0.5, 0.1, 0.1, 0.1]}],
            [{"id": 1, "label": 1, "probabilities": [0.3, 0.4, 0.1, 0.1, 0.1]}],
        ]
        got = calibrated_vote(rows, [1, 1, 1], 1, 1)
        self.assertEqual(got[0]["label"], 1)
        self.assertAlmostEqual(got[0]["probabilities"][1], 0.4)
        with self.assertRaises(ValueError):
            calibrated_vote(rows, [0, 0, 0], 1, 1)
        with self.assertRaises(ValueError):
            calibrated_vote(rows, [1, 1, 1], 0, 1)
        bad = copy.deepcopy(rows)
        bad[1][0]["id"] = "1"
        with self.assertRaises(ValueError):
            calibrated_vote(bad, [1, 1, 1], 1, 1)

    def test_strategy_matrix_covers_distinct_decoders_with_bounded_beams(self):
        from model.tuning import decoding_strategies

        strategies = decoding_strategies()
        self.assertEqual(len({s["name"] for s in strategies}), len(strategies))
        self.assertEqual(strategies[0]["name"], "greedy_reference")
        self.assertTrue(any(s.get("schema") for s in strategies))
        self.assertTrue(any(s.get("edit_rerank") for s in strategies))
        self.assertTrue(
            any(s["generation"].get("num_beam_groups", 1) > 1 for s in strategies)
        )
        self.assertTrue(any(s["generation"].get("do_sample") for s in strategies))
        self.assertTrue(
            all(s["generation"].get("num_beams", 1) <= 8 for s in strategies)
        )

    def test_screening_is_seeded_stratified_and_gold_is_stripped(self):
        from model.tuning import panel, inference_rows

        rows = [
            {"id": 10 * k + i, "source": "toy", "target": "reference", "label": k}
            for k in range(5)
            for i in range(5)
        ]
        self.assertEqual(panel(rows, 2, 42), panel(rows, 2, 42))
        self.assertEqual(len(panel(rows, 2, 42)), 10)
        self.assertEqual(set(inference_rows(rows)[0]), {"id", "source"})


class CacheIdentityTests(unittest.TestCase):
    def test_saved_prompt_changes_invalidate_decode_cache(self):
        import tempfile, json
        from pathlib import Path
        from tools.decoding_sweep import cache_identity

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "adapter_model.safetensors").write_bytes(b"weights")
            (root / "inference.json").write_text(json.dumps({"prompt": "first"}))
            first = cache_identity(root, [], False)
            (root / "inference.json").write_text(json.dumps({"prompt": "changed"}))
            self.assertNotEqual(first, cache_identity(root, [], False))
