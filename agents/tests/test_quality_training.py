import unittest
from collections import Counter


class QualityTrainingTests(unittest.TestCase):
    def test_real_bundle_round_trip_preserves_validation_and_rejects_resealed_injection(
        self,
    ):
        import copy
        from agents.tests.test_grouped_validation import grouped_packet
        from model.quality_data import seal, load_quality
        from model.retained_rules import digest

        source, _ = grouped_packet()
        bundle = seal(source)
        train, validation, info = load_quality(bundle, 0.2, 42)
        self.assertEqual(
            [r["id"] for r in validation], [r["ID"] for r in source["validation"]]
        )
        self.assertTrue(info["leakage_audit"]["passed"])
        self.assertTrue(train)
        broken = copy.deepcopy(bundle)
        broken["train"].append(source["validation"][0])
        broken["content_sha256"] = digest(
            {k: v for k, v in broken.items() if k != "content_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "deterministic policy"):
            load_quality(broken, 0.2, 42)

    def test_mechanism_cap_original_retention_controls_and_repeat_caps(self):
        from model.quality_data import curate

        rows = [
            {
                "ID": i,
                "Input": f"original {i}",
                "Output": {"Error_Type_Id": i, "Fixed Sentence": f"target {i}"},
            }
            for i in range(5)
        ]
        rows += [
            {
                "ID": f"balanced:{i}",
                "Input": f"error {i}",
                "Output": {"Error_Type_Id": 2, "Fixed Sentence": f"complete {i}."},
                "Provenance": {"template_rule": "one_rule", "parent_ids": [i + 100]},
            }
            for i in range(20)
        ]
        selected, train, report = curate(rows, "classification")
        self.assertTrue(all(r in selected for r in rows[:5]))
        self.assertLessEqual(
            sum(str(r["ID"]).startswith("balanced:") for r in selected), 8
        )
        controls = [r for r in selected if str(r["ID"]).startswith("control:")]
        self.assertTrue(controls)
        for r in controls:
            self.assertEqual(r["Input"], r["Output"]["Fixed Sentence"])
            self.assertIn(
                r["Input"],
                {
                    x["Output"]["Fixed Sentence"]
                    for x in selected
                    if str(x["ID"]).startswith("balanced:")
                },
            )
        counts = Counter(
            r.get("Provenance", {}).get("resampling", {}).get("source_id", r["ID"])
            for r in train
        )
        self.assertTrue(
            all(
                counts[r["ID"]] <= 2
                for r in selected
                if str(r["ID"]).startswith("balanced:")
            )
        )
        self.assertEqual(curate(rows, "classification"), (selected, train, report))

    def test_abbreviated_correction_targets_are_never_oversampled(self):
        from model.quality_data import curate

        rows = [
            {
                "ID": i,
                "Input": f"clean {i}.",
                "Output": {"Error_Type_Id": 0, "Fixed Sentence": f"clean {i}."},
            }
            for i in range(5)
        ]
        rows += [
            {
                "ID": 20,
                "Input": "error .",
                "Output": {"Error_Type_Id": 2, "Fixed Sentence": "partial..."},
            }
        ]
        _, train, _ = curate(rows, "correction")
        self.assertEqual(
            sum(
                r["ID"] == 20
                or r.get("Provenance", {}).get("resampling", {}).get("source_id") == 20
                for r in train
            ),
            1,
        )

    def test_calibration_is_applied_to_real_voting_probabilities(self):
        from model.classifier_ensemble import vote

        predictions = [
            [{"id": 1, "label": max(range(5), key=p.__getitem__), "probabilities": p}]
            for p in [
                [0.6, 0.1, 0.1, 0.1, 0.1],
                [0.6, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.05, 0.05],
            ]
        ]
        self.assertEqual(vote(predictions)[0]["label"], 0)
        got = vote(
            predictions,
            {"weights": [1, 1, 2], "temperature": 1.0, "phrase_scale": 1.25},
        )
        self.assertEqual(got[0]["label"], 1)
        self.assertAlmostEqual(sum(got[0]["probabilities"]), 1.0)
        with self.assertRaises(ValueError):
            vote(
                predictions,
                {"weights": [1, -1, 2], "temperature": 1.0, "phrase_scale": 1.25},
            )

    def test_private_inference_hash_is_enforced_and_corrector_selection_matches_objective(
        self,
    ):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from tools.run_experiment import resolve_inputs
        from model.correction import correction_selection_metrics
        from model.runtime import selection_key

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = [
                {
                    "ID": 1,
                    "Input": "fixture",
                    "Output": {"Error_Type_Id": 0, "Fixed Sentence": "fixture"},
                }
            ]
            for filename, data in [
                ("VGEC 2026_v1.json", original),
                ("VGEC_classifier.json", {}),
                ("VGEC_balanced.json", {}),
                ("private_test.json", [{"ID": 99, "Input": "unlabeled fixture"}]),
            ]:
                (root / filename).write_text(json.dumps(data))
            recipe = {
                "mode": "both",
                "data_archive": "missing.zip",
                "submission_filename": "private_test.json",
                "data_sha256": {},
                "submission_sha256": hashlib.sha256(
                    (root / "private_test.json").read_bytes()
                ).hexdigest(),
            }
            self.assertEqual(
                resolve_inputs(recipe, root, root / "unused")["submission"],
                root / "private_test.json",
            )
            (root / "private_test.json").write_text(
                json.dumps([{"ID": 98, "Input": "wrong fixture"}])
            )
            with self.assertRaisesRegex(ValueError, "private-test hash"):
                resolve_inputs(recipe, root, root / "unused")
        fields = correction_selection_metrics(
            {
                "evaluation": {
                    "selection_metrics": [
                        "corrections.all.gleu_plus",
                        "corrections.error_only.gleu_plus",
                        "clean_preservation.exact_source_preservation",
                    ]
                }
            }
        )
        better_all = {
            "corrections": {
                "all": {"gleu_plus": 0.7},
                "error_only": {"gleu_plus": 0.2},
            },
            "clean_preservation": {"exact_source_preservation": 1.0},
        }
        better_error = {
            "corrections": {
                "all": {"gleu_plus": 0.6},
                "error_only": {"gleu_plus": 0.3},
            },
            "clean_preservation": {"exact_source_preservation": 1.0},
        }
        self.assertGreater(
            selection_key(better_all, fields), selection_key(better_error, fields)
        )
        with self.assertRaises(ValueError):
            correction_selection_metrics(
                {"evaluation": {"selection_metrics": ["invented"]}}
            )

    def test_teacher_native_audit_resolves_lazy_import_before_meta_context(self):
        import tempfile
        from pathlib import Path
        from transformers import Qwen2Config
        from tools.diversity_teacher import native_parameter_count

        with tempfile.TemporaryDirectory() as folder:
            Qwen2Config(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            ).save_pretrained(folder)
            count = native_parameter_count({"name": folder, "revision": "main"})
            self.assertGreater(count, 0)
