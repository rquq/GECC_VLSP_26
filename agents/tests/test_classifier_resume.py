"""Partial recovery never treats an incomplete or changed classifier as completed."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from model.classifier_resume import (
    MANIFEST,
    package,
    materialize,
    plan,
    sha256,
    validate_files,
    resume_output,
)
from model.runtime import write_json


class ClassifierResumeTests(unittest.TestCase):
    def fixture(self, root):
        from agents.tests.test_voting_classifier import EnsembleArtifactTests

        source = root / "source"
        config = EnsembleArtifactTests().fixture(source / "classification")
        shutil.rmtree(source / "classification/members/mistral7b")
        original, mixed = root / "original.json", root / "mixed.json"
        original.write_text("original")
        mixed.write_text("mixed")
        for name in ["qwen7b", "sailor7b"]:
            p = source / "classification/members" / name / "run_manifest.json"
            m = json.loads(p.read_text())
            m.update(data_sha256=sha256(original), classifier_data_sha256=sha256(mixed))
            write_json(p, m)
        data = root / "data.zip"
        with zipfile.ZipFile(data, "w") as archive:
            for name in [
                "VGEC 2026_v1.json",
                "VGEC_classifier.json",
                "VGEC_balanced.json",
                "public_test.json",
            ]:
                archive.writestr(name, "[]")
        archive = package(source, data, root / "packet")
        return source, config, original, mixed, archive

    def test_zip_and_extracted_packets_reuse_only_completed_members(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, config, original, mixed, archive = self.fixture(root)
            digest = sha256(root / "packet" / MANIFEST)
            mounted = root / "input"
            mounted.mkdir()
            shutil.copy2(archive, mounted / archive.name)
            output = materialize(mounted, archive.name, digest, root / "output")
            self.assertEqual(validate_files(output)["members"], ["qwen7b", "sailor7b"])
            self.assertFalse((output / "classification/members/mistral7b").exists())
            self.assertEqual(resume_output(output), output.resolve())
            (output / "unexpected.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "unexpected/prior-run"):
                resume_output(output)
            (output / "unexpected.json").unlink()
            weights = (
                "classification/members/qwen7b/classifier/adapter_model.safetensors"
            )
            self.assertEqual(sha256(source / weights), sha256(output / weights))
            (mounted / archive.name).unlink()
            shutil.copytree(root / "packet", mounted / "arbitrary-upload-name")
            materialize(mounted, archive.name, digest, root / "extracted")
            self.assertEqual(
                sha256(output / weights), sha256(root / "extracted" / weights)
            )

    def test_original_notebook_output_can_be_reused_without_packet_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, _, _, _, archive = self.fixture(root)
            packet = json.loads((root / "packet" / MANIFEST).read_text())
            digest = sha256(root / "packet" / MANIFEST)
            proof = {
                **packet,
                "packet_manifest_sha256": digest,
                "file_sha256": {
                    name: sha256(source / name) for name in packet["file_sha256"]
                },
            }
            proof_path = root / "expected-source.json"
            write_json(proof_path, proof)
            inputs = root / "kaggle/input"
            mounted = (
                inputs / "notebooks/some-user/odd-version/outputs/runs/previous run/run"
            )
            shutil.copytree(source, mounted)
            output = materialize(
                inputs,
                archive.name,
                digest,
                root / "raw-output",
                source_manifest=proof_path,
            )
            self.assertEqual(validate_files(output)["members"], ["qwen7b", "sailor7b"])
            resume_output(output)
            retry = materialize(
                output,
                archive.name,
                digest,
                root / "retry-output",
                source_manifest=proof_path,
            )
            self.assertEqual(validate_files(retry)["members"], ["qwen7b", "sailor7b"])
            weight = (
                "classification/members/qwen7b/classifier/adapter_model.safetensors"
            )
            self.assertEqual(sha256(source / weight), sha256(output / weight))
            duplicate = inputs / "another-version/run"
            shutil.copytree(mounted, duplicate)
            with self.assertRaisesRegex(ValueError, "exactly one original"):
                materialize(
                    inputs,
                    archive.name,
                    digest,
                    root / "ambiguous",
                    source_manifest=proof_path,
                )
            shutil.rmtree(duplicate)
            (mounted / weight).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                materialize(
                    inputs,
                    archive.name,
                    digest,
                    root / "changed-output",
                    source_manifest=proof_path,
                )

    def test_changed_manifest_weights_and_escaping_members_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, _, _, _, archive = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                materialize(root, archive.name, "0" * 64, root / "bad")
            packet = root / "packet"
            weight = (
                packet
                / "classification/members/qwen7b/classifier/adapter_model.safetensors"
            )
            weight.write_bytes(weight.read_bytes() + b"bad")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                validate_files(packet)
            manifest = json.loads((packet / MANIFEST).read_text())
            manifest["file_sha256"] = {"../escape": "0" * 64}
            write_json(packet / MANIFEST, manifest)
            with self.assertRaisesRegex(ValueError, "path/hash"):
                validate_files(packet)

    def test_plan_requires_identical_rows_recipe_and_pinned_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, config, original, mixed, _ = self.fixture(root)
            packet = root / "packet"
            member = packet / "classification/members/qwen7b"
            train = json.loads((member / "train.json").read_text())
            validation = json.loads((member / "validation.json").read_text())
            # Dataset decoding has separate coverage; here vary the requested
            # population/config while preserving valid packet hashes.
            with patch(
                "model.data.load_classifier_split", return_value=(train, {})
            ), patch(
                "model.classification._organizer_partition",
                return_value=([], validation),
            ):
                self.assertEqual(
                    list(plan(packet, config, original, mixed)), ["qwen7b", "sailor7b"]
                )
                changed = copy.deepcopy(config)
                changed["training"]["max_length"] += 1
                with self.assertRaisesRegex(ValueError, "model/data/recipe"):
                    plan(packet, changed, original, mixed)
                changed = copy.deepcopy(config)
                changed["ensemble"]["members"][0]["model"]["revision"] = "changed"
                with self.assertRaisesRegex(ValueError, "model/data/recipe"):
                    plan(packet, changed, original, mixed)
                with patch(
                    "model.data.load_classifier_split", return_value=(train[:1], {})
                ):
                    with self.assertRaisesRegex(ValueError, "model/data/recipe"):
                        plan(packet, config, original, mixed)
