"""Legacy nested ZIPs migrate to plain JSON; JSON and weight hashes remain strict."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.run_focused import materialize_package


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ExpandedControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "odd mount/dataset/version"
        self.inputs.mkdir(parents=True)
        self.control = self.inputs / "baseline/quality_control.zip"
        self.control.parent.mkdir()
        self.payload = b'[{"ID":1,"Output":{"Error_Type_Id":0,"Fixed Sentence":["fixture"]}}]\n'
        info = zipfile.ZipInfo("results.json", (2026, 10, 17, 16, 50, 18))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        with zipfile.ZipFile(self.control, "w") as z:
            z.writestr(info, self.payload)
        self.original = self.control.read_bytes()
        self.proof = {"zip_sha256": sha(self.original), "json_sha256": sha(self.payload),
                      "date_time": list(info.date_time), "external_attr": info.external_attr,
                      "create_system": info.create_system}
        (self.inputs / "weights").write_bytes(b"unchanged artificial weights")
        manifest = {"format": "vgec-focused-package-v1", "file_sha256": {
            "weights": sha((self.inputs / "weights").read_bytes()),
            "baseline/quality_control.zip": sha(self.original)}}
        raw = json.dumps(manifest).encode()
        (self.inputs / "focused_package_manifest.json").write_bytes(raw)
        self.manifest_sha = sha(raw)

    def expand(self, location="baseline/quality_control/results.json"):
        self.control.unlink()
        payload = self.inputs / location
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(self.payload)
        return payload

    def materialize(self):
        return materialize_package(self.root / "odd mount", self.root / "output", "upload.zip",
                                   "f" * 64, legacy_manifest_sha256=self.manifest_sha,
                                   legacy_control_json_sha256=sha(self.payload))

    def test_expanded_archive_becomes_verified_json_without_mutating_input(self):
        self.expand()
        root, _ = self.materialize()
        self.assertEqual((root / "baseline/quality_control/results.json").read_bytes(), self.payload)
        self.assertFalse(self.control.exists())
        self.assertEqual((root / "weights").read_bytes(), (self.inputs / "weights").read_bytes())

    def test_sibling_expansion_is_supported(self):
        self.expand("baseline/results.json")
        root, _ = self.materialize()
        self.assertEqual((root / "baseline/quality_control/results.json").read_bytes(), self.payload)

    def test_arbitrarily_nested_expansion_is_supported(self):
        self.expand("baseline/quality_control/quality_control/another/results.json")
        root, _ = self.materialize()
        self.assertEqual((root / "baseline/quality_control/results.json").read_bytes(), self.payload)

    def test_modified_payload_is_rejected(self):
        self.expand().write_bytes(self.payload + b" ")
        with self.assertRaisesRegex(ValueError, "payload"):
            self.materialize()

    def test_modified_other_weight_is_rejected(self):
        self.expand()
        (self.inputs / "weights").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.materialize()

    def test_irrelevant_zip_container_bytes_are_not_hashed(self):
        self.expand()
        self.control.write_bytes(b"container bytes are irrelevant to verified JSON")
        root, manifest = self.materialize()
        self.assertEqual((root / "baseline/quality_control/results.json").read_bytes(), self.payload)
        self.assertFalse((root / "baseline/quality_control.zip").exists())
        self.assertNotIn("baseline/quality_control.zip", manifest["file_sha256"])
        self.assertFalse(manifest["upload_migration"]["nested_archive_hash_checked"])

    def test_old_outer_archive_uses_json_even_if_inner_zip_container_changed(self):
        with zipfile.ZipFile(self.control, "w", zipfile.ZIP_STORED) as control:
            control.writestr("results.json", self.payload)
        self.assertNotEqual(sha(self.control.read_bytes()), sha(self.original))
        with zipfile.ZipFile(self.inputs / "upload.zip", "w") as archive:
            for name in ["focused_package_manifest.json", "weights", "baseline/quality_control.zip"]:
                archive.write(self.inputs / name, name)
        root, manifest = self.materialize()
        self.assertEqual((root / "baseline/quality_control/results.json").read_bytes(), self.payload)
        self.assertFalse((root / "baseline/quality_control.zip").exists())

    def test_modern_plain_json_package_needs_no_migration_or_zip(self):
        payload = self.expand()
        manifest_path = self.inputs / "focused_package_manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        del manifest["file_sha256"]["baseline/quality_control.zip"]
        manifest["file_sha256"]["baseline/quality_control/results.json"] = sha(self.payload)
        raw = json.dumps(manifest).encode()
        manifest_path.write_bytes(raw)
        root, actual = materialize_package(self.root / "odd mount", self.root / "output", "upload.zip", sha(raw))
        self.assertEqual(root, self.inputs)
        self.assertNotIn("upload_migration", actual)
        self.assertEqual(payload.read_bytes(), self.payload)

    def test_expanded_payload_symlink_cannot_escape_mount(self):
        payload = self.expand()
        outside = self.root / "outside.json"
        outside.write_bytes(self.payload)
        payload.unlink()
        payload.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "path"):
            self.materialize()


if __name__ == "__main__":
    unittest.main()
