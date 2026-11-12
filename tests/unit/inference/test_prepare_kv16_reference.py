"""The KV16 oracle import never silently accepts absent or changed assets."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("prepare_kv16_reference", ROOT / "tools/prepare_kv16_reference.py")
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class Kv16ReferenceIdentity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.bundle = self.base / "bundle"
        self.bundle.mkdir()
        self.payload = self.bundle / "oracle.safetensors"
        self.payload.write_bytes(b"reviewed independent reference")
        record = {"schema_version": 1, "files": {self.payload.name: PREPARE.identity(self.payload)}}
        self.expected = self.base / "reviewed.json"
        self.expected.write_text(json.dumps(record))
        (self.bundle / "manifest.json").write_text(json.dumps(record))

    def testReviewedFilesPass(self):
        self.assertEqual(PREPARE.verify(self.bundle, self.expected)["files"], 1)

    def testSameSizePayloadMutationFails(self):
        data = bytearray(self.payload.read_bytes())
        data[-1] ^= 1
        self.payload.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            PREPARE.verify(self.bundle, self.expected)

    def testMissingOracleFails(self):
        self.payload.unlink()
        with self.assertRaisesRegex(ValueError, "file set mismatch"):
            PREPARE.verify(self.bundle, self.expected)

    def testExtraOracleFails(self):
        (self.bundle / "unreviewed.safetensors").write_bytes(b"unreviewed")
        with self.assertRaisesRegex(ValueError, "file set mismatch"):
            PREPARE.verify(self.bundle, self.expected)

    def testExtraNestedManifestFails(self):
        (self.bundle / "unreviewed").mkdir()
        (self.bundle / "unreviewed/manifest.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "file set mismatch"):
            PREPARE.verify(self.bundle, self.expected)

    def testChangedManifestCannotAcceptChangedPayload(self):
        self.payload.write_bytes(b"unreviewed replacement")
        record = {"schema_version": 1, "files": {self.payload.name: PREPARE.identity(self.payload)}}
        (self.bundle / "manifest.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "reviewed expected identity"):
            PREPARE.verify(self.bundle, self.expected)


if __name__ == "__main__":
    unittest.main()
