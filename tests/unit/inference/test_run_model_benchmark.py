"""Identity-collection contracts for model benchmark inputs; stdlib only."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import run_model_benchmark as runner


class ModelBenchmarkIdentity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "model"
        self.root.mkdir()
        (self.root / "config.json").write_text("{}")

    def write(self, name, data=b"weights"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def index(self, weight_map):
        self.write("model.safetensors.index.json", json.dumps({"weight_map": weight_map}).encode())

    def records(self):
        return {item["path"]: item for item in runner.model_identity(self.root)}

    def test_flat_model_files_have_actual_payload_identities(self):
        self.write("model.safetensors", b"known weights")
        record = self.records()["model.safetensors"]
        self.assertEqual(record["bytes"], 13)
        self.assertEqual(record["sha256"], hashlib.sha256(b"known weights").hexdigest())

    def test_nested_indexed_shard_is_collected_without_a_root_weight_file(self):
        self.write("weights/part-1.safetensors")
        self.index({"model.weight": "weights/part-1.safetensors"})
        self.assertEqual(set(self.records()),
                         {"config.json", "model.safetensors.index.json", "weights/part-1.safetensors"})

    def test_unused_root_file_cannot_hide_a_consumed_nested_shard(self):
        self.write("unused.safetensors")
        self.write("weights/actual.safetensors", b"actual data")
        self.index({"model.weight": "weights/actual.safetensors"})
        records = self.records()
        self.assertIn("unused.safetensors", records)
        self.assertEqual(records["weights/actual.safetensors"]["bytes"], 11)

    def test_changed_nested_payload_changes_the_identity(self):
        actual = self.write("weights/actual.safetensors", b"before")
        self.index({"model.weight": "weights/actual.safetensors"})
        before = self.records()
        actual.write_bytes(b"after!")
        after = self.records()
        self.assertNotEqual(before["weights/actual.safetensors"]["sha256"],
                            after["weights/actual.safetensors"]["sha256"])

    def test_shared_shard_is_recorded_once(self):
        self.write("weights/part.safetensors")
        self.index({"first": "weights/part.safetensors", "second": "weights/part.safetensors"})
        self.assertEqual(list(self.records()).count("weights/part.safetensors"), 1)

    def test_missing_indexed_shard_is_rejected_even_with_unused_weights(self):
        self.write("unused.safetensors")
        self.index({"weight": "missing/part.safetensors"})
        with self.assertRaises((OSError, ValueError)):
            self.records()

    def test_parent_traversal_is_rejected(self):
        self.write("unused.safetensors")
        self.index({"weight": "../outside.safetensors"})
        with self.assertRaises(ValueError):
            self.records()

    def test_absolute_shard_is_rejected(self):
        outside = Path(self.temporary.name) / "outside.safetensors"
        outside.write_bytes(b"outside")
        self.index({"weight": str(outside)})
        with self.assertRaises(ValueError):
            self.records()

    def test_file_symlink_is_rejected(self):
        target = self.write("weights/part.safetensors")
        (self.root / "alias.safetensors").symlink_to(target)
        with self.assertRaises(ValueError):
            self.records()

    def test_directory_symlink_to_an_internal_shard_is_rejected(self):
        self.write("weights/part.safetensors")
        (self.root / "alias").symlink_to(self.root / "weights", target_is_directory=True)
        self.index({"weight": "alias/part.safetensors"})
        with self.assertRaises(ValueError):
            self.records()

    def test_duplicate_index_json_is_rejected(self):
        self.write("model.safetensors")
        self.write("model.safetensors.index.json",
                   b'{"weight_map":{"weight":"model.safetensors","weight":"other.safetensors"}}')
        with self.assertRaises(ValueError):
            self.records()

    def test_uri_shard_is_rejected(self):
        self.write("unused.safetensors")
        self.index({"weight": "https://example.test/weights.safetensors"})
        with self.assertRaises(ValueError):
            self.records()

    def test_empty_index_weight_map_is_rejected(self):
        self.write("unused.safetensors")
        self.index({})
        with self.assertRaises(ValueError):
            self.records()


if __name__ == "__main__":
    unittest.main()
