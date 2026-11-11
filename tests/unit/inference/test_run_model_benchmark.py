"""Identity-collection and stub-engine command contracts for the model benchmark driver; stdlib only."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

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


def stub_engine(options):
    """Contract-valid engine document for the fixed 1,7,3 -> 7,8,9 workload."""
    context = len(options.prompt) + max(options.max_new_tokens - 1, 0)
    sample = {
        "index": 0,
        "preparation": {"load_ns": 100, "compile_ns": 200, "runtime_ns": 300, "prepare_ns": 400,
                        "conversion_ns": None if options.weights == "require-fp32" else 50},
        "memory": {"raw_unique_visible_bytes": 1000, "packed_unique_storage_bytes": 200,
                   "kv_capacity_bytes": 512 * context, "workspace_bytes": 512,
                   "process_peak_rss_bytes": 1048576},
        "internal": {
            "generated_tokens": [7, 8, 9], "stop_reason": "length", "final_watermark": 5,
            "request_setup_ns": 300, "first_token_request_ns": 1400,
            "prefill_execute_ns": 1000, "decode_setup_ns": 200,
            "decode_steps": [
                {"index": 0, "kv_before": 3, "kv_after": 4, "input_token": 7, "output_token": 8, "execute_ns": 2000},
                {"index": 1, "kv_before": 4, "kv_after": 5, "input_token": 8, "output_token": 9, "execute_ns": 2100},
            ],
            "allocator_requests": {"prefill_execute": None, "decode_execute": [None, None]},
        },
        "generate": {"tokens": [7, 8, 9], "stop_reason": "length", "duration_ns": 10000},
    }
    samples = []
    for index in range(options.repetitions):
        item = copy.deepcopy(sample)
        item["index"] = index
        samples.append(item)
    return {
        "schema_version": 1, "status": "complete", "mode": options.mode,
        "build": {"compiler": "GCC 14.2", "build_type": "Release", "sanitizer": "none",
                  "compiled_avx2_fma": True, "cxx_flags": "-O3"},
        "cpu": {"usable_features": ["avx2", "fma"], "effective_features": ["avx2", "fma"]},
        "geometry": {"hidden_size": 16, "intermediate_size": 37, "vocab_size": 64,
                     "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "head_dim": 4, "max_position_embeddings": 32},
        "workload": {"preset": options.preset, "weights": options.weights,
                     "prompt_tokens": options.prompt, "max_new_tokens": options.max_new_tokens,
                     "eos_token_id": options.eos_token, "context_capacity": context,
                     "warmup": options.warmup, "repetitions": options.repetitions},
        "samples": samples,
    }


class ModelBenchmarkDriverCommand(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.model = base / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        (self.model / "model.safetensors").write_bytes(b"weights")
        self.binary = base / "model-bench"
        self.binary.write_text("#!/bin/sh\n")
        self.binary.chmod(0o755)
        self.library = base / "libAetherMind.so"
        self.library.write_bytes(b"library")
        self.output = base / "results"
        machine = {"cpu_model": "test-cpu", "logical_cpus": 2, "os_release": "test Linux",
                   "kernel": "test-kernel", "virtualization": "test-vm",
                   "memory_total_bytes": 67108864, "governors": ["performance"],
                   "turbo": "unavailable", "numa_nodes": [0]}
        for target, value in (("source_snapshot", ("a" * 40, False, b"patch")),
                              ("linked_library", (self.library, "ldd output")),
                              ("machine_identity", machine)):
            patcher = mock.patch.object(runner, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        environment = mock.patch.dict(runner.os.environ,
                                      {key: "" for key in runner.ENVIRONMENT_KEYS}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def options(self, **changes):
        result = argparse.Namespace(
            binary=self.binary, model=self.model, prompt=[1, 7, 3], max_new_tokens=3,
            eos_token=None, preset="o1-plain", warmup=0, repetitions=1, mode="timing",
            weights="require-fp32", output_root=self.output, variant="", timeout_seconds=60)
        for name, value in changes.items():
            setattr(result, name, value)
        return result

    def run_driver(self, options):
        def launch(command, **kwargs):
            return subprocess.CompletedProcess(command, 0,
                                               stdout=json.dumps(stub_engine(options)), stderr="")
        with mock.patch.object(runner.subprocess, "run", side_effect=launch):
            return runner.run(options)

    def test_default_command_and_workload_record_require_fp32(self):
        path = self.run_driver(self.options())
        invocation = json.loads((path.parent / "invocation.json").read_text())
        command = invocation["engine_command"]
        self.assertEqual(command[command.index("--weights") + 1], "require-fp32")
        report = json.loads(path.read_text())
        self.assertEqual(report["provenance"]["command"], command)
        self.assertEqual(report["engine"]["workload"]["weights"], "require-fp32")
        self.assertIsNone(report["engine"]["samples"][0]["preparation"]["conversion_ns"])

    def test_keep_bf16_is_forwarded_and_recorded(self):
        path = self.run_driver(self.options(weights="keep-bf16"))
        invocation = json.loads((path.parent / "invocation.json").read_text())
        command = invocation["engine_command"]
        self.assertEqual(command[command.index("--weights") + 1], "keep-bf16")
        report = json.loads(path.read_text())
        self.assertEqual(report["provenance"]["command"], command)
        self.assertEqual(report["engine"]["workload"]["weights"], "keep-bf16")
        self.assertEqual(report["engine"]["samples"][0]["preparation"]["conversion_ns"], 50)


if __name__ == "__main__":
    unittest.main()
