"""Tests for offline fixture conversion; no third-party Python packages required."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "admission_fixture_generator", ROOT / "tools/generate_fp32_admission_hf_fixtures.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATOR
SPEC.loader.exec_module(GENERATOR)


def source_config():
    return {"model_type": "llama", "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4, "intermediate_size": 8, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
            "vocab_size": 8, "max_position_embeddings": 32, "hidden_act": "silu",
            "tie_word_embeddings": False, "dtype": "float32", "torch_dtype": "bfloat16"}


def tensor_records(config):
    records = {"model.embed_tokens.weight": [8, 4], "lm_head.weight": [8, 4],
               "model.norm.weight": [4]}
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        records[prefix + "input_layernorm.weight"] = [4]
        records[prefix + "post_attention_layernorm.weight"] = [4]
        for name, shape in (("q", [4, 4]), ("k", [2, 4]), ("v", [2, 4]), ("o", [4, 4])):
            records[prefix + f"self_attn.{name}_proj.weight"] = shape
        for name, shape in (("gate", [8, 4]), ("up", [8, 4]), ("down", [4, 8])):
            records[prefix + f"mlp.{name}_proj.weight"] = shape
    return {name: ("F32", shape, b"".join(struct.pack("<f", ((i % 7) - 3) / 8)
                                       for i in range(math.prod(shape))))
            for name, shape in records.items()}


def safetensors_bytes(records):
    header = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, (dtype, shape, values) in sorted(records.items()):
        start = len(payload)
        payload.extend(values)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


def independent_read(path):
    data = path.read_bytes()
    size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + size])
    payload = data[8 + size:]
    result = {}
    cursor = 0
    for name in sorted(key for key in header if key != "__metadata__"):
        record = header[name]
        start, end = record["data_offsets"]
        if start != cursor or end > len(payload):
            raise AssertionError("Invalid independent tensor offsets")
        size_per_element = {"F32": 4, "F16": 2, "BF16": 2}[record["dtype"]]
        if end - start != math.prod(record["shape"]) * size_per_element:
            raise AssertionError("Invalid independent tensor bytes")
        result[name] = (record["dtype"], record["shape"], payload[start:end])
        cursor = end
    if cursor != len(payload):
        raise AssertionError("Unclaimed independent tensor bytes")
    return result


class ConversionTests(unittest.TestCase):
    def test_signed_zero_and_exact_values(self):
        payload = struct.pack("<4f", 0.0, -0.0, 1.0, -2.0)
        self.assertEqual(GENERATOR.convert_f32(payload, "F32"), payload)
        self.assertEqual(GENERATOR.convert_f32(payload, "F16"), bytes.fromhex("00000080003c00c0"))
        self.assertEqual(GENERATOR.convert_f32(payload, "BF16"), bytes.fromhex("00000080803f00c0"))

    def test_f16_rounds_halfway_values_to_even(self):
        values = (1 + 2 ** -11, 1 + 3 * 2 ** -11, -1 - 2 ** -11, -1 - 3 * 2 ** -11)
        self.assertEqual(struct.unpack("<4H", GENERATOR.convert_f32(struct.pack("<4f", *values), "F16")),
                         (0x3c00, 0x3c02, 0xbc00, 0xbc02))

    def test_bf16_rounds_halfway_values_to_even(self):
        payload = struct.pack("<4I", 0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000)
        self.assertEqual(struct.unpack("<4H", GENERATOR.convert_f32(payload, "BF16")),
                         (0x3f80, 0x3f82, 0xbf80, 0xbf82))

    def test_rejects_nonfinite_inputs_for_every_dtype(self):
        for bits in (0x7f800000, 0xff800000, 0x7fc00000):
            for dtype in ("F32", "F16", "BF16"):
                with self.subTest(bits=bits, dtype=dtype), self.assertRaises(GENERATOR.FixtureError):
                    GENERATOR.convert_f32(struct.pack("<I", bits), dtype)

    def test_rejects_f16_overflow_and_bf16_rounding_to_infinity(self):
        for dtype, bits in (("F16", 0x47800000), ("F16", 0xc7800000),
                            ("BF16", 0x7f7fffff), ("BF16", 0xff7fffff)):
            with self.subTest(dtype=dtype, bits=bits), self.assertRaises(GENERATOR.FixtureError):
                GENERATOR.convert_f32(struct.pack("<I", bits), dtype)

    def test_retains_f16_subnormal_values(self):
        self.assertEqual(GENERATOR.convert_f32(struct.pack("<f", 2 ** -24), "F16"), b"\x01\x00")


class SafetensorsValidationTests(unittest.TestCase):
    def test_rejects_truncated_or_invalid_header(self):
        for data in (b"", b"\x00" * 7, struct.pack("<Q", 99),
                     struct.pack("<Q", 8) + b"null    "):
            with self.subTest(data=data), self.assertRaises(GENERATOR.FixtureError):
                GENERATOR.read_safetensors(data)

    def test_rejects_duplicate_json_keys(self):
        with self.assertRaises(GENERATOR.FixtureError):
            GENERATOR.parse_json(b'{"x":1,"x":2}')

    def test_rejects_overlap_gap_shape_and_unclaimed_bytes(self):
        valid = safetensors_bytes({"a": ("F32", [1], b"\x00" * 4),
                                  "b": ("F32", [1], b"\x00" * 4)})
        length = struct.unpack_from("<Q", valid)[0]
        original = json.loads(valid[8:8 + length])
        payload = valid[8 + length:]
        changes = (("data_offsets", [0, 4]), ("data_offsets", [5, 9]), ("shape", [2]),
                   ("data_offsets", [-1, 3]), ("dtype", "I32"), ("shape", [True]))
        for key, value in changes:
            header = copy.deepcopy(original)
            header["b"][key] = value
            encoded = json.dumps(header).encode()
            encoded += b" " * (-len(encoded) % 8)
            with self.subTest(key=key, value=value), self.assertRaises(GENERATOR.FixtureError):
                GENERATOR.read_safetensors(struct.pack("<Q", len(encoded)) + encoded + payload)
        with self.assertRaises(GENERATOR.FixtureError):
            GENERATOR.read_safetensors(valid + b"trailing")


class FixtureGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="admission-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.config = source_config()
        self.records = tensor_records(self.config)
        self.recipe = json.loads((ROOT / "tests/fixtures/inference_admission/recipe.json").read_text())
        self.recipe["source"]["repository"] = "unit-test/synthetic-llama"
        self.recipe["source"]["revision"] = "0" * 40
        self.recipe_path = self.root / "recipe.json"
        self.output = self.root / "output"
        self.expected = self.root / "checksums.json"
        self.write_source()

    def write_source(self):
        (self.source / "config.json").write_text(json.dumps(self.config))
        (self.source / "model.safetensors").write_bytes(safetensors_bytes(self.records))
        self.recipe["source"]["files"] = {
            name: {"bytes": (self.source / name).stat().st_size,
                   "sha256": hashlib.sha256((self.source / name).read_bytes()).hexdigest()}
            for name in ("config.json", "model.safetensors")}
        self.recipe_path.write_text(json.dumps(self.recipe))

    def generate(self):
        manifest = GENERATOR.generate(self.source, self.recipe_path, self.output)
        self.expected.write_bytes(GENERATOR.json_bytes(manifest))
        return manifest

    def snapshot(self, root):
        return {str(path.relative_to(root)): (path.stat().st_mtime_ns, path.read_bytes())
                for path in root.rglob("*") if path.is_file()}

    def test_variants_have_real_dtype_payload_and_preserved_schema(self):
        manifest = self.generate()
        self.assertEqual(len(manifest["variants"]), 20)
        self.assertNotIn(str(self.root), json.dumps(manifest))
        for spec in self.recipe["variants"]:
            with self.subTest(variant=spec["name"]):
                path = self.output / spec["name"]
                config = json.loads((path / "config.json").read_bytes())
                self.assertNotIn("dtype", config)
                if spec["hint"] is None:
                    self.assertNotIn("torch_dtype", config)
                else:
                    self.assertEqual(config["torch_dtype"], spec["hint"])
                actual = independent_read(path / "model.safetensors")
                expected_names = set(self.records) - ({"lm_head.weight"} if spec["tied"] else set())
                self.assertEqual(set(actual), expected_names)
                self.assertEqual(config["tie_word_embeddings"], spec["tied"])
                for name, (dtype, shape, payload) in actual.items():
                    expected_dtype = spec["dtype"]
                    if spec["norms"] == "all" and not (
                            name == "model.norm.weight" or name.endswith("layernorm.weight")):
                        expected_dtype = "F32"
                    if spec["norms"] == "last-post-attention" and name != \
                            "model.layers.1.post_attention_layernorm.weight":
                        expected_dtype = "F32"
                    self.assertEqual(dtype, expected_dtype)
                    self.assertEqual(shape, self.records[name][1])
                    source_values = struct.unpack("<" + "f" * math.prod(shape), self.records[name][2])
                    if dtype == "BF16":
                        values = tuple(struct.unpack("<f", struct.pack("<I", bits << 16))[0]
                                       for (bits,) in struct.iter_unpack("<H", payload))
                    else:
                        values = tuple(value for (value,) in struct.iter_unpack("<f" if dtype == "F32" else "<e", payload))
                    self.assertEqual(values, source_values)

    def test_check_is_read_only_and_regeneration_is_repeatable(self):
        self.generate()
        before = self.snapshot(self.root)
        GENERATOR.check(self.source, self.recipe_path, self.output, self.expected)
        self.assertEqual(self.snapshot(self.root), before)
        GENERATOR.generate(self.source, self.recipe_path, self.root / "second-output")
        first_bytes = {name: data for name, (_, data) in self.snapshot(self.output).items()}
        second_bytes = {name: data for name, (_, data) in self.snapshot(self.root / "second-output").items()}
        self.assertEqual(first_bytes, second_bytes)

    def test_python_environment_provenance_does_not_change_content_checks(self):
        manifest = self.generate()
        manifest["python"]["version"] = "3.12.999"
        self.expected.write_bytes(GENERATOR.json_bytes(manifest))
        GENERATOR.check(self.source, self.recipe_path, self.output, self.expected)

    def test_check_rejects_tampered_payload_without_repairing(self):
        self.generate()
        path = self.output / "f16-auto/model.safetensors"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        before = self.snapshot(self.root)
        with self.assertRaisesRegex(GENERATOR.FixtureError, "differs"):
            GENERATOR.check(self.source, self.recipe_path, self.output, self.expected)
        self.assertEqual(self.snapshot(self.root), before)

    def test_source_identity_and_recipe_revision_are_enforced(self):
        self.generate()
        (self.source / "config.json").write_bytes(b"{}")
        with self.assertRaisesRegex(GENERATOR.FixtureError, "identity"):
            GENERATOR.check(self.source, self.recipe_path, self.output, self.expected)
        self.recipe["source"]["revision"] = "main"
        self.recipe_path.write_text(json.dumps(self.recipe))
        with self.assertRaisesRegex(GENERATOR.FixtureError, "revision"):
            GENERATOR.load_recipe(self.recipe_path)

    def test_check_rejects_rule_identity_drift(self):
        self.generate()
        self.recipe_path.write_bytes(self.recipe_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(GENERATOR.FixtureError, "identity"):
            GENERATOR.check(self.source, self.recipe_path, self.output, self.expected)

    def test_source_must_be_complete_f32_untied_and_unsharded(self):
        for kind in ("missing", "dtype", "shape", "nonfinite", "tied", "sharded"):
            with self.subTest(kind=kind):
                records = tensor_records(self.config)
                self.config["tie_word_embeddings"] = False
                if kind == "missing":
                    records.pop("model.layers.1.post_attention_layernorm.weight")
                elif kind == "dtype":
                    records["model.norm.weight"] = ("F16", [4], b"\x00" * 8)
                elif kind == "shape":
                    records["model.norm.weight"] = ("F32", [2, 2], b"\x00" * 16)
                elif kind == "nonfinite":
                    records["model.norm.weight"] = ("F32", [4], struct.pack("<4f", 1, 2, 3, float("inf")))
                elif kind == "tied":
                    self.config["tie_word_embeddings"] = True
                self.records = records
                self.write_source()
                extra = self.source / "model.safetensors.index.json"
                if kind == "sharded":
                    extra.write_text("{}")
                with self.assertRaises(GENERATOR.FixtureError):
                    GENERATOR.generate(self.source, self.recipe_path, self.output)
                self.assertFalse(self.output.exists())
                extra.unlink(missing_ok=True)

    def test_protects_source_recipe_checksums_and_repository_sources(self):
        for output in (self.source, self.source / "nested", self.root, self.recipe_path,
                       ROOT / "src/generated-admission"):
            with self.subTest(output=output), self.assertRaises(GENERATOR.FixtureError):
                GENERATOR.generate(self.source, self.recipe_path, output)
        self.generate()
        with self.assertRaises(GENERATOR.FixtureError):
            GENERATOR.check(self.source, self.recipe_path, self.output,
                            self.output / "generated-manifest.json")

    def test_rejects_output_symlinks_and_unexpected_files(self):
        self.output.mkdir()
        (self.output / "keep.txt").write_text("preserve")
        with self.assertRaises(GENERATOR.FixtureError):
            GENERATOR.generate(self.source, self.recipe_path, self.output)
        (self.output / "keep.txt").unlink()
        (self.output / "f32-auto").symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(GENERATOR.FixtureError):
            GENERATOR.generate(self.source, self.recipe_path, self.output)
        self.assertTrue((self.source / "config.json").is_file())

    def test_protects_source_files_resolved_through_symlinks(self):
        self.output.mkdir()
        for name in ("config.json", "model.safetensors"):
            with self.subTest(name=name):
                source_file = self.source / name
                original = source_file.read_bytes()
                target = self.output / name
                target.write_bytes(original)
                source_file.unlink()
                source_file.symlink_to(target)
                with self.assertRaisesRegex(GENERATOR.FixtureError, "overlaps"):
                    GENERATOR.validate_paths(self.source, self.recipe_path, self.output)
                self.assertEqual(source_file.read_bytes(), original)
                source_file.unlink()
                source_file.write_bytes(original)

    def test_cli_reports_missing_source_as_failure(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/generate_fp32_admission_hf_fixtures.py"),
                                 "--source", str(self.root / "missing"), "--recipe", str(self.recipe_path),
                                 "--output-dir", str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("fixture error", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
