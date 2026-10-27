"""Numerical asset checker contracts; no torch or downloaded assets required."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "llama_numerical_checker", ROOT / "tools/verify_llama_numerical_reference.py")
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)

CASES = ("mha-untied", "mha-tied", "gqa-tail", "mqa-tail", "mha-near-limit")
STAGES = ("prefill", "decode_0", "decode_1")
TOLERANCE = {"logits": [2e-5, 2e-4], "kv": [2e-6, 2e-4]}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def file_record(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def tensor(dtype, shape, values=None):
    count = math.prod(shape)
    if values is None:
        values = [0] * count
    assert len(values) == count
    return dtype, list(shape), struct.pack("<" + str(count) + ("f" if dtype == "F32" else "q"), *values)


def safetensors_bytes(tensors):
    header, payload = {"__metadata__": {"format": "pt"}}, bytearray()
    for name, (dtype, shape, data) in sorted(tensors.items()):
        begin = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [begin, len(payload)]}
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


def tensor_records(tensors):
    return {name: {"dtype": dtype, "shape": shape, "bytes": len(data),
                   "payload_sha256": hashlib.sha256(data).hexdigest()}
            for name, (dtype, shape, data) in tensors.items()}


def model_weights(geometry):
    h, intermediate, vocab = (geometry[key] for key in ("hidden_size", "intermediate_size", "vocab_size"))
    kv_width = geometry["num_key_value_heads"] * geometry["head_dim"]
    shapes = {"model.embed_tokens.weight": [vocab, h], "model.norm.weight": [h]}
    if not geometry["tie_word_embeddings"]:
        shapes["lm_head.weight"] = [vocab, h]
    for layer in range(geometry["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        shapes[prefix + "input_layernorm.weight"] = [h]
        shapes[prefix + "post_attention_layernorm.weight"] = [h]
        for name, shape in (("q", [h, h]), ("k", [kv_width, h]), ("v", [kv_width, h]), ("o", [h, h])):
            shapes[prefix + f"self_attn.{name}_proj.weight"] = shape
        for name, shape in (("gate", [intermediate, h]), ("up", [intermediate, h]), ("down", [h, intermediate])):
            shapes[prefix + f"mlp.{name}_proj.weight"] = shape
    return {name: tensor("F32", shape) for name, shape in shapes.items()}


def reference_tensors(geometry, trajectory):
    prompt = trajectory["prompt"]
    tensors = {"prompt": tensor("I64", [len(prompt)], prompt),
               "decode_inputs": tensor("I64", [2], [17, 23]),
               "greedy_tokens": tensor("I64", [3], trajectory["greedy_tokens"]),
               "greedy_margins": tensor("F32", [3], trajectory["greedy_margins"]),
               "tolerance.logits": tensor("F32", [2], [2e-5, 2e-4]),
               "tolerance.kv": tensor("F32", [2], [2e-6, 2e-4])}
    for index, stage in enumerate(STAGES):
        rows = trajectory["logit_rows"][stage]
        count = len(prompt) + index
        tensors[stage + ".logit_rows"] = tensor("I64", [len(rows)], rows)
        tensors[stage + ".logits"] = tensor("F32", [len(rows), geometry["vocab_size"]])
        tensors[stage + ".tokens"] = tensor("I64", [len(prompt) if index == 0 else 1])
        shape = [geometry["num_hidden_layers"], count, geometry["num_key_value_heads"], geometry["head_dim"]]
        tensors[stage + ".keys"] = tensor("F32", shape)
        tensors[stage + ".values"] = tensor("F32", shape)
        tensors[stage + ".position"] = tensor("I64", [1], [count])
    return tensors


def write_reference(root, case_id, record, tensors):
    case_root = root / case_id
    record["tensors"] = tensor_records(tensors)
    data = safetensors_bytes(tensors)
    (case_root / "reference.safetensors").write_bytes(data)
    without_files = {key: value for key, value in record.items() if key != "files"}
    metadata = json_bytes(without_files)
    (case_root / "reference.json").write_bytes(metadata)
    record["files"] = {"reference.safetensors": file_record(data), "reference.json": file_record(metadata)}


def make_bundle(root):
    root.mkdir()
    manifest = {
        "schema_version": 1,
        "generator": {"version": 1, "sha256": "1" * 64, "recipe_sha256": "2" * 64,
                      "requirements_sha256": "3" * 64},
        "reference": {"versions": {name: "test" for name in
                                   ("torch", "transformers", "numpy", "safetensors", "python", "python_implementation")},
                      "execution": {"device": "cpu", "dtype": "float32", "attention_implementation": "eager",
                                    "threads": 1, "interop_threads": 1, "deterministic_algorithms": True,
                                    "training": False, "gradient_mode": "inference_mode", "seed": 20261027,
                                    "environment": {"ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "COMPATIBLE",
                                                    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                                    "OPENBLAS_NUM_THREADS": "1"}, "torch_cpu_capability": "NO AVX"},
                      "source_files": {"torch/__init__.py": file_record(b"source")}},
        "admission": {"checksums_sha256": "4" * 64,
                      "source": {"repository": "test/tiny-llama", "revision": "5" * 40,
                                 "files": {"config.json": file_record(b"config"),
                                           "model.safetensors": file_record(b"weights")}}},
        "cases": {},
    }
    for case_id in CASES:
        is_gqa, is_mqa = case_id == "gqa-tail", case_id == "mqa-tail"
        qh, kvh = (4, 2) if is_gqa else (3, 1) if is_mqa else (2, 2)
        geometry = {"hidden_size": qh * 2, "intermediate_size": qh * 4,
                    "vocab_size": 31, "num_hidden_layers": 1, "num_attention_heads": qh,
                    "num_key_value_heads": kvh, "head_dim": 2, "max_position_embeddings": 2048,
                    "rms_norm_eps": 1e-5, "tie_word_embeddings": case_id == "mha-tied"}
        config = {**geometry, "model_type": "llama", "hidden_act": "silu", "rope_theta": 10000.0}
        count = 2046 if case_id == "mha-near-limit" else 3
        trajectory = {"prompt": ([1, 7, 3] * ((count + 2) // 3))[:count], "decode_inputs": [17, 23],
                      "positions": {"prefill": list(range(count)), "decode_0": [count], "decode_1": [count + 1]},
                      "logit_rows": {"prefill": [count - 1] if case_id == "mha-near-limit" else list(range(count)),
                                     "decode_0": [0], "decode_1": [0]},
                      "commit_positions": [count, count + 1, count + 2],
                      "greedy_tokens": [0, 0, 0], "greedy_margins": [0.0, 0.0, 0.0]}
        model_dir = root / case_id / "model"
        model_dir.mkdir(parents=True)
        config_data = json_bytes(config)
        weights = safetensors_bytes(model_weights(geometry))
        (model_dir / "config.json").write_bytes(config_data)
        (model_dir / "model.safetensors").write_bytes(weights)
        synthetic = is_gqa or is_mqa
        record = {"schema_version": 1, "id": case_id,
                  "kind": "synthetic_hf" if synthetic else "hf_admission",
                  "source_variant": None if synthetic else "f32-tied" if case_id == "mha-tied" else "f32-explicit",
                  "synthetic_seed": (20261027 if is_gqa else 20261028) if synthetic else None,
                  "geometry": geometry, "trajectory": trajectory, "tolerance": copy.deepcopy(TOLERANCE),
                  "layout": {"kv": "layer,position,kv_head,dim", "key": "post_rope", "value": "unrotated",
                             "logits": "selected_row,vocab", "rope": "split_half"},
                  "model_files": {"model/config.json": file_record(config_data),
                                  "model/model.safetensors": file_record(weights)}}
        write_reference(root, case_id, record, reference_tensors(geometry, trajectory))
        manifest["cases"][case_id] = record
    (root / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def load_tensors(path):
    data = path.read_bytes()
    length = struct.unpack_from("<Q", data)[0]
    header = json.loads(data[8:8 + length])
    payload = data[8 + length:]
    return {name: (entry["dtype"], entry["shape"], payload[slice(*entry["data_offsets"])])
            for name, entry in header.items() if name != "__metadata__"}


class LlamaNumericalReferenceChecker(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="aethermind-numerical-check-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "bundle"
        self.manifest = make_bundle(self.root)
        self.expected = self.directory / "reviewed.json"
        self.expected.write_bytes(json_bytes(self.manifest))

    def write_manifest(self, root=None, manifest=None):
        (root or self.root).joinpath("manifest.json").write_bytes(json_bytes(manifest or self.manifest))

    def rewrite_case(self, case_id="mha-untied", change=None, root=None, manifest=None):
        root = root or self.root
        manifest = manifest or self.manifest
        record = manifest["cases"][case_id]
        tensors = load_tensors(root / case_id / "reference.safetensors")
        if change:
            change(record, tensors)
        write_reference(root, case_id, record, tensors)
        self.write_manifest(root, manifest)

    def assert_invalid(self, text, expected=None):
        with self.assertRaisesRegex(CHECKER.VerificationError, text):
            CHECKER.validate_bundle(self.root, expected)

    def rewrite_config(self, change):
        path = self.root / "mha-untied/model/config.json"
        config = json.loads(path.read_bytes())
        change(config)
        data = json_bytes(config)
        path.write_bytes(data)
        self.manifest["cases"]["mha-untied"]["model_files"]["model/config.json"] = file_record(data)
        self.rewrite_case()

    def test_valid_bundle_has_five_cases_and_120_tensors(self):
        manifest, tensors = CHECKER.validate_bundle(self.root, self.expected)
        self.assertEqual(set(manifest["cases"]), set(CASES))
        self.assertEqual(sum(map(len, tensors.values())), 120)

    def test_current_manifest_cannot_be_its_own_reviewed_expectation(self):
        self.assert_invalid("must be separate", self.root / "manifest.json")

    def test_hardlinked_current_manifest_is_not_a_reviewed_copy(self):
        hardlink = self.directory / "hardlinked.json"
        hardlink.hardlink_to(self.root / "manifest.json")
        self.assert_invalid("must be separate", hardlink)

    def test_missing_case_is_rejected(self):
        self.manifest["cases"].pop("mqa-tail")
        self.write_manifest()
        self.assert_invalid("cases: missing")

    def test_extra_case_is_rejected(self):
        self.manifest["cases"]["another-case"] = {}
        self.write_manifest()
        self.assert_invalid("extra=.*another-case")

    def test_duplicate_case_is_rejected(self):
        path = self.root / "manifest.json"
        path.write_text(path.read_text().replace('"mqa-tail": {', '"mqa-tail": {}, "mqa-tail": {', 1))
        self.assert_invalid("Duplicate JSON key: mqa-tail")

    def test_missing_tensor_is_rejected(self):
        self.rewrite_case(change=lambda record, tensors: tensors.pop("decode_0.keys"))
        self.assert_invalid("reference tensors: missing=.*decode_0.keys")

    def test_extra_tensor_is_rejected(self):
        self.rewrite_case(change=lambda record, tensors: tensors.update({"extra": tensor("F32", [1])}))
        self.assert_invalid("extra=.*extra")

    def test_wrong_tensor_shape_is_rejected(self):
        def change(record, tensors):
            dtype, shape, data = tensors["prefill.logits"]
            tensors["prefill.logits"] = dtype, [1, math.prod(shape)], data
        self.rewrite_case(change=change)
        self.assert_invalid("prefill.logits: dtype/shape mismatch")

    def test_nonfinite_tensor_values_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                def change(record, tensors):
                    dtype, shape, data = tensors["prefill.logits"]
                    tensors["prefill.logits"] = dtype, shape, struct.pack("<f", value) + data[4:]
                self.rewrite_case(change=change)
                self.assert_invalid("non-finite value at index 0")

    def test_wrong_position_is_rejected(self):
        self.rewrite_case(change=lambda record, tensors: tensors.update({"decode_0.position": tensor("I64", [1], [5])}))
        self.assert_invalid("decode_0.position: value/position mismatch")

    def test_wrong_metadata_position_is_rejected(self):
        self.rewrite_case(change=lambda record, tensors: record["trajectory"]["positions"].update({"decode_0": [4]}))
        self.assert_invalid("wrong stage positions")

    def test_boolean_position_is_not_an_integer(self):
        self.rewrite_case(change=lambda record, tensors: record["trajectory"]["positions"]["prefill"].__setitem__(0, False))
        self.assert_invalid("wrong stage positions")

    def test_wrong_kv_layout_is_rejected(self):
        self.rewrite_case(change=lambda record, tensors: record["layout"].update({"kv": "layer,kv_head,position,dim"}))
        self.assert_invalid("invalid KV/logits layout")

    def test_pre_rope_key_identity_is_rejected(self):
        self.rewrite_case(change=lambda record, tensors: record["layout"].update({"key": "pre_rope"}))
        self.assert_invalid("invalid KV/logits layout")

    def test_frozen_tolerance_metadata_cannot_be_widened(self):
        self.rewrite_case(change=lambda record, tensors: record["tolerance"].update({"logits": [1.0, 1.0]}))
        self.assert_invalid("frozen tolerance changed")

    def test_frozen_tolerance_payload_cannot_be_widened(self):
        self.rewrite_case(change=lambda record, tensors: tensors.update({"tolerance.kv": tensor("F32", [2], [1.0, 1.0])}))
        self.assert_invalid("frozen tolerance payload changed")

    def test_changed_committed_kv_prefix_is_rejected(self):
        def change(record, tensors):
            dtype, shape, data = tensors["decode_1.keys"]
            tensors["decode_1.keys"] = dtype, shape, struct.pack("<f", 0.25) + data[4:]
        self.rewrite_case(change=change)
        self.assert_invalid("committed KV prefix changed")

    def test_wrong_stage_argmax_is_rejected(self):
        self.rewrite_case(change=lambda record, tensors: tensors.update({"decode_0.tokens": tensor("I64", [1], [1])}))
        self.assert_invalid("logits/Argmax mismatch")

    def test_source_version_must_match_reviewed_identity(self):
        self.manifest["reference"]["versions"]["torch"] = "different"
        self.write_manifest()
        self.assert_invalid("Reviewed identity mismatch: reference.versions.torch", self.expected)

    def test_synthetic_seed_requires_an_integer_identity(self):
        self.rewrite_case("gqa-tail", change=lambda record, tensors: record.update({"synthetic_seed": 20261027.0}))
        self.assert_invalid("wrong source variant/seed identity")

    def test_source_file_hash_must_match_reviewed_identity(self):
        self.manifest["reference"]["source_files"]["torch/__init__.py"]["sha256"] = "a" * 64
        self.write_manifest()
        self.assert_invalid("Reviewed identity mismatch: reference.source_files", self.expected)

    def test_raw_file_corruption_is_rejected(self):
        path = self.root / "mha-untied/reference.safetensors"
        path.write_bytes(path.read_bytes()[:-1] + b"\x01")
        self.assert_invalid("file sha256 mismatch")

    def test_model_weight_identity_must_match_reviewed_manifest(self):
        path = self.root / "mha-untied/model/model.safetensors"
        data = path.read_bytes()[:-4] + struct.pack("<f", 0.5)
        path.write_bytes(data)
        self.manifest["cases"]["mha-untied"]["model_files"]["model/model.safetensors"] = file_record(data)
        self.rewrite_case()
        self.assert_invalid("Reviewed identity mismatch: cases.mha-untied", self.expected)

    def test_model_config_geometry_mismatch_is_rejected(self):
        self.rewrite_config(lambda config: config.update({"hidden_size": config["hidden_size"] + 2}))
        self.assert_invalid("config/geometry mismatch for hidden_size")

    def test_explicit_standard_rope_parameters_are_accepted(self):
        self.rewrite_config(lambda config: config.update({
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}}))
        CHECKER.validate_bundle(self.root)

    def test_unsupported_rope_parameters_are_rejected(self):
        self.rewrite_config(lambda config: config.update({
            "rope_parameters": {"rope_type": "linear", "factor": 2.0}}))
        self.assert_invalid("unsupported attention/RoPE")

    def test_nonboolean_bias_flag_is_rejected(self):
        self.rewrite_config(lambda config: config.update({"attention_bias": 0}))
        self.assert_invalid("unsupported attention/RoPE")

    def test_nonfinite_model_weight_is_rejected(self):
        path = self.root / "mha-untied/model/model.safetensors"
        data = path.read_bytes()
        data = data[:-4] + struct.pack("<f", float("nan"))
        path.write_bytes(data)
        self.manifest["cases"]["mha-untied"]["model_files"]["model/model.safetensors"] = file_record(data)
        self.rewrite_case()
        self.assert_invalid("model:.*non-finite value")

    def test_reference_json_cannot_change_integer_to_boolean(self):
        path = self.root / "mha-untied/reference.json"
        record = json.loads(path.read_bytes())
        record["trajectory"]["positions"]["prefill"][0] = False
        data = json_bytes(record)
        path.write_bytes(data)
        self.manifest["cases"]["mha-untied"]["files"]["reference.json"] = file_record(data)
        self.write_manifest()
        self.assert_invalid("reference.json/manifest record mismatch")

    def test_file_record_cannot_use_parent_path(self):
        record = self.manifest["cases"]["mha-untied"]["model_files"]
        record["../config.json"] = record.pop("model/config.json")
        self.rewrite_case()
        self.assert_invalid("model_files: missing")

    def test_symlink_to_external_identical_file_is_rejected(self):
        path = self.root / "mha-untied/model/config.json"
        target = self.directory / "external-config.json"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
        self.assert_invalid("Symlink asset is not allowed")

    def test_extra_unclaimed_case_file_is_rejected(self):
        (self.root / "mha-untied/extra.bin").write_bytes(b"extra")
        self.assert_invalid("unexpected/missing case files")

    def candidate(self, change=None):
        root = self.directory / "candidate"
        shutil.copytree(self.root, root)
        manifest = copy.deepcopy(self.manifest)
        if change:
            self.rewrite_case(root=root, manifest=manifest, change=change)
        return CHECKER.validate_bundle(root)

    def test_identical_candidate_has_no_differences(self):
        report = CHECKER.compare_bundles(CHECKER.validate_bundle(self.root), self.candidate())
        self.assertFalse(report["needs_review"])
        self.assertTrue(report["numerical_passed"])
        self.assertEqual(report["compared_tensors"], 120)

    def test_numerical_perturbation_reports_first_tensor_coordinate(self):
        def change(record, tensors):
            dtype, shape, data = tensors["prefill.logits"]
            tensors["prefill.logits"] = dtype, shape, struct.pack("<f", 0.25) + data[4:]
        report = CHECKER.compare_bundles(CHECKER.validate_bundle(self.root), self.candidate(change))
        self.assertTrue(report["needs_review"])
        self.assertFalse(report["numerical_passed"])
        delta = report["tensor_differences"][0]
        self.assertEqual(delta["tensor"], "prefill.logits")
        self.assertEqual(delta["first_failure"]["coordinate"], [0, 0])
        self.assertEqual(delta["failed_elements"], 1)

    def test_within_tolerance_candidate_still_requires_review(self):
        def change(record, tensors):
            dtype, shape, data = tensors["prefill.logits"]
            tensors["prefill.logits"] = dtype, shape, struct.pack("<f", 1e-6) + data[4:]
        report = CHECKER.compare_bundles(CHECKER.validate_bundle(self.root), self.candidate(change))
        self.assertTrue(report["needs_review"])
        self.assertTrue(report["numerical_passed"])
        self.assertTrue(report["asset_differences"])

    def test_metadata_difference_is_separate_from_tensor_difference(self):
        candidate_root = self.directory / "candidate"
        shutil.copytree(self.root, candidate_root)
        manifest = copy.deepcopy(self.manifest)
        manifest["reference"]["versions"]["torch"] = "candidate-version"
        self.write_manifest(candidate_root, manifest)
        report = CHECKER.compare_bundles(CHECKER.validate_bundle(self.root), CHECKER.validate_bundle(candidate_root))
        self.assertTrue(report["needs_review"])
        self.assertEqual(report["tensor_differences"], [])
        self.assertEqual(report["metadata_differences"][0]["path"], "reference.versions.torch")

    def test_integer_comparison_is_exact(self):
        baseline = CHECKER.Tensor("I64", (1,), memoryview(struct.pack("<q", 1)))
        candidate = CHECKER.Tensor("I64", (1,), memoryview(struct.pack("<q", 2)))
        result = CHECKER.tensor_difference("test", "greedy_tokens", baseline, candidate)
        self.assertFalse(result["passed"])
        self.assertEqual(result["atol"], 0)

    def test_dtype_or_shape_difference_is_reported(self):
        baseline = CHECKER.Tensor("F32", (2,), memoryview(struct.pack("<2f", 0, 0)))
        candidate = CHECKER.Tensor("F32", (1, 2), baseline.payload)
        result = CHECKER.tensor_difference("test", "prefill.logits", baseline, candidate)
        self.assertEqual(result["reason"], "dtype_or_shape_mismatch")

    def test_cli_success_and_candidate_review_do_not_modify_expected(self):
        before = self.expected.read_bytes()
        command = [sys.executable, str(ROOT / "tools/verify_llama_numerical_reference.py"),
                   "--root", str(self.root), "--expected", str(self.expected)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(json.loads(result.stdout)["status"], "verified")
        candidate_root = self.directory / "candidate"
        shutil.copytree(self.root, candidate_root)
        manifest = copy.deepcopy(self.manifest)
        manifest["generator"]["sha256"] = "a" * 64
        self.write_manifest(candidate_root, manifest)
        result = subprocess.run(command + ["--compare", str(candidate_root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        self.assertEqual(json.loads(result.stdout)["status"], "review_required")
        self.assertEqual(self.expected.read_bytes(), before)

    def test_cli_invalid_bundle_returns_json_error(self):
        path = self.root / "manifest.json"
        path.write_text(path.read_text().replace('"mqa-tail": {', '"mqa-tail": {}, "mqa-tail": {', 1))
        result = subprocess.run([sys.executable, str(ROOT / "tools/verify_llama_numerical_reference.py"),
                                 "--root", str(self.root), "--expected", str(self.expected)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr + result.stdout)
        self.assertEqual(json.loads(result.stdout)["status"], "invalid")
        self.assertEqual(result.stderr, "")


class StrictNumericalSafetensorsParser(unittest.TestCase):
    def test_duplicate_tensor_key_is_rejected(self):
        header = b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
        header += b" " * (-len(header) % 8)
        with self.assertRaisesRegex(CHECKER.VerificationError, "Duplicate JSON key"):
            CHECKER.read_safetensors(struct.pack("<Q", len(header)) + header + b"\0" * 4)

    def test_overlapping_offsets_are_rejected(self):
        header = {name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for name in ("a", "b")}
        data = json.dumps(header).encode()
        data += b" " * (-len(data) % 8)
        with self.assertRaisesRegex(CHECKER.VerificationError, "overlapping"):
            CHECKER.read_safetensors(struct.pack("<Q", len(data)) + data + b"\0" * 4)

    def test_unclaimed_payload_is_rejected(self):
        data = safetensors_bytes({"x": tensor("F32", [1])}) + b"\0" * 4
        with self.assertRaisesRegex(CHECKER.VerificationError, "unclaimed payload"):
            CHECKER.read_safetensors(data)

    def test_shape_byte_count_mismatch_is_rejected(self):
        data = safetensors_bytes({"x": ("F32", [2], b"\0" * 4)})
        with self.assertRaisesRegex(CHECKER.VerificationError, "shape/dtype byte count mismatch"):
            CHECKER.read_safetensors(data)

    def test_json_rejects_nonfinite_and_overflowing_float(self):
        for value in (b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}'):
            with self.subTest(value=value), self.assertRaisesRegex(CHECKER.VerificationError, "Non-finite"):
                CHECKER.parse_json(value)


if __name__ == "__main__":
    unittest.main()
