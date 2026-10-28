#!/usr/bin/env python3
"""Verify reviewed Llama numerical assets, or report candidate differences.

Only the Python standard library is required. Verification never regenerates
assets, changes tolerances, or updates the reviewed manifest.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys

CASE_IDS = {"mha-untied", "mha-tied", "gqa-tail", "mqa-tail", "mha-near-limit"}
STAGES = ("prefill", "decode_0", "decode_1")
MODEL_FILES = {"model/config.json", "model/model.safetensors"}
REFERENCE_FILES = {"reference.json", "reference.safetensors"}
TOLERANCE = {"logits": [2e-5, 2e-4], "kv": [2e-6, 2e-4]}
LAYOUT = {"kv": "layer,position,kv_head,dim", "key": "post_rope",
          "value": "unrotated", "logits": "selected_row,vocab", "rope": "split_half"}
GEOMETRY_KEYS = {"hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers",
                 "num_attention_heads", "num_key_value_heads", "head_dim",
                 "max_position_embeddings", "rms_norm_eps", "tie_word_embeddings"}
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_ELEMENTS = 128 * 1024 * 1024


class VerificationError(ValueError):
    """The bundle, reviewed identity, or numerical contract is invalid."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def exact_keys(value, keys, context):
    require(isinstance(value, dict), f"{context}: expected object")
    require(set(value) == set(keys),
            f"{context}: missing={sorted(set(keys) - set(value))}, "
            f"extra={sorted(set(value) - set(keys))}")


def integer(value, minimum=0):
    return type(value) is int and minimum <= value <= (1 << 63) - 1


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(data):
    def reject_constant(value):
        raise VerificationError(f"Non-finite JSON constant: {value}")

    def finite(value):
        if isinstance(value, float):
            require(math.isfinite(value), "Non-finite JSON number")
        elif isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)

    try:
        value = json.loads(data, object_pairs_hook=_unique_object,
                           parse_constant=reject_constant)
        finite(value)
        return value
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise VerificationError(f"Invalid JSON: {error}") from error


def read_bytes(path):
    require(path.is_file(), f"Missing regular file: {path}")
    require(path.stat().st_size <= MAX_FILE_BYTES, f"File exceeds size limit: {path}")
    return path.read_bytes()


def safe_path(root, relative):
    path = PurePosixPath(relative)
    require(not path.is_absolute() and ".." not in path.parts and
            str(path) == relative and "\\" not in relative,
            f"Unsafe asset path: {relative}")
    current = root
    for part in path.parts:
        current = current / part
        require(not current.is_symlink(), f"Symlink asset is not allowed: {relative}")
    require(current.resolve().is_relative_to(root.resolve()),
            f"Asset escapes bundle root: {relative}")
    return current


def file_record(value, context):
    exact_keys(value, {"bytes", "sha256"}, context)
    require(integer(value["bytes"], 1) and value["bytes"] <= MAX_FILE_BYTES,
            f"{context}: invalid byte count")
    require(isinstance(value["sha256"], str) and
            re.fullmatch(r"[0-9a-f]{64}", value["sha256"]), f"{context}: invalid sha256")


def verify_file(root, relative, record):
    file_record(record, relative)
    data = read_bytes(safe_path(root, relative))
    require(len(data) == record["bytes"], f"{relative}: file byte count mismatch")
    require(sha256(data) == record["sha256"], f"{relative}: file sha256 mismatch")
    return data


@dataclass(frozen=True)
class Tensor:
    dtype: str
    shape: tuple
    payload: memoryview

    def values(self):
        return (value for (value,) in struct.iter_unpack(
            "<f" if self.dtype == "F32" else "<q", self.payload))


def read_safetensors(data, context="safetensors"):
    require(len(data) >= 8, f"{context}: truncated length prefix")
    header_bytes = struct.unpack_from("<Q", data)[0]
    require(0 < header_bytes <= min(MAX_HEADER_BYTES, len(data) - 8) and
            header_bytes % 8 == 0, f"{context}: invalid header length")
    header_data = data[8:8 + header_bytes]
    require(header_data.startswith(b"{"), f"{context}: invalid header start")
    header = parse_json(header_data)
    require(isinstance(header, dict), f"{context}: header must be an object")
    metadata = header.pop("__metadata__", None)
    require(metadata is None or isinstance(metadata, dict) and all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()),
            f"{context}: invalid safetensors metadata")
    payload = memoryview(data)[8 + header_bytes:]
    tensors = {}
    spans = []
    for name, descriptor in header.items():
        require(isinstance(name, str) and name, f"{context}: invalid tensor name")
        label = f"{context}:{name}"
        exact_keys(descriptor, {"dtype", "shape", "data_offsets"}, label)
        dtype, shape, offsets = (descriptor[key] for key in ("dtype", "shape", "data_offsets"))
        require(dtype in ("F32", "I64"), f"{label}: unsupported dtype")
        require(isinstance(shape, list) and 1 <= len(shape) <= 4 and
                all(integer(dim, 1) for dim in shape), f"{label}: invalid shape")
        count = math.prod(shape)
        require(count <= MAX_ELEMENTS, f"{label}: element count exceeds limit")
        require(isinstance(offsets, list) and len(offsets) == 2 and
                all(integer(offset) for offset in offsets), f"{label}: invalid offsets")
        begin, end = offsets
        require(begin <= end <= len(payload), f"{label}: offsets exceed payload")
        require(end - begin == count * (4 if dtype == "F32" else 8),
                f"{label}: shape/dtype byte count mismatch")
        tensor = Tensor(dtype, tuple(shape), payload[begin:end])
        if dtype == "F32":
            for index, value in enumerate(tensor.values()):
                require(math.isfinite(value), f"{label}: non-finite value at index {index}")
        tensors[name] = tensor
        spans.append((begin, end))
    cursor = 0
    for begin, end in sorted(spans):
        require(begin == cursor, f"{context}: overlapping tensors or payload gap")
        cursor = end
    require(tensors and cursor == len(payload), f"{context}: empty or unclaimed payload")
    return tensors


def validate_provenance(manifest):
    exact_keys(manifest, {"schema_version", "generator", "reference", "admission", "cases"}, "manifest")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1,
            "Unsupported manifest schema_version")
    generator = manifest["generator"]
    exact_keys(generator, {"version", "sha256", "recipe_sha256", "requirements_sha256"}, "generator")
    require(type(generator["version"]) is int and generator["version"] == 1, "Invalid generator version")
    for name in ("sha256", "recipe_sha256", "requirements_sha256"):
        require(isinstance(generator[name], str) and re.fullmatch(r"[0-9a-f]{64}", generator[name]),
                f"generator.{name}: invalid sha256")
    reference = manifest["reference"]
    exact_keys(reference, {"versions", "execution", "source_files"}, "reference")
    exact_keys(reference["versions"], {"torch", "transformers", "numpy", "safetensors", "python",
                                       "python_implementation"}, "reference.versions")
    require(all(isinstance(value, str) and value for value in reference["versions"].values()),
            "reference.versions: versions must be nonempty strings")
    execution = reference["execution"]
    required = {"device": "cpu", "dtype": "float32", "attention_implementation": "eager",
                "threads": 1, "interop_threads": 1, "deterministic_algorithms": True,
                "training": False, "gradient_mode": "inference_mode", "seed": 20261027,
                "environment": {"ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "COMPATIBLE",
                                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                "OPENBLAS_NUM_THREADS": "1"}}
    exact_keys(execution, set(required) | {"torch_cpu_capability"}, "reference.execution")
    for key, value in required.items():
        require(type(execution[key]) is type(value) and execution[key] == value,
                f"reference.execution.{key}: contract mismatch")
    require(isinstance(execution["torch_cpu_capability"], str) and execution["torch_cpu_capability"],
            "reference.execution.torch_cpu_capability: missing identity")
    source_files = reference["source_files"]
    require(isinstance(source_files, dict) and source_files, "reference.source_files: empty identity")
    for name, record in source_files.items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and str(path) == name,
                f"reference.source_files: unsafe source identity {name}")
        file_record(record, f"reference.source_files.{name}")
    admission = manifest["admission"]
    exact_keys(admission, {"checksums_sha256", "source"}, "admission")
    require(isinstance(admission["checksums_sha256"], str) and
            re.fullmatch(r"[0-9a-f]{64}", admission["checksums_sha256"]), "Invalid admission checksum")
    source = admission["source"]
    exact_keys(source, {"repository", "revision", "files"}, "admission.source")
    require(isinstance(source["repository"], str) and source["repository"], "Missing source repository")
    require(isinstance(source["revision"], str) and re.fullmatch(r"[0-9a-f]{40}", source["revision"]),
            "Source revision must be a full commit identity")
    exact_keys(source["files"], {"config.json", "model.safetensors"}, "admission.source.files")
    for name, record in source["files"].items():
        file_record(record, f"admission.source.files.{name}")
    exact_keys(manifest["cases"], CASE_IDS, "cases")


def validate_geometry(case_id, record, config):
    geometry = record["geometry"]
    exact_keys(geometry, GEOMETRY_KEYS, f"{case_id}.geometry")
    for key in GEOMETRY_KEYS - {"rms_norm_eps", "tie_word_embeddings"}:
        require(integer(geometry[key], 1), f"{case_id}.geometry.{key}: expected positive integer")
    epsilon = geometry["rms_norm_eps"]
    require(type(epsilon) in (int, float) and math.isfinite(epsilon) and epsilon > 0,
            f"{case_id}.geometry.rms_norm_eps: invalid epsilon")
    require(type(geometry["tie_word_embeddings"]) is bool, f"{case_id}: invalid tie flag")
    require(geometry["hidden_size"] == geometry["num_attention_heads"] * geometry["head_dim"] and
            geometry["num_attention_heads"] % geometry["num_key_value_heads"] == 0 and
            geometry["head_dim"] % 2 == 0 and geometry["intermediate_size"] >= geometry["hidden_size"],
            f"{case_id}: inconsistent attention geometry")
    require(geometry["num_hidden_layers"] <= 128 and
            geometry["num_attention_heads"] <= 256 and
            geometry["hidden_size"] * max(geometry["vocab_size"], geometry["intermediate_size"]) <= MAX_ELEMENTS,
            f"{case_id}: geometry exceeds fixture limits")
    require(isinstance(config, dict) and config.get("model_type") == "llama" and
            config.get("hidden_act", "silu") == "silu", f"{case_id}: unsupported model config")
    for key, value in geometry.items():
        actual = config.get(key, geometry["hidden_size"] // geometry["num_attention_heads"]
        if key == "head_dim" else None)
        require(type(actual) is type(value) and actual == value,
                f"{case_id}: config/geometry mismatch for {key}")
    rope_parameters = config.get("rope_parameters")
    standard_rope = rope_parameters is None or not differences(
        {"rope_type": "default", "rope_theta": 10000.0}, rope_parameters)
    require(config.get("sliding_window") is None and config.get("attention_bias", False) is False and
            config.get("mlp_bias", False) is False and config.get("rope_scaling") is None and
            standard_rope and config.get("rope_theta", 10000.0) == 10000.0,
            f"{case_id}: unsupported attention/RoPE configuration")
    qh, kvh = geometry["num_attention_heads"], geometry["num_key_value_heads"]
    require((case_id.startswith("mha-") and qh == kvh) or
            (case_id == "gqa-tail" and 1 < kvh < qh) or
            (case_id == "mqa-tail" and kvh == 1 < qh), f"{case_id}: wrong attention grouping")
    require(geometry["tie_word_embeddings"] == (case_id == "mha-tied"), f"{case_id}: wrong tie identity")
    return geometry


def validate_model_tensors(tensors, geometry, case_id):
    h, i, v, layers, kvh, d = (geometry[key] for key in (
        "hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers", "num_key_value_heads", "head_dim"))
    expected = {"model.embed_tokens.weight": (v, h), "model.norm.weight": (h,)}
    if not geometry["tie_word_embeddings"]:
        expected["lm_head.weight"] = (v, h)
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        expected.update({prefix + "input_layernorm.weight": (h,),
                         prefix + "post_attention_layernorm.weight": (h,)})
        for name, shape in (("q", (h, h)), ("k", (kvh * d, h)),
                            ("v", (kvh * d, h)), ("o", (h, h))):
            expected[prefix + f"self_attn.{name}_proj.weight"] = shape
        for name, shape in (("gate", (i, h)), ("up", (i, h)), ("down", (h, i))):
            expected[prefix + f"mlp.{name}_proj.weight"] = shape
    exact_keys(tensors, expected, f"{case_id}.model tensors")
    for name, shape in expected.items():
        require(tensors[name].dtype == "F32" and tensors[name].shape == shape,
                f"{case_id}.model:{name}: dtype/shape mismatch")


def require_tensor(tensors, name, dtype, shape, case_id):
    tensor = tensors[name]
    require(tensor.dtype == dtype and tensor.shape == tuple(shape),
            f"{case_id}:{name}: dtype/shape mismatch")
    return tensor


def require_values(tensor, expected, context):
    require(list(tensor.values()) == expected, f"{context}: value/position mismatch")


def validate_reference_tensors(case_id, record, tensors, geometry):
    names = {"prompt", "decode_inputs", "greedy_tokens", "greedy_margins", "tolerance.logits", "tolerance.kv"}
    names.update(f"{stage}.{name}" for stage in STAGES for name in
                 ("logit_rows", "logits", "tokens", "keys", "values", "position"))
    exact_keys(tensors, names, f"{case_id}.reference tensors")
    exact_keys(record["tensors"], names, f"{case_id}.tensor records")
    for name, tensor in tensors.items():
        descriptor = record["tensors"][name]
        exact_keys(descriptor, {"dtype", "shape", "bytes", "payload_sha256"}, f"{case_id}:{name}")
        require(not differences({"dtype": tensor.dtype, "shape": list(tensor.shape),
                                 "bytes": len(tensor.payload), "payload_sha256": sha256(tensor.payload)}, descriptor),
                f"{case_id}:{name}: tensor record/hash mismatch")
    trajectory = record["trajectory"]
    exact_keys(trajectory, {"prompt", "decode_inputs", "positions", "logit_rows", "commit_positions",
                            "greedy_tokens", "greedy_margins"}, f"{case_id}.trajectory")
    p = 2046 if case_id == "mha-near-limit" else 3
    prompt = ([1, 7, 3] * ((p + 2) // 3))[:p]
    require(not differences(prompt, trajectory["prompt"]) and
            not differences([17, 23], trajectory["decode_inputs"]),
            f"{case_id}: unexpected fixed trajectory")
    require(geometry["max_position_embeddings"] >= p + 2, f"{case_id}: trajectory exceeds context")
    if case_id == "mha-near-limit":
        require(geometry["max_position_embeddings"] == p + 2, f"{case_id}: not at context boundary")
    expected_positions = {"prefill": list(range(p)), "decode_0": [p], "decode_1": [p + 1]}
    expected_rows = {"prefill": [p - 1] if case_id == "mha-near-limit" else list(range(p)),
                     "decode_0": [0], "decode_1": [0]}
    require(not differences(expected_positions, trajectory["positions"]) and
            not differences(expected_rows, trajectory["logit_rows"]) and
            not differences([p, p + 1, p + 2], trajectory["commit_positions"]),
            f"{case_id}: wrong stage positions/rows")
    v = geometry["vocab_size"]
    greedy = trajectory["greedy_tokens"]
    require(isinstance(greedy, list) and len(greedy) == 3 and
            all(integer(token) and token < v for token in greedy), f"{case_id}: invalid greedy tokens")
    margins = trajectory["greedy_margins"]
    require(isinstance(margins, list) and len(margins) == 3 and all(
        type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in margins),
            f"{case_id}: invalid greedy margins")
    for name, values in (("prompt", prompt), ("decode_inputs", [17, 23]), ("greedy_tokens", greedy)):
        require_values(require_tensor(tensors, name, "I64", [len(values)], case_id), values, f"{case_id}:{name}")
    margin_tensor = require_tensor(tensors, "greedy_margins", "F32", [3], case_id)
    require(bytes(margin_tensor.payload) == struct.pack("<3f", *margins), f"{case_id}: margin metadata mismatch")
    require(record["tolerance"] == TOLERANCE, f"{case_id}: frozen tolerance changed")
    require(record["layout"] == LAYOUT, f"{case_id}: invalid KV/logits layout")
    for name, values in TOLERANCE.items():
        tensor = require_tensor(tensors, f"tolerance.{name}", "F32", [2], case_id)
        require(bytes(tensor.payload) == struct.pack("<2f", *values), f"{case_id}: frozen tolerance payload changed")
    previous = None
    for index, stage in enumerate(STAGES):
        seq, committed = (p if index == 0 else 1), p + index
        rows = expected_rows[stage]
        require_values(require_tensor(tensors, f"{stage}.logit_rows", "I64", [len(rows)], case_id), rows,
                       f"{case_id}:{stage}.logit_rows")
        require_values(require_tensor(tensors, f"{stage}.position", "I64", [1], case_id), [committed],
                       f"{case_id}:{stage}.position")
        tokens = list(require_tensor(tensors, f"{stage}.tokens", "I64", [seq], case_id).values())
        require(all(0 <= token < v for token in tokens), f"{case_id}:{stage}: token outside vocabulary")
        logits = require_tensor(tensors, f"{stage}.logits", "F32", [len(rows), v], case_id)
        logits_values = list(logits.values())
        for row_index, source_row in enumerate(rows):
            row = logits_values[row_index * v:(row_index + 1) * v]
            require(max(range(v), key=row.__getitem__) == tokens[source_row],
                    f"{case_id}:{stage}: logits/Argmax mismatch at row {source_row}")
        if index == 0:
            require(tokens[-1] == greedy[0], f"{case_id}: greedy first token differs from Prefill")
        kv_shape = [geometry["num_hidden_layers"], committed, geometry["num_key_value_heads"], geometry["head_dim"]]
        for kind in ("keys", "values"):
            current = require_tensor(tensors, f"{stage}.{kind}", "F32", kv_shape, case_id)
            if previous is not None:
                old = tensors[f"{previous}.{kind}"]
                old_stride = math.prod(old.shape[1:]) * 4
                stride = math.prod(current.shape[1:]) * 4
                for layer in range(kv_shape[0]):
                    require(old.payload[layer * old_stride:(layer + 1) * old_stride] ==
                            current.payload[layer * stride:layer * stride + old_stride],
                            f"{case_id}:{stage}.{kind}: committed KV prefix changed at layer {layer}")
        previous = stage


def validate_case(root, case_id, record):
    exact_keys(record, {"schema_version", "id", "kind", "source_variant", "synthetic_seed", "geometry",
                        "trajectory", "tolerance", "layout", "model_files", "tensors", "files"}, case_id)
    require(type(record["schema_version"]) is int and record["schema_version"] == 1 and record["id"] == case_id,
            f"{case_id}: invalid case identity/schema")
    synthetic = case_id in {"gqa-tail", "mqa-tail"}
    seed = 20261027 if case_id == "gqa-tail" else 20261028
    require(record["kind"] == ("synthetic_hf" if synthetic else "hf_admission") and
            (type(record["synthetic_seed"]) is int if synthetic else record["synthetic_seed"] is None) and
            record["synthetic_seed"] == (seed if synthetic else None) and
            record["source_variant"] == (
                None if synthetic else "f32-tied" if case_id == "mha-tied" else "f32-explicit"),
            f"{case_id}: wrong source variant/seed identity")
    case_root = safe_path(root, case_id)
    require(case_root.is_dir(), f"Missing case directory: {case_id}")
    require({path.name for path in case_root.iterdir()} == {"model", *REFERENCE_FILES},
            f"{case_id}: unexpected/missing case files")
    model_root = safe_path(root, f"{case_id}/model")
    require(
        model_root.is_dir() and {path.name for path in model_root.iterdir()} == {"config.json", "model.safetensors"},
        f"{case_id}: unexpected/missing model files")
    exact_keys(record["model_files"], MODEL_FILES, f"{case_id}.model_files")
    exact_keys(record["files"], REFERENCE_FILES, f"{case_id}.files")
    model_data = {name: verify_file(root, f"{case_id}/{name}", value)
                  for name, value in record["model_files"].items()}
    reference_data = {name: verify_file(root, f"{case_id}/{name}", value)
                      for name, value in record["files"].items()}
    reference_json = parse_json(reference_data["reference.json"])
    require(not differences({key: value for key, value in record.items() if key != "files"}, reference_json),
            f"{case_id}: reference.json/manifest record mismatch")
    geometry = validate_geometry(case_id, record, parse_json(model_data["model/config.json"]))
    validate_model_tensors(read_safetensors(model_data["model/model.safetensors"], f"{case_id}:model"), geometry,
                           case_id)
    tensors = read_safetensors(reference_data["reference.safetensors"], f"{case_id}:reference")
    validate_reference_tensors(case_id, record, tensors, geometry)
    return tensors


def differences(baseline, candidate, path=""):
    result = []
    if type(baseline) is not type(candidate):
        return [{"path": path, "baseline": baseline, "candidate": candidate}]
    if isinstance(baseline, dict):
        for key in sorted(set(baseline) | set(candidate)):
            child = f"{path}.{key}" if path else key
            if key not in baseline or key not in candidate:
                result.append({"path": child, "baseline": baseline.get(key), "candidate": candidate.get(key)})
            else:
                result.extend(differences(baseline[key], candidate[key], child))
    elif isinstance(baseline, list):
        if len(baseline) != len(candidate):
            result.append({"path": path, "baseline": baseline, "candidate": candidate})
        else:
            for index, (old, new) in enumerate(zip(baseline, candidate)):
                result.extend(differences(old, new, f"{path}[{index}]"))
    elif baseline != candidate:
        result.append({"path": path, "baseline": baseline, "candidate": candidate})
    return result


def validate_bundle(root, expected=None):
    root = Path(root)
    require(root.is_dir(), f"Missing bundle root: {root}")
    require({path.name for path in root.iterdir()} == {"manifest.json", *CASE_IDS},
            "Bundle contains missing/extra cases or files")
    manifest = parse_json(read_bytes(safe_path(root, "manifest.json")))
    validate_provenance(manifest)
    if expected is not None:
        require(not Path(expected).samefile(root / "manifest.json"),
                "Reviewed expected manifest must be separate from the current bundle manifest")
        reviewed = parse_json(read_bytes(Path(expected)))
        validate_provenance(reviewed)
        mismatch = differences(reviewed, manifest)
        require(not mismatch, f"Reviewed identity mismatch: {mismatch[0]['path']}" if mismatch else "")
    tensors = {case_id: validate_case(root, case_id, manifest["cases"][case_id]) for case_id in sorted(CASE_IDS)}
    return manifest, tensors


def tensor_difference(case_id, name, baseline, candidate):
    result = {"case": case_id, "tensor": name, "dtype": baseline.dtype, "shape": list(baseline.shape)}
    if baseline.dtype != candidate.dtype or baseline.shape != candidate.shape:
        return {**result, "passed": False, "reason": "dtype_or_shape_mismatch",
                "candidate_dtype": candidate.dtype, "candidate_shape": list(candidate.shape)}
    if baseline.payload == candidate.payload:
        return None
    is_float = baseline.dtype == "F32"
    atol, rtol = TOLERANCE["kv" if name.endswith((".keys", ".values")) else "logits"] if is_float else (0, 0)
    first_failure = None
    max_abs = max_rel = 0.0
    changed = failed = 0
    for index, (expected, actual) in enumerate(zip(baseline.values(), candidate.values())):
        error = abs(actual - expected)
        relative = error / max(abs(expected), 1e-30)
        max_abs, max_rel = max(max_abs, error), max(max_rel, relative)
        changed += actual != expected
        if error > atol + rtol * abs(expected):
            failed += 1
            if first_failure is None:
                coordinate = []
                remainder = index
                for dimension in reversed(baseline.shape):
                    coordinate.append(remainder % dimension)
                    remainder //= dimension
                first_failure = {"index": index, "coordinate": list(reversed(coordinate)),
                                 "baseline": expected, "candidate": actual}
    return {**result, "passed": failed == 0, "changed_elements": changed, "failed_elements": failed,
            "max_absolute_error": max_abs, "max_relative_error": max_rel,
            "atol": atol, "rtol": rtol, "first_failure": first_failure}


def compare_bundles(baseline, candidate):
    baseline_manifest, baseline_tensors = baseline
    candidate_manifest, candidate_tensors = candidate

    def metadata(manifest):
        result = {key: value for key, value in manifest.items() if key != "cases"}
        result["cases"] = {case: {key: value for key, value in record.items() if key not in {"files", "tensors"}}
                           for case, record in manifest["cases"].items()}
        return result

    metadata_diff = differences(metadata(baseline_manifest), metadata(candidate_manifest))
    asset_diff = []
    numerical_diff = []
    compared = 0
    for case_id in sorted(CASE_IDS):
        old, new = baseline_manifest["cases"][case_id], candidate_manifest["cases"][case_id]
        asset_diff.extend(differences(old["files"], new["files"], f"cases.{case_id}.files"))
        for name in sorted(baseline_tensors[case_id]):
            compared += 1
            delta = tensor_difference(case_id, name, baseline_tensors[case_id][name], candidate_tensors[case_id][name])
            if delta is not None:
                numerical_diff.append(delta)
    needs_review = bool(metadata_diff or asset_diff or numerical_diff)
    return {"schema_version": 1, "status": "review_required" if needs_review else "identical",
            "needs_review": needs_review, "numerical_passed": all(item["passed"] for item in numerical_diff),
            "compared_tensors": compared, "metadata_differences": metadata_diff,
            "asset_differences": asset_diff, "tensor_differences": numerical_diff}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Reviewed baseline bundle")
    parser.add_argument("--expected", required=True, type=Path, help="Separately reviewed manifest copy")
    parser.add_argument("--compare", type=Path, help="Candidate bundle; emit differences without accepting it")
    arguments = parser.parse_args(argv)
    try:
        baseline = validate_bundle(arguments.root, arguments.expected)
        if arguments.compare is not None:
            report = compare_bundles(baseline, validate_bundle(arguments.compare))
            code = 1 if report["needs_review"] else 0
        else:
            report = {"schema_version": 1, "status": "verified", "cases": len(CASE_IDS),
                      "tensors": sum(len(value) for value in baseline[1].values())}
            code = 0
    except (VerificationError, OSError, OverflowError, struct.error) as error:
        report, code = {"schema_version": 1, "status": "invalid", "error": str(error)}, 2
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
