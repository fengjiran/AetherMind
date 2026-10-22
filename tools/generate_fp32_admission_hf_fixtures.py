#!/usr/bin/env python3
"""Generate pinned HF dtype-admission fixtures using only the Python standard library.

This offline test-data converter is not a production weight normalizer. Run with
--check to verify an existing output against reviewed checksums without changing
either the output or the expected manifest.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import struct
import tempfile


GENERATOR_VERSION = 1
SCHEMA_VERSION = 1
DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2}
SERIALIZATION = {
    "config_json": "utf8-sort_keys-indent2-final-lf",
    "safetensors_header": "utf8-sort_keys-compact-space-pad8-u64le",
    "tensor_order": "lexicographic",
    "payload_byte_order": "little-endian",
}
CONVERSION = {
    "F32": "preserve-original-bits",
    "F16": "struct.pack('<e')-round-to-nearest-even",
    "BF16": "bits-plus-0x7fff-plus-lsb-round-to-nearest-even",
    "nonfinite_or_overflow": "reject",
}


class FixtureError(ValueError):
    """An input, generated fixture, or reviewed checksum violates the contract."""


@dataclass(frozen=True)
class Tensor:
    dtype: str
    shape: tuple
    payload: bytes


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FixtureError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(data):
    def reject_constant(value):
        raise FixtureError(f"Non-finite JSON constant: {value}")

    try:
        return json.loads(data, object_pairs_hook=_unique_object,
                          parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FixtureError(f"Invalid JSON: {error}") from error


def read_json(path):
    return parse_json(Path(path).read_bytes())


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def read_safetensors(data):
    """Validate a complete, densely stored Safetensors file before slicing bytes."""
    if len(data) < 8:
        raise FixtureError("Safetensors length prefix is truncated")
    header_bytes = struct.unpack_from("<Q", data)[0]
    if not header_bytes or header_bytes % 8 or header_bytes > len(data) - 8:
        raise FixtureError("Invalid Safetensors header length")
    header_data = data[8:8 + header_bytes]
    if not header_data.startswith(b"{"):
        raise FixtureError("Safetensors header must start with an object")
    header = parse_json(header_data)
    if not isinstance(header, dict):
        raise FixtureError("Safetensors header is not an object")
    metadata = header.pop("__metadata__", None)
    if metadata is not None and (not isinstance(metadata, dict) or
                                 any(not isinstance(v, str) for v in metadata.values())):
        raise FixtureError("Safetensors metadata must contain string values")
    payload = data[8 + header_bytes:]
    tensors = {}
    spans = []
    for name, descriptor in header.items():
        if not name or not isinstance(descriptor, dict) or set(descriptor) != {
                "dtype", "shape", "data_offsets"}:
            raise FixtureError(f"Invalid tensor descriptor: {name}")
        dtype = descriptor["dtype"]
        shape = descriptor["shape"]
        offsets = descriptor["data_offsets"]
        if dtype not in DTYPE_BYTES:
            raise FixtureError(f"Unsupported tensor dtype for {name}: {dtype}")
        if not isinstance(shape, list) or any(not _integer(n) for n in shape):
            raise FixtureError(f"Invalid tensor shape: {name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(
                not _integer(n) for n in offsets):
            raise FixtureError(f"Invalid tensor offsets: {name}")
        begin, end = offsets
        if begin > end or end > len(payload):
            raise FixtureError(f"Tensor offsets exceed payload: {name}")
        if end - begin != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise FixtureError(f"Tensor byte count does not match shape/dtype: {name}")
        tensors[name] = Tensor(dtype, tuple(shape), payload[begin:end])
        spans.append((begin, end))
    cursor = 0
    for begin, end in sorted(spans):
        if begin != cursor:
            raise FixtureError("Safetensors payload has gaps or overlapping tensors")
        cursor = end
    if not tensors or cursor != len(payload):
        raise FixtureError("Safetensors contains no tensors or unclaimed payload bytes")
    return tensors, metadata


def encode_safetensors(tensors, metadata=None):
    header = {} if metadata is None else {"__metadata__": metadata}
    payloads = []
    offset = 0
    for name, tensor in sorted(tensors.items()):
        end = offset + len(tensor.payload)
        header[name] = {"dtype": tensor.dtype, "shape": list(tensor.shape),
                        "data_offsets": [offset, end]}
        payloads.append(tensor.payload)
        offset = end
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads)


def convert_f32(payload, dtype):
    if dtype not in DTYPE_BYTES or len(payload) % 4:
        raise FixtureError("Invalid F32 conversion input")
    converted = bytearray()
    for (bits,) in struct.iter_unpack("<I", payload):
        value = struct.unpack("<f", struct.pack("<I", bits))[0]
        if not math.isfinite(value):
            raise FixtureError("Non-finite source F32 value")
        if dtype == "F32":
            continue
        if dtype == "F16":
            try:
                packed = struct.pack("<e", value)
            except OverflowError as error:
                raise FixtureError("F16 conversion overflow") from error
            if not math.isfinite(struct.unpack("<e", packed)[0]):
                raise FixtureError("Non-finite F16 conversion result")
            converted.extend(packed)
        else:
            rounded = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
            if (rounded & 0x7f80) == 0x7f80:
                raise FixtureError("Non-finite BF16 conversion result")
            converted.extend(struct.pack("<H", rounded))
    return payload if dtype == "F32" else bytes(converted)


def variant_specs():
    variants = []

    def add(name, dtype="F32", hint=None, norms="none", tied=False):
        variants.append({"name": name, "dtype": dtype, "hint": hint,
                         "norms": norms, "tied": tied})

    for suffix, hint in (("missing", None), ("auto", "auto"), ("explicit", "float32")):
        add(f"f32-{suffix}", hint=hint)
    for dtype in ("F16", "BF16"):
        for suffix, hint in (("missing", None), ("auto", "auto")):
            add(f"{dtype.lower()}-{suffix}", dtype, hint)
    for dtype in ("F16", "BF16"):
        add(f"{dtype.lower()}-f32-hint", dtype, "float32")
        add(f"f32-{dtype.lower()}-hint", hint="float16" if dtype == "F16" else "bfloat16")
    for scope, norms in (("mixed", "all"), ("single", "last-post-attention")):
        for dtype in ("F16", "BF16"):
            for suffix, hint in (("missing", None), ("auto", "auto")):
                add(f"{scope}-norm-{dtype.lower()}-{suffix}", dtype, hint, norms)
    add("f32-tied", hint="float32", tied=True)
    return sorted(variants, key=lambda variant: variant["name"])


def load_recipe(path):
    recipe = read_json(path)
    if not isinstance(recipe, dict) or recipe.get("schema_version") != SCHEMA_VERSION:
        raise FixtureError("Unsupported recipe schema")
    if recipe.get("recipe_version") != 1 or recipe.get("serialization") != SERIALIZATION or \
            recipe.get("conversion") != CONVERSION or recipe.get("variants") != variant_specs():
        raise FixtureError("Recipe rules or variant matrix differ from the supported contract")
    source = recipe.get("source", {})
    if not isinstance(source, dict):
        raise FixtureError("Recipe source must be an object")
    if not isinstance(source.get("repository"), str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source["repository"]):
        raise FixtureError("Recipe must pin a source repository")
    if not isinstance(source.get("revision"), str) or not re.fullmatch(
            r"[0-9a-f]{40}", source["revision"]):
        raise FixtureError("Recipe must pin an immutable source revision")
    files = source.get("files", {})
    if not isinstance(files, dict) or set(files) != {"config.json", "model.safetensors"}:
        raise FixtureError("Recipe must pin config.json and model.safetensors")
    for name, info in files.items():
        if not isinstance(info, dict) or not _integer(info.get("bytes"), 1) or not isinstance(
                info.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", info["sha256"]):
            raise FixtureError(f"Invalid source file identity: {name}")
    request = recipe.get("request", {})
    if request != {"prompt_tokens": [7], "context_tokens": 8, "max_new_tokens": 3,
                   "eos_token_id": None, "presets": ["o1-plain", "o2-packed"]}:
        raise FixtureError("Recipe request must match the frozen integration baseline")
    return recipe


def expected_shapes(config):
    keys = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings")
    if any(not _integer(config.get(key), 1) for key in keys):
        raise FixtureError("Source config requires positive Llama dimensions")
    if config.get("model_type") != "llama" or config.get("hidden_act") != "silu" or \
            config.get("tie_word_embeddings") is not False or config.get("attention_bias", False) or \
            config.get("mlp_bias", False):
        raise FixtureError("Source must be an untied, bias-free Llama SwiGLU model")
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    dim = config["head_dim"]
    intermediate = config["intermediate_size"]
    if hidden != heads * dim or heads % kv_heads:
        raise FixtureError("Invalid source attention geometry")
    shapes = {"model.embed_tokens.weight": (config["vocab_size"], hidden),
              "lm_head.weight": (config["vocab_size"], hidden), "model.norm.weight": (hidden,)}
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        for role in ("input_layernorm", "post_attention_layernorm"):
            shapes[prefix + role + ".weight"] = (hidden,)
        for role, rows, columns in (("q", heads * dim, hidden), ("k", kv_heads * dim, hidden),
                                    ("v", kv_heads * dim, hidden), ("o", hidden, heads * dim)):
            shapes[prefix + f"self_attn.{role}_proj.weight"] = (rows, columns)
        for role, rows, columns in (("gate", intermediate, hidden), ("up", intermediate, hidden),
                                    ("down", hidden, intermediate)):
            shapes[prefix + f"mlp.{role}_proj.weight"] = (rows, columns)
    return shapes


def load_source(source_dir, recipe):
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FixtureError("Source HF directory is missing")
    model_files = {p.name for p in source_dir.iterdir() if p.suffix in {
        ".safetensors", ".bin", ".pt", ".pth", ".gguf"} or p.name.endswith(".index.json")}
    if model_files != {"model.safetensors"}:
        raise FixtureError("Source must contain exactly one unsharded model.safetensors")
    contents = {}
    for name, identity in recipe["source"]["files"].items():
        data = (source_dir / name).read_bytes()
        if {"bytes": len(data), "sha256": sha256(data)} != identity:
            raise FixtureError(f"Source identity mismatch: {name}")
        contents[name] = data
    config = parse_json(contents["config.json"])
    if not isinstance(config, dict):
        raise FixtureError("Source config must be an object")
    shapes = expected_shapes(config)
    tensors, metadata = read_safetensors(contents["model.safetensors"])
    if set(tensors) != set(shapes):
        raise FixtureError("Source must contain exactly the complete untied Llama weight schema")
    for name, tensor in tensors.items():
        if tensor.dtype != "F32" or tensor.shape != shapes[name]:
            raise FixtureError(f"Source tensor must have the required F32 shape: {name}")
        convert_f32(tensor.payload, "F32")
    return config, tensors, metadata


def _config_changes(config, generated):
    return {"removed": sorted(set(config) - set(generated)),
            "set": {key: value for key, value in generated.items()
                    if key not in config or config[key] != value}}


def _variant(config, tensors, spec, cache):
    generated = dict(config)
    generated.pop("dtype", None)
    generated.pop("torch_dtype", None)
    if spec["hint"] is not None:
        generated["torch_dtype"] = spec["hint"]
    if spec["tied"]:
        generated["tie_word_embeddings"] = True
    result = {}
    last_norm = f"model.layers.{config['num_hidden_layers'] - 1}.post_attention_layernorm.weight"
    for name, tensor in tensors.items():
        if spec["tied"] and name == "lm_head.weight":
            continue
        dtype = spec["dtype"]
        if spec["norms"] == "all" and not (
                name == "model.norm.weight" or name.endswith("layernorm.weight")):
            dtype = "F32"
        elif spec["norms"] == "last-post-attention" and name != last_norm:
            dtype = "F32"
        key = (name, dtype)
        if key not in cache:
            cache[key] = convert_f32(tensor.payload, dtype)
        result[name] = Tensor(dtype, tensor.shape, cache[key])
    return generated, result


def _file_record(data):
    return {"bytes": len(data), "sha256": sha256(data)}


def inspect_variant(directory):
    config_data = (directory / "config.json").read_bytes()
    weights_data = (directory / "model.safetensors").read_bytes()
    tensors, _ = read_safetensors(weights_data)
    return {"files": {"config.json": _file_record(config_data),
                      "model.safetensors": _file_record(weights_data)},
            "tensors": [{"name": name, "shape": list(tensor.shape), "dtype": tensor.dtype,
                         "bytes": len(tensor.payload), "payload_sha256": sha256(tensor.payload)}
                        for name, tensor in sorted(tensors.items())]}


def _manifest(recipe, recipe_path):
    return {"schema_version": SCHEMA_VERSION,
            "generator": {"version": GENERATOR_VERSION, "sha256": sha256(Path(__file__).read_bytes())},
            "recipe": {"version": recipe["recipe_version"], "sha256": sha256(Path(recipe_path).read_bytes())},
            "python": {"implementation": platform.python_implementation(), "version": platform.python_version()},
            "source": recipe["source"], "request": recipe["request"], "variants": {}}


def generate_into(source_dir, recipe_path, output_dir):
    """Generate into an empty private staging directory; callers publish on success."""
    recipe = load_recipe(recipe_path)
    config, tensors, metadata = load_source(source_dir, recipe)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FixtureError("Generation staging directory must be empty")
    manifest = _manifest(recipe, recipe_path)
    cache = {}
    for spec in recipe["variants"]:
        generated, converted = _variant(config, tensors, spec, cache)
        destination = output_dir / spec["name"]
        destination.mkdir()
        (destination / "config.json").write_bytes(json_bytes(generated))
        (destination / "model.safetensors").write_bytes(encode_safetensors(converted, metadata))
        manifest["variants"][spec["name"]] = {
            "config_changes": _config_changes(config, generated), **inspect_variant(destination)}
    (output_dir / "generated-manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def _overlaps(left, right):
    return left == right or left in right.parents or right in left.parents


def validate_paths(source_dir, recipe_path, output_dir, expected_path=None):
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    protected = [source, (source / "config.json").resolve(),
                 (source / "model.safetensors").resolve(),
                 Path(recipe_path).resolve(), Path(__file__).resolve()]
    if expected_path is not None:
        protected.append(Path(expected_path).resolve())
    if any(_overlaps(output, path) for path in protected):
        raise FixtureError("Output overlaps source, generator, recipe, or expected checksums")
    repo = Path(__file__).resolve().parents[1]
    if output == repo or repo in output.parents:
        ignored_root = repo / ".models"
        if ignored_root not in output.parents:
            raise FixtureError("Repository-local generated output must be under .models/")
    if Path(output_dir).is_symlink():
        raise FixtureError("Output directory must not be a symlink")
    return output


def validate_output_tree(output, names, require_complete):
    if not output.exists():
        if require_complete:
            raise FixtureError("Generated output directory is missing")
        return
    if not output.is_dir():
        raise FixtureError("Generated output path must be a directory")
    allowed = set(names) | {"generated-manifest.json"}
    actual = {path.name for path in output.iterdir()}
    if actual - allowed or (require_complete and actual != allowed):
        raise FixtureError("Output contains missing or unexpected entries")
    for path in output.iterdir():
        if path.is_symlink():
            raise FixtureError("Generated output must not contain symlinks")
        if path.name == "generated-manifest.json":
            if not path.is_file():
                raise FixtureError("Generated manifest must be a file")
            continue
        if not path.is_dir():
            raise FixtureError("Variant output must be a directory")
        children = {child.name for child in path.iterdir()}
        if children - {"config.json", "model.safetensors"} or (require_complete and children != {
                "config.json", "model.safetensors"}):
            raise FixtureError(f"Missing or unexpected files in variant: {path.name}")
        if any(child.is_symlink() or not child.is_file() for child in path.iterdir()):
            raise FixtureError("Generated variant files must be regular files")


def _content_manifest(manifest):
    # Environment provenance is recorded, but Python patch differences are not
    # asset drift when all source/rule identities and generated bytes agree.
    result = dict(manifest)
    result.pop("python", None)
    return result


def check(source_dir, recipe_path, output_dir, expected_path):
    output = validate_paths(source_dir, recipe_path, output_dir, expected_path)
    recipe = load_recipe(recipe_path)
    names = [spec["name"] for spec in recipe["variants"]]
    validate_output_tree(output, names, require_complete=True)
    expected = read_json(expected_path)
    actual = read_json(output / "generated-manifest.json")
    if not isinstance(expected, dict) or not isinstance(actual, dict) or \
            _content_manifest(actual) != _content_manifest(expected):
        raise FixtureError("Generated manifest differs from reviewed checksums")
    for name in names:
        record = inspect_variant(output / name)
        expected_record = expected.get("variants", {}).get(name, {})
        if any(record[key] != expected_record.get(key) for key in ("files", "tensors")):
            raise FixtureError(f"Generated fixture differs from reviewed checksums: {name}")
    with tempfile.TemporaryDirectory(prefix="aethermind-admission-check-") as temporary:
        regenerated = generate_into(source_dir, recipe_path, temporary)
        if _content_manifest(regenerated) != _content_manifest(expected):
            raise FixtureError("Regenerated content or source/generator/recipe identity differs")
    return actual


def generate(source_dir, recipe_path, output_dir):
    output = validate_paths(source_dir, recipe_path, output_dir)
    recipe = load_recipe(recipe_path)
    validate_output_tree(output, [spec["name"] for spec in recipe["variants"]], require_complete=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".admission-staging-", dir=output.parent) as temporary:
        staging = Path(temporary)
        manifest = generate_into(source_dir, recipe_path, staging)
        output.mkdir(exist_ok=True)
        for directory in sorted(staging.iterdir()):
            if directory.is_dir():
                target = output / directory.name
                target.mkdir(exist_ok=True)
                for path in directory.iterdir():
                    path.replace(target / path.name)
            else:
                directory.replace(output / directory.name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", type=Path, metavar="CHECKSUMS_JSON")
    args = parser.parse_args()
    try:
        if args.check is None:
            manifest = generate(args.source, args.recipe, args.output_dir)
            print(f"Generated {len(manifest['variants'])} HF variants and generated-manifest.json")
        else:
            manifest = check(args.source, args.recipe, args.output_dir, args.check)
            print(f"Verified {len(manifest['variants'])} HF variants against reviewed checksums")
    except (FixtureError, OSError, TypeError, KeyError) as error:
        parser.exit(1, f"fixture error: {error}\n")


if __name__ == "__main__":
    main()
