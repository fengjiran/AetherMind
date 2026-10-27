#!/usr/bin/env python3
"""Generate a new candidate CPU FP32 Llama numerical reference bundle.

The pinned Hugging Face implementation reads checkpoint files independently of
AetherMind. This development tool never promotes a candidate or overwrites an
existing bundle. Normal C++ tests consume the files without importing PyTorch.
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import sys
import tempfile

import generate_fp32_admission_hf_fixtures as admission


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SOURCE = ROOT / "tests/fixtures/inference_numerical"
SCHEMA_VERSION = 1
GENERATOR_VERSION = 1
REFERENCE_ENVIRONMENT = {
    "ATEN_CPU_CAPABILITY": "default",
    "MKL_CBWR": "COMPATIBLE",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
VERSIONS = {"torch": "2.9.0+cpu", "transformers": "4.57.1",
            "numpy": "2.3.4", "safetensors": "0.6.2"}
LAYOUT = {"kv": "layer,position,kv_head,dim", "key": "post_rope",
          "value": "unrotated", "logits": "selected_row,vocab", "rope": "split_half"}
GEOMETRY_FIELDS = (
    "hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "max_position_embeddings", "rms_norm_eps", "tie_word_embeddings",
)


class ReferenceError(ValueError):
    """The source, reference environment, or generated data violates its contract."""


def file_identity(path):
    data = Path(path).read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def read_recipe(path):
    recipe = admission.read_json(path)
    if recipe != admission.read_json(FIXTURE_SOURCE / "recipe.json"):
        raise ReferenceError("Use the checked-in numerical reference recipe")
    if recipe["schema_version"] != SCHEMA_VERSION or recipe["generator_version"] != GENERATOR_VERSION:
        raise ReferenceError("Unsupported recipe version")
    return recipe


def load_modules():
    # These settings affect only this reference-generation process. Apply them
    # before importing torch so its CPU dispatcher and BLAS initialize once.
    os.environ.update(REFERENCE_ENVIRONMENT)
    try:
        import numpy
        import safetensors
        import torch
        import transformers
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as error:
        raise ReferenceError("Install requirements-reference.txt in an isolated reference environment") from error

    modules = {"torch": torch, "transformers": transformers,
               "numpy": numpy, "safetensors": safetensors}
    for name, expected in VERSIONS.items():
        if modules[name].__version__ != expected:
            raise ReferenceError(f"{name} must be {expected}, got {modules[name].__version__}")
    if sys.byteorder != "little":
        raise ReferenceError("This reference environment requires a little-endian host")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    return torch, transformers, numpy, safetensors, LlamaConfig, LlamaForCausalLM


def reference_identity(modules, seed):
    torch, transformers, numpy, safetensors, _, _ = modules
    import transformers.cache_utils as cache_utils
    import transformers.modeling_rope_utils as rope_utils
    import transformers.models.llama.configuration_llama as configuration_llama
    import transformers.models.llama.modeling_llama as modeling_llama

    sources = {
        "torch/__init__.py": Path(torch.__file__),
        "torch/version.py": Path(inspect.getfile(torch.version)),
        "transformers/models/llama/modeling_llama.py": Path(inspect.getfile(modeling_llama)),
        "transformers/models/llama/configuration_llama.py": Path(inspect.getfile(configuration_llama)),
        "transformers/cache_utils.py": Path(inspect.getfile(cache_utils)),
        "transformers/modeling_rope_utils.py": Path(inspect.getfile(rope_utils)),
    }
    binary = Path(torch.__file__).parent / "lib/libtorch_cpu.so"
    if binary.is_file():
        sources["torch/lib/libtorch_cpu.so"] = binary
    return {
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "numpy": numpy.__version__, "safetensors": safetensors.__version__,
                     "python": platform.python_version(),
                     "python_implementation": platform.python_implementation()},
        "execution": {"device": "cpu", "dtype": "float32",
                      "attention_implementation": "eager", "threads": 1,
                      "interop_threads": 1, "deterministic_algorithms": True,
                      "training": False, "gradient_mode": "inference_mode", "seed": seed,
                      "environment": dict(REFERENCE_ENVIRONMENT),
                      "torch_cpu_capability": torch.backends.cpu.get_cpu_capability()},
        "source_files": {name: file_identity(path) for name, path in sorted(sources.items())},
    }


def check_admission(admission_root, expected):
    for variant in ("f32-explicit", "f32-tied"):
        directory = admission_root / variant
        record = admission.inspect_variant(directory)
        reviewed = expected["variants"][variant]
        if record["files"] != reviewed["files"] or record["tensors"] != reviewed["tensors"]:
            raise ReferenceError(f"Admission source identity mismatch: {variant}")


def synthetic_config(spec):
    result = {name: spec[name] for name in GEOMETRY_FIELDS if name in spec}
    result.update({"architectures": ["LlamaForCausalLM"], "model_type": "llama",
                   "rms_norm_eps": 1.0e-5, "hidden_act": "silu",
                   "tie_word_embeddings": False, "attention_bias": False,
                   "mlp_bias": False, "attention_dropout": 0.0,
                   "rope_theta": 10000.0, "torch_dtype": "float32",
                   "use_cache": True, "bos_token_id": 0, "eos_token_id": 1,
                   "pad_token_id": 0, "pretraining_tp": 1})
    return result


def encode_tensors(torch, tensors):
    header = {}
    payload = []
    records = {}
    offset = 0
    # A dense mixed-dtype safetensors payload can otherwise place I64 after an
    # odd number of F32 values. Emit the wider alignment first without padding.
    ordered = sorted(tensors.items(), key=lambda item: (0 if item[1].dtype == torch.int64 else 1, item[0]))
    for name, value in ordered:
        value = value.detach().cpu().contiguous()
        if value.dtype == torch.float32:
            dtype = "F32"
            if not torch.isfinite(value).all().item():
                raise ReferenceError(f"Non-finite reference tensor: {name}")
        elif value.dtype == torch.int64:
            dtype = "I64"
        else:
            raise ReferenceError(f"Unsupported reference tensor dtype: {name}: {value.dtype}")
        raw = value.numpy().tobytes(order="C")
        shape = list(value.shape)
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(raw)]}
        records[name] = {"dtype": dtype, "shape": shape, "bytes": len(raw),
                         "payload_sha256": hashlib.sha256(raw).hexdigest()}
        payload.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(payload), records


def make_model(modules, spec, model_dir, admission_root):
    torch, _, _, _, LlamaConfig, LlamaForCausalLM = modules
    model_dir.mkdir()
    if "source_variant" in spec:
        for filename in ("config.json", "model.safetensors"):
            shutil.copyfile(admission_root / spec["source_variant"] / filename, model_dir / filename)
        config_data = admission.read_json(model_dir / "config.json")
    else:
        config_data = synthetic_config(spec)
        config = LlamaConfig(**config_data)
        config._attn_implementation = "eager"
        torch.manual_seed(spec["seed"])
        model = LlamaForCausalLM(config)
        generator = torch.Generator(device="cpu").manual_seed(spec["seed"])
        with torch.no_grad():
            for _, parameter in sorted(model.named_parameters()):
                if parameter.ndim == 1:
                    parameter.uniform_(0.8, 1.2, generator=generator)
                else:
                    parameter.uniform_(-0.15, 0.15, generator=generator)
        state = model.state_dict()
        expected_shapes = admission.expected_shapes(config_data)
        if set(state) != set(expected_shapes):
            raise ReferenceError("HF synthetic state_dict does not match the complete Llama weight schema")
        for name, value in state.items():
            if tuple(value.shape) != expected_shapes[name] or value.dtype != torch.float32:
                raise ReferenceError(f"Unexpected HF synthetic tensor: {name}")
        # Only model tensors belong here; reference files live one level above.
        encoded, _ = encode_tensors(torch, state)
        (model_dir / "config.json").write_bytes(json_bytes(config_data))
        (model_dir / "model.safetensors").write_bytes(encoded)
    # Both real and synthetic references consume the serialized checkpoint via
    # HF's own loader, rather than keeping the just-initialized synthetic model.
    config = LlamaConfig.from_pretrained(model_dir, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        model_dir, config=config, local_files_only=True, use_safetensors=True,
        dtype=torch.float32, attn_implementation="eager")
    model.to(device="cpu", dtype=torch.float32).eval()
    if model.config._attn_implementation != "eager":
        raise ReferenceError("Reference attention must use the eager implementation")
    geometry = {name: config_data[name] for name in GEOMETRY_FIELDS}
    return model, geometry


def cache_planes(torch, cache, geometry, committed):
    # The pinned DynamicCache exposes each layer's post-RoPE keys and raw values.
    layers = cache.to_legacy_cache()
    if len(layers) != geometry["num_hidden_layers"]:
        raise ReferenceError("Reference cache has the wrong layer count")
    expected = (1, geometry["num_key_value_heads"], committed, geometry["head_dim"])
    keys = []
    values = []
    for key, value in layers:
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ReferenceError("Reference cache shape does not match committed geometry")
        keys.append(key[0].permute(1, 0, 2).contiguous())
        values.append(value[0].permute(1, 0, 2).contiguous())
    return torch.stack(keys).clone(), torch.stack(values).clone()


def forward(torch, model, tokens, position, cache):
    inputs = torch.tensor([tokens], dtype=torch.int64)
    positions = torch.arange(position, position + len(tokens), dtype=torch.int64)
    return model(input_ids=inputs, position_ids=positions.unsqueeze(0),
                 attention_mask=torch.ones((1, position + len(tokens)), dtype=torch.int64),
                 cache_position=positions, past_key_values=cache,
                 use_cache=True, return_dict=True)


def trajectories(torch, model, geometry, prompt, decode_inputs, near_limit, tolerance):
    tensors = {"prompt": torch.tensor(prompt, dtype=torch.int64),
               "decode_inputs": torch.tensor(decode_inputs, dtype=torch.int64),
               "tolerance.logits": torch.tensor(tolerance["logits"], dtype=torch.float32),
               "tolerance.kv": torch.tensor(tolerance["kv"], dtype=torch.float32)}
    positions = {}
    selected_rows = {}
    committed_positions = []
    cache = None
    stages = [("prefill", prompt, 0), ("decode_0", [decode_inputs[0]], len(prompt)),
              ("decode_1", [decode_inputs[1]], len(prompt) + 1)]
    with torch.inference_mode():
        for stage, inputs, position in stages:
            output = forward(torch, model, inputs, position, cache)
            cache = output.past_key_values
            committed = position + len(inputs)
            logits = output.logits[0]
            rows = [len(inputs) - 1] if stage == "prefill" and near_limit else list(range(len(inputs)))
            key, value = cache_planes(torch, cache, geometry, committed)
            tensors[stage + ".logit_rows"] = torch.tensor(rows, dtype=torch.int64)
            tensors[stage + ".logits"] = logits[rows].contiguous().clone()
            tensors[stage + ".tokens"] = logits.argmax(dim=-1).to(torch.int64).clone()
            tensors[stage + ".keys"] = key
            tensors[stage + ".values"] = value
            tensors[stage + ".position"] = torch.tensor([committed], dtype=torch.int64)
            positions[stage] = list(range(position, committed))
            selected_rows[stage] = rows
            committed_positions.append(committed)

        # Independent closed-loop greedy run: fixed teacher-forcing inputs above
        # never determine these cache contents or generated outputs.
        cache = None
        inputs = prompt
        position = 0
        greedy = []
        margins = []
        for _ in range(3):
            output = forward(torch, model, inputs, position, cache)
            cache = output.past_key_values
            last = output.logits[0, -1]
            token = int(last.argmax().item())
            top_two = torch.topk(last, 2).values
            greedy.append(token)
            margins.append(float((top_two[0] - top_two[1]).item()))
            position += len(inputs)
            inputs = [token]
    tensors["greedy_tokens"] = torch.tensor(greedy, dtype=torch.int64)
    tensors["greedy_margins"] = torch.tensor(margins, dtype=torch.float32)
    return tensors, {"prompt": prompt, "decode_inputs": decode_inputs,
                     "positions": positions, "logit_rows": selected_rows,
                     "commit_positions": committed_positions,
                     "greedy_tokens": greedy, "greedy_margins": margins}


def generate(admission_root, output_dir, recipe_path, checksums_path):
    admission_root = Path(admission_root).resolve()
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ReferenceError("Candidate output must not already exist")
    output_dir = output_dir.resolve()
    if output_dir == admission_root or admission_root in output_dir.parents or output_dir in admission_root.parents:
        raise ReferenceError("Candidate output must not overlap admission source assets")
    recipe = read_recipe(recipe_path)
    expected = admission.read_json(checksums_path)
    if expected.get("schema_version") != 1:
        raise ReferenceError("Unsupported admission checksum schema")
    check_admission(admission_root, expected)
    modules = load_modules()
    torch = modules[0]
    torch.manual_seed(recipe["seed"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": {"version": GENERATOR_VERSION,
                      "sha256": file_identity(__file__)["sha256"],
                      "recipe_sha256": file_identity(recipe_path)["sha256"],
                      "requirements_sha256": file_identity(FIXTURE_SOURCE / "requirements-reference.txt")["sha256"]},
        "reference": reference_identity(modules, recipe["seed"]),
        "admission": {"checksums_sha256": file_identity(checksums_path)["sha256"], "source": expected["source"]},
        "cases": {},
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".numerical-reference-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        for spec in recipe["cases"]:
            print(f"Generating independent HF reference: {spec['id']}", flush=True)
            case_dir = staging / spec["id"]
            case_dir.mkdir()
            model, geometry = make_model(modules, spec, case_dir / "model", admission_root)
            near_limit = spec.get("near_limit", False)
            prompt_length = geometry["max_position_embeddings"] - 2 if near_limit else len(recipe["short_prompt"])
            prompt = [recipe["short_prompt"][index % len(recipe["short_prompt"])] for index in range(prompt_length)]
            tensors, trajectory = trajectories(torch, model, geometry, prompt, recipe["decode_inputs"], near_limit, recipe["tolerance"])
            encoded, tensor_records = encode_tensors(torch, tensors)
            (case_dir / "reference.safetensors").write_bytes(encoded)
            record = {"schema_version": SCHEMA_VERSION, "id": spec["id"],
                      "kind": "hf_admission" if "source_variant" in spec else "synthetic_hf",
                      "source_variant": spec.get("source_variant"), "synthetic_seed": spec.get("seed"),
                      "geometry": geometry, "trajectory": trajectory, "tolerance": recipe["tolerance"],
                      "layout": dict(LAYOUT),
                      "model_files": {"model/" + name: file_identity(case_dir / "model" / name)
                                      for name in ("config.json", "model.safetensors")},
                      "tensors": tensor_records}
            (case_dir / "reference.json").write_bytes(json_bytes(record))
            record["files"] = {name: file_identity(case_dir / name)
                               for name in ("reference.safetensors", "reference.json")}
            manifest["cases"][spec["id"]] = record
            del model, tensors
        (staging / "manifest.json").write_bytes(json_bytes(manifest))
        staging.rename(output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=FIXTURE_SOURCE / "recipe.json")
    parser.add_argument("--admission-checksums", type=Path, default=ROOT / "tests/fixtures/inference_admission/checksums.json")
    options = parser.parse_args()
    try:
        generate(options.admission_root, options.output_dir, options.recipe, options.admission_checksums)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Numerical reference generation failed: {error}\n")
    print(options.output_dir / "manifest.json")


if __name__ == "__main__":
    main()
