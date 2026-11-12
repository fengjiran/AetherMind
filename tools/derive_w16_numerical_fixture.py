#!/usr/bin/env python3
"""Derive a mathematical W16 checkpoint from an F32 numerical fixture.

Step-10 formal acceptance needs W16 versions of the calibrated numerical
fixtures. This converts every matrix (2D) weight to F16/BF16 with fixed RNE and
keeps Norm (1D) weights F32, matching the retained-W16 production policy. The
conversion is independent of AetherMind and is cross-checked against the integer
RNE oracle used by the step-1 calibrator. It does not modify the accepted F32
bundle; it writes a fresh gitignored candidate directory.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from calibrate_kv16_reference import oracle_f32_to_f16, oracle_f32_to_bf16


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def narrow_matrix(tensor, weight_dtype):
    narrow = torch.float16 if weight_dtype == "f16" else torch.bfloat16
    oracle = oracle_f32_to_f16 if weight_dtype == "f16" else oracle_f32_to_bf16
    narrowed = tensor.to(narrow)
    # Cross-check the library narrowing against the independent integer oracle
    # on every element (finite fixture weights). View as int16 then truncate to
    # uint16 before widening to uint32, so negative patterns are not sign-extended.
    bits = narrowed.view(torch.int16).numpy().astype(np.uint16).astype(np.uint32)
    src_bits = tensor.numpy().view(np.uint32)
    expected = np.array([oracle(int(b)) for b in src_bits.ravel()], dtype=np.uint32)
    if not np.array_equal(bits.ravel(), expected):
        raise ValueError("narrowing disagrees with the independent RNE oracle")
    return narrowed


def derive(source_dir, output_dir, weight_dtype):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    config = json.loads((source_dir / "config.json").read_text())
    tensors = {}
    converted = []
    with safe_open(str(source_dir / "model.safetensors"), "pt") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            if tensor.ndim == 2:
                tensors[name] = narrow_matrix(tensor, weight_dtype)
                converted.append(name)
            elif tensor.ndim == 1:
                tensors[name] = tensor.to(torch.float32)
            else:
                raise ValueError(f"unexpected tensor rank: {name}")

    save_file(tensors, str(output_dir / "model.safetensors"))
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    provenance = {
        "source_model_sha256": file_sha256(source_dir / "model.safetensors"),
        "source_config_sha256": file_sha256(source_dir / "config.json"),
        "output_model_sha256": file_sha256(output_dir / "model.safetensors"),
        "weight_dtype": weight_dtype,
        "converted_matrix_weights": converted,
        "norm_weights_kept_fp32": [n for n in tensors if n not in converted],
    }
    (output_dir / "derivation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-dtype", choices=("f16", "bf16"), required=True)
    options = parser.parse_args()
    provenance = derive(options.source, options.output, options.weight_dtype)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()