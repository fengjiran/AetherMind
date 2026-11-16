#!/usr/bin/env python3
"""Import the reviewed A16 calibration evidence into a production fixture.

The A16 calibration (proposal 11 section 4.5) produced, per numerical case and
per same-dtype combination, three safetensors bundles:

  impl_f32.safetensors       F32-arithmetic quantized data flow
  impl_f64_oracle.safetensors FP64-reduction isomorphic oracle
  quality_w16_a32.safetensors  the same W16 checkpoint executed as A32/KV32

Production acceptance compares real CPU execution with the FP64 oracle within
the frozen implementation budgets, and with a live W16/A32 run within the frozen
quality budgets. Only the FP64 oracle is imported here; the quality comparison
runs production code directly.

The tool verifies every oracle payload against the calibration manifest (which
records each tensor's dtype, shape, byte count, and SHA-256), requires all
frozen budgets to match, and requires every measured case to have passed. It
then writes a self-contained fixture: the W16 checkpoint plus the A16 oracle per
case and weight dtype.

Usage:
  python3 tools/prepare_a16_reference.py prepare \
      --calibration benchmark-results/models/a16-calibration/run-a \
      --checkpoints .models/kv16-reference \
      --output .models/a16-reference
  python3 tools/prepare_a16_reference.py verify --output .models/a16-reference
  python3 tools/prepare_a16_reference.py record-identities \
      --output .models/a16-reference \
      --header tests/unit/inference/fixtures/a16_reference_identities.h

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

# Frozen identity of the reviewed calibrator (proposal 11 section 4.5).
CALIBRATOR_SHA256 = "11adc1b141ae5e4da6f6132aebbdf4c42be3f995e35831dc55486d3d50ce6d74"

# Frozen budgets from proposal 11 section 4.5. The manifest must agree.
FROZEN_BUDGETS = {
    "impl-f16": {"logits": [1e-4, 1e-3], "kv": [2e-4, 2e-3]},
    "impl-bf16": {"logits": [2e-4, 2e-3], "kv": [2e-3, 2e-2]},
    "quality-f16": {"logits": [2e-3, 6e-3], "kv": [1e-3, 1e-2]},
    "quality-bf16": {"logits": [1e-2, 3e-2], "kv": [5e-3, 3e-2]},
}

CASES = ("mha-untied", "mha-tied", "gqa-tail", "mqa-tail", "mha-near-limit")

# (weight directory, calibration combo directory, fixture oracle file name)
COMBINATIONS = (
    ("wf16", "wf16-af16-kvf16", "oracle-af16.safetensors"),
    ("wbf16", "wbf16-abf16-kvbf16", "oracle-af16.safetensors"),
)

CHECKPOINT_FILES = ("config.json", "model.safetensors")
CHECKSUMS_NAME = "checksums.json"
ORACLE_LABEL = "impl_f64_oracle"
ORACLE_TENSOR_COUNT = 22


class PreparationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fnv1a64_file(path: Path) -> int:
    value = 14695981039346656037
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            for byte in chunk:
                value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def load_safetensors(path: Path):
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise PreparationError(f"{path}: truncated safetensors header length")
        (header_length,) = struct.unpack("<Q", raw)
        header = json.loads(handle.read(header_length))
        data_start = 8 + header_length
    tensors = {name: entry for name, entry in header.items() if name != "__metadata__"}
    return tensors, data_start


def verify_oracle(path: Path, records: dict) -> None:
    tensors, data_start = load_safetensors(path)
    if len(tensors) != ORACLE_TENSOR_COUNT:
        raise PreparationError(
            f"{path}: expected {ORACLE_TENSOR_COUNT} tensors, found {len(tensors)}")
    if set(tensors) != set(records):
        missing = sorted(set(records) - set(tensors))
        extra = sorted(set(tensors) - set(records))
        raise PreparationError(f"{path}: tensor set mismatch, missing={missing} extra={extra}")
    with path.open("rb") as handle:
        for name, record in records.items():
            entry = tensors[name]
            if entry["dtype"] != record["dtype"]:
                raise PreparationError(f"{path}: {name} dtype drift")
            if list(entry["shape"]) != list(record["shape"]):
                raise PreparationError(f"{path}: {name} shape drift")
            begin, end = entry["data_offsets"]
            if end - begin != record["bytes"]:
                raise PreparationError(f"{path}: {name} byte count drift")
            handle.seek(data_start + begin)
            payload = handle.read(end - begin)
            if hashlib.sha256(payload).hexdigest() != record["payload_sha256"]:
                raise PreparationError(f"{path}: {name} payload hash mismatch")


def load_manifest(calibration: Path) -> dict:
    manifest_path = calibration / "manifest.json"
    if not manifest_path.is_file():
        raise PreparationError(f"missing calibration manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    calibrator = manifest.get("calibrator", {})
    if calibrator.get("sha256") != CALIBRATOR_SHA256:
        raise PreparationError(
            "calibration manifest does not reference the reviewed calibrator")
    if manifest.get("budgets") != FROZEN_BUDGETS:
        raise PreparationError("calibration budgets differ from the frozen proposal values")
    cases = manifest.get("measurements", {}).get("cases", {})
    if set(cases) != set(CASES):
        raise PreparationError(f"calibration case set mismatch: {sorted(cases)}")
    for case, case_entry in cases.items():
        for weight_dir, combo_name, _ in COMBINATIONS:
            combo = case_entry.get("combos", {}).get(combo_name)
            if combo is None:
                raise PreparationError(f"{case}: missing combination {combo_name}")
            if not combo.get("pass"):
                raise PreparationError(f"{case}/{combo_name}: calibration did not pass")
            if not combo.get("impl_correctness", {}).get("pass"):
                raise PreparationError(f"{case}/{combo_name}: implementation budget failed")
            if not combo.get("relative_w16_a32", {}).get("pass"):
                raise PreparationError(f"{case}/{combo_name}: quality budget failed")
            if weight_dir == "wf16" and combo.get("activation_dtype") != "f16":
                raise PreparationError(f"{case}/{combo_name}: activation dtype mismatch")
            if weight_dir == "wbf16" and combo.get("activation_dtype") != "bf16":
                raise PreparationError(f"{case}/{combo_name}: activation dtype mismatch")
    return manifest


def prepare(calibration: Path, checkpoints: Path, output: Path, force: bool) -> None:
    manifest = load_manifest(calibration)
    if output.exists() and any(output.iterdir()):
        if not force:
            raise PreparationError(f"{output} is not empty; pass --force to rebuild")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        for weight_dir, combo_name, oracle_name in COMBINATIONS:
            source_dir = checkpoints / case / weight_dir
            for name in CHECKPOINT_FILES:
                if not (source_dir / name).is_file():
                    raise PreparationError(f"missing source checkpoint file: {source_dir / name}")
            oracle_source = calibration / case / combo_name / f"{ORACLE_LABEL}.safetensors"
            if not oracle_source.is_file():
                raise PreparationError(f"missing calibration oracle: {oracle_source}")
            records = manifest["measurements"]["cases"][case]["combos"][combo_name]["tensors"][ORACLE_LABEL]
            verify_oracle(oracle_source, records)

            target_dir = output / case / weight_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            for name in CHECKPOINT_FILES:
                shutil.copyfile(source_dir / name, target_dir / name)
            shutil.copyfile(oracle_source, target_dir / oracle_name)

    files = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        files.append({
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    checksums = {
        "provenance": {
            "calibration_manifest_sha256": sha256_file(calibration / "manifest.json"),
            "calibrator_sha256": CALIBRATOR_SHA256,
            "oracle_label": ORACLE_LABEL,
            "source_checkpoints": checkpoints.name,
        },
        "files": files,
    }
    (output / CHECKSUMS_NAME).write_text(json.dumps(checksums, indent=1) + "\n")
    print(f"prepared {len(files)} files in {output}")


def verify(output: Path) -> None:
    checksums_path = output / CHECKSUMS_NAME
    if not checksums_path.is_file():
        raise PreparationError(f"missing {checksums_path}")
    checksums = json.loads(checksums_path.read_text())
    if checksums.get("provenance", {}).get("calibrator_sha256") != CALIBRATOR_SHA256:
        raise PreparationError("fixture provenance does not reference the reviewed calibrator")
    expected = {entry["path"]: entry for entry in checksums["files"]}
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != CHECKSUMS_NAME
    }
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise PreparationError(f"file set mismatch, missing={missing} extra={extra}")
    for path, entry in expected.items():
        file_path = output / path
        if file_path.stat().st_size != entry["bytes"]:
            raise PreparationError(f"{path}: byte count drift")
        if sha256_file(file_path) != entry["sha256"]:
            raise PreparationError(f"{path}: sha256 drift")
    print(f"verified {len(expected)} files in {output}")


def record_identities(output: Path, header: Path) -> None:
    verify(output)
    entries = []
    for path_text in sorted(
            p.relative_to(output).as_posix()
            for p in output.rglob("*")
            if p.is_file() and p.name != CHECKSUMS_NAME):
        file_path = output / path_text
        entries.append(
            f'        FileIdentity{{"{path_text}", {file_path.stat().st_size}ULL, '
            f'0x{fnv1a64_file(file_path):016x}ULL}},')
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(
        "#ifndef AETHERMIND_TEST_A16_REFERENCE_IDENTITIES_H\n"
        "#define AETHERMIND_TEST_A16_REFERENCE_IDENTITIES_H\n"
        "\n"
        "// Generated by tools/prepare_a16_reference.py from the reviewed A16\n"
        "// calibration evidence. The direct GTest path checks size/FNV to catch\n"
        "// absent or changed assets even when the SHA-256 verification step was\n"
        "// not invoked.\n"
        "#include <array>\n"
        "#include <cstdint>\n"
        "#include <string_view>\n"
        "\n"
        "namespace aethermind::test::a16 {\n"
        "struct FileIdentity {\n"
        "    std::string_view path;\n"
        "    uint64_t bytes;\n"
        "    uint64_t fnv1a64;\n"
        "};\n"
        "inline constexpr std::array kFileIdentities{\n"
        + "\n".join(entries) + "\n"
        "};\n"
        "} // namespace aethermind::test::a16\n"
        "#endif\n")
    print(f"recorded {len(entries)} identities in {header}")


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--calibration", type=Path, required=True)
    prepare_parser.add_argument("--checkpoints", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--force", action="store_true")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output", type=Path, required=True)

    record_parser = subparsers.add_parser("record-identities")
    record_parser.add_argument("--output", type=Path, required=True)
    record_parser.add_argument("--header", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.calibration, args.checkpoints, args.output, args.force)
        elif args.command == "verify":
            verify(args.output)
        else:
            record_identities(args.output, args.header)
    except PreparationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
