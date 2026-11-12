#!/usr/bin/env python3
"""Materialize/verify the pinned independent KV16 production-test bundle.

Preparation needs the reference Python environment. Verification uses only the
standard library and never accepts a changed identity. Existing FP32 assets and
the independently generated calibration bundle remain untouched.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "tests/fixtures/kv16_reference/checksums.json"
HEADER = ROOT / "tests/unit/inference/fixtures/kv16_reference_identities.h"
CASES = ("mha-untied", "mha-tied", "gqa-tail", "mqa-tail", "mha-near-limit")
CALIBRATOR_SHA256 = "0c1b0be328c4dcb2453631ec27b125138119dad97df652cc48eccffb8affaba6"
CALIBRATION_MANIFEST_SHA256 = "5c9dc628f4fbe90fde8a28bdd30d313d0c4a17ecf12d79b3875d39bb474f93e7"


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def identity(path):
    payload = Path(path).read_bytes()
    fnv = 14695981039346656037
    for byte in payload:
        fnv = ((fnv ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
            "fnv1a64": f"{fnv:016x}"}


def record_bundle(root, calibration_manifest):
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p != root / "manifest.json")
    return {"schema_version": 1,
            "calibrator_sha256": CALIBRATOR_SHA256,
            "calibration_manifest_sha256": CALIBRATION_MANIFEST_SHA256,
            "source_bundle_sha256": calibration_manifest["source_bundle_sha256"],
            "files": {p.relative_to(root).as_posix(): identity(p) for p in paths}}


def verify(root, expected=EXPECTED):
    root = Path(root)
    trusted = read_json(expected)
    manifest = read_json(root / "manifest.json")
    if manifest != trusted:
        raise ValueError("KV16 manifest differs from the reviewed expected identity")
    actual_names = {p.relative_to(root).as_posix() for p in root.rglob("*")
                    if p.is_file() and p != root / "manifest.json"}
    if actual_names != set(trusted["files"]):
        raise ValueError("KV16 reference file set mismatch")
    for relative, expected_identity in trusted["files"].items():
        if identity(root / relative) != expected_identity:
            raise ValueError(f"KV16 reference identity mismatch: {relative}")
    return {"status": "verified", "files": len(actual_names),
            "manifest_sha256": identity(root / "manifest.json")["sha256"]}


def write_identity_header(record):
    rows = []
    for relative, item in sorted(record["files"].items()):
        rows.append(f'        FileIdentity{{"{relative}", {item["bytes"]}ULL, 0x{item["fnv1a64"]}ULL}},')
    content = """#ifndef AETHERMIND_TEST_KV16_REFERENCE_IDENTITIES_H
#define AETHERMIND_TEST_KV16_REFERENCE_IDENTITIES_H

// Generated from the reviewed SHA-256 manifest by prepare_kv16_reference.py.
// The direct GTest path checks size/FNV to catch absent or changed assets even
// when the separate SHA-256 preparation/verification step was not invoked.
#include <array>
#include <cstdint>
#include <string_view>

namespace aethermind::test::kv16 {
struct FileIdentity {
    std::string_view path;
    uint64_t bytes;
    uint64_t fnv1a64;
};
inline constexpr std::array kFileIdentities{
""" + "\n".join(rows) + """
};
} // namespace aethermind::test::kv16
#endif
"""
    HEADER.parent.mkdir(parents=True, exist_ok=True)
    HEADER.write_text(content)


def prepare(source, calibration, output, record_identities):
    import derive_w16_numerical_fixture as derive

    source, calibration, output = Path(source), Path(calibration), Path(output)
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if identity(calibration / "manifest.json")["sha256"] != CALIBRATION_MANIFEST_SHA256:
        raise ValueError("independent calibration manifest is not the reviewed bundle")
    manifest = read_json(calibration / "manifest.json")
    if manifest["calibrator"]["sha256"] != CALIBRATOR_SHA256:
        raise ValueError("independent calibrator source identity mismatch")
    if identity(source / "manifest.json")["sha256"] != manifest["source_bundle_sha256"]:
        raise ValueError("FP32 source bundle identity mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kv16-reference-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        for case in CASES:
            measured = manifest["measurements"]["cases"][case]
            model = source / case / "model"
            for name, expected in measured["model_files"].items():
                actual = identity(model / name)
                if any(actual[key] != value for key, value in expected.items()):
                    raise ValueError(f"source model identity mismatch: {case}/{name}")
            for weight in ("f16", "bf16"):
                derived = staging / case / ("w" + weight)
                derived.parent.mkdir(parents=True, exist_ok=True)
                derive.derive(model, derived, weight)
                for kv in ("f16", "bf16"):
                    combo = f"w{weight}-kv{kv}"
                    oracle = calibration / case / combo / "impl_f64_oracle.safetensors"
                    # Check every oracle tensor's original payload identity;
                    # no inference or conversion is performed during import.
                    data = oracle.read_bytes()
                    import struct
                    header_size = struct.unpack("<Q", data[:8])[0]
                    header = json.loads(data[8:8 + header_size])
                    expected_tensors = measured["combos"][combo]["tensors"]["impl_f64_oracle"]
                    if set(header) != set(expected_tensors):
                        raise ValueError(f"oracle tensor set mismatch: {case}/{combo}")
                    for name, expected in expected_tensors.items():
                        tensor = header[name]
                        begin, end = tensor["data_offsets"]
                        payload = data[8 + header_size + begin:8 + header_size + end]
                        if (tensor["dtype"] != expected["dtype"] or tensor["shape"] != expected["shape"]
                                or len(payload) != expected["bytes"]
                                or hashlib.sha256(payload).hexdigest() != expected["payload_sha256"]):
                            raise ValueError(f"oracle payload mismatch: {case}/{combo}/{name}")
                    shutil.copyfile(oracle, derived / ("oracle-kv" + kv + ".safetensors"))
        record = record_bundle(staging, manifest)
        encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
        (staging / "manifest.json").write_text(encoded)
        if record_identities:
            if EXPECTED.exists() or HEADER.exists():
                raise ValueError("reviewed identities already exist; refusing automatic replacement")
            EXPECTED.parent.mkdir(parents=True, exist_ok=True)
            EXPECTED.write_text(encoded)
            write_identity_header(record)
        verify(staging)
        staging.rename(output)
    return verify(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("--output", type=Path, default=ROOT / ".models/kv16-reference")
    parser.add_argument("--source", type=Path, default=ROOT / ".models/inference-numerical")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--record-identities", action="store_true",
                        help="initial reviewed bundle only; refuses to overwrite existing pins")
    options = parser.parse_args()
    try:
        if options.mode == "verify":
            result = verify(options.output)
        else:
            if options.calibration is None:
                parser.error("prepare requires --calibration")
            result = prepare(options.source, options.calibration, options.output, options.record_identities)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"KV16 fixture error: {error}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
