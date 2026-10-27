#!/usr/bin/env python3
"""Verify optional AVX2/FMA translation-unit boundaries in compile_commands.json."""

import argparse
import json
from pathlib import Path
import re
import shlex
import sys


LIBRARY_SOURCES = {
    "src/backend/cpu/kernels/common/dot_product_f32_avx2.cpp",
    "src/backend/cpu/kernels/rmsnorm/rmsnorm_f32_avx2.cpp",
    "src/backend/cpu/kernels/gemm/gemm_f32_avx2.cpp",
    "src/backend/cpu/kernels/gemm/gemm_packed_b_avx2.cpp",
}
TEST_SOURCE = "tests/unit/backend/cpu/kernels/avx_learning/test_avx_learning_avx2.cpp"
AVX_MACROS = ("GEMM_HAS_AVX2_FMA_KERNEL", "RMSNORM_HAS_AVX2_FMA_KERNEL",
              "DOT_PRODUCT_HAS_AVX2_FMA_KERNEL", "AETHERMIND_TEST_HAS_AVX2_FMA")


def uplift_options(arguments):
    result = set()
    for argument in arguments:
        lower = argument.lower()
        if lower.startswith(("-mavx", "-mfma", "-mf16c", "-msse3", "-mssse3",
                             "-msse4", "-mbmi", "-mlzcnt", "-mpopcnt", "-maes",
                             "-mpclmul", "/arch:avx", "+avx", "+fma")):
            result.add(argument)
        elif lower.startswith("-march=") and lower.split("=", 1)[1] not in {
                "x86-64", "i386", "i486", "i586", "i686", "armv8-a"}:
            result.add(argument)
        elif lower.startswith("-mcpu=") and lower.split("=", 1)[1] != "generic":
            result.add(argument)
        elif lower in {"-march", "-mcpu", "-target-cpu"}:
            result.add(argument)
    return result


def project_sources(source, source_dir):
    def relative(path):
        try:
            value = path.resolve().relative_to(source_dir).as_posix()
        except ValueError:
            return None
        return value if value.startswith(("src/", "apps/", "tests/")) else None

    direct = relative(source)
    if direct is not None:
        return {direct}, False
    if "Unity" not in source.parts:
        return set(), False
    # CMake-generated Unity files live in the build tree. Attribute them by
    # their included sources so a target-wide ISA flag cannot escape checking.
    included = set()
    for name in re.findall(r'^\s*#\s*include\s+"([^"]+)"', source.read_text(), re.MULTILINE):
        path = Path(name)
        value = relative(path if path.is_absolute() else source.parent / path)
        if value is not None:
            included.add(value)
    return included, True


def verify(source_dir, build_dir, expect_avx2):
    source_dir = source_dir.resolve()
    entries = json.loads((build_dir / "compile_commands.json").read_text())
    cache = (build_dir / "CMakeCache.txt").read_text()
    tests_value = next((line.split("=", 1)[1] for line in cache.splitlines()
                        if line.startswith("BUILD_TESTS:BOOL=")), "OFF").upper()
    tests_enabled = tests_value not in {"", "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND"}
    expected = (LIBRARY_SOURCES | ({TEST_SOURCE} if tests_enabled else set())) if expect_avx2 else set()
    seen = set()
    project_entries = 0
    for entry in entries:
        source = Path(entry["file"])
        if not source.is_absolute():
            source = Path(entry["directory"]) / source
        owned_sources, unity = project_sources(source, source_dir)
        if not owned_sources:
            continue
        if unity and owned_sources & (LIBRARY_SOURCES | {TEST_SOURCE}):
            raise ValueError(f"Optional ISA implementation merged into Unity: {source}")
        relative = str(source) if unity else next(iter(owned_sources))
        project_entries += 1
        arguments = entry.get("arguments")
        if arguments is None:
            arguments = shlex.split(entry["command"])
        options = uplift_options(arguments)
        if relative in LIBRARY_SOURCES | {TEST_SOURCE}:
            if relative not in expected:
                raise ValueError(f"Optional AVX2 source compiled in baseline: {relative}")
            lto_options = [argument for argument in arguments
                           if argument.startswith("-flto") or argument == "-fno-lto"]
            if options != {"-mavx2", "-mfma"} or not lto_options or lto_options[-1] != "-fno-lto":
                raise ValueError(f"AVX2 source has invalid ISA/LTO options: {relative}: {sorted(options)}")
            seen.add(relative)
        elif options:
            raise ValueError(f"Baseline translation unit has ISA uplift options: {relative}: {sorted(options)}")
        if not expect_avx2:
            definitions = [argument[2:].split("=", 1)[0] for argument in arguments
                           if argument.startswith("-D")]
            if any(macro in definitions for macro in AVX_MACROS):
                raise ValueError(f"Optional AVX2 availability macro in baseline: {relative}")
    if project_entries == 0:
        raise ValueError("No project compile commands found")
    if seen != expected:
        raise ValueError(f"Missing optional implementation commands: {sorted(expected - seen)}")
    return {"optional_avx2_compiled": expect_avx2,
            "project_compile_commands": project_entries,
            "isa_translation_units": sorted(seen)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--expect-avx2", required=True, choices=("ON", "OFF"))
    options = parser.parse_args()
    try:
        result = verify(options.source_dir, options.build_dir, options.expect_avx2 == "ON")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"CPU ISA build verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
