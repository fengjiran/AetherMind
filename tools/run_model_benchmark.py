#!/usr/bin/env python3
"""Run one isolated model benchmark and preserve its complete provenance.

Identity collection reads model files before launch. Preparation measurements
therefore describe pre-read filesystem cache state, never a cold-start claim.
The engine probe owns all timed intervals; Python performs no work inside them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
import uuid

import verify_model_benchmark as contract

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_KEYS = contract.ENVIRONMENT_KEYS
SOURCE_PATHS = (
    "CMakeLists.txt", "CMakePresets.json", "cmake", "include", "src",
    "tests/benchmark", "tools/run_model_benchmark.py",
    "tools/verify_model_benchmark.py",
)


def unsigned(text):
    if not re.fullmatch(r"0|[1-9][0-9]*", text):
        raise argparse.ArgumentTypeError("expected an unsigned decimal integer")
    value = int(text)
    if value > (1 << 63) - 1:
        raise argparse.ArgumentTypeError("integer exceeds the supported Int64 range")
    return value


def token(text):
    value = unsigned(text)
    if value > (1 << 32) - 1:
        raise argparse.ArgumentTypeError("token exceeds uint32")
    return value


def prompt(text):
    try:
        result = [token(part) for part in text.split(",")]
    except argparse.ArgumentTypeError as error:
        raise argparse.ArgumentTypeError("prompt requires comma-separated uint32 tokens") from error
    if not result:
        raise argparse.ArgumentTypeError("prompt must not be empty")
    return result


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def model_identity(directory):
    directory = Path(directory).resolve(strict=True)
    paths = {
        path.name: path for path in directory.iterdir()
        if path.name == "config.json" or path.name.endswith(".safetensors")
        or path.name.endswith(".safetensors.index.json")
    }
    if "config.json" not in paths:
        raise ValueError("model directory requires config.json")

    def regular_contained(path):
        relative = path.relative_to(directory)
        component = directory
        for part in relative.parts:
            component /= part
            if component.is_symlink():
                raise ValueError(f"model identity rejects symlink paths: {relative}")
        if not path.is_file() or not path.resolve(strict=True).is_relative_to(directory):
            raise ValueError(f"model identity requires regular contained files: {relative}")

    # The loader may follow index entries into nested shard directories. Hash
    # every referenced file, rather than assuming a root-level glob is complete.
    for name, index in list(paths.items()):
        regular_contained(index)
        if not name.endswith(".safetensors.index.json"):
            continue
        document = strict_json(index.read_text(encoding="utf-8"))
        weight_map = document.get("weight_map") if isinstance(document, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model identity requires a nonempty index weight_map")
        for shard in weight_map.values():
            if not isinstance(shard, str) or not shard or "://" in shard or "\\0" in shard:
                raise ValueError("index shard identity requires a relative filename")
            relative = Path(shard)
            if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
                raise ValueError("index shard identity requires a contained relative filename")
            path = directory / relative
            regular_contained(path)
            paths[relative.as_posix()] = path
    if not any(name.endswith(".safetensors") for name in paths):
        raise ValueError("model directory requires Safetensors weights")
    result = []
    for name, path in sorted(paths.items()):
        regular_contained(path)
        result.append({"path": name, **file_identity(path)})
    return result


def git_bytes(arguments, allowed=(0,)):
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode not in allowed:
        raise ValueError("git provenance collection failed: " + result.stderr.decode(errors="replace"))
    return result.stdout


def source_snapshot():
    commit = git_bytes(["rev-parse", "HEAD"]).decode().strip()
    dirty = bool(git_bytes(["status", "--porcelain", "--untracked-files=normal"]))
    patch = git_bytes(["diff", "--binary", "HEAD", "--", *SOURCE_PATHS])
    extra = git_bytes(["ls-files", "-z", "--others", "--exclude-standard", "--", *SOURCE_PATHS])
    for name in sorted(item for item in extra.split(b"\0") if item):
        path = ROOT / os.fsdecode(name)
        if path.is_file() and not path.is_symlink():
            patch += git_bytes(
                ["diff", "--no-index", "--binary", "--", "/dev/null", str(path)],
                allowed=(0, 1),
            )
    return commit, dirty, patch


def linked_library(binary, environment):
    result = subprocess.run(
        ["ldd", str(binary)], text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment, check=False,
    )
    if result.returncode:
        raise ValueError("ldd could not resolve the benchmark's AetherMind library")
    matches = re.findall(r"^\s*libAetherMind\.so\s+=>\s+(.+?)\s+\(", result.stdout, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("benchmark must dynamically resolve exactly one libAetherMind.so")
    library = Path(matches[0]).resolve(strict=True)
    if not library.is_file():
        raise ValueError("resolved AetherMind library is not a regular file")
    return library, result.stdout


def machine_identity():
    """Collect stable visible topology; virtualization limits interpretation."""
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return ""

    cpuinfo = read("/proc/cpuinfo")
    models = [
        line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
        if line.startswith("model name") and ":" in line
    ]
    kernel = platform.release()
    try:
        os_release = platform.freedesktop_os_release().get("PRETTY_NAME", "unavailable")
    except OSError:
        os_release = "unavailable"
    memory = re.search(r"^MemTotal:\s+(\d+)\s+kB$", read("/proc/meminfo"), re.MULTILINE)
    governors = sorted({
        value for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")
        if (value := read(path))
    })
    turbo = "unavailable"
    for path in ("/sys/devices/system/cpu/intel_pstate/no_turbo",
                 "/sys/devices/system/cpu/cpufreq/boost"):
        value = read(path)
        if value:
            turbo = path.rsplit("/", 1)[-1] + "=" + value
            break
    nodes = sorted(
        int(path.name[4:]) for path in Path("/sys/devices/system/node").glob("node[0-9]*")
        if path.name[4:].isdigit()
    )
    virtualization = ("wsl" if "microsoft" in kernel.lower() else
                      "hypervisor_visible" if "hypervisor" in cpuinfo else "not_detected")
    return {
        "cpu_model": models[0] if models else platform.processor() or "unavailable",
        "logical_cpus": os.cpu_count() or 1,
        "os_release": os_release, "kernel": kernel, "virtualization": virtualization,
        "memory_total_bytes": int(memory[1]) * 1024 if memory else 0,
        "governors": governors, "turbo": turbo, "numa_nodes": nodes,
    }


def json_text(document):
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def run(options):
    if platform.system() != "Linux":
        raise ValueError("model benchmark provenance currently requires Linux")
    binary = options.binary.resolve(strict=True)
    directory = options.model.resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("benchmark binary must be an executable file")
    if not directory.is_dir():
        raise ValueError("model path must be a directory")
    environment = os.environ.copy()
    if options.mode == "timing" and any(environment.get(name) for name in ("LD_PRELOAD", "LD_AUDIT")):
        raise ValueError("timing requires an unmodified loader; use a separate profile process")
    commit, dirty, patch = source_snapshot()
    model_files = model_identity(directory)
    library, dependencies = linked_library(binary, environment)
    machine = machine_identity()
    binary_identity = file_identity(binary)
    library_identity = file_identity(library)
    command = [
        str(binary), "--model", str(directory),
        "--prompt", ",".join(map(str, options.prompt)),
        "--max-new-tokens", str(options.max_new_tokens),
        "--preset", options.preset, "--warmup", str(options.warmup),
        "--repetitions", str(options.repetitions), "--mode", options.mode,
    ]
    if options.eos_token is not None:
        command += ["--eos-token", str(options.eos_token)]
    started = datetime.now(timezone.utc)
    host_id = re.sub(r"[^a-z0-9-]+", "-", socket.gethostname().lower()).strip("-")[:32] or "host"
    variant = options.variant or f"{options.preset}-{options.mode}"
    run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}_{commit[:12]}_{host_id}_{variant}-{uuid.uuid4().hex[:8]}"
    destination = options.output_root.resolve() / run_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "source.patch").write_bytes(patch)
    (destination / "dynamic-libraries.txt").write_text(dependencies)
    (destination / "machine.json").write_text(json_text(machine))
    (destination / "invocation.json").write_text(json_text({
        "runner_command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "engine_command": command, "cwd": str(ROOT),
        "timeout_seconds": options.timeout_seconds,
    }))
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=environment, text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=options.timeout_seconds, check=False,
        )
        (destination / "engine.stdout").write_text(completed.stdout)
        (destination / "engine.stderr").write_text(completed.stderr)
        if completed.returncode:
            raise ValueError(f"engine exited {completed.returncode}; see engine.stderr")
        engine = strict_json(completed.stdout)
        (destination / "engine.json").write_text(json_text(engine))
        expected_workload = {
            "preset": options.preset, "prompt_tokens": options.prompt,
            "max_new_tokens": options.max_new_tokens, "eos_token_id": options.eos_token,
            "context_capacity": len(options.prompt) + max(options.max_new_tokens - 1, 0),
            "warmup": options.warmup, "repetitions": options.repetitions,
        }
        if engine.get("workload") != expected_workload or engine.get("mode") != options.mode:
            raise ValueError("engine workload/mode differs from the requested workload")
        if model_identity(directory) != model_files:
            raise ValueError("model files changed during the benchmark")
        if file_identity(binary) != binary_identity or file_identity(library) != library_identity:
            raise ValueError("benchmark executable or library changed during the run")
        if machine_identity() != machine:
            raise ValueError("visible machine configuration changed during the benchmark")
        if source_snapshot() != (commit, dirty, patch):
            raise ValueError("relevant source state changed during the benchmark")
        report = {
            "schema_version": 1, "status": "complete", "run_id": run_id, "mode": options.mode,
            "provenance": {
                "commit": commit, "dirty": dirty, "command": command,
                "host_id": host_id, "platform": platform.platform(), "machine": machine,
                "affinity": sorted(os.sched_getaffinity(0)),
                "environment": {name: environment.get(name, "") for name in ENVIRONMENT_KEYS},
                "executable": {"path": str(binary), **binary_identity},
                "library": {"path": str(library), **library_identity},
                "source_patch_sha256": hashlib.sha256(patch).hexdigest(),
                "started_utc": started.isoformat(),
            },
            "model": {"directory": str(directory), "files": model_files},
            "page_cache": "pre_read_by_identity_collection", "engine": engine,
        }
        contract.verify_report(report)
        summary = contract.summarize(report) if options.mode == "timing" else None
        (destination / "report.json").write_text(json_text(report))
        if summary is not None:
            (destination / "summary.json").write_text(json_text(summary))
        (destination / "status.json").write_text(json_text({"status": "complete"}))
    except (OSError, ValueError, contract.VerificationError, subprocess.TimeoutExpired) as error:
        if isinstance(error, subprocess.TimeoutExpired):
            for name, value in (("engine.stdout", error.stdout), ("engine.stderr", error.stderr)):
                (destination / name).write_text(
                    value.decode(errors="replace") if isinstance(value, bytes) else value or ""
                )
        (destination / "status.json").write_text(json_text({"status": "failed", "error": str(error)}))
        raise ValueError(f"{error}; artifacts: {destination}") from error
    return destination / "report.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", type=prompt, required=True)
    parser.add_argument("--max-new-tokens", type=unsigned, required=True)
    parser.add_argument("--eos-token", type=token)
    parser.add_argument("--preset", choices=("o1-plain", "o2-packed"), required=True)
    parser.add_argument("--warmup", type=unsigned, default=1)
    parser.add_argument("--repetitions", type=unsigned, default=5)
    parser.add_argument("--mode", choices=("timing", "profile", "allocations"), default="timing")
    parser.add_argument("--output-root", type=Path, default=ROOT / "benchmark-results/models")
    parser.add_argument("--variant", default="")
    parser.add_argument("--timeout-seconds", type=unsigned, default=600)
    options = parser.parse_args()
    if not 1 <= options.repetitions <= 10000 or options.warmup > 10000:
        parser.error("repetitions must be 1..10000 and warmup must be 0..10000")
    if not options.timeout_seconds:
        parser.error("timeout-seconds must be positive")
    if options.variant and not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,63}", options.variant):
        parser.error("variant must be 1..64 lowercase letters, digits, dots or hyphens")
    try:
        path = run(options)
    except (OSError, ValueError, contract.VerificationError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Model benchmark failed: {error}\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
