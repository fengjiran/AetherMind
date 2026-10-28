#!/usr/bin/env python3
"""Run the committed five-batch validation matrix and preserve evidence.

This is an orchestrator, not a second test framework.  It only calls existing
fixture checkers, CMake presets, CTest and black-box probes, then verifies that
their inventories and reports prove a complete run.  It never downloads assets,
generates a new golden, or turns benchmark observations into a performance gate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_ctest_report as ctest_report

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("baseline", "avx2", "asan", "tsan", "cli", "benchmark", "benchmark-baseline")
TEST_SLICES = {
    "baseline": (("baseline-production", "validation-baseline", "^validation$", "^isa-avx2$"),
                 ("baseline-numerical", "validation-baseline", "^numerical-model$", "")),
    "avx2": (("avx-production", "validation-avx2", "^validation$", "^isa-avx2$"),
             ("avx-isa", "validation-avx2", "^isa-avx2$", ""),
             ("avx-numerical", "validation-avx2", "^numerical-model$", "")),
    "asan": (("asan-production", "validation-asan-ubsan", "^validation$", "^(interposer|isa-avx2|preload)$"),
             ("asan-numerical", "validation-asan-ubsan", "^numerical-model$", ""),
             ("asan-isa", "validation-asan-ubsan", "^isa-avx2$", "")),
    "tsan": (("tsan-applicable", "validation-tsan", "^(validation|thread-smoke)$", "^(interposer|isa-avx2|preload|cli)$"),),
}
PRESET_FOR_PROFILE = {"baseline": "validation-baseline", "avx2": "validation-avx2",
                      "asan": "validation-asan-ubsan", "tsan": "validation-tsan",
                      "cli": "validation-cli-only", "benchmark": "benchmark-model",
                      "benchmark-baseline": "benchmark-model-baseline"}
TARGETS = {"baseline": ("aethermind_unit_tests", "aethermind_infer"),
           "avx2": ("aethermind_unit_tests", "aethermind_infer"),
           "asan": ("aethermind_unit_tests", "aethermind_infer"),
           "tsan": ("aethermind_unit_tests", "aethermind_infer"),
           "cli": ("aethermind_infer",), "benchmark": ("aethermind_model_benchmark",),
           "benchmark-baseline": ("aethermind_model_benchmark",)}
ACTIVE_RUN = None
ACTIVE_RESULT = None
ACTIVE_STARTED = None


class FinalValidationError(ValueError):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _identity(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _run(argv, log: Path, environment=None, cwd=ROOT):
    log.parent.mkdir(parents=True, exist_ok=True)
    selected = {name: (environment or os.environ).get(name, "") for name in
                ("ASAN_OPTIONS", "UBSAN_OPTIONS", "TSAN_OPTIONS", "LD_PRELOAD", "LD_AUDIT")}
    started = datetime.now(timezone.utc).isoformat()
    interrupted = False
    try:
        with log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(argv, cwd=cwd, env=environment, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                with process.stdout:
                    for line in process.stdout:
                        output.write(line)
                        output.flush()
                code = process.wait()
            except KeyboardInterrupt:
                interrupted = True
                process.terminate()
                process.wait()
                output.write("final validation interrupted by user\n")
                code = 130
    except OSError as error:
        log.write_text(f"command launch failed: {error}\n", encoding="utf-8")
        code = 127
    return {"argv": list(map(str, argv)), "returncode": code, "log": str(log), "ok": code == 0,
            "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "environment": selected, "interrupted": interrupted}


def _contains_interruption(value):
    if isinstance(value, dict):
        return bool(value.get("interrupted")) or any(_contains_interruption(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_interruption(item) for item in value)
    return False


def _source_state():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    scope = ("CMakeLists.txt", "CMakePresets.json", "cmake", "include", "src", "apps", "tests", "tools")
    tracked = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", *scope], cwd=ROOT)
    untracked = subprocess.check_output(["git", "ls-files", "-z", "--others", "--exclude-standard", "--", *scope], cwd=ROOT)
    patch = bytearray(tracked)
    for raw in sorted(item for item in untracked.split(b"\0") if item):
        rel = os.fsdecode(raw)
        path = ROOT / rel
        if path.is_file() and not path.is_symlink():
            diff = subprocess.run(["git", "diff", "--no-index", "--binary", "--", "/dev/null", str(path)],
                                  cwd=ROOT, stdout=subprocess.PIPE, check=False)
            patch.extend(diff.stdout)
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT)
    return {"head": head, "dirty": bool(status), "source_scope": list(scope),
            "source_patch_sha256": hashlib.sha256(patch).hexdigest(), "_patch": bytes(patch)}


def _record_source_snapshot(run: Path, label: str):
    state = _source_state()
    (run / f"source-{label}.patch").write_bytes(state["_patch"])
    public = {key: value for key, value in state.items() if key != "_patch"}
    (run / f"source-{label}.json").write_text(_json(public))
    return state


def _public_state(state):
    return {key: value for key, value in state.items() if key != "_patch"}


def _emergency_finish(error):
    """Leave an inspectable terminal result even if orchestration itself fails."""
    if ACTIVE_RUN is None or ACTIVE_RESULT is None:
        return
    result = ACTIVE_RESULT
    result.update(status="failed", error=str(error), interrupted=isinstance(error, KeyboardInterrupt),
                  milestone_complete=False)
    try:
        after = _record_source_snapshot(ACTIVE_RUN, "after")
        result["source_after"] = _public_state(after)
        result["source_unchanged"] = (_public_state(ACTIVE_STARTED) == _public_state(after)
                                      if ACTIVE_STARTED is not None else False)
    except OSError as snapshot_error:
        result["source_after_error"] = str(snapshot_error)
    (ACTIVE_RUN / "summary.json").write_text(_json(result))
    (ACTIVE_RUN / "status.json").write_text(_json(result))


def _load_policy(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise FinalValidationError(f"duplicate policy key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise FinalValidationError(f"non-finite policy value: {value}")
    try:
        result = json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalValidationError(f"cannot load final validation policy: {error}") from error
    if (not isinstance(result, dict) or result.get("schema_version") != 1 or
            not isinstance(result.get("ctest_slices"), dict) or
            not isinstance(result.get("unittest_suites"), dict)):
        raise FinalValidationError("final validation policy has an unsupported schema")
    return result


def _validate_policy(policy, requested):
    for profile in requested:
        for name, *_ in TEST_SLICES.get(profile, ()):
            value = policy["ctest_slices"].get(name)
            if not isinstance(value, dict) or type(value.get("expected_count")) is not int or value["expected_count"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("names_sha256", ""))):
                raise FinalValidationError(f"policy has no valid frozen CTest inventory for {name}")
        suite = "cli" if profile == "cli" else "benchmark" if profile.startswith("benchmark") else None
        if suite:
            value = policy["unittest_suites"].get(suite)
            if (not isinstance(value, dict) or type(value.get("expected_count")) is not int or
                    value["expected_count"] <= 0 or
                    not re.fullmatch(r"[0-9a-f]{64}", str(value.get("names_sha256", "")))):
                raise FinalValidationError(f"policy has no valid unittest count for {suite}")


def _require_absolute(path: Path, name: str):
    if not path.is_absolute():
        raise FinalValidationError(f"{name} must be absolute")
    return path


def _preflight(models: Path, run: Path, label="before"):
    source = models / "tiny-random-LlamaForCausalLM"
    commands = [
        [sys.executable, "tools/generate_fp32_admission_hf_fixtures.py", "--source", str(source),
         "--recipe", "tests/fixtures/inference_admission/recipe.json", "--output-dir", str(models / "inference-admission"),
         "--check", "tests/fixtures/inference_admission/checksums.json"],
        [sys.executable, "tools/verify_llama_numerical_reference.py", "--root", str(models / "inference-numerical"),
         "--expected", "tests/fixtures/inference_numerical/checksums.json"],
    ]
    records = []
    for index, command in enumerate(commands):
        record = _run(command, run / "assets" / f"{label}-check-{index}.log")
        records.append(record)
        if not record["ok"]:
            raise FinalValidationError(f"asset preflight failed; see {record['log']}")
    return {"checks": records, "source": _identity(source / "model.safetensors"),
            "admission_manifest": _identity(models / "inference-admission" / "generated-manifest.json"),
            "numerical_manifest": _identity(models / "inference-numerical" / "manifest.json")}


def _asset_identities(result):
    return {key: result[key] for key in ("source", "admission_manifest", "numerical_manifest")}


def _configure(profile, options, run):
    preset = PRESET_FOR_PROFILE[profile]
    command = ["cmake", "--preset", preset,
               f"-DAETHERMIND_TEST_MODELS_DIR={options.models_dir}",
               f"-DAETHERMIND_TEST_NUMERICAL_DIR={options.models_dir / 'inference-numerical'}"]
    if options.spdlog_source:
        command.append(f"-DFETCHCONTENT_SOURCE_DIR_SPDLOG={options.spdlog_source}")
    if options.google_benchmark_source and profile.startswith("benchmark"):
        command.append(f"-DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK={options.google_benchmark_source}")
    if profile == "cli":
        command += ["-DCMAKE_DISABLE_FIND_PACKAGE_GTest=ON", "-DCMAKE_DISABLE_FIND_PACKAGE_Python3=ON"]
    if options.fresh_build:
        command += ["-B", str(_build_dir(options, preset))]
    return _run(command, run / profile / "configure.log")


def _build(profile, options, run):
    if options.fresh_build:
        command = ["cmake", "--build", str(_build_dir(options, PRESET_FOR_PROFILE[profile])), "--parallel", str(options.jobs),
                   "--target", *TARGETS[profile]]
    else:
        command = ["cmake", "--build", "--preset", PRESET_FOR_PROFILE[profile], "--parallel", str(options.jobs),
                   "--target", *TARGETS[profile]]
    return _run(command, run / profile / "build.log")


def _build_dir(options, preset):
    return options.build_dirs.get(preset, ROOT / "build" / "validation" / preset)


def _test_slice(spec, options, policy, run, prefix=()):
    name, preset, include, exclude = spec
    expected = policy["ctest_slices"].get(name)
    if not isinstance(expected, dict):
        raise FinalValidationError(f"policy does not define CTest slice {name}")
    build = _build_dir(options, preset)
    environment = dict(os.environ)
    if preset == "validation-asan-ubsan":
        environment.update(ASAN_OPTIONS="halt_on_error=1:detect_leaks=1",
                           UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
    if preset == "validation-tsan":
        environment["TSAN_OPTIONS"] = "halt_on_error=1"
    inventory = run / "ctest" / f"{name}.inventory.json"
    command = [*prefix, "ctest", "--test-dir", str(build), "--show-only=json-v1", "-L", include]
    if exclude:
        command += ["-LE", exclude]
    record = _run(command, inventory, environment)
    if not record["ok"]:
        return {"status": "failed", "inventory": record}
    try:
        discovered = ctest_report.verify_inventory(ctest_report.load_inventory(inventory), expected["expected_count"], expected["names_sha256"])
    except (ctest_report.VerificationError, KeyError) as error:
        return {"status": "failed", "inventory": record, "error": str(error)}
    junit = run / "ctest" / f"{name}.junit.xml"
    command = [*prefix, "ctest", "--test-dir", str(build), "-L", include, "--output-on-failure", "--no-tests=error", "-j", "2", "--output-junit", str(junit)]
    if exclude:
        command += ["-LE", exclude]
    executed = _run(command, run / "ctest" / f"{name}.log", environment)
    try:
        summary = ctest_report.verify_run(ctest_report.load_inventory(inventory), junit, expected["expected_count"], expected["names_sha256"])
    except (ctest_report.VerificationError, OSError) as error:
        return {"status": "failed", "inventory": record, "execution": executed, "error": str(error), "discovered": len(discovered)}
    return {"status": "passed" if executed["ok"] else "failed", "inventory": record, "execution": executed, "summary": summary}


def _unittest(name, command, expected_count, expected_names_sha256, run, environment=None):
    result = _run(command, run / "probes" / f"{name}.log", environment)
    output = Path(result["log"]).read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Ran (\d+) tests? in [^\n]+\n\nOK\s*$", output)
    cases = re.findall(r"^(test_[A-Za-z0-9_]+ \([^\n]+\)) \.\.\. ok$", output, re.MULTILINE)
    forbidden = re.search(r"\.\.\. (?:FAIL|ERROR|skipped|expected failure|unexpected success)$", output, re.MULTILINE | re.IGNORECASE)
    digest = ctest_report.names_digest(cases)
    ok = (result["ok"] and match is not None and int(match.group(1)) == expected_count and
          len(cases) == expected_count and len(cases) == len(set(cases)) and forbidden is None and
          digest == expected_names_sha256)
    return {"status": "passed" if ok else "failed", "command": result, "expected": expected_count,
            "executed": int(match.group(1)) if match else None, "case_ids": cases,
            "names_sha256": digest}


def _profile_binary(profile, options):
    base = _build_dir(options, PRESET_FOR_PROFILE[profile])
    if profile == "cli":
        return base / "apps" / "inference" / "aethermind_infer"
    return base / "tests" / "benchmark" / "aethermind_model_benchmark"


def _artifacts(profile, options):
    """Record the produced files, without treating their mere presence as a pass."""
    base = _build_dir(options, PRESET_FOR_PROFILE[profile])
    candidates = [base / "CMakeCache.txt", base / "compile_commands.json", base / "src" / "libAetherMind.so"]
    if profile in {"baseline", "avx2", "asan", "tsan"}:
        candidates += [base / "tests" / "unit" / "aethermind_unit_tests",
                       base / "apps" / "inference" / "aethermind_infer"]
    elif profile == "cli":
        candidates.append(base / "apps" / "inference" / "aethermind_infer")
    else:
        candidates.append(base / "tests" / "benchmark" / "aethermind_model_benchmark")
    missing = [str(path) for path in candidates if not path.is_file() or path.is_symlink()]
    if missing:
        raise FinalValidationError(f"profile {profile} is missing required build artifacts: {missing}")
    return [_identity(path) for path in candidates]


def _tsan_prefix(options, run):
    """Select the documented WSL workaround only for its known startup fault."""
    if options.tsan_launcher == "setarch-no-aslr":
        return ("setarch", platform.machine(), "-R"), {"mode": "setarch-no-aslr"}
    if options.tsan_launcher == "native":
        return (), {"mode": "native"}
    unit = _build_dir(options, "validation-tsan") / "tests" / "unit" / "aethermind_unit_tests"
    probe = _run([str(unit), "--gtest_list_tests"], run / "tsan" / "native-startup.log",
                 {**os.environ, "TSAN_OPTIONS": "halt_on_error=1"})
    if probe["ok"]:
        return (), {"mode": "native", "probe": probe}
    output = Path(probe["log"]).read_text(encoding="utf-8", errors="replace").lower()
    if "unexpected memory mapping" not in output:
        raise FinalValidationError("TSAN startup failed for a reason other than unexpected memory mapping")
    retry = _run(["setarch", platform.machine(), "-R", str(unit), "--gtest_list_tests"],
                 run / "tsan" / "setarch-startup.log", {**os.environ, "TSAN_OPTIONS": "halt_on_error=1"})
    if not retry["ok"]:
        raise FinalValidationError("TSAN setarch retry failed")
    return ("setarch", platform.machine(), "-R"), {"mode": "setarch-no-aslr", "probe": probe, "retry": retry}


def run_validation(options):
    global ACTIVE_RUN, ACTIVE_RESULT, ACTIVE_STARTED
    _require_absolute(options.models_dir, "models-dir")
    requested = tuple(options.profiles)
    if len(set(requested)) != len(requested):
        raise FinalValidationError("profiles must not repeat")
    started = _source_state()
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{started['head'][:12]}_{socket.gethostname().lower()}_{uuid.uuid4().hex[:8]}"
    run = options.output_root / run_id
    run.mkdir(parents=True, exist_ok=False)
    options.build_dirs = {preset: (run / "builds" / preset if options.fresh_build else
                                   ROOT / "build" / "validation" / preset)
                          for preset in set(PRESET_FOR_PROFILE.values())}
    result = {"schema_version": 1, "status": "running", "run_id": run_id, "requested_profiles": list(requested),
              "source_before": _public_state(started), "profiles": {}, "milestone_complete": False}
    ACTIVE_RUN, ACTIVE_RESULT, ACTIVE_STARTED = run, result, started
    (run / "source-before.patch").write_bytes(started["_patch"])
    (run / "source-before.json").write_text(_json(_public_state(started)))
    (run / "status.json").write_text(_json(result))
    try:
        policy = _load_policy(options.policy)
        _validate_policy(policy, requested)
        result["policy"] = _identity(options.policy)
        result["assets"] = _preflight(options.models_dir, run, "before")
    except (FinalValidationError, OSError) as error:
        after = _record_source_snapshot(run, "after")
        result.update(status="failed", error=str(error), source_after=_public_state(after),
                      source_unchanged=_public_state(started) == _public_state(after))
        (run / "summary.json").write_text(_json(result)); (run / "status.json").write_text(_json(result)); return result
    configured = []
    for profile in requested:
        record = result["profiles"].setdefault(profile, {})
        record["stage"] = "configuring"
        (run / "status.json").write_text(_json(result))
        record["configure"] = _configure(profile, options, run)
        if not record["configure"]["ok"]:
            record["status"] = "failed"
        else:
            configured.append(profile)
    with ThreadPoolExecutor(max_workers=options.build_concurrency) as executor:
        futures = {}
        for profile in configured:
            result["profiles"][profile]["stage"] = "building"
            futures[executor.submit(_build, profile, options, run)] = profile
        (run / "status.json").write_text(_json(result))
        for future in as_completed(futures):
            profile = futures[future]
            record = result["profiles"][profile]
            try:
                record["build"] = future.result()
            except Exception as error:  # A single profile cannot abandon independent evidence.
                record["build"] = {"ok": False, "error": str(error)}
            if not record["build"]["ok"]:
                record["status"] = "failed"; continue
            try:
                record["artifacts"] = _artifacts(profile, options)
            except FinalValidationError as error:
                record.update(status="failed", error=str(error)); continue
    for profile in requested:
        record = result["profiles"].setdefault(profile, {})
        if record.get("status") == "failed":
            continue
        record["stage"] = "testing"
        (run / "status.json").write_text(_json(result))
        if profile in TEST_SLICES:
            prefix = ()
            if profile == "tsan":
                try:
                    prefix, record["launcher"] = _tsan_prefix(options, run)
                except FinalValidationError as error:
                    record.update(status="failed", error=str(error)); continue
            record["slices"] = {spec[0]: _test_slice(spec, options, policy, run, prefix) for spec in TEST_SLICES[profile]}
            record["status"] = "passed" if all(item["status"] == "passed" for item in record["slices"].values()) else "failed"
        elif profile == "cli":
            expected = policy["unittest_suites"]["cli"]
            record["probe"] = _unittest("cli", [sys.executable, "tests/unit/apps/inference/test_inference_cli.py", "--infer", str(_profile_binary(profile, options)), "--fixtures", str(options.models_dir / "inference-admission"), "-v"], expected["expected_count"], expected["names_sha256"], run)
            record["status"] = record["probe"]["status"]
        else:
            expected = policy["unittest_suites"]["benchmark"]
            record["probe"] = _unittest(profile, [sys.executable, "tests/unit/inference/test_model_benchmark_cli.py", "--binary", str(_profile_binary(profile, options)), "--fixtures", str(options.models_dir / "inference-numerical"), "-v"], expected["expected_count"], expected["names_sha256"], run)
            record["status"] = record["probe"]["status"]
        (run / "status.json").write_text(_json(result))
    try:
        result["assets_after"] = _preflight(options.models_dir, run, "after")
        result["assets_unchanged"] = _asset_identities(result["assets"]) == _asset_identities(result["assets_after"])
        if not result["assets_unchanged"]:
            result["assets_after_error"] = "validation assets changed during execution"
    except (FinalValidationError, OSError) as error:
        result["assets_after_error"] = str(error)
        result["all_requested_passed"] = False
    after = _record_source_snapshot(run, "after")
    result["source_after"] = _public_state(after)
    result["source_unchanged"] = _public_state(started) == _public_state(after)
    result["all_requested_passed"] = (not result.get("assets_after_error") and
                                       all(item.get("status") == "passed" for item in result["profiles"].values()))
    result["interrupted"] = _contains_interruption(result)
    if result["interrupted"]:
        result["all_requested_passed"] = False
    result["milestone_complete"] = set(requested) == set(PROFILES) and result["all_requested_passed"] and result["source_unchanged"]
    result["status"] = "passed" if result["all_requested_passed"] and result["source_unchanged"] else "failed"
    (run / "summary.json").write_text(_json(result)); (run / "status.json").write_text(_json(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--output-root", type=Path, default=ROOT / "build" / "validation" / "final")
    parser.add_argument("--models-dir", type=Path, default=ROOT / ".models")
    parser.add_argument("--policy", type=Path, default=ROOT / "tests" / "fixtures" / "final_validation" / "policy.json")
    parser.add_argument("--spdlog-source", type=Path)
    parser.add_argument("--google-benchmark-source", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--build-concurrency", type=int, default=1)
    parser.add_argument("--fresh-build", action="store_true", help="configure each selected profile below this run directory")
    parser.add_argument("--tsan-launcher", choices=("auto", "native", "setarch-no-aslr"), default="auto")
    options = parser.parse_args()
    if not 1 <= options.jobs <= 64 or not 1 <= options.build_concurrency <= 3:
        parser.error("jobs must be 1..64 and build-concurrency must be 1..3")
    options.models_dir = options.models_dir.resolve()
    options.output_root = options.output_root.resolve()
    options.policy = options.policy.resolve()
    for name in ("spdlog_source", "google_benchmark_source"):
        value = getattr(options, name)
        if value is not None:
            value = value.resolve()
            if not value.is_dir(): parser.error(f"--{name.replace('_', '-')} must name a directory")
            setattr(options, name, value)
    try:
        result = run_validation(options)
    except KeyboardInterrupt as error:
        _emergency_finish(error)
        return 130
    except Exception as error:
        _emergency_finish(error)
        parser.exit(2, f"Final validation failed before completion: {error}\n")
    print(_json({"run_id": result["run_id"], "status": result["status"], "milestone_complete": result["milestone_complete"]}), end="")
    if result.get("interrupted"):
        return 130
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
