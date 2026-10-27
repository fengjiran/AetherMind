#!/usr/bin/env python3
"""Validate complete model benchmark reports and summarize observed timing.

Only the standard library is required. Validation checks the recorded identities;
the collecting runner hashes actual files. This tool does not accept candidates,
set regression thresholds, or infer model quality.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys

MAX_INTEGER = (1 << 63) - 1
MODES = {"timing", "profile", "allocations"}
# Inputs that can change allocation, dispatch, instrumentation or loader behavior.
# The collector imports this tuple so collection and validation share one list.
ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH", "GLIBC_TUNABLES",
    "MALLOC_PERTURB_", "MALLOC_CHECK_", "ASAN_OPTIONS", "UBSAN_OPTIONS",
    "TSAN_OPTIONS", "THP_MEM_ALLOC_ENABLE", "TRACEBACK_LIMIT",
)
GEOMETRY = {
    "hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings",
}
PREPARATION = {"load_ns", "compile_ns", "runtime_ns", "prepare_ns"}
MEMORY = {
    "raw_unique_visible_bytes", "packed_unique_storage_bytes", "kv_capacity_bytes",
    "workspace_bytes", "process_peak_rss_bytes",
}


class VerificationError(ValueError):
    """A report violates the frozen collection or comparison contract."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def exact_keys(value, expected, context):
    require(isinstance(value, dict), f"{context}: expected object")
    require(set(value) == set(expected),
            f"{context}: missing={sorted(set(expected) - set(value))}, "
            f"extra={sorted(set(value) - set(expected))}")


def integer(value, context, minimum=0, maximum=MAX_INTEGER):
    require(type(value) is int and minimum <= value <= maximum,
            f"{context}: expected integer in [{minimum}, {maximum}]")
    return value


def string(value, context, allow_empty=False):
    require(isinstance(value, str) and "\0" not in value
            and (allow_empty or bool(value.strip())),
            f"{context}: expected {'possibly empty ' if allow_empty else ''}string")
    return value


def number(value, context, positive=False):
    require(type(value) in (int, float) and value <= MAX_INTEGER and math.isfinite(value)
            and (value > 0 if positive else value >= 0),
            f"{context}: expected finite {'positive' if positive else 'nonnegative'} number")
    return value


def duration(value, mode, context, required=True, positive=False):
    if mode != "timing" or not required:
        require(value is None, f"{context}: timing is unavailable and must be null")
    else:
        number(value, context, positive)


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
        if type(value) is float:
            require(math.isfinite(value), "Non-finite JSON number")
        elif isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)

    try:
        result = json.loads(data, object_pairs_hook=_unique_object,
                            parse_constant=reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise VerificationError(f"Invalid JSON: {error}") from error
    finite(result)
    return result


def load_report(path):
    try:
        return parse_json(Path(path).read_bytes())
    except OSError as error:
        raise VerificationError(f"Cannot read report {path}: {error}") from error


def _tokens(values, vocab, context, nonempty=False):
    require(isinstance(values, list) and (values or not nonempty),
            f"{context}: expected {'nonempty ' if nonempty else ''}token array")
    for index, value in enumerate(values):
        integer(value, f"{context}[{index}]", maximum=min(vocab - 1, (1 << 32) - 1))


def _feature_list(values, context):
    require(isinstance(values, list), f"{context}: expected array")
    for value in values:
        string(value, context)
    require(len(values) == len(set(values)), f"{context}: duplicate feature")


def _stop(tokens, reason, maximum, eos, context):
    if maximum == 0:
        require(not tokens and reason == "zero_limit",
                f"{context}: zero limit requires no tokens and zero_limit")
        return
    require(1 <= len(tokens) <= maximum, f"{context}: generated token count")
    require(isinstance(reason, str) and reason in {"length", "eos"}, f"{context}: invalid stop_reason")
    if eos is not None:
        require(eos not in tokens[:-1], f"{context}: continued after EOS")
    ended_on_eos = eos is not None and tokens[-1] == eos
    if ended_on_eos:
        require(reason == "eos", f"{context}: EOS terminal token requires eos reason")
    else:
        require(reason == "length" and len(tokens) == maximum,
                f"{context}: length stop must reach max_new_tokens")


def verify_engine(engine):
    """Validate the benchmark executable's report, returning the input object."""
    exact_keys(engine, {
        "schema_version", "status", "mode", "build", "cpu", "geometry", "workload", "samples",
    }, "engine")
    require(type(engine["schema_version"]) is int and engine["schema_version"] == 1,
            "engine: unsupported schema_version")
    require(engine["status"] == "complete", "engine: incomplete run")
    mode = engine["mode"]
    require(isinstance(mode, str) and mode in MODES, "engine: invalid mode")
    build = engine["build"]
    exact_keys(build, {"compiler", "build_type", "sanitizer", "compiled_avx2_fma", "cxx_flags"}, "build")
    for key in ("compiler", "build_type", "sanitizer"):
        string(build[key], f"build.{key}")
    string(build["cxx_flags"], "build.cxx_flags", allow_empty=True)
    require(type(build["compiled_avx2_fma"]) is bool, "build.compiled_avx2_fma: expected bool")
    if mode == "timing":
        require(build["build_type"] == "Release" and build["sanitizer"] == "none",
                "timing requires Release and sanitizer=none")
    cpu = engine["cpu"]
    exact_keys(cpu, {"usable_features", "effective_features"}, "cpu")
    for key, values in cpu.items():
        _feature_list(values, f"cpu.{key}")
    require(set(cpu["effective_features"]) <= set(cpu["usable_features"]),
            "cpu: effective features exceed usable features")
    geometry = engine["geometry"]
    exact_keys(geometry, GEOMETRY, "geometry")
    for key, value in geometry.items():
        integer(value, f"geometry.{key}", minimum=1)
    require(geometry["hidden_size"] == geometry["num_attention_heads"] * geometry["head_dim"],
            "geometry: hidden_size/head geometry mismatch")
    require(geometry["num_attention_heads"] % geometry["num_key_value_heads"] == 0,
            "geometry: query heads must be divisible by KV heads")
    workload = engine["workload"]
    exact_keys(workload, {
        "preset", "prompt_tokens", "max_new_tokens", "eos_token_id",
        "context_capacity", "warmup", "repetitions",
    }, "workload")
    require(isinstance(workload["preset"], str)
            and workload["preset"] in {"o1-plain", "o2-packed"}, "workload: invalid preset")
    prompt = workload["prompt_tokens"]
    vocab = geometry["vocab_size"]
    _tokens(prompt, vocab, "workload.prompt_tokens", nonempty=True)
    integer(workload["max_new_tokens"], "workload.max_new_tokens")
    integer(workload["context_capacity"], "workload.context_capacity", minimum=1,
            maximum=geometry["max_position_embeddings"])
    integer(workload["warmup"], "workload.warmup", maximum=10000)
    integer(workload["repetitions"], "workload.repetitions", minimum=1, maximum=10000)
    eos = workload["eos_token_id"]
    if eos is not None:
        integer(eos, "workload.eos_token_id", maximum=min(vocab - 1, (1 << 32) - 1))
    prompt_count = len(prompt)
    maximum = workload["max_new_tokens"]
    require(prompt_count <= geometry["max_position_embeddings"],
            "workload: prompt exceeds model context")
    if maximum:
        require(prompt_count + maximum - 1 <= workload["context_capacity"],
                "workload: worst-case append exceeds physical context capacity")
    require(workload["context_capacity"] == prompt_count + max(maximum - 1, 0),
            "workload: context_capacity must exactly match the probe reservation policy")
    samples = engine["samples"]
    require(isinstance(samples, list) and len(samples) == workload["repetitions"],
            "samples: missing or extra repetitions")
    # MakeFp32KVCacheOptions fixes 64-byte row alignment. KVCacheManager pads
    # every FP32 head row to a multiple of sixteen elements.
    padded_dim = (geometry["head_dim"] + 15) // 16 * 16
    kv_bytes = (2 * geometry["num_hidden_layers"] * geometry["num_key_value_heads"]
                * padded_dim * workload["context_capacity"] * 4)
    previous_tokens = None
    for index, sample in enumerate(samples):
        context = f"samples[{index}]"
        exact_keys(sample, {"index", "preparation", "memory", "internal", "generate"}, context)
        integer(sample["index"], f"{context}.index")
        require(sample["index"] == index, f"{context}: sample indices must be complete and ordered")
        preparation = sample["preparation"]
        exact_keys(preparation, PREPARATION, f"{context}.preparation")
        for key, value in preparation.items():
            duration(value, mode, f"{context}.preparation.{key}", positive=True)
        memory = sample["memory"]
        exact_keys(memory, MEMORY, f"{context}.memory")
        for key, value in memory.items():
            integer(value, f"{context}.memory.{key}")
        require(memory["kv_capacity_bytes"] == kv_bytes,
                f"{context}: kv_capacity_bytes differs from 64-byte padded FP32 storage")
        internal = sample["internal"]
        exact_keys(internal, {
            "generated_tokens", "stop_reason", "final_watermark", "request_setup_ns",
            "first_token_request_ns", "prefill_execute_ns", "decode_setup_ns",
            "decode_steps", "allocator_requests",
        }, f"{context}.internal")
        tokens = internal["generated_tokens"]
        _tokens(tokens, vocab, f"{context}.internal.generated_tokens")
        _stop(tokens, internal["stop_reason"], maximum, eos, f"{context}.internal")
        integer(internal["final_watermark"], f"{context}.internal.final_watermark")
        steps = internal["decode_steps"]
        require(isinstance(steps, list) and len(steps) == max(0, len(tokens) - 1),
                f"{context}: Decode steps must equal actual generated tokens minus one")
        require(internal["final_watermark"] == (prompt_count + len(steps) if tokens else 0),
                f"{context}: incorrect final watermark")
        duration(internal["request_setup_ns"], mode, f"{context}.internal.request_setup_ns")
        if not tokens and mode == "timing":
            require(internal["request_setup_ns"] == 0,
                    f"{context}: zero-limit request must not reserve or prepare execution")
        duration(internal["first_token_request_ns"], mode,
                 f"{context}.internal.first_token_request_ns", required=bool(tokens))
        duration(internal["prefill_execute_ns"], mode,
                 f"{context}.internal.prefill_execute_ns", required=bool(tokens))
        duration(internal["decode_setup_ns"], mode,
                 f"{context}.internal.decode_setup_ns", required=bool(steps))
        if mode == "timing" and tokens:
            included = internal["request_setup_ns"] + internal["prefill_execute_ns"]
            epsilon = max(1.0, abs(included) * 1e-12)
            require(internal["first_token_request_ns"] + epsilon >= included,
                    f"{context}: first-token request time excludes setup or Prefill")
        for step_index, step in enumerate(steps):
            step_context = f"{context}.decode_steps[{step_index}]"
            exact_keys(step, {"index", "kv_before", "kv_after", "input_token",
                              "output_token", "execute_ns"}, step_context)
            for key in ("index", "kv_before", "kv_after", "input_token", "output_token"):
                integer(step[key], f"{step_context}.{key}")
            require(step["index"] == step_index, f"{step_context}: non-contiguous index")
            require(step["kv_before"] == prompt_count + step_index
                    and step["kv_after"] == prompt_count + step_index + 1,
                    f"{step_context}: wrong KV interval")
            require(step["input_token"] == tokens[step_index]
                    and step["output_token"] == tokens[step_index + 1],
                    f"{step_context}: token trajectory mismatch")
            duration(step["execute_ns"], mode, f"{step_context}.execute_ns")
        allocations = internal["allocator_requests"]
        exact_keys(allocations, {"prefill_execute", "decode_execute"},
                   f"{context}.allocator_requests")
        require(isinstance(allocations["decode_execute"], list)
                and len(allocations["decode_execute"]) == len(steps),
                f"{context}: allocation counts do not cover Decode steps")
        if mode == "allocations":
            if tokens:
                integer(allocations["prefill_execute"], f"{context}.prefill allocation count")
            else:
                require(allocations["prefill_execute"] is None,
                        f"{context}: zero-limit request has no Prefill allocation count")
            for value in allocations["decode_execute"]:
                integer(value, f"{context}.Decode allocation count")
        else:
            require(allocations["prefill_execute"] is None
                    and all(value is None for value in allocations["decode_execute"]),
                    f"{context}: allocation instrumentation is forbidden in {mode} mode")
        generate = sample["generate"]
        exact_keys(generate, {"tokens", "stop_reason", "duration_ns"}, f"{context}.generate")
        _tokens(generate["tokens"], vocab, f"{context}.generate.tokens")
        _stop(generate["tokens"], generate["stop_reason"], maximum, eos, f"{context}.generate")
        require(generate["tokens"] == tokens and generate["stop_reason"] == internal["stop_reason"],
                f"{context}: public Generate and internal trajectory differ")
        duration(generate["duration_ns"], mode, f"{context}.generate.duration_ns", positive=True)
        if previous_tokens is not None:
            require(tokens == previous_tokens, f"{context}: non-deterministic repeated trajectory")
        previous_tokens = tokens
    return engine


def _sha(value, context):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{context}: expected lowercase SHA256")


def _identity(value, context):
    exact_keys(value, {"path", "bytes", "sha256"}, context)
    string(value["path"], f"{context}.path")
    integer(value["bytes"], f"{context}.bytes", minimum=1)
    _sha(value["sha256"], f"{context}.sha256")


def verify_report(document):
    """Validate a collected report, returning the same document without mutation."""
    exact_keys(document, {
        "schema_version", "status", "run_id", "mode", "provenance", "model", "page_cache", "engine",
    }, "report")
    require(type(document["schema_version"]) is int and document["schema_version"] == 1,
            "report: unsupported schema_version")
    require(document["status"] == "complete", "report: incomplete run")
    string(document["run_id"], "run_id")
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", document["run_id"]) is not None,
            "run_id: unsafe identifier")
    require(isinstance(document["mode"], str) and document["mode"] in MODES, "report: invalid mode")
    require(document["page_cache"] == "pre_read_by_identity_collection",
            "report: unknown page-cache boundary")
    provenance = document["provenance"]
    exact_keys(provenance, {
        "commit", "dirty", "command", "host_id", "platform", "affinity", "environment",
        "executable", "library", "source_patch_sha256", "started_utc", "machine",
    }, "provenance")
    require(isinstance(provenance["commit"], str)
            and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", provenance["commit"]) is not None,
            "provenance.commit: expected full Git object id")
    require(type(provenance["dirty"]) is bool, "provenance.dirty: expected bool")
    command = provenance["command"]
    require(isinstance(command, list) and bool(command), "provenance.command: expected argv")
    for value in command:
        string(value, "provenance.command argument", allow_empty=True)
    require(bool(command[0]), "provenance.command: missing executable")
    string(provenance["host_id"], "provenance.host_id")
    string(provenance["platform"], "provenance.platform")
    machine = provenance["machine"]
    exact_keys(machine, {
        "cpu_model", "logical_cpus", "os_release", "kernel", "virtualization",
        "memory_total_bytes", "governors", "turbo", "numa_nodes",
    }, "provenance.machine")
    for key in ("cpu_model", "os_release", "kernel", "virtualization", "turbo"):
        string(machine[key], f"provenance.machine.{key}")
    integer(machine["logical_cpus"], "provenance.machine.logical_cpus", minimum=1)
    integer(machine["memory_total_bytes"], "provenance.machine.memory_total_bytes")
    require(isinstance(machine["governors"], list), "provenance.machine.governors: expected array")
    for value in machine["governors"]:
        string(value, "provenance.machine.governors")
    require(isinstance(machine["numa_nodes"], list), "provenance.machine.numa_nodes: expected array")
    for value in machine["numa_nodes"]:
        integer(value, "provenance.machine.numa_nodes")
    require(len(machine["numa_nodes"]) == len(set(machine["numa_nodes"])),
            "provenance.machine.numa_nodes: duplicate node")
    affinity = provenance["affinity"]
    require(isinstance(affinity, list), "provenance.affinity: expected array")
    for value in affinity:
        integer(value, "provenance.affinity")
    require(len(affinity) == len(set(affinity)), "provenance.affinity: duplicate CPU")
    environment = provenance["environment"]
    require(isinstance(environment, dict), "provenance.environment: expected object")
    missing = set(ENVIRONMENT_KEYS) - set(environment)
    require(not missing, f"provenance.environment: missing required environment keys {sorted(missing)}")
    for key, value in environment.items():
        string(key, "environment key")
        string(value, f"environment.{key}", allow_empty=True)
    if document["mode"] == "timing":
        require(environment["LD_PRELOAD"] == "" and environment["LD_AUDIT"] == "",
                "timing requires empty LD_PRELOAD and LD_AUDIT")
    for key in ("executable", "library"):
        _identity(provenance[key], f"provenance.{key}")
    _sha(provenance["source_patch_sha256"], "provenance.source_patch_sha256")
    string(provenance["started_utc"], "provenance.started_utc")
    try:
        started = datetime.fromisoformat(provenance["started_utc"].replace("Z", "+00:00"))
    except ValueError as error:
        raise VerificationError("provenance.started_utc: invalid timestamp") from error
    require(started.tzinfo is not None and started.utcoffset() == timedelta(0),
            "provenance.started_utc: timestamp must be UTC")
    model = document["model"]
    exact_keys(model, {"directory", "files"}, "model")
    string(model["directory"], "model.directory")
    require(isinstance(model["files"], list) and bool(model["files"]),
            "model.files: expected nonempty array")
    paths = []
    for record in model["files"]:
        _identity(record, "model.files")
        value = record["path"]
        path = PurePosixPath(value)
        require(not path.is_absolute() and ".." not in path.parts
                and path.as_posix() == value and value not in {".", ""},
                "model.files: unsafe relative path")
        paths.append(value)
    require(len(paths) == len(set(paths)), "model.files: duplicate file")
    require("config.json" in paths and any(path.endswith(".safetensors") for path in paths),
            "model.files: config and Safetensors weights are required")
    verify_engine(document["engine"])
    require(document["mode"] == document["engine"]["mode"], "report/engine mode mismatch")
    return document


def _statistics(values):
    if not values:
        return None
    ordered = sorted(values)
    result = {"n": len(ordered), "min": ordered[0],
              "median": statistics.median(ordered), "max": ordered[-1]}
    # Observed nearest-rank percentiles are not stable tail estimates.
    for percentile, minimum in ((90, 10), (99, 100)):
        if len(ordered) >= minimum:
            result[f"p{percentile}_observed"] = ordered[math.ceil(percentile / 100 * len(ordered)) - 1]
    return result


def _rate(count, elapsed):
    if count == 0 or elapsed == 0:
        return None
    result = count * 1e9 / elapsed
    require(math.isfinite(result), "Throughput cannot be represented as a finite number")
    return result


def _rate_statistics(values):
    # A zero-duration execution has no defined rate. Do not silently discard a
    # repetition: retain every raw timing, but leave the entire rate group null.
    return None if any(value is None for value in values) else _statistics(values)


def summarize(document):
    """Summarize all raw repetitions of a verified timing report."""
    verify_report(document)
    require(document["mode"] == "timing", "Summary requires timing mode")
    engine = document["engine"]
    samples = engine["samples"]
    workload = engine["workload"]
    metrics = {}
    for key in sorted(PREPARATION):
        metrics[f"preparation.{key}"] = _statistics([sample["preparation"][key] for sample in samples])
    for key in ("request_setup_ns", "first_token_request_ns", "prefill_execute_ns", "decode_setup_ns"):
        metrics[f"internal.{key}"] = _statistics(
            [sample["internal"][key] for sample in samples if sample["internal"][key] is not None])
    metrics["generate.duration_ns"] = _statistics([sample["generate"]["duration_ns"] for sample in samples])
    rates = [_rate(len(sample["generate"]["tokens"]), sample["generate"]["duration_ns"]) for sample in samples]
    metrics["generate.actual_tokens_per_second"] = _rate_statistics(rates)
    rates = [_rate(len(workload["prompt_tokens"]), sample["internal"]["prefill_execute_ns"])
             for sample in samples if sample["internal"]["prefill_execute_ns"] is not None]
    metrics["internal.prefill_tokens_per_second"] = _rate_statistics(rates)
    totals = [sum(step["execute_ns"] for step in sample["internal"]["decode_steps"])
              for sample in samples if sample["internal"]["decode_steps"]]
    metrics["internal.decode_execute_total_ns"] = _statistics(totals)
    rates = [_rate(len(sample["internal"]["decode_steps"]),
                   sum(step["execute_ns"] for step in sample["internal"]["decode_steps"]))
             for sample in samples if sample["internal"]["decode_steps"]]
    metrics["internal.decode_actual_tokens_per_second"] = _rate_statistics(rates)
    contexts = {}
    for sample in samples:
        for step in sample["internal"]["decode_steps"]:
            contexts.setdefault(step["kv_before"], []).append(step["execute_ns"])
    for position, elapsed in sorted(contexts.items()):
        metrics[f"internal.decode_execute_ns.kv_before_{position}"] = _statistics(elapsed)
    return {
        "schema_version": 1, "status": "summarized", "run_id": document["run_id"],
        "mode": "timing", "repetitions": len(samples),
        "percentile_policy": "observed_nearest_rank_p90_n10_p99_n100",
        "actual_generated_tokens_per_request": len(samples[0]["generate"]["tokens"]),
        "actual_decode_steps_per_request": len(samples[0]["internal"]["decode_steps"]),
        "decode_contexts": [{"kv_before": key, "kv_after": key + 1, "n": len(values)}
                            for key, values in sorted(contexts.items())],
        "metrics": metrics, "performance_claim": "none",
    }


def _comparison_identity(document, include_preset):
    provenance = document["provenance"]
    engine = document["engine"]
    workload = dict(engine["workload"])
    if not include_preset:
        workload.pop("preset")
    return {
        "machine": {key: provenance[key] for key in ("host_id", "platform", "environment", "machine")},
        "affinity": sorted(provenance["affinity"]),
        "build": engine["build"],
        "cpu": {key: sorted(value) for key, value in engine["cpu"].items()},
        "geometry": engine["geometry"], "workload": workload,
        "model_files": sorted(document["model"]["files"], key=lambda record: record["path"]),
        "page_cache": document["page_cache"],
        "observed_tokens": engine["samples"][0]["internal"]["generated_tokens"],
    }


def _matching_metrics(summaries):
    names = set(summaries[0]["metrics"])
    require(all(set(summary["metrics"]) == names for summary in summaries),
            "Comparison has missing metric groups")
    return sorted(names)


def compare_reports(baseline, candidate):
    baseline_summary = summarize(baseline)
    candidate_summary = summarize(candidate)
    require(_comparison_identity(baseline, False) == _comparison_identity(candidate, False),
            "Comparison requires the same machine, build configuration, model and non-preset workload")
    metrics = {}
    for name in _matching_metrics((baseline_summary, candidate_summary)):
        left = baseline_summary["metrics"][name]
        right = candidate_summary["metrics"][name]
        require((left is None) == (right is None), f"Comparison metric {name}: missing samples")
        delta = None if left is None or left["median"] == 0 else (
            (right["median"] - left["median"]) / left["median"] * 100)
        metrics[name] = {"baseline": left, "candidate": right, "observed_median_delta_percent": delta}
    return {
        "schema_version": 1, "status": "compared",
        "baseline_run_id": baseline["run_id"], "candidate_run_id": candidate["run_id"],
        "baseline_preset": baseline["engine"]["workload"]["preset"],
        "candidate_preset": candidate["engine"]["workload"]["preset"],
        "metrics": metrics, "performance_claim": "none",
        "interpretation": "Needs More Data: observed differences do not establish an improvement or regression",
    }


def noise_floor(reports):
    """Observe same-implementation process dispersion without setting a gate."""
    require(isinstance(reports, list) and len(reports) >= 4,
            "Noise floor requires at least four independent reports")
    summaries = [summarize(report) for report in reports]
    ids = [report["run_id"] for report in reports]
    require(len(set(ids)) == len(ids), "Noise floor requires distinct run_id values")
    expected = _comparison_identity(reports[0], True)
    reference = reports[0]["provenance"]
    for report in reports[1:]:
        require(_comparison_identity(report, True) == expected,
                "Noise floor requires identical machine, build, model and workload")
        provenance = report["provenance"]
        for key in ("commit", "dirty", "source_patch_sha256"):
            require(provenance[key] == reference[key], f"Noise floor requires identical {key}")
        for key in ("executable", "library"):
            require({field: provenance[key][field] for field in ("bytes", "sha256")}
                    == {field: reference[key][field] for field in ("bytes", "sha256")},
                    f"Noise floor requires the same {key} binary")
    metrics = {}
    for name in _matching_metrics(summaries):
        values = [summary["metrics"][name] for summary in summaries]
        require(all(value is None for value in values) or all(value is not None for value in values),
                f"Noise floor metric {name}: missing samples")
        if values[0] is None:
            metrics[name] = None
            continue
        medians = [value["median"] for value in values]
        pairs = []
        for left, right in itertools.combinations(range(len(reports)), 2):
            denominator = min(medians[left], medians[right])
            difference = abs(medians[right] - medians[left])
            relative = (difference / denominator * 100 if denominator else
                        0.0 if difference == 0 else None)
            pairs.append({"left_run_id": ids[left], "right_run_id": ids[right],
                          "absolute_median_difference_percent": relative})
        available = [pair["absolute_median_difference_percent"] for pair in pairs]
        floor = max(available) if all(value is not None for value in available) else None
        metrics[name] = {
            "run_medians": [{"run_id": run_id, "median": value} for run_id, value in zip(ids, medians)],
            "pairwise_differences": pairs, "observed_noise_floor_percent": floor,
        }
    return {
        "schema_version": 1, "status": "observed_noise", "run_ids": ids,
        "process_reports": len(reports), "metrics": metrics,
        "threshold_policy": "none", "performance_claim": "none",
        "interpretation": "Needs More Data: observed same-implementation dispersion is not a production threshold",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--compare", type=Path)
    group.add_argument("--noise-floor", type=Path, nargs="+",
                       help="Additional independent reports; at least four including --report")
    arguments = parser.parse_args()
    try:
        report = load_report(arguments.report)
        verify_report(report)
        if arguments.compare:
            result = compare_reports(report, load_report(arguments.compare))
        elif arguments.noise_floor:
            result = noise_floor([report] + [load_report(path) for path in arguments.noise_floor])
        elif report["mode"] == "timing":
            result = summarize(report)
        else:
            result = {"schema_version": 1, "status": "verified", "run_id": report["run_id"],
                      "mode": report["mode"], "repetitions": report["engine"]["workload"]["repetitions"],
                      "performance_claim": "none"}
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (VerificationError, OSError) as error:
        print(f"Invalid model benchmark: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
