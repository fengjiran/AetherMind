"""Model benchmark report contracts; no engine or downloaded fixtures required."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("model_benchmark_checker", ROOT / "tools/verify_model_benchmark.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def report(repetitions=3):
    sample = {
        "index": 0,
        "preparation": {"load_ns": 100, "compile_ns": 200, "runtime_ns": 300, "prepare_ns": 400},
        "memory": {"raw_unique_visible_bytes": 1000, "packed_unique_storage_bytes": 200,
                   "kv_capacity_bytes": 2560, "workspace_bytes": 512, "process_peak_rss_bytes": 1048576},
        "internal": {
            "generated_tokens": [7, 8, 9], "stop_reason": "length", "final_watermark": 5,
            "request_setup_ns": 300, "first_token_request_ns": 1400,
            "prefill_execute_ns": 1000, "decode_setup_ns": 200,
            "decode_steps": [
                {"index": 0, "kv_before": 3, "kv_after": 4, "input_token": 7, "output_token": 8, "execute_ns": 2000},
                {"index": 1, "kv_before": 4, "kv_after": 5, "input_token": 8, "output_token": 9, "execute_ns": 2100},
            ],
            "allocator_requests": {"prefill_execute": None, "decode_execute": [None, None]},
        },
        "generate": {"tokens": [7, 8, 9], "stop_reason": "length", "duration_ns": 10000},
    }
    samples = []
    for index in range(repetitions):
        item = copy.deepcopy(sample)
        item["index"] = index
        samples.append(item)
    return {
        "schema_version": 1, "status": "complete", "run_id": "20261027T000000Z_abc_host_baseline",
        "mode": "timing", "page_cache": "pre_read_by_identity_collection",
        "provenance": {
            "commit": "a" * 40, "dirty": True, "command": ["/build/model-bench", "--model", "/model"],
            "host_id": "host", "platform": "Linux-x86_64", "affinity": [0, 1],
            "machine": {"cpu_model": "test-cpu", "logical_cpus": 2, "os_release": "test Linux",
                        "kernel": "test-kernel", "virtualization": "test-vm",
                        "memory_total_bytes": 67108864, "governors": ["performance"],
                        "turbo": "unavailable", "numa_nodes": [0]},
            "environment": {**{key: "" for key in CHECKER.ENVIRONMENT_KEYS},
                            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            "executable": {"path": "/build/model-bench", "bytes": 10000, "sha256": "1" * 64},
            "library": {"path": "/build/libAetherMind.so", "bytes": 20000, "sha256": "2" * 64},
            "source_patch_sha256": "3" * 64, "started_utc": "2026-10-27T00:00:00Z",
        },
        "model": {"directory": "/model", "files": [
            {"path": "config.json", "bytes": 100, "sha256": "4" * 64},
            {"path": "model.safetensors", "bytes": 900, "sha256": "5" * 64},
        ]},
        "engine": {
            "schema_version": 1, "status": "complete", "mode": "timing",
            "build": {"compiler": "GCC 14.2", "build_type": "Release", "sanitizer": "none",
                      "compiled_avx2_fma": True, "cxx_flags": "-O3"},
            "cpu": {"usable_features": ["avx2", "fma"], "effective_features": ["avx2", "fma"]},
            "geometry": {"hidden_size": 16, "intermediate_size": 37, "vocab_size": 64,
                         "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                         "head_dim": 4, "max_position_embeddings": 32},
            "workload": {"preset": "o1-plain", "prompt_tokens": [1, 7, 3], "max_new_tokens": 3,
                         "eos_token_id": None, "context_capacity": 5, "warmup": 1, "repetitions": repetitions},
            "samples": samples,
        },
    }


def set_mode(document, mode):
    document["mode"] = document["engine"]["mode"] = mode

    def clear(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.endswith("_ns"):
                    value[key] = None
                else:
                    clear(child)
        elif isinstance(value, list):
            for child in value:
                clear(child)
    clear(document["engine"]["samples"])
    if mode == "allocations":
        for sample in document["engine"]["samples"]:
            counts = sample["internal"]["allocator_requests"]
            counts["prefill_execute"] = 2 if sample["internal"]["generated_tokens"] else None
            counts["decode_execute"] = [0] * len(sample["internal"]["decode_steps"])
    return document


def end_on(document, count, eos=None, maximum=None):
    workload = document["engine"]["workload"]
    if maximum is not None:
        workload["max_new_tokens"] = maximum
    workload["eos_token_id"] = eos
    workload["context_capacity"] = len(workload["prompt_tokens"]) + max(workload["max_new_tokens"] - 1, 0)
    for sample in document["engine"]["samples"]:
        sample["memory"]["kv_capacity_bytes"] = 512 * workload["context_capacity"]
        internal = sample["internal"]
        internal["generated_tokens"] = internal["generated_tokens"][:count]
        internal["decode_steps"] = internal["decode_steps"][:max(0, count - 1)]
        internal["allocator_requests"]["decode_execute"] = [None] * max(0, count - 1)
        reason = "zero_limit" if count == 0 else "eos" if eos is not None else "length"
        internal["stop_reason"] = sample["generate"]["stop_reason"] = reason
        sample["generate"]["tokens"] = list(internal["generated_tokens"])
        internal["final_watermark"] = 3 + max(0, count - 1) if count else 0
        if count < 2:
            internal["decode_setup_ns"] = None
        if count == 0:
            internal["request_setup_ns"] = 0
            internal["first_token_request_ns"] = internal["prefill_execute_ns"] = None
    return document


def noise_reports(count=4):
    result = []
    for index in range(count):
        item = report()
        item["run_id"] += f"-{index}"
        for sample in item["engine"]["samples"]:
            sample["generate"]["duration_ns"] = [10000, 11000, 9000, 12000][index % 4]
        result.append(item)
    return result


class ReportContracts(unittest.TestCase):
    def invalid(self, path, value, match):
        document = report()
        target = document
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        with self.assertRaisesRegex(CHECKER.VerificationError, match):
            CHECKER.verify_report(document)

    def test_valid_report_is_returned_without_mutation(self):
        document = report()
        before = copy.deepcopy(document)
        self.assertIs(CHECKER.verify_report(document), document)
        self.assertEqual(document, before)
        self.assertIs(CHECKER.verify_engine(document["engine"]), document["engine"])

    def test_profile_verifies_without_timing_summary(self):
        document = set_mode(report(), "profile")
        CHECKER.verify_report(document)
        with self.assertRaisesRegex(CHECKER.VerificationError, "timing mode"):
            CHECKER.summarize(document)

    def test_allocation_mode_counts_are_separate_from_timing(self):
        document = set_mode(report(), "allocations")
        CHECKER.verify_report(document)
        with self.assertRaisesRegex(CHECKER.VerificationError, "timing mode"):
            CHECKER.summarize(document)

    def test_zero_generation_has_no_prefill_or_decode(self):
        CHECKER.verify_report(end_on(report(), 0, maximum=0))

    def test_probe_capacity_policy_is_fixed_even_for_zero_generation(self):
        document = end_on(report(), 0, maximum=0)
        document["engine"]["workload"]["context_capacity"] = 2
        for sample in document["engine"]["samples"]:
            sample["memory"]["kv_capacity_bytes"] = 1024
        with self.assertRaisesRegex(CHECKER.VerificationError, "exactly match"):
            CHECKER.verify_report(document)

    def test_prefill_eos_has_no_decode(self):
        CHECKER.verify_report(end_on(report(), 1, eos=7))

    def test_decode_eos_uses_actual_step_count(self):
        CHECKER.verify_report(end_on(report(), 2, eos=8))

    def test_eos_at_length_limit_is_still_reported_as_eos(self):
        CHECKER.verify_report(end_on(report(), 3, eos=9))

    def test_exact_context_capacity_is_valid(self):
        document = report()
        document["engine"]["workload"]["context_capacity"] = 5
        for sample in document["engine"]["samples"]:
            sample["memory"]["kv_capacity_bytes"] = 2560
        CHECKER.verify_report(document)

    def test_missing_repetition_fails(self):
        self.invalid(("engine", "samples"), report()["engine"]["samples"][:-1], "repetitions")

    def test_extra_repetition_fails(self):
        self.invalid(("engine", "samples"), report(4)["engine"]["samples"], "repetitions")

    def test_duplicate_and_unordered_indices_fail(self):
        for value in (0, 2):
            with self.subTest(value=value):
                self.invalid(("engine", "samples", 1, "index"), value, "indices")

    def test_bool_cannot_masquerade_as_integer(self):
        for path in (("schema_version",), ("engine", "samples", 0, "index"),
                     ("engine", "workload", "repetitions")):
            with self.subTest(path=path):
                self.invalid(path, True, "schema|integer")

    def test_zero_repetitions_are_invalid(self):
        self.invalid(("engine", "workload", "repetitions"), 0, "integer")

    def test_unknown_fields_do_not_silently_change_contract(self):
        document = report()
        document["engine"]["samples"][0]["generate"]["ttft_ns"] = 100
        with self.assertRaisesRegex(CHECKER.VerificationError, "extra"):
            CHECKER.verify_report(document)

    def test_invalid_mode_and_wrong_mode_types_fail_cleanly(self):
        for value in ("unknown", [], {}):
            with self.subTest(value=value):
                self.invalid(("engine", "mode"), value, "mode")

    def test_profile_must_not_contain_timing_samples(self):
        document = set_mode(report(), "profile")
        document["engine"]["samples"][0]["internal"]["prefill_execute_ns"] = 10
        with self.assertRaisesRegex(CHECKER.VerificationError, "must be null"):
            CHECKER.verify_report(document)

    def test_timing_must_not_include_allocation_instrumentation(self):
        self.invalid(("engine", "samples", 0, "internal", "allocator_requests", "prefill_execute"),
                     0, "instrumentation")

    def test_allocation_mode_requires_every_execute_count(self):
        document = set_mode(report(), "allocations")
        document["engine"]["samples"][0]["internal"]["allocator_requests"]["decode_execute"][0] = None
        with self.assertRaisesRegex(CHECKER.VerificationError, "allocation count"):
            CHECKER.verify_report(document)

    def test_missing_decode_allocation_sample_fails(self):
        self.invalid(("engine", "samples", 0, "internal", "allocator_requests", "decode_execute"),
                     [None], "allocation counts")

    def test_zero_limit_cannot_claim_request_preparation(self):
        document = end_on(report(), 0, maximum=0)
        document["engine"]["samples"][0]["internal"]["request_setup_ns"] = 100
        with self.assertRaisesRegex(CHECKER.VerificationError, "must not reserve"):
            CHECKER.verify_report(document)

    def test_negative_and_nonfinite_durations_fail(self):
        for value in (-1, math.nan, math.inf, True):
            with self.subTest(value=value):
                self.invalid(("engine", "samples", 0, "generate", "duration_ns"), value, "finite")

    def test_preparation_and_public_generate_durations_must_be_positive(self):
        for path in (("engine", "samples", 0, "preparation", "load_ns"),
                     ("engine", "samples", 0, "generate", "duration_ns")):
            with self.subTest(path=path):
                self.invalid(path, 0, "positive")

    def test_first_token_latency_must_include_request_setup(self):
        self.invalid(("engine", "samples", 0, "internal", "first_token_request_ns"), 1000,
                     "excludes setup")

    def test_first_token_is_not_a_decode_step(self):
        self.invalid(("engine", "samples", 0, "internal", "decode_steps"), [], "minus one")

    def test_decode_positions_must_be_contiguous(self):
        self.invalid(("engine", "samples", 0, "internal", "decode_steps", 0, "kv_before"), 2,
                     "KV interval")

    def test_decode_tokens_must_follow_the_recorded_trajectory(self):
        for field in ("input_token", "output_token"):
            with self.subTest(field=field):
                self.invalid(("engine", "samples", 0, "internal", "decode_steps", 0, field),
                             22, "trajectory mismatch")

    def test_final_watermark_must_cover_all_committed_steps(self):
        self.invalid(("engine", "samples", 0, "internal", "final_watermark"), 6, "watermark")

    def test_public_generate_must_match_internal_result(self):
        self.invalid(("engine", "samples", 0, "generate", "tokens"), [7, 8, 10], "trajectory differ")

    def test_repetitions_must_have_deterministic_trajectories(self):
        document = report()
        item = document["engine"]["samples"][1]
        item["internal"]["generated_tokens"][-1] = item["generate"]["tokens"][-1] = 10
        item["internal"]["decode_steps"][-1]["output_token"] = 10
        with self.assertRaisesRegex(CHECKER.VerificationError, "non-deterministic"):
            CHECKER.verify_report(document)

    def test_eos_cannot_appear_before_the_last_generated_token(self):
        self.invalid(("engine", "workload", "eos_token_id"), 8, "continued after EOS")

    def test_eos_stop_requires_a_matching_terminal_token(self):
        self.invalid(("engine", "samples", 0, "internal", "stop_reason"), "eos", "length stop")

    def test_length_stop_cannot_end_early(self):
        document = end_on(report(), 2)
        with self.assertRaisesRegex(CHECKER.VerificationError, "length stop"):
            CHECKER.verify_report(document)

    def test_worst_case_context_must_fit_before_eos_optimization(self):
        self.invalid(("engine", "workload", "max_new_tokens"), 7, "physical context")

    def test_prompt_and_eos_ids_must_fit_the_vocab(self):
        for path, value in ((("engine", "workload", "prompt_tokens"), [64]),
                            (("engine", "workload", "eos_token_id"), 64)):
            with self.subTest(path=path):
                self.invalid(path, value, "integer")

    def test_kv_memory_uses_real_padded_storage_not_logical_bytes(self):
        self.invalid(("engine", "samples", 0, "memory", "kv_capacity_bytes"), 1024, "padded FP32")

    def test_memory_counters_must_be_nonnegative_integers(self):
        self.invalid(("engine", "samples", 0, "memory", "workspace_bytes"), -1, "integer")

    def test_effective_cpu_features_cannot_exceed_usable_features(self):
        self.invalid(("engine", "cpu", "effective_features"), ["avx512"], "exceed usable")

    def test_debug_and_sanitizer_runs_cannot_be_used_as_timings(self):
        for key, value in (("build_type", "Debug"), ("sanitizer", "asan-ubsan")):
            with self.subTest(key=key):
                self.invalid(("engine", "build", key), value, "Release")

    def test_root_and_engine_modes_must_match(self):
        self.invalid(("mode",), "profile", "mode mismatch")

    def test_provenance_commit_hash_and_utc_timestamp_are_required(self):
        for path, value, pattern in (
            (("provenance", "commit"), "abc", "Git object"),
            (("provenance", "source_patch_sha256"), "BAD", "SHA256"),
            (("provenance", "started_utc"), "2026-10-27T00:00:00", "UTC"),
        ):
            with self.subTest(path=path):
                self.invalid(path, value, pattern)

    def test_model_paths_must_be_safe_and_unique(self):
        for value in ("../model.safetensors", "/model.safetensors", "nested//model.safetensors"):
            with self.subTest(value=value):
                self.invalid(("model", "files", 1, "path"), value, "unsafe relative")
        document = report()
        document["model"]["files"].append(copy.deepcopy(document["model"]["files"][0]))
        with self.assertRaisesRegex(CHECKER.VerificationError, "duplicate"):
            CHECKER.verify_report(document)

    def test_allocator_environment_inputs_are_required_and_extra_metadata_is_allowed(self):
        for key in ("THP_MEM_ALLOC_ENABLE", "TRACEBACK_LIMIT"):
            with self.subTest(key=key):
                document = report()
                del document["provenance"]["environment"][key]
                with self.assertRaisesRegex(CHECKER.VerificationError, "missing required environment"):
                    CHECKER.verify_report(document)
        document = report()
        document["provenance"]["environment"]["EXTRA_CONTROL"] = "recorded"
        CHECKER.verify_report(document)

    def test_timing_rejects_preload_and_audit_instrumentation(self):
        for key in ("LD_PRELOAD", "LD_AUDIT"):
            with self.subTest(key=key):
                self.invalid(("provenance", "environment", key), "/instrumentation.so",
                             "empty LD_PRELOAD and LD_AUDIT")

    def test_thp_configuration_mixing_is_rejected_in_comparison_and_noise(self):
        baseline, candidate = report(), report()
        baseline["provenance"]["environment"]["THP_MEM_ALLOC_ENABLE"] = "0"
        candidate["provenance"]["environment"]["THP_MEM_ALLOC_ENABLE"] = "1"
        with self.assertRaisesRegex(CHECKER.VerificationError, "same machine"):
            CHECKER.compare_reports(baseline, candidate)
        documents = noise_reports()
        for document in documents:
            document["provenance"]["environment"]["THP_MEM_ALLOC_ENABLE"] = "0"
        documents[-1]["provenance"]["environment"]["THP_MEM_ALLOC_ENABLE"] = "1"
        with self.assertRaisesRegex(CHECKER.VerificationError, "identical machine"):
            CHECKER.noise_floor(documents)

    def test_missing_model_weight_identity_fails(self):
        self.invalid(("model", "files"), report()["model"]["files"][:1], "Safetensors")

    def test_unavailable_optional_machine_information_is_explicit(self):
        document = report()
        machine = document["provenance"]["machine"]
        machine["memory_total_bytes"] = 0
        machine["virtualization"] = machine["turbo"] = "unavailable"
        machine["governors"] = machine["numa_nodes"] = []
        CHECKER.verify_report(document)

    def test_machine_metadata_has_strict_types_and_is_part_of_comparison_identity(self):
        for key, value in (("logical_cpus", 0), ("memory_total_bytes", -1),
                           ("kernel", None), ("governors", ["performance", 1]),
                           ("numa_nodes", [0, 0])):
            with self.subTest(key=key):
                self.invalid(("provenance", "machine", key), value, "machine")
        baseline, candidate = report(), report()
        candidate["provenance"]["machine"]["cpu_model"] = "other-cpu"
        with self.assertRaisesRegex(CHECKER.VerificationError, "same machine"):
            CHECKER.compare_reports(baseline, candidate)

    def test_json_parser_rejects_duplicate_keys(self):
        with self.assertRaisesRegex(CHECKER.VerificationError, "Duplicate"):
            CHECKER.parse_json('{"a":1,"a":2}')

    def test_json_parser_rejects_nonfinite_literals_and_overflow(self):
        for text in ('{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'):
            with self.subTest(text=text):
                with self.assertRaisesRegex(CHECKER.VerificationError, "Non-finite"):
                    CHECKER.parse_json(text)


class SummaryContracts(unittest.TestCase):
    def test_all_repetitions_are_used_and_decode_contexts_are_separate(self):
        document = report()
        for sample, value in zip(document["engine"]["samples"], (10000, 1000000, 20000)):
            sample["generate"]["duration_ns"] = value
        result = CHECKER.summarize(document)
        self.assertEqual(result["metrics"]["generate.duration_ns"],
                         {"n": 3, "min": 10000, "median": 20000, "max": 1000000})
        self.assertEqual(result["metrics"]["internal.decode_execute_total_ns"]["median"], 4100)
        self.assertEqual(result["metrics"]["internal.decode_execute_ns.kv_before_3"]["n"], 3)
        self.assertEqual(result["metrics"]["internal.decode_execute_ns.kv_before_4"]["median"], 2100)
        self.assertEqual(result["actual_decode_steps_per_request"], 2)
        self.assertEqual(result["performance_claim"], "none")

    def test_actual_eos_count_is_the_generate_throughput_numerator(self):
        result = CHECKER.summarize(end_on(report(), 1, eos=7))
        self.assertEqual(result["metrics"]["generate.actual_tokens_per_second"]["median"], 100000)
        self.assertEqual(result["actual_generated_tokens_per_request"], 1)
        self.assertEqual(result["actual_decode_steps_per_request"], 0)
        self.assertIsNone(result["metrics"]["internal.decode_execute_total_ns"])
        self.assertEqual(result["decode_contexts"], [])

    def test_zero_limit_does_not_report_fake_prefill_or_token_throughput(self):
        result = CHECKER.summarize(end_on(report(), 0, maximum=0))
        self.assertIsNone(result["metrics"]["internal.prefill_execute_ns"])
        self.assertIsNone(result["metrics"]["generate.actual_tokens_per_second"])
        self.assertIsNotNone(result["metrics"]["generate.duration_ns"])

    def test_undefined_throughput_keeps_all_timings_without_selecting_repetitions(self):
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                document = report()
                internal = document["engine"]["samples"][1]["internal"]
                if phase == "prefill":
                    internal["prefill_execute_ns"] = 0
                    metric = "internal.prefill_tokens_per_second"
                    timing = "internal.prefill_execute_ns"
                else:
                    for step in internal["decode_steps"]:
                        step["execute_ns"] = 0
                    metric = "internal.decode_actual_tokens_per_second"
                    timing = "internal.decode_execute_total_ns"
                result = CHECKER.summarize(document)
                self.assertIsNone(result["metrics"][metric])
                self.assertEqual(result["metrics"][timing]["n"], 3)
                self.assertEqual(result["metrics"][timing]["min"], 0)
                self.assertEqual(result["metrics"]["generate.duration_ns"]["n"], 3)
                self.assertEqual(result["metrics"]["generate.actual_tokens_per_second"]["n"], 3)

    def test_observed_percentiles_require_their_declared_sample_count(self):
        for count in (1, 9, 10, 99, 100):
            with self.subTest(count=count):
                metric = CHECKER.summarize(report(count))["metrics"]["generate.duration_ns"]
                self.assertEqual("p90_observed" in metric, count >= 10)
                self.assertEqual("p99_observed" in metric, count >= 100)

    def test_comparison_allows_preset_and_candidate_binary_changes_without_acceptance(self):
        baseline = report()
        candidate = report()
        candidate["run_id"] += "-candidate"
        candidate["engine"]["workload"]["preset"] = "o2-packed"
        candidate["provenance"]["executable"]["sha256"] = "6" * 64
        for sample in candidate["engine"]["samples"]:
            sample["generate"]["duration_ns"] = 5000
        result = CHECKER.compare_reports(baseline, candidate)
        self.assertEqual(result["metrics"]["generate.duration_ns"]["observed_median_delta_percent"], -50)
        self.assertEqual(result["performance_claim"], "none")
        self.assertIn("Needs More Data", result["interpretation"])

    def test_comparison_rejects_machine_model_and_control_mixing(self):
        changes = [
            (("provenance", "host_id"), "other"),
            (("provenance", "affinity"), [2]),
            (("provenance", "environment"), {
                **report()["provenance"]["environment"], "OMP_NUM_THREADS": "2"}),
            (("model", "files", 1, "sha256"), "6" * 64),
            (("engine", "cpu", "effective_features"), []),
            (("engine", "workload", "warmup"), 2),
            (("engine", "build", "cxx_flags"), "-O2"),
        ]
        baseline = report()
        for path, value in changes:
            with self.subTest(path=path):
                candidate = report()
                target = candidate
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaisesRegex(CHECKER.VerificationError, "same machine"):
                    CHECKER.compare_reports(baseline, candidate)

    def test_noise_floor_preserves_all_process_medians_and_pairs(self):
        result = CHECKER.noise_floor(noise_reports())
        metric = result["metrics"]["generate.duration_ns"]
        self.assertEqual(len(metric["run_medians"]), 4)
        self.assertEqual(len(metric["pairwise_differences"]), 6)
        self.assertAlmostEqual(metric["observed_noise_floor_percent"], 100 / 3)
        self.assertEqual(result["threshold_policy"], "none")
        self.assertEqual(result["performance_claim"], "none")

    def test_noise_floor_rejects_insufficient_duplicate_and_mixed_runs(self):
        groups = [noise_reports(3), noise_reports()]
        groups[1][1]["run_id"] = groups[1][0]["run_id"]
        changed_binary = noise_reports()
        changed_binary[1]["provenance"]["library"]["sha256"] = "6" * 64
        groups.append(changed_binary)
        changed_preset = noise_reports()
        changed_preset[1]["engine"]["workload"]["preset"] = "o2-packed"
        groups.append(changed_preset)
        changed_patch = noise_reports()
        changed_patch[1]["provenance"]["source_patch_sha256"] = "6" * 64
        groups.append(changed_patch)
        for documents in groups:
            with self.subTest(ids=[item["run_id"] for item in documents]):
                with self.assertRaises(CHECKER.VerificationError):
                    CHECKER.noise_floor(documents)


class CliContracts(unittest.TestCase):
    def run_cli(self, document, extra=()):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(document, allow_nan=False), encoding="utf-8")
            return subprocess.run([sys.executable, str(ROOT / "tools/verify_model_benchmark.py"),
                                   "--report", str(path), *extra],
                                  text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_cli_timing_summary_has_stable_json_and_success_status(self):
        result = self.run_cli(report())
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["status"], "summarized")
        self.assertEqual(output["repetitions"], 3)

    def test_cli_non_timing_modes_only_emit_verification(self):
        for mode in ("profile", "allocations"):
            with self.subTest(mode=mode):
                result = self.run_cli(set_mode(report(), mode))
                self.assertEqual(result.returncode, 0, result.stderr)
                output = json.loads(result.stdout)
                self.assertEqual(output["status"], "verified")
                self.assertNotIn("metrics", output)

    def test_cli_invalid_report_returns_two_without_a_performance_summary(self):
        document = report()
        document["engine"]["samples"].pop()
        result = self.run_cli(document)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("Invalid model benchmark", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_warmup_and_repetition_limits_match_probe_cli(self):
        for field in ("warmup", "repetitions"):
            with self.subTest(field=field):
                document = report()
                document["engine"]["workload"][field] = 10001
                with self.assertRaisesRegex(CHECKER.VerificationError, "integer"):
                    CHECKER.verify_report(document)

    def test_larger_physical_capacity_cannot_change_the_probe_workload(self):
        document = report()
        document["engine"]["workload"]["context_capacity"] = 8
        for sample in document["engine"]["samples"]:
            sample["memory"]["kv_capacity_bytes"] = 4096
        with self.assertRaisesRegex(CHECKER.VerificationError, "exactly match"):
            CHECKER.verify_report(document)


if __name__ == "__main__":
    unittest.main()
