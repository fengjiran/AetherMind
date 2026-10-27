#!/usr/bin/env python3
"""Exercise the real model benchmark probe and its artifact driver.

These are correctness checks, not performance gates. The fixture bundle must
already pass its independent numerical asset checker.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import verify_model_benchmark as contract

OPTIONS = None


class ModelBenchmarkCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checked = subprocess.run([
            sys.executable, str(ROOT / "tools/verify_llama_numerical_reference.py"),
            "--root", str(OPTIONS.fixtures),
            "--expected", str(ROOT / "tests/fixtures/inference_numerical/checksums.json"),
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if checked.returncode:
            raise RuntimeError("numerical asset prerequisite failed: " + checked.stderr)
        cls.manifest = json.loads((OPTIONS.fixtures / "manifest.json").read_text())

    def arguments(self, case="mha-untied", preset="o1-plain", limit=3, mode=None,
                  repetitions=1, warmup=0, eos=None):
        mode = mode or ("profile" if OPTIONS.instrumented else "timing")
        result = [
            str(OPTIONS.binary), "--model", str(OPTIONS.fixtures / case / "model"),
            "--prompt", "1,7,3", "--max-new-tokens", str(limit),
            "--preset", preset, "--warmup", str(warmup),
            "--repetitions", str(repetitions), "--mode", mode,
        ]
        if eos is not None:
            result += ["--eos-token", str(eos)]
        return result

    def engine(self, **kwargs):
        result = subprocess.run(
            self.arguments(**kwargs), text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        engine = json.loads(result.stdout)
        contract.verify_engine(engine)
        return engine

    def expected(self, case):
        return self.manifest["cases"][case]["trajectory"]["greedy_tokens"]

    def test_short_models_match_independent_greedy_in_both_presets(self):
        for case in ("mha-untied", "mha-tied", "gqa-tail", "mqa-tail"):
            for preset in ("o1-plain", "o2-packed"):
                with self.subTest(case=case, preset=preset):
                    engine = self.engine(case=case, preset=preset, repetitions=2)
                    self.assertEqual(len(engine["samples"]), 2)
                    for sample in engine["samples"]:
                        self.assertEqual(sample["internal"]["generated_tokens"], self.expected(case))
                        self.assertEqual(sample["generate"]["tokens"], self.expected(case))

    def test_warmup_does_not_replace_or_renumber_repetitions(self):
        engine = self.engine(case="gqa-tail", repetitions=3, warmup=2)
        self.assertEqual(engine["workload"]["warmup"], 2)
        self.assertEqual([sample["index"] for sample in engine["samples"]], [0, 1, 2])

    def test_zero_limit_has_no_internal_execution_or_kv_commit(self):
        engine = self.engine(limit=0)
        internal = engine["samples"][0]["internal"]
        self.assertEqual(internal["generated_tokens"], [])
        self.assertEqual(internal["final_watermark"], 0)
        self.assertEqual(internal["decode_steps"], [])
        self.assertIsNone(internal["prefill_execute_ns"])
        self.assertEqual(engine["samples"][0]["generate"]["tokens"], [])

    def test_first_token_limit_has_no_decode(self):
        engine = self.engine(limit=1)
        internal = engine["samples"][0]["internal"]
        self.assertEqual(internal["generated_tokens"], self.expected("mha-untied")[:1])
        self.assertEqual(internal["final_watermark"], 3)
        self.assertEqual(internal["decode_steps"], [])
        self.assertIsNone(internal["decode_setup_ns"])

    def test_prefill_eos_stops_without_decode(self):
        engine = self.engine(eos=self.expected("mha-untied")[0])
        sample = engine["samples"][0]
        self.assertEqual(sample["internal"]["stop_reason"], "eos")
        self.assertEqual(sample["internal"]["generated_tokens"], self.expected("mha-untied")[:1])
        self.assertEqual(sample["internal"]["decode_steps"], [])
        self.assertEqual(sample["generate"]["stop_reason"], "eos")

    def test_decode_eos_uses_actual_steps_and_watermark(self):
        self.assertNotEqual(*self.expected("mha-untied")[:2])
        engine = self.engine(eos=self.expected("mha-untied")[1])
        internal = engine["samples"][0]["internal"]
        self.assertEqual(internal["generated_tokens"], self.expected("mha-untied")[:2])
        self.assertEqual(len(internal["decode_steps"]), 1)
        self.assertEqual(internal["decode_steps"][0]["kv_before"], 3)
        self.assertEqual(internal["decode_steps"][0]["kv_after"], 4)
        self.assertEqual(internal["final_watermark"], 4)

    def test_profile_has_no_timing_or_allocation_instrumentation(self):
        engine = self.engine(mode="profile")
        sample = engine["samples"][0]
        self.assertTrue(all(value is None for value in sample["preparation"].values()))
        self.assertIsNone(sample["internal"]["prefill_execute_ns"])
        self.assertTrue(all(step["execute_ns"] is None for step in sample["internal"]["decode_steps"]))
        self.assertIsNone(sample["generate"]["duration_ns"])
        self.assertIsNone(sample["internal"]["allocator_requests"]["prefill_execute"])

    def test_allocation_diagnostic_is_separate_and_decode_is_prepared_once(self):
        engine = self.engine(mode="allocations")
        sample = engine["samples"][0]
        self.assertIsNone(sample["generate"]["duration_ns"])
        self.assertEqual(sample["internal"]["allocator_requests"]["prefill_execute"], 0)
        self.assertEqual(sample["internal"]["allocator_requests"]["decode_execute"], [0, 0])

    def test_context_limit_is_rejected_before_a_success_report(self):
        result = subprocess.run(
            self.arguments(limit=2049), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "")

    def test_invalid_tokens_are_rejected(self):
        for option, value in (("--prompt", "999999"), ("--eos-token", "999999")):
            command = self.arguments()
            if option == "--prompt":
                command[command.index(option) + 1] = value
            else:
                command += [option, value]
            with self.subTest(option=option):
                result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout.strip(), "")

    def test_malformed_and_duplicate_arguments_are_rejected(self):
        changes = (
            ("replace", "--max-new-tokens", "-1"),
            ("replace", "--max-new-tokens", "1.5"),
            ("replace", "--max-new-tokens", "18446744073709551616"),
            ("replace", "--prompt", "1,,3"),
            ("replace", "--prompt", "4294967296"),
            ("replace", "--repetitions", "0"),
            ("append", "--unknown", "1"),
            ("append", "--max-new-tokens", "3"),
        )
        for action, option, value in changes:
            command = self.arguments()
            if action == "replace":
                command[command.index(option) + 1] = value
            else:
                command += [option, value]
            with self.subTest(option=option, value=value):
                result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout.strip(), "")

    def driver(self, destination, prompt="1,7,3", mode=None):
        return subprocess.run([
            sys.executable, str(ROOT / "tools/run_model_benchmark.py"),
            "--binary", str(OPTIONS.binary), "--model", str(OPTIONS.fixtures / "gqa-tail/model"),
            "--prompt", prompt, "--max-new-tokens", "3", "--preset", "o1-plain",
            "--warmup", "0", "--repetitions", "2",
            "--mode", mode or ("profile" if OPTIONS.instrumented else "timing"),
            "--output-root", str(destination),
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)

    def test_driver_preserves_provenance_and_all_repetitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.driver(Path(temporary))
            self.assertEqual(result.returncode, 0, result.stderr)
            path = Path(result.stdout.strip())
            report = json.loads(path.read_text())
            contract.verify_report(report)
            self.assertEqual(len(report["engine"]["samples"]), 2)
            for name in ("engine.stdout", "engine.stderr", "source.patch",
                         "dynamic-libraries.txt", "invocation.json", "status.json"):
                self.assertTrue((path.parent / name).is_file(), name)
            self.assertEqual((path.parent / "summary.json").is_file(), not OPTIONS.instrumented)

    def test_driver_preserves_failure_without_accepting_a_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.driver(Path(temporary), prompt="999999")
            self.assertNotEqual(result.returncode, 0)
            runs = list(Path(temporary).iterdir())
            self.assertEqual(len(runs), 1)
            self.assertEqual(json.loads((runs[0] / "status.json").read_text())["status"], "failed")
            self.assertTrue((runs[0] / "engine.stderr").read_text())
            self.assertFalse((runs[0] / "report.json").exists())

    def test_sanitizer_build_cannot_enter_timing_statistics(self):
        if OPTIONS.instrumented:
            with tempfile.TemporaryDirectory() as temporary:
                result = self.driver(Path(temporary), mode="timing")
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(list(Path(temporary).glob("*/report.json")))
        else:
            engine = self.engine(case="gqa-tail")
            self.assertEqual(engine["build"]["build_type"], "Release")
            self.assertEqual(engine["build"]["sanitizer"], "none")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--instrumented", action="store_true")
    OPTIONS, remaining = parser.parse_known_args()
    OPTIONS.binary = OPTIONS.binary.resolve()
    OPTIONS.fixtures = OPTIONS.fixtures.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
