"""Small contracts for the final-validation orchestrator; it never builds here."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("final_validation", ROOT / "tools/run_final_validation.py")
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class FinalValidationContracts(unittest.TestCase):
    def test_policy_requires_schema_and_both_sections(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            path.write_text("{}")
            with self.assertRaises(DRIVER.FinalValidationError):
                DRIVER._load_policy(path)

    def test_policy_rejects_non_object_duplicate_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            for encoded in ("[]", '{"schema_version":1,"schema_version":1}',
                            '{"schema_version":NaN,"ctest_slices":{},"unittest_suites":{}}'):
                with self.subTest(encoded=encoded):
                    path.write_text(encoded)
                    with self.assertRaises(DRIVER.FinalValidationError):
                        DRIVER._load_policy(path)

    def test_policy_accepts_minimal_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            value = {"schema_version": 1, "ctest_slices": {}, "unittest_suites": {}}
            path.write_text(json.dumps(value))
            self.assertEqual(DRIVER._load_policy(path), value)

    def test_policy_requires_selected_slice_and_positive_frozen_values(self):
        policy = {"schema_version": 1, "ctest_slices": {}, "unittest_suites": {}}
        with self.assertRaisesRegex(DRIVER.FinalValidationError, "baseline-production"):
            DRIVER._validate_policy(policy, ("baseline",))

    def test_source_state_has_full_commit_and_patch_digest(self):
        state = DRIVER._source_state()
        self.assertRegex(state["head"], r"^[0-9a-f]{40}$")
        self.assertRegex(state["source_patch_sha256"], r"^[0-9a-f]{64}$")
        self.assertIsInstance(state["dirty"], bool)

    def test_identity_hashes_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "x"
            path.write_bytes(b"abc")
            identity = DRIVER._identity(path)
            self.assertEqual(identity["bytes"], 3)
            self.assertEqual(len(identity["sha256"]), 64)

    def test_absolute_path_requirement_is_fail_closed(self):
        with self.assertRaises(DRIVER.FinalValidationError):
            DRIVER._require_absolute(Path("models"), "models-dir")

    def test_run_records_stdout_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "command.log"
            result = DRIVER._run(["/bin/sh", "-c", "printf hello; exit 3"], log)
            self.assertFalse(result["ok"])
            self.assertEqual(result["returncode"], 3)
            self.assertEqual(log.read_text(), "hello")

    def test_slice_requires_frozen_policy_entry(self):
        options = type("O", (), {})()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(DRIVER.FinalValidationError):
                DRIVER._test_slice(("absent", "validation-baseline", "^x$", ""), options,
                                   {"ctest_slices": {}}, Path(temporary))

    def test_unittest_zero_or_bad_output_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = DRIVER._run
            try:
                DRIVER._run = lambda *args, **kwargs: {"ok": True, "returncode": 0, "log": str(Path(temporary) / "x")}
                Path(temporary, "x").write_text("Ran 0 tests in 0.0s\n\nOK\n")
                result = DRIVER._unittest("x", ["ignored"], 1, "0" * 64, Path(temporary))
                self.assertEqual(result["status"], "failed")
            finally:
                DRIVER._run = original

    def test_unittest_fake_summary_without_verbose_case_ids_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = DRIVER._run
            try:
                log = Path(temporary) / "x"
                log.write_text("Ran 14 tests in 0.1s\n\nOK\n")
                DRIVER._run = lambda *args, **kwargs: {"ok": True, "returncode": 0, "log": str(log)}
                self.assertEqual(DRIVER._unittest("x", ["ignored"], 14, "0" * 64, Path(temporary))["status"], "failed")
            finally:
                DRIVER._run = original

    def test_fresh_build_directory_is_keyed_by_preset_not_profile(self):
        custom = Path("/tmp/final-build")
        options = type("O", (), {"build_dirs": {"validation-baseline": custom}})()
        self.assertEqual(DRIVER._build_dir(options, "validation-baseline"), custom)

    def test_artifacts_fail_when_required_binary_or_compile_database_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            options = type("O", (), {"build_dirs": {"validation-baseline": base}})()
            with self.assertRaisesRegex(DRIVER.FinalValidationError, "missing required"):
                DRIVER._artifacts("baseline", options)

    def test_tsan_auto_only_retries_the_known_startup_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = type("O", (), {"tsan_launcher": "auto", "build_dirs": {}})()
            original = DRIVER._run
            try:
                DRIVER._run = lambda *args, **kwargs: {"ok": False, "log": str(Path(temporary) / "x")}
                Path(temporary, "x").write_text("unrelated startup error")
                with self.assertRaisesRegex(DRIVER.FinalValidationError, "other than"):
                    DRIVER._tsan_prefix(options, Path(temporary))
            finally:
                DRIVER._run = original

    def test_run_launch_failure_is_a_recorded_nonzero_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = DRIVER._run(["/definitely/not/a/command"], Path(temporary) / "failure.log")
            self.assertFalse(record["ok"])
            self.assertEqual(record["returncode"], 127)
            self.assertIn("launch failed", Path(record["log"]).read_text())

    def test_interrupt_marker_recurses_through_profile_records(self):
        self.assertTrue(DRIVER._contains_interruption({"profiles": {"baseline": {"build": {"interrupted": True}}}}))
        self.assertFalse(DRIVER._contains_interruption({"profiles": {"baseline": {"build": {"interrupted": False}}}}))

    def test_emergency_interrupt_writes_terminal_failed_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            old = (DRIVER.ACTIVE_RUN, DRIVER.ACTIVE_RESULT, DRIVER.ACTIVE_STARTED)
            try:
                DRIVER.ACTIVE_RUN = run
                DRIVER.ACTIVE_RESULT = {"status": "running", "profiles": {}}
                DRIVER.ACTIVE_STARTED = None
                DRIVER._emergency_finish(KeyboardInterrupt())
                result = __import__("json").loads((run / "summary.json").read_text())
                self.assertEqual(result["status"], "failed")
                self.assertTrue(result["interrupted"])
                self.assertFalse(result["milestone_complete"])
            finally:
                DRIVER.ACTIVE_RUN, DRIVER.ACTIVE_RESULT, DRIVER.ACTIVE_STARTED = old

    def test_public_source_state_never_serializes_patch_bytes(self):
        state = {"head": "a" * 40, "dirty": True, "source_patch_sha256": "b" * 64, "_patch": b"secret"}
        self.assertNotIn("_patch", DRIVER._public_state(state))

    def test_unittest_same_count_with_different_case_names_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = DRIVER._run
            try:
                log = Path(temporary) / "x"
                log.write_text("test_wrong (__main__.Suite.test_wrong) ... ok\n"
                               "Ran 1 test in 0.1s\n\nOK\n")
                DRIVER._run = lambda *args, **kwargs: {"ok": True, "returncode": 0, "log": str(log)}
                expected = DRIVER.ctest_report.names_digest(["test_expected (__main__.Suite.test_expected)"])
                self.assertEqual(DRIVER._unittest("x", ["ignored"], 1, expected, Path(temporary))["status"], "failed")
            finally:
                DRIVER._run = original

    def test_asset_identity_difference_is_visible(self):
        before = {"source": {"sha256": "a"}, "admission_manifest": {"sha256": "b"}, "numerical_manifest": {"sha256": "c"}}
        after = dict(before)
        after["numerical_manifest"] = {"sha256": "changed"}
        self.assertNotEqual(DRIVER._asset_identities(before), DRIVER._asset_identities(after))

    def test_profile_binary_locations_match_existing_preset_layout(self):
        options = type("O", (), {"build_dirs": {}})()
        self.assertTrue(str(DRIVER._profile_binary("cli", options)).endswith("apps/inference/aethermind_infer"))
        self.assertTrue(str(DRIVER._profile_binary("benchmark", options)).endswith("tests/benchmark/aethermind_model_benchmark"))

    def test_profile_lists_are_complete_and_unique(self):
        self.assertEqual(len(DRIVER.PROFILES), len(set(DRIVER.PROFILES)))
        self.assertEqual(set(DRIVER.TEST_SLICES), {"baseline", "avx2", "asan", "tsan"})


if __name__ == "__main__":
    unittest.main()
