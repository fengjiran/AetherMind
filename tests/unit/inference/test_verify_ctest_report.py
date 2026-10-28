"""Contracts for the fail-closed CTest inventory/JUnit verifier."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("ctest_report_checker", ROOT / "tools/verify_ctest_report.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def inventory(names=("Alpha.One", "Beta.Two"), disabled=None):
    tests = [{"name": name, "properties": []} for name in names]
    if disabled is not None:
        tests[0]["properties"] = [{"name": "DISABLED", "value": disabled}]
    return {"kind": "ctestInfo", "version": {"major": 1, "minor": 0}, "tests": tests}


def junit(cases=("Alpha.One", "Beta.Two"), root="testsuite", attributes=None, body=""):
    attributes = {"tests": str(len(cases)), "failures": "0", "errors": "0", "skipped": "0", **(attributes or {})}
    attrs = " ".join(f'{key}="{value}"' for key, value in attributes.items())
    tests = "".join(f'<testcase name="{case}" classname="{case}" time="0.1" status="run"/>' for case in cases)
    return f"<{root} {attrs}>{tests}{body}</{root}>"


class CTestReportContracts(unittest.TestCase):
    def setUp(self):
        self.expected = CHECKER.names_digest(["Alpha.One", "Beta.Two"])

    def verify(self, inv=None, xml=None):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "junit.xml"
            path.write_text(xml or junit())
            return CHECKER.verify_run(inv or inventory(), path, 2, self.expected)

    def test_valid_report_returns_complete_counts(self):
        result = self.verify()
        self.assertEqual(result, {"expected": 2, "discovered": 2, "executed": 2, "passed": 2,
                                  "failed": 0, "skipped": 0, "disabled": 0, "not_run": 0,
                                  "names_sha256": self.expected})

    def test_digest_is_sorted_canonical_utf8(self):
        self.assertEqual(CHECKER.names_digest(["β", "a"]), CHECKER.names_digest(["a", "β"]))
        self.assertNotEqual(CHECKER.names_digest(["a", "b"]), CHECKER.names_digest(["a", "B"]))

    def test_inventory_count_and_digest_are_exact(self):
        with self.assertRaisesRegex(CHECKER.VerificationError, "count mismatch"):
            CHECKER.verify_inventory(inventory(), 3, self.expected)
        with self.assertRaisesRegex(CHECKER.VerificationError, "SHA256 mismatch"):
            CHECKER.verify_inventory(inventory(), 2, "0" * 64)

    def test_inventory_rejects_empty_or_duplicate_names(self):
        for names in (("", "Beta.Two"), ("Alpha.One", "Alpha.One")):
            with self.subTest(names=names), self.assertRaises(CHECKER.VerificationError):
                CHECKER.verify_inventory(inventory(names), 2, self.expected)

    def test_inventory_rejects_disabled_boolean_and_string(self):
        for value in (True, "TRUE", " true ", "ON", "YES", "1", "custom-nonempty"):
            with self.subTest(value=value), self.assertRaisesRegex(CHECKER.VerificationError, "disabled"):
                CHECKER.verify_inventory(inventory(disabled=value), 2, self.expected)

    def test_inventory_accepts_cmake_false_values(self):
        for value in (False, "", "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND", "custom-NOTFOUND"):
            with self.subTest(value=value):
                self.assertEqual(CHECKER.verify_inventory(inventory(disabled=value), 2, self.expected),
                                 ["Alpha.One", "Beta.Two"])

    def test_inventory_allows_false_disabled_property(self):
        inv = inventory(disabled="FALSE")
        self.assertEqual(CHECKER.verify_inventory(inv, 2, self.expected), ["Alpha.One", "Beta.Two"])

    def test_inventory_requires_tests_array(self):
        with self.assertRaisesRegex(CHECKER.VerificationError, "tests array"):
            CHECKER.verify_inventory({"tests": {}}, 2, self.expected)

    def test_inventory_rejects_non_object_test(self):
        with self.assertRaisesRegex(CHECKER.VerificationError, "must be an object"):
            CHECKER.verify_inventory({"tests": ["Alpha.One", {"name": "Beta.Two"}]}, 2, self.expected)

    def test_inventory_rejects_malformed_properties(self):
        inv = inventory()
        inv["tests"][0]["properties"] = {"DISABLED": ["TRUE"]}
        with self.assertRaisesRegex(CHECKER.VerificationError, "property"):
            CHECKER.verify_inventory(inv, 2, self.expected)

    def test_inventory_expected_values_must_be_valid(self):
        with self.assertRaises(CHECKER.VerificationError):
            CHECKER.verify_inventory(inventory(), 0, self.expected)
        with self.assertRaises(CHECKER.VerificationError):
            CHECKER.verify_inventory(inventory(), 2, "A" * 64)

    def test_junit_name_set_must_match_inventory(self):
        with self.assertRaisesRegex(CHECKER.VerificationError, "differ"):
            self.verify(xml=junit(("Alpha.One", "Other")))

    def test_junit_duplicate_case_is_rejected(self):
        with self.assertRaisesRegex(CHECKER.VerificationError, "duplicate"):
            self.verify(xml=junit(("Alpha.One", "Alpha.One")))

    def test_junit_failure_error_and_skipped_are_rejected(self):
        for child in ("<failure/>", "<error/>", "<skipped/>"):
            xml = junit(("Alpha.One", "Beta.Two"), body="")
            xml = xml.replace('/>', f'>{child}</testcase>', 1)
            with self.subTest(child=child), self.assertRaises(CHECKER.VerificationError):
                self.verify(xml=xml)

    def test_junit_notrun_and_disabled_status_are_rejected(self):
        for status in ("notrun", "not_run", "disabled", "skipped", "suppressed", "unknown"):
            with self.subTest(status=status), self.assertRaisesRegex(CHECKER.VerificationError, "non-passing status|disabled"):
                self.verify(xml=junit().replace('status="run"', f'status="{status}"', 1))

    def test_junit_allows_known_passing_statuses(self):
        for status in ("passed", "completed"):
            with self.subTest(status=status):
                self.assertEqual(self.verify(xml=junit().replace('status="run"', f'status="{status}"'))["passed"], 2)

    def test_junit_result_only_allows_passing_values(self):
        for result in ("skipped", "suppressed", "failed", "unknown"):
            with self.subTest(result=result), self.assertRaisesRegex(CHECKER.VerificationError, "non-passing result"):
                self.verify(xml=junit().replace('status="run"', f'status="run" result="{result}"', 1))
        self.assertEqual(self.verify(xml=junit().replace('status="run"', 'status="run" result="completed"', 1))["passed"], 2)

    def test_junit_rejects_direct_suite_outcome_nodes(self):
        for node in ("failure", "error", "skipped"):
            with self.subTest(node=node), self.assertRaisesRegex(CHECKER.VerificationError, "direct"):
                self.verify(xml=junit(body=f'<{node}/>'))

    def test_junit_declared_counts_are_verified(self):
        for attribute, value in (("tests", "3"), ("failures", "1"), ("errors", "1"), ("skipped", "1"), ("disabled", "1")):
            with self.subTest(attribute=attribute), self.assertRaises(CHECKER.VerificationError):
                self.verify(xml=junit(attributes={attribute: value}))

    def test_junit_testsuites_aggregate_count_is_verified(self):
        xml = '<testsuites tests="3" failures="0" errors="0"><testsuite name="part" tests="2"><testcase name="Alpha.One"/><testcase name="Beta.Two"/></testsuite></testsuites>'
        with self.assertRaisesRegex(CHECKER.VerificationError, "tests mismatch"):
            self.verify(xml=xml)

    def test_junit_missing_case_name_is_rejected(self):
        with self.assertRaisesRegex(CHECKER.VerificationError, "empty name"):
            self.verify(xml='<testsuite tests="2"><testcase/><testcase name="Beta.Two"/></testsuite>')

    def test_junit_suite_time_must_be_valid(self):
        with self.assertRaisesRegex(CHECKER.VerificationError, "testsuite.time"):
            self.verify(xml=junit(attributes={"time": "NaN"}))

    def test_junit_status_without_a_time_is_allowed(self):
        self.assertEqual(self.verify(xml=junit().replace(' time="0.1"', ""))["executed"], 2)

    def test_load_inventory_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text('{"tests": [], "tests": []}')
            with self.assertRaisesRegex(CHECKER.VerificationError, "duplicate JSON"):
                CHECKER.load_inventory(path)

    def test_cli_inventory_only_is_read_only_candidate_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inventory.json"
            path.write_text(json.dumps(inventory()))
            result = subprocess.run(["python3", str(ROOT / "tools/verify_ctest_report.py"),
                                     "--inventory", str(path), "--inventory-only"],
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["candidate_names_sha256"], self.expected)

    def test_cli_invalid_report_returns_exit_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inventory_path = directory / "inventory.json"
            junit_path = directory / "junit.xml"
            inventory_path.write_text(json.dumps(inventory()))
            junit_path.write_text(junit(("Alpha.One", "Other")))
            result = subprocess.run(["python3", str(ROOT / "tools/verify_ctest_report.py"),
                                     "--inventory", str(inventory_path), "--junit", str(junit_path),
                                     "--expected-count", "2", "--expected-sha256", self.expected],
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Invalid CTest report", result.stderr)

    def test_junit_time_must_be_finite_nonnegative(self):
        for value in ("NaN", "inf", "-1", "broken"):
            with self.subTest(value=value), self.assertRaisesRegex(CHECKER.VerificationError, "time"):
                self.verify(xml=junit().replace('time="0.1"', f'time="{value}"', 1))

    def test_testsuites_root_and_nested_suite_are_supported(self):
        xml = '<testsuites tests="2" failures="0" errors="0"><testsuite name="part" tests="2" failures="0" errors="0" skipped="0"><testcase name="Alpha.One"/><testcase name="Beta.Two"/></testsuite></testsuites>'
        self.assertEqual(self.verify(xml=xml)["passed"], 2)

    def test_malformed_xml_and_invalid_root_are_rejected(self):
        for xml in ("<testsuite>", "<other/>"):
            with self.subTest(xml=xml), self.assertRaises(CHECKER.VerificationError):
                self.verify(xml=xml)

    def test_inventory_only_candidate_does_not_freeze_expected_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inventory.json"
            path.write_text(json.dumps(inventory()))
            loaded = CHECKER.load_inventory(path)
            self.assertEqual(CHECKER.names_digest([test["name"] for test in loaded["tests"]]), self.expected)


if __name__ == "__main__":
    unittest.main()
