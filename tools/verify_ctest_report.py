#!/usr/bin/env python3
"""Fail closed verification for a CTest inventory and its JUnit result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


class VerificationError(ValueError):
    """The inventory or JUnit report cannot prove a complete successful run."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise VerificationError(f"non-finite JSON constant: {value}")


def load_inventory(path: Path) -> dict:
    """Load a CTest --show-only=json-v1 result without silently accepting JSON quirks."""
    try:
        inventory = json.loads(Path(path).read_bytes(), object_pairs_hook=_pairs,
                               parse_constant=_reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read CTest inventory: {error}") from error
    _require(isinstance(inventory, dict), "inventory root must be an object")
    return inventory


def names_digest(names: list[str]) -> str:
    """SHA256 of the canonical, sorted JSON list of exact CTest names."""
    encoded = json.dumps(sorted(names), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_disabled(properties) -> bool:
    if isinstance(properties, dict):
        properties = [{"name": key, "value": value} for key, value in properties.items()]
    _require(isinstance(properties, list), "test properties must be an array")
    for property_ in properties:
        _require(isinstance(property_, dict), "test property must be an object")
        name = property_.get("name")
        _require(isinstance(name, str) and bool(name), "test property must have a name")
        _require("value" in property_, "test property must have a value")
        if not isinstance(name, str) or name.upper() != "DISABLED":
            continue
        value = property_.get("value")
        _require(isinstance(value, (bool, str)), "DISABLED property value must be boolean or string")
        if value is True or (isinstance(value, str) and _cmake_truthy(value)):
            return True
    return False


def _cmake_truthy(value: str) -> bool:
    """The CMake if() truth values relevant to a DISABLED property."""
    normalized = value.strip().upper()
    if normalized in {"", "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND"}:
        return False
    return not normalized.endswith("-NOTFOUND")


def verify_inventory(inventory: dict, expected_count: int, expected_names_sha256: str) -> list[str]:
    """Validate the complete discovered test set and return its sorted names."""
    _require(type(expected_count) is int and expected_count > 0, "expected count must be positive")
    _require(isinstance(expected_names_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_names_sha256),
             "expected names SHA256 must be lowercase hexadecimal")
    _require(isinstance(inventory, dict), "inventory root must be an object")
    tests = inventory.get("tests")
    _require(isinstance(tests, list), "inventory must contain a tests array")
    names = []
    for index, test in enumerate(tests):
        _require(isinstance(test, dict), f"inventory tests[{index}] must be an object")
        name = test.get("name")
        _require(isinstance(name, str) and bool(name), f"inventory tests[{index}] has an empty name")
        _require(not _is_disabled(test.get("properties", [])), f"inventory test is disabled: {name}")
        names.append(name)
    _require(len(names) == len(set(names)), "inventory contains duplicate test names")
    _require(len(names) == expected_count,
             f"inventory count mismatch: actual={len(names)} expected={expected_count}")
    names.sort()
    actual_digest = names_digest(names)
    _require(actual_digest == expected_names_sha256,
             f"inventory names SHA256 mismatch: actual={actual_digest} expected={expected_names_sha256}")
    return names


def _number(value: str, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise VerificationError(f"{context} must be a number") from error
    _require(math.isfinite(result) and result >= 0, f"{context} must be finite and nonnegative")
    return result


def _declared_count(element: ET.Element, attribute: str, actual: int) -> None:
    if attribute not in element.attrib:
        return
    value = element.attrib[attribute]
    _require(re.fullmatch(r"[0-9]+", value) is not None, f"{element.tag}.{attribute} must be an integer")
    _require(int(value) == actual,
             f"{element.tag}.{attribute} mismatch: actual={actual} declared={value}")


def _testcases(element: ET.Element) -> list[ET.Element]:
    return list(element.iter("testcase"))


def _validate_suite(element: ET.Element) -> None:
    if "time" in element.attrib:
        _number(element.attrib["time"], f"{element.tag}.time")
    cases = _testcases(element)
    _require(not (element.findall("failure") or element.findall("error") or element.findall("skipped")),
             f"{element.tag} contains a direct failure, error, or skipped node")
    _declared_count(element, "tests", len(cases))
    failed = sum(bool(case.findall("failure") or case.findall("error")) for case in cases)
    skipped = sum(bool(case.findall("skipped")) for case in cases)
    disabled = sum(case.attrib.get("status", "").lower() == "disabled" for case in cases)
    _declared_count(element, "failures", sum(bool(case.findall("failure")) for case in cases))
    _declared_count(element, "errors", sum(bool(case.findall("error")) for case in cases))
    _declared_count(element, "skipped", skipped)
    _declared_count(element, "disabled", disabled)
    _require(failed == 0 and skipped == 0 and disabled == 0,
             f"{element.tag} contains failure, error, skipped, or disabled cases")


def _load_junit(path: Path) -> ET.Element:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise VerificationError(f"cannot read JUnit report: {error}") from error
    _require(root.tag in {"testsuite", "testsuites"}, "JUnit root must be testsuite or testsuites")
    return root


def verify_run(inventory: dict, junit_path: Path, expected_count: int,
               expected_names_sha256: str) -> dict:
    """Verify that JUnit reports every discovered test exactly once and none failed."""
    names = verify_inventory(inventory, expected_count, expected_names_sha256)
    root = _load_junit(junit_path)
    for suite in root.iter("testsuite"):
        _validate_suite(suite)
    if root.tag == "testsuites":
        _validate_suite(root)
    cases = _testcases(root)
    actual_names = []
    not_run = 0
    for index, case in enumerate(cases):
        name = case.attrib.get("name")
        _require(isinstance(name, str) and bool(name), f"JUnit testcase[{index}] has an empty name")
        status = case.attrib.get("status", "run").lower()
        if status in {"notrun", "not_run"}:
            not_run += 1
        _require(status in {"run", "passed", "completed"},
                 f"JUnit testcase has a non-passing status: {name}={status}")
        if "result" in case.attrib:
            result = case.attrib["result"].lower()
            _require(result in {"passed", "completed"},
                     f"JUnit testcase has a non-passing result: {name}={result}")
        if "time" in case.attrib:
            _number(case.attrib["time"], f"JUnit testcase {name}.time")
        _require(not (case.findall("failure") or case.findall("error") or case.findall("skipped")),
                 f"JUnit testcase did not pass: {name}")
        actual_names.append(name)
    _require(len(actual_names) == len(set(actual_names)), "JUnit contains duplicate testcase names")
    _require(set(actual_names) == set(names), "JUnit testcase names differ from the discovered inventory")
    return {
        "expected": expected_count,
        "discovered": len(names),
        "executed": len(actual_names),
        "passed": len(actual_names),
        "failed": 0,
        "skipped": 0,
        "disabled": 0,
        "not_run": not_run,
        "names_sha256": names_digest(names),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--inventory-only", action="store_true")
    options = parser.parse_args()
    try:
        inventory = load_inventory(options.inventory)
        if options.inventory_only:
            _require(options.junit is None and options.expected_count is None and options.expected_sha256 is None,
                     "inventory-only cannot accept JUnit or expected values")
            tests = inventory.get("tests")
            _require(isinstance(tests, list), "inventory must contain a tests array")
            names = [test.get("name") for test in tests if isinstance(test, dict) and isinstance(test.get("name"), str)]
            _require(len(names) == len(tests) and len(names) == len(set(names)), "inventory names are invalid")
            print(json.dumps({"candidate_count": len(names), "candidate_names_sha256": names_digest(names)}, sort_keys=True))
            return 0
        _require(options.junit is not None and options.expected_count is not None and options.expected_sha256 is not None,
                 "run verification requires JUnit, expected count, and expected SHA256")
        print(json.dumps(verify_run(inventory, options.junit, options.expected_count, options.expected_sha256),
                         sort_keys=True))
        return 0
    except VerificationError as error:
        print(f"Invalid CTest report: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
