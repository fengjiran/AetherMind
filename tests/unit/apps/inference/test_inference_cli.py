"""Black-box CLI contracts; numerical oracles remain in the C++ execution tests.

Run with --infer /path/to/aethermind_infer, or set AETHERMIND_INFER_BINARY.
Only the Python standard library and the existing C toolchain are required.
"""

import argparse
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
INFER_BINARY = os.environ.get("AETHERMIND_INFER_BINARY")
FIXTURE_ROOT = ROOT / ".models/inference-admission"
TIMEOUT_SECONDS = 30
FP32_FIXTURES = ("f32-missing", "f32-auto", "f32-explicit", "f32-tied")
REJECTED_PRECISION_FIXTURES = (
    "f16-missing", "f16-auto", "bf16-missing", "bf16-auto",
    "f16-f32-hint", "bf16-f32-hint", "f32-f16-hint", "f32-bf16-hint",
    "mixed-norm-f16-missing", "mixed-norm-f16-auto",
    "mixed-norm-bf16-missing", "mixed-norm-bf16-auto",
    "single-norm-f16-missing", "single-norm-f16-auto",
    "single-norm-bf16-missing", "single-norm-bf16-auto",
)


class InferenceCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not INFER_BINARY:
            raise RuntimeError("Supply --infer or AETHERMIND_INFER_BINARY")
        cls.binary = Path(INFER_BINARY).resolve()
        if not cls.binary.is_file() or not os.access(cls.binary, os.X_OK):
            raise RuntimeError(f"Inference executable is missing or not executable: {cls.binary}")
        cls.fixtures = Path(FIXTURE_ROOT).resolve()
        for name in FP32_FIXTURES + REJECTED_PRECISION_FIXTURES:
            for filename in ("config.json", "model.safetensors"):
                path = cls.fixtures / name / filename
                if not path.is_file():
                    raise RuntimeError(f"Generate and check M0 fixtures before CLI tests: {path}")
        cls.model = cls.fixtures / "f32-explicit"
        config = json.loads((cls.model / "config.json").read_text(encoding="utf-8"))
        cls.config = config
        cls.vocab_size = config["vocab_size"]
        cls.model_context = config["max_position_embeddings"]

    @contextmanager
    def model_directory(self, changes=None, *, name="model"):
        # Preserve valid FP32 bytes: each rejection changes only config semantics.
        with tempfile.TemporaryDirectory(prefix="aethermind-cli-model-") as temporary:
            directory = Path(temporary) / name
            directory.mkdir()
            config = json.loads((self.model / "config.json").read_text(encoding="utf-8"))
            config.update(changes or {})
            (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
            shutil.copyfile(self.model / "model.safetensors", directory / "model.safetensors")
            yield directory

    def arguments(self, *, model=None, tokens="7", capacity="8", new_tokens="3",
                  eos=None, preset=None):
        result = ["--model", str(self.model if model is None else model),
                  "--tokens", str(tokens), "--context-tokens", str(capacity),
                  "--max-new-tokens", str(new_tokens)]
        if eos is not None:
            result += ["--eos-token-id", str(eos)]
        if preset is not None:
            result += ["--preset", preset]
        return result

    def invoke(self, arguments, *, environment=None):
        return subprocess.run([str(self.binary), *arguments], capture_output=True,
                              text=True, timeout=TIMEOUT_SECONDS,
                              env=environment, cwd=ROOT)

    def generate(self, arguments):
        result = self.invoke(arguments)
        return self.expect_success(result)

    def expect_success(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.endswith("\n"), repr(result.stdout))
        self.assertEqual(len(result.stdout.splitlines()), 1, repr(result.stdout))
        tokens = json.loads(result.stdout)
        self.assertIsInstance(tokens, list)
        for token in tokens:
            self.assertIs(type(token), int)
            self.assertGreaterEqual(token, 0)
            self.assertLess(token, self.vocab_size)
        return tokens

    def expect_error(self, arguments, exit_code=2, *, stage=None):
        result = self.invoke(arguments)
        self.assertEqual(result.returncode, exit_code, result.stderr)
        self.assertEqual(result.stdout, "", repr(result.stdout))
        self.assertTrue(result.stderr.strip(), "Errors require a stderr diagnostic")
        if stage is not None:
            self.assertIn(stage, result.stderr.lower())
        return result

    def test_default_generation_is_one_json_line(self):
        self.assertEqual(len(self.generate(self.arguments())), 3)

    def test_explicit_o1_matches_default(self):
        self.assertEqual(self.generate(self.arguments()),
                         self.generate(self.arguments(preset="o1-plain")))

    def test_o2_is_deterministic_independently(self):
        arguments = self.arguments(preset="o2-packed")
        first = self.generate(arguments)
        self.assertEqual(len(first), 3)
        self.assertEqual(first, self.generate(arguments))

    def test_zero_generation_returns_empty_json_array(self):
        self.assertEqual(self.generate(self.arguments(new_tokens="0")), [])

    def test_zero_generation_allows_prompt_beyond_physical_capacity(self):
        self.assertEqual(self.generate(self.arguments(tokens="7,7", capacity="1",
                                                      new_tokens="0")), [])

    def test_eos_is_included_and_stops_generation(self):
        first = self.generate(self.arguments())[0]
        self.assertEqual(self.generate(self.arguments(eos=first)), [first])

    def test_exact_capacity_counts_only_decode_appends(self):
        self.assertEqual(len(self.generate(self.arguments(capacity="3"))), 3)

    def test_generation_beyond_capacity_fails_without_partial_output(self):
        self.expect_error(self.arguments(capacity="2"), stage="parameters")

    def test_prompt_and_eos_vocabulary_boundaries_fail(self):
        for arguments in (self.arguments(tokens=self.vocab_size),
                          self.arguments(eos=self.vocab_size)):
            with self.subTest(arguments=arguments):
                self.expect_error(arguments, stage="parameters")

    def test_zero_generation_keeps_prompt_and_eos_validation(self):
        for arguments in (self.arguments(tokens=self.vocab_size, new_tokens="0"),
                          self.arguments(eos=self.vocab_size, new_tokens="0")):
            with self.subTest(arguments=arguments):
                self.expect_error(arguments, stage="parameters")

    def test_capacity_cannot_exceed_model_context(self):
        self.expect_error(self.arguments(capacity=self.model_context + 1), stage="parameters")

    def test_prompt_cannot_exceed_model_context_even_for_zero_generation(self):
        self.expect_error(self.arguments(tokens=",".join(["7"] * (self.model_context + 1)),
                                         capacity=self.model_context, new_tokens="0"),
                          stage="parameters")

    def test_requested_token_count_overflow_is_a_parameter_failure(self):
        maximum_size = (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1
        self.expect_error(self.arguments(tokens="7,7", new_tokens=maximum_size),
                          stage="parameters")

    def test_uint32_maximum_is_parsed_then_rejected_by_vocabulary(self):
        self.expect_error(self.arguments(tokens=(1 << 32) - 1), stage="parameters")

    def test_token_zero_is_valid(self):
        self.assertEqual(self.generate(self.arguments(tokens="0", new_tokens="0")), [])

    def test_missing_model_is_a_load_failure(self):
        self.expect_error(self.arguments(model=self.fixtures / "missing-model"),
                          1, stage="load")

    def test_unsupported_family_is_a_compile_failure(self):
        with self.model_directory({"model_type": "qwen2"}) as model:
            result = self.expect_error(self.arguments(model=model), 1, stage="compile")
            self.assertIn("unsupported model type", result.stderr)

    def test_unknown_rope_is_a_compile_failure(self):
        rope = dict(self.config["rope_parameters"])
        rope["rope_type"] = "unsupported_cli_fixture"
        with self.model_directory({"rope_parameters": rope}) as model:
            result = self.expect_error(self.arguments(model=model), 1, stage="compile")
            self.assertIn("RoPE", result.stderr)

    def test_unsupported_activation_is_rejected_during_loading(self):
        with self.model_directory({"hidden_act": "gelu"}) as model:
            result = self.expect_error(self.arguments(model=model), 1, stage="load")
            self.assertIn("hidden_act", result.stderr)

    def test_model_directory_with_spaces_preserves_literal_path(self):
        expected = self.generate(self.arguments())
        with self.model_directory(name="model directory with spaces") as model:
            self.assertEqual(self.generate(self.arguments(model=model)), expected)

    def test_loader_supported_non_fp32_inputs_fail_admission(self):
        for name in REJECTED_PRECISION_FIXTURES:
            with self.subTest(fixture=name):
                result = self.expect_error(self.arguments(model=self.fixtures / name),
                                           1, stage="model admission")
                self.assertIn("Float32", result.stderr)

    def test_help_is_an_explicit_non_json_success(self):
        result = self.invoke(["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model", result.stdout)
        self.assertIn("--max-new-tokens", result.stdout)

    def test_help_cannot_be_combined_with_other_arguments(self):
        for arguments in (["--help", *self.arguments()],
                          [*self.arguments(), "--help"], ["--help", "--help"]):
            with self.subTest(arguments=arguments):
                self.expect_error(arguments)

    def test_options_require_separate_flag_value_arguments(self):
        for option, value in (("--model", str(self.model)), ("--tokens", "7"),
                              ("--context-tokens", "8"), ("--max-new-tokens", "3"),
                              ("--eos-token-id", "1"), ("--preset", "o1-plain")):
            arguments = self.arguments()
            combined = f"{option}={value}"
            if option in arguments:
                index = arguments.index(option)
                arguments[index:index + 2] = [combined]
            else:
                arguments.append(combined)
            with self.subTest(option=option):
                self.expect_error(arguments)

    def test_no_arguments_is_a_parameter_error(self):
        self.expect_error([])

    def test_each_required_option_is_required(self):
        for option in ("--model", "--tokens", "--context-tokens", "--max-new-tokens"):
            arguments = self.arguments()
            index = arguments.index(option)
            del arguments[index:index + 2]
            with self.subTest(option=option):
                self.expect_error(arguments)

    def test_missing_option_values_are_rejected(self):
        for option in ("--model", "--tokens", "--context-tokens", "--max-new-tokens",
                       "--eos-token-id", "--preset"):
            with self.subTest(option=option):
                self.expect_error(self.arguments() + [option])

    def test_duplicate_options_are_rejected(self):
        duplicates = (("--model", str(self.model)), ("--tokens", "7"),
                      ("--context-tokens", "8"), ("--max-new-tokens", "3"),
                      ("--eos-token-id", "1"), ("--preset", "o1-plain"))
        for option, value in duplicates:
            arguments = self.arguments()
            if option not in arguments:
                arguments += [option, value]
            with self.subTest(option=option):
                self.expect_error(arguments + [option, value])

    def test_unknown_options_and_positional_arguments_are_rejected(self):
        for extra in (["--unknown", "1"], ["extra"]):
            with self.subTest(extra=extra):
                self.expect_error(self.arguments() + extra)

    def test_invalid_presets_are_rejected(self):
        for preset in ("", "o1", "o2-plain", "O1-Plain", " o1-plain"):
            with self.subTest(preset=preset):
                self.expect_error(self.arguments(preset=preset))

    def test_empty_model_path_is_rejected(self):
        self.expect_error(self.arguments(model=""))

    def test_tokens_require_strict_unsigned_decimal_items(self):
        values = ("", "-1", "+7", " 7", "7 ", "7, 8", "7\t", ",7",
                  "7,", "7,,8", "7x", "7,8x", "0x7", "7.0")
        for value in values:
            with self.subTest(value=value):
                self.expect_error(self.arguments(tokens=value))

    def test_tokens_reject_uint32_and_arbitrary_integer_overflow(self):
        for value in (str(1 << 32), "9" * 1000):
            with self.subTest(value=value):
                self.expect_error(self.arguments(tokens=value))

    def test_capacity_requires_a_positive_unsigned_decimal_integer(self):
        for value in ("", "0", "-1", "+8", " 8", "8 ", "8x", "8.0", "9" * 1000):
            with self.subTest(value=value):
                self.expect_error(self.arguments(capacity=value))

    def test_generation_limit_requires_a_nonnegative_unsigned_decimal_integer(self):
        for value in ("", "-1", "+3", " 3", "3 ", "3x", "3.0", "9" * 1000):
            with self.subTest(value=value):
                self.expect_error(self.arguments(new_tokens=value))

    def test_eos_requires_a_uint32_decimal_integer(self):
        for value in ("", "-1", "+1", " 1", "1 ", "1x", "1.0",
                      str(1 << 32), "9" * 1000):
            with self.subTest(value=value):
                self.expect_error(self.arguments(eos=value))

    def test_logger_initialization_allocation_failure_precedes_model_loading(self):
        self.assertTrue(sys.platform.startswith("linux"),
                        "The logger allocation gate requires Linux/glibc")
        arguments = self.arguments(model=self.fixtures / "missing-logger-injection-model")
        self.expect_error(arguments, 1, stage="load")
        source = Path(__file__).with_name("fixtures") / "fail_logger_allocation.c"
        compiler = shlex.split(os.environ.get("CC", "cc"))
        self.assertTrue(compiler, "A C compiler is required for logger allocation injection")
        with tempfile.TemporaryDirectory(prefix="aethermind-cli-logger-oom-", dir="/tmp") as temporary:
            shim = Path(temporary) / "fail_logger_allocation.so"
            compiled = subprocess.run([*compiler, "-shared", "-fPIC", "-Wall", "-Wextra",
                                       "-Werror", str(source), "-ldl", "-o", str(shim)],
                                      capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            environment = os.environ.copy()
            environment["LD_PRELOAD"] = str(shim)
            for attempt in range(3):
                with self.subTest(attempt=attempt):
                    result = self.invoke(arguments, environment=environment)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertEqual(result.stdout, "", repr(result.stdout))
                    self.assertIn("logger initialization", result.stderr)
                    self.assertIn("RESOURCE_EXHAUSTED", result.stderr)
                    self.assertNotIn(": load:", result.stderr)

    def test_real_library_warning_uses_stderr_without_polluting_json(self):
        self.assertTrue(sys.platform.startswith("linux"),
                        "The real THP warning gate requires Linux; evidence is unavailable here")
        source = Path(__file__).with_name("fixtures") / "fail_hugepage_advice.c"
        compiler = shlex.split(os.environ.get("CC", "cc"))
        self.assertTrue(compiler, "A C compiler is required for the real library-warning gate")
        # LD_PRELOAD treats whitespace as a separator rather than a path escape.
        with tempfile.TemporaryDirectory(prefix="aethermind-cli-logger-", dir="/tmp") as temporary:
            shim = Path(temporary) / "fail_hugepage_advice.so"
            compiled = subprocess.run([*compiler, "-shared", "-fPIC", "-Wall", "-Wextra",
                                       "-Werror", str(source), "-o", str(shim)],
                                      capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            environment = os.environ.copy()
            environment["LD_PRELOAD"] = str(shim)
            environment["THP_MEM_ALLOC_ENABLE"] = "1"
            result = self.invoke(self.arguments(tokens=",".join(["7"] * 17),
                                                 capacity="17", new_tokens="1"),
                                 environment=environment)
            self.assertEqual(len(self.expect_success(result)), 1)
            self.assertIn("thp madvise for HUGEPAGE failed", result.stderr)
            self.assertNotIn("thp madvise for HUGEPAGE failed", result.stdout)


def main():
    global INFER_BINARY, FIXTURE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer", default=INFER_BINARY,
                        help="Path to the built aethermind_infer executable")
    parser.add_argument("--fixtures", type=Path, default=FIXTURE_ROOT,
                        help="Generated and checked M0 HF fixture directory")
    arguments, unittest_arguments = parser.parse_known_args()
    if not arguments.infer:
        parser.error("--infer or AETHERMIND_INFER_BINARY is required")
    INFER_BINARY = arguments.infer
    FIXTURE_ROOT = arguments.fixtures
    unittest.main(argv=[sys.argv[0], *unittest_arguments])


if __name__ == "__main__":
    main()
