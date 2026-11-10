import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE_DIR = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "verify_cpu_isa_build", SOURCE_DIR / "tools/verify_cpu_isa_build.py")
ISA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ISA)


class CpuIsaBuildContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="aethermind-isa-build-")
        self.addCleanup(self.temporary.cleanup)
        self.build_dir = Path(self.temporary.name)
        (self.build_dir / "CMakeCache.txt").write_text("BUILD_TESTS:BOOL=ON\n")

    def entry(self, name, flags=()):
        return {"directory": str(self.build_dir), "file": str(SOURCE_DIR / name),
                "arguments": ["c++", "-std=c++20", *flags, "-c", str(SOURCE_DIR / name)]}

    def commands(self, avx2):
        entries = [self.entry("tests/unit/backend/cpu/kernels/test_avx_learning.cpp")]
        if avx2:
            entries.extend(self.entry(name, ("-mavx2", "-mfma", "-fno-lto"))
                           for name in sorted(ISA.LIBRARY_SOURCES | {ISA.TEST_SOURCE}))
        return entries

    def verify(self, entries, avx2, f16c=False, conversion=False):
        (self.build_dir / "compile_commands.json").write_text(json.dumps(entries))
        return ISA.verify(SOURCE_DIR, self.build_dir, avx2, f16c, conversion)

    def test_conversion_implementations_are_independently_optional(self):
        for f16c, conversion in ((False, True), (True, False), (True, True)):
            with self.subTest(f16c=f16c, conversion=conversion):
                entries = self.commands(False)
                if f16c:
                    entries.append(self.entry(ISA.F16C_SOURCE, ("-mavx", "-mf16c", "-fno-lto")))
                if conversion:
                    entries.append(self.entry(ISA.AVX2_CONVERSION_SOURCE, ("-mavx2", "-fno-lto")))
                result = self.verify(entries, False, f16c, conversion)
                self.assertEqual(len(result["isa_translation_units"]), int(f16c) + int(conversion))

    def test_conversion_sources_reject_extra_isa_and_lto_override(self):
        for flag in ("-mfma", "-flto"):
            entries = self.commands(False)
            entries.append(self.entry(ISA.F16C_SOURCE, ("-mavx", "-mf16c", "-fno-lto", flag)))
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "invalid ISA/LTO"):
                self.verify(entries, False, True)

    def test_disabled_conversion_macros_and_sources_fail(self):
        for macro in ("FP16_HAS_F16C_CONVERSION", "FP16_HAS_AVX2_CONVERSION"):
            entries = self.commands(False)
            entries[0]["arguments"].append("-D" + macro + "=1")
            with self.subTest(macro=macro), self.assertRaisesRegex(ValueError, "availability macro"):
                self.verify(entries, False)
        with self.assertRaisesRegex(ValueError, "Optional FP16 source"):
            self.verify([self.entry(ISA.F16C_SOURCE, ("-mavx", "-mf16c", "-fno-lto"))], False)

    def test_conversion_implementation_cannot_merge_into_unity(self):
        unity = self.build_dir / "CMakeFiles/unit.dir/Unity/unity_0_cxx.cxx"
        unity.parent.mkdir(parents=True)
        unity.write_text('#include "' + str(SOURCE_DIR / ISA.F16C_SOURCE) + '"\n')
        entries = self.commands(False)
        entries.append(self.entry(ISA.F16C_SOURCE, ("-mavx", "-mf16c", "-fno-lto")))
        entries.append({"directory": str(self.build_dir), "file": str(unity),
                        "arguments": ["c++", "-c", str(unity)]})
        with self.assertRaisesRegex(ValueError, "merged into Unity"):
            self.verify(entries, False, True)

    def test_baseline_has_no_optional_implementations(self):
        result = self.verify(self.commands(False), False)
        self.assertEqual(result["isa_translation_units"], [])

    def test_avx2_implementations_have_isolated_options(self):
        result = self.verify(self.commands(True), True)
        self.assertEqual(len(result["isa_translation_units"]), 5)

    def test_baseline_rejects_optional_source(self):
        with self.assertRaisesRegex(ValueError, "Optional AVX2 source"):
            self.verify(self.commands(True), False)

    def test_wrappers_reject_global_isa_uplift(self):
        for avx2 in (False, True):
            for flag in ("-mavx2", "-mfma", "-mf16c", "-mavx", "-march=native", "-mcpu=native", "+avx2"):
                with self.subTest(avx2=avx2, flag=flag):
                    entries = self.commands(avx2)
                    entries[0]["arguments"].append(flag)
                    with self.assertRaisesRegex(ValueError, "Baseline translation unit"):
                        self.verify(entries, avx2)

    def test_missing_implementation_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing optional implementation"):
            self.verify(self.commands(True)[:-1], True)

    def test_lto_cannot_override_isa_boundary(self):
        entries = self.commands(True)
        entries[-1]["arguments"].append("-flto")
        with self.assertRaisesRegex(ValueError, "invalid ISA/LTO"):
            self.verify(entries, True)

    def test_baseline_rejects_availability_macro(self):
        entries = self.commands(False)
        entries[0]["arguments"].append("-DGEMM_HAS_AVX2_FMA_KERNEL=1")
        with self.assertRaisesRegex(ValueError, "availability macro"):
            self.verify(entries, False)

    def test_generated_unity_wrapper_is_checked(self):
        for avx2 in (False, True):
            with self.subTest(avx2=avx2):
                unity = self.build_dir / "CMakeFiles/unit.dir/Unity/unity_0_cxx.cxx"
                unity.parent.mkdir(parents=True, exist_ok=True)
                unity.write_text('#include "' + str(SOURCE_DIR / "src/model/model_loader.cpp") + '"\n')
                entries = self.commands(avx2)
                entries.append({"directory": str(self.build_dir), "file": str(unity),
                                "arguments": ["c++", "-mavx2", "-mfma", "-c", str(unity)]})
                with self.assertRaisesRegex(ValueError, "Baseline translation unit"):
                    self.verify(entries, avx2)

    def test_isa_implementation_cannot_merge_into_unity(self):
        unity = self.build_dir / "CMakeFiles/unit.dir/Unity/unity_0_cxx.cxx"
        unity.parent.mkdir(parents=True)
        unity.write_text('#include "' + str(SOURCE_DIR / ISA.TEST_SOURCE) + '"\n')
        entries = self.commands(True)
        entries.append({"directory": str(self.build_dir), "file": str(unity),
                        "arguments": ["c++", "-c", str(unity)]})
        with self.assertRaisesRegex(ValueError, "merged into Unity"):
            self.verify(entries, True)


if __name__ == "__main__":
    unittest.main()
