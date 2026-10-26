# FP32 admission HF fixtures

This directory contains the source recipe and reviewed checksums for milestone M0 of the [upper-level Runtime assembly plan](../../../docs/improvement-plan/10-upper-level-runtime-assembly.md). Test sources live in `tests/unit/apps/inference/`; generated model directories live under the gitignored `.models/inference-admission/`.

These assets verify the inputs to the first FP32 admission policy. They do not demonstrate FP16/BF16 execution support or provide generation goldens.

## Pinned source

The base is the untied, single-file FP32 checkpoint from [peft-internal-testing/tiny-random-LlamaForCausalLM at d0b81c5](https://huggingface.co/peft-internal-testing/tiny-random-LlamaForCausalLM/tree/d0b81c5e23f2d61bb7a5f430a8cd6f3e711c0cb3). Only `config.json` and `model.safetensors` are needed; tokenizer files are not used.

| File | Bytes | SHA-256 |
|---|---:|---|
| `config.json` | 720 | `ea68007f47752b68f0f58e5fc2d51fa380bb730af51faff8c1df9fccbe608bc2` |
| `model.safetensors` | 4131280 | `49c20f32c6c597480fcaec5df2f86c645eabea765cbea1e67886dbae45e5c992` |

The remote config bytes and the weight file's Git LFS pointer were checked against the local base. The recipe enforces both file hashes before generation; a directory name or a floating branch is insufficient source identity.

## Prepare and verify

Run these commands from the repository root. Python uses only the standard library; no torch, transformers, or safetensors package is required.

```bash
MODEL_DIR=.models/tiny-random-LlamaForCausalLM
HF_REVISION=d0b81c5e23f2d61bb7a5f430a8cd6f3e711c0cb3
mkdir -p "$MODEL_DIR"
curl --fail --location "https://huggingface.co/peft-internal-testing/tiny-random-LlamaForCausalLM/resolve/$HF_REVISION/config.json" --output "$MODEL_DIR/config.json"
curl --fail --location "https://huggingface.co/peft-internal-testing/tiny-random-LlamaForCausalLM/resolve/$HF_REVISION/model.safetensors" --output "$MODEL_DIR/model.safetensors"

python3 tools/generate_fp32_admission_hf_fixtures.py \
  --source "$MODEL_DIR" --recipe tests/fixtures/inference_admission/recipe.json \
  --output-dir .models/inference-admission
python3 tools/generate_fp32_admission_hf_fixtures.py \
  --source "$MODEL_DIR" --recipe tests/fixtures/inference_admission/recipe.json \
  --output-dir .models/inference-admission \
  --check tests/fixtures/inference_admission/checksums.json
```

Generation preserves the base and writes 20 complete HF variants plus `generated-manifest.json`. The variants cover missing/auto/explicit hints, real F16/BF16 payloads, hint mismatches, mixed and single non-first-layer norms, and a tied lm-head. All must pass `ModelLoader::Load`; the application's FP32 admission helper rejects unsupported execution inputs.

`--check` reads the reviewed checksums, checks the existing output, and regenerates into a temporary directory for comparison. It never updates expected values. Python implementation/version are recorded as provenance; environment descriptions do not override the source, script, recipe, file, or tensor content checks.

If the script, recipe, or base changes, regenerate, review the actual manifest, and explicitly update `checksums.json`. CI must not accept a new manifest automatically. Missing assets or checksum mismatches fail the gate.

## Focused validation

```bash
python3 -m unittest discover -s tests/unit/apps/inference -p 'test_generate_fp32_admission_hf_fixtures.py' -v
cmake --build build --target aethermind_unit_tests -j
./build/tests/unit/aethermind_unit_tests --gtest_filter='AdmissionFixtures/*:InferenceAdmissionFixtures.*'
```

The Python tests independently inspect serialized tensors and conversion boundaries. The M0 C++ tests load every variant through the real loader and check the resolved roles, shapes, byte counts, dtype hints, and tied binding. M1 adds the application's metadata-only admission policy and configuration helpers:

```bash
./build/tests/unit/aethermind_unit_tests --gtest_filter='InferenceRunOptions.*:InferenceRuntimeOptions.*:InferenceCpuPreset.*:Fp32ExecutionInputs.*:AdmissionFixtures/Fp32HfAdmission.*:AdmissionFixtures/Fp32HfCompile.*:RequiredRoles/Fp32RequiredRoleAdmission.*'
```

The real Load→admission tests accept four FP32 variants and reject the other sixteen. Eight Compile tests cover the four accepted fixtures and both CPU presets without constructing Runtime. Synthetic role-by-role tests locate missing dtype checks; they do not replace the real HF cases.

M2 adds the real synchronous runner and a builder injection seam compiled only for tests:

```bash
TMPDIR=/tmp ./build/tests/unit/aethermind_unit_tests --gtest_filter='HfPresets/InferenceRunnerEquivalence.*:InferenceRunnerSuccess.*:InferenceRunnerFailure.*:KvPlanes/RunnerKvAllocationFailure.*:AdmissionFixtures/RunnerFp32Admission.*'
```

The 39 M2 tests cover independent Session equivalence, failure stages, partial allocation cleanup, and rejection before KV allocation. CLI black-box validation and the later C ABI shared-policy gate remain separate.

M3 provides the optional standalone executable and its Python black-box tests. Build with tests/benchmarks disabled as described in the [root README](../../../README.md), then run:

```bash
python3 tests/unit/apps/inference/test_inference_cli.py \
  --infer "$PWD/build-infer/apps/inference/aethermind_infer" -v
```

M3's 31 tests covered strict flag/number/CSV parsing, JSON output, exit codes, EOS/capacity/zero generation, and representative precision rejection. The Linux library-log test compiles `tests/unit/apps/inference/fixtures/fail_hugepage_advice.c` into a temporary preload library, enables the existing THP allocator gate, and forces only `madvise(MADV_HUGEPAGE)` to fail. A 17-token request triggers the real shared-library allocator warning; the test requires that warning on stderr and an unpolluted JSON result on stdout. The shim is never part of the application target. Python and the existing C compiler are required for this test; unavailable fixtures, compiler, or Linux log evidence fail explicitly.

Thirty ordinary CLI cases also passed with the sanitizer executable in M3; the external-preload log case was checked separately in the normal build. Leak checking was disabled for that address/UB run under the existing M2 LSAN limitation below.

M4 completed the C++ integration matrix and expanded the Python CLI tests to 38 methods (some matrices use subTests). All 16 unsupported precision variants now require admission-stage rejection, exit 1, and empty stdout. Additional valid-FP32-payload cases verify unsupported family/unknown RoPE at Compile, activation rejection at Load, literal paths containing spaces, and strict help/flag syntax. A Linux/glibc preload test rejects one allocation immediately before main and verifies logger initialization fails before loading the missing model; the application has no failure-injection option.

Run the independent CLI/direct-Session matrix after building both targets:

```bash
AETHERMIND_INFER_BINARY="$PWD/build-infer/apps/inference/aethermind_infer" TMPDIR=/tmp \
  ./build/tests/unit/aethermind_unit_tests --gtest_filter='CliMatrix/InferenceCliIntegration.*'
```

The 48 cases cover four F32 fixtures × two presets × six behaviors: three new tokens, repeated processes, exact KV capacity, zero generation beyond physical prompt capacity, included EOS, and capacity rejection. Expected results come from independently configured public Load/Compile/TryBuild/Prepare/Create/Generate calls; they do not reuse application helpers or claim an HF numerical golden. Missing `AETHERMIND_INFER_BINARY`, executable, fixtures, or Linux support fails explicitly.

M4 passed 48 CLI/Session comparisons, 271 focused C++ regressions, 38 Python CLI tests, and 20 converter tests. ASAN/UBSAN passed the same 48 comparisons and 36 ordinary Python CLI methods, excluding both external-preload cases. LSAN's zero-test and 48-test runs reported identical existing 1384-byte/17-allocation baselines. Reproduction commands, source identities, gate mapping, and limitations are in the [M4 validation report](../../../docs/tests/upper_level_runtime_assembly_m4_validation_2026-10-22.md).

M5 moved the unique admission implementation to `inference/model_execution_policy.h` and added shared `MakeFp32KVCacheOptions`. Existing admission tests consume that public header directly; the application has no forwarding implementation. Eleven additional `Fp32KVCacheOptions.*` tests cover the mapping. The [M5 report](../../../docs/tests/upper_level_runtime_assembly_m5_validation_2026-10-22.md) records its 290-test focused C++ run, CLI checks, C11 contract, and sanitizer boundaries.

M6 implemented the frozen C ABI and moved CPU compile presets into shared `inference/cpu_inference_options.h`. CLI and C adapter use the same admission, geometry mapping, and preset implementation. Run the new C gates with:

```bash
TMPDIR=/tmp ./build/tests/unit/aethermind_unit_tests \
  --gtest_filter='CAbiRuntime.*:CAbiModel.*:CAbiSession.*:CAbiGenerate.*:CAbiKVValidation.*:KvPlanes/CAbiRuntimeAllocationFailure.*:HfPresets/CAbiRuntimeIntegration.*'
```

M6 passed 100 C ABI cases (52 boundary/failure cases and 48 pure-C/direct-Session comparisons), the existing 290 focused regressions, 48 CLI comparisons, and 38 Python CLI methods. All 13 allocation-injection cases actually hit their failure points in the normal build; the remaining 87 C ABI cases passed ASAN/UBSAN. The unchanged LSAN baseline remains open. See the [M6 report](../../../docs/tests/upper_level_runtime_assembly_m6_validation_2026-10-22.md) for the independent C executable, exact commands, source hashes, and evidence limits.

## Sanitizer validation boundary

The [CTest/sanitizer second-batch validation](../../../docs/tests/ctest_sanitizer_validation_2026-10-26.md) provides the current reproducible presets. The full test executable now compiles inert interposer helpers under ASAN/TSAN; incompatible fault-injection cases are explicitly excluded from those gates. The current ASAN/UBSAN slice also passed with leak detection enabled. The historical startup conflict and removed function-registry leak baseline below describe their original snapshots and are not current failure claims.

The M2 run rebuilt `build-asan` with ASAN/UBSAN. The full unit executable exited 139 before listing tests, including outside the sandbox. An A/B link using the same three M2 objects and otherwise identical link arguments started normally without the global malloc interposer object and exited 139 with it. An isolated Linux harness uses the same runner/test sources and the instrumented core while excluding that interposer:

```bash
M2_ROOT="$PWD"
cmake -S . -B build-asan -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer'
cmake --build build-asan --target AetherMind -j
c++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -DSPDLOG_USE_STD_FORMAT -DAETHERMIND_INFERENCE_RUNNER_TESTING=1 \
  -DAETHERMIND_TEST_MODELS_DIR="\"$M2_ROOT/.models\"" \
  -Iinclude -Iapps/inference -Itests/unit -Ibuild-asan/_deps/spdlog-src/include \
  apps/inference/inference_runner.cpp \
  tests/unit/apps/inference/test_inference_runner.cpp \
  tests/unit/apps/inference/test_inference_runner_success.cpp \
  build-asan/src/libAetherMind.so build-asan/_deps/spdlog-build/libspdlog.a \
  -lgtest_main -lgtest -pthread -Wl,-rpath,"$M2_ROOT/build-asan/src" \
  -o /tmp/aethermind-m2-asan-focused

TMPDIR=/tmp ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/aethermind-m2-asan-focused
```

All 39 tests passed with no address/UB report. Leak checks require a process not traced with ptrace; the sandbox's LSAN failure is not a passed leak check. Run both the zero-test baseline and the same 39 tests with `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`. On this baseline, both exit 1 with the same 1384 bytes / 17 allocations and identical normalized stacks from `GlobalFunctionTable` static registration in `src/function.cpp:38`. No new leak was reported by M2, but the overall zero-leak gate remains open. Do not suppress that baseline or present `detect_leaks=0` as leak validation.
