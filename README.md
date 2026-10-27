# AetherMind

> A desktop/server CPU local inference runtime for the Llama family of dense models, with explicitly scoped long-term evolution directions.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/)
[![CMake](https://img.shields.io/badge/CMake-%E2%89%A53.28-green.svg)](https://cmake.org/)

## Overview

AetherMind is a C++20 desktop/server CPU inference engine. Its current executable provides synchronous, single-request FP32 Llama Dense greedy generation with static KV storage and Token IDs I/O. The validated integration slice uses a pinned tiny HF checkpoint; it does not establish large-model performance or completion of every product hard gate.

GPU offloading, serving, and distributed execution are direction-only and documented in the [PRD appendix](docs/products/aethermind_prd.md); they are not current delivery commitments.

### Delivered C++ Slice

| Included | Excluded |
|----------|----------|
| CPU execution, validated on Linux/x86_64 | GPU / CUDA execution |
| Llama Dense FP32, O1/plain and explicit O2/packed | Other families, MoE, encoder-decoder |
| Greedy sampling | temperature / top-k / top-p |
| Static pre-allocated KV cache | PagedAttention, continuous batching |
| Scalar FP32 execution inputs | FP16/BF16 and INT8/INT4 end-to-end execution |
| Token IDs I/O (no tokenizer) | HTTP/gRPC serving |

The [PRD](docs/products/aethermind_prd.md) defines broader product targets, including quantized execution. HF loader acceptance of a dtype or family does not imply executable support. The [runtime C ABI v1](docs/api/runtime-c-abi.md) is implemented for the same CPU FP32 slice; its scoped validation does not establish every product or release gate.

## Key Features

- **Model loading**: HuggingFace `config.json` + `*.safetensors` I/O, validation, and raw weight resolution; graph-driven packing occurs during model preparation.
- **Graph IR**: A model-graph intermediate representation with optimization passes (constant folding, dead-code elimination, SiLU-Mul fusion).
- **Operator layer**: Reference and SIMD-optimized kernels (RMSNorm with AVX2, embedding lookup, dot product) with static dispatch.
- **Execution engine**: Synchronous Prefill/Decode executor with a static KV-cache manager and per-layer runner.
- **Validation**: GoogleTest, CLI black-box checks, scalar-oracle execution tests, and Google Benchmark targets, with explicit sanitizer baseline limits recorded in the assembly plan.

## Architecture

The application explicitly assembles the public C++ building blocks in this order:

```text
ModelLoader::Load → ModelCompiler::Compile → RuntimeBuilder::TryBuild
  → PrepareExecutableModel → InferenceSession::Create → Generate
```

Compilation precedes KV allocation. Runtime remains at a stable address from preparation onward and outlives the prepared model and session; the application returns only owning Token IDs. Source ownership and dependency rules are defined in [AGENTS.md](AGENTS.md), with the detailed [architecture overview](docs/designs/architecture/architecture_overview.md) and [assembly plan](docs/improvement-plan/10-upper-level-runtime-assembly.md) covering the contracts.

## Project Structure

```
AetherMind/
├── include/aethermind/      # Public headers
│   ├── base/                # Tensor, status, mmap, shape/stride
│   ├── dtypes/              # half, bfloat16, float8, complex
│   ├── shape_inference/     # Symbolic shape & constraint solver
│   ├── model/               # HF loader, raw weights, family graph builders
│   │   └── formats/hf/      # HuggingFace safetensors/config
│   ├── graph/               # Graph IR and backend-independent semantic passes
│   ├── compiler/            # Semantic pipeline composition and lowering artifacts
│   ├── inference/           # Model preparation, Session, Generate
│   ├── operators/           # Operator registry, schema, op_params, inference
│   ├── backend/             # Kernel selector/registry, CPU backend
│   │   └── cpu/kernels/     # RMSNorm (AVX2), embedding, dot product
│   ├── execution/           # Executor, execution plan, prepared bindings
│   ├── kv_cache/            # Static KV layout, storage, reservation, session view
│   ├── runtime/             # Runtime context, builder, workspace
│   └── memory/              # Allocator abstractions (CPU/CUDA/CANN)
├── src/                     # Implementation (mirrors include/)
├── apps/                    # Standalone application sources
│   └── inference/           # aethermind_infer CLI, synchronous runner, private helpers
├── tests/
│   ├── unit/                # GoogleTest and application test sources
│   ├── fixtures/            # Versioned fixture recipes, checksums, and instructions
│   └── benchmark/           # Google Benchmark targets
├── cmake/                   # CMake helper modules
├── 3rdparty/                # Third-party code (libbacktrace)
├── docs/                    # Architecture, design, and guide documents
│   ├── products/            # PRD
│   ├── designs/              # Architecture & module designs
│   ├── guides/               # Coding style & test writing guides
│   └── reviews/              # Design and code reviews
└── tools/                   # Development and maintenance tools
```

The optional `aethermind_infer` executable uses the `apps/inference/` runner to load, compile, build Runtime, prepare a model, and generate owning Token IDs through public C++ APIs. CLI and C ABI share FP32 admission/model-to-KV mapping in `inference/model_execution_policy.h` and compilation presets in `inference/cpu_inference_options.h`; application helpers retain request validation, CPU resource assembly, and I/O. The [assembly plan](docs/improvement-plan/10-upper-level-runtime-assembly.md) records staged validation. Application code stays outside the core shared library; custom builder injection is enabled only for the runner's test compilation.

## Requirements

- **Compiler**: A C++20-compliant compiler (GCC ≥ 11, Clang ≥ 14, or MSVC ≥ 19.30)
- **CMake**: ≥ 3.28
- **C++ runtime**: pthreads
- **GoogleTest**: system-installed (for unit tests)
- **libbacktrace** (bundled under `3rdparty/`, enabled by default)

Third-party dependencies fetched automatically via CMake `FetchContent`:

| Dependency | Version | Purpose |
|------------|---------|---------|
| [spdlog](https://github.com/gabime/spdlog) | v1.17.0 | Logging |
| [Google Benchmark](https://github.com/google/benchmark) | v1.9.5 | Fetched/configured only when `BUILD_BENCHMARKS=ON` |

## Build

```bash
# Default configuration (tests and benchmarks ON)
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Full build
cmake --build build -j

# Build a specific target (recommended for faster iteration)
cmake --build build --target AetherMind -j            # Core shared library
cmake --build build --target aethermind_unit_tests -j  # Unit tests
cmake --build build --target aethermind_benchmark -j  # Benchmarks
```

### Model Measurements

The Linux developer target `aethermind_model_benchmark` separates preparation, internal prepared Prefill/Decode, and public `Generate` measurements. The Release presets build it with `BUILD_TESTS=OFF`; the current `BUILD_BENCHMARKS` option still configures the existing Google Benchmark dependency.

```bash
cmake --preset benchmark-model
cmake --build --preset benchmark-model
python3 tools/run_model_benchmark.py \
  --binary build/validation/benchmark-model/tests/benchmark/aethermind_model_benchmark \
  --model .models/inference-numerical/mha-untied/model \
  --prompt 1,7,3 --max-new-tokens 3 --preset o1-plain \
  --warmup 1 --repetitions 20 --mode timing
```

Prepare the [numerical fixtures](tests/fixtures/inference_numerical/README.md) first. The [model measurement guide](tests/benchmark/models/README.md) defines timing boundaries, separate profiling/allocation modes, baseline builds, complete raw artifacts, report checks, and same-implementation noise observations. Raw runs go under gitignored `benchmark-results/models/`; identity collection reads model files before execution, so loading times are not cold-cache startup measurements. Performance capture is explicit and does not run in the default correctness presets.

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | `ON` | Build unit test targets |
| `BUILD_BENCHMARKS` | `ON` | Build benchmark targets |
| `BUILD_INFERENCE_TOOL` | `OFF` | Build the standalone `aethermind_infer` CLI |
| `AETHERMIND_ENABLE_AVX2` | `ON` | Build optional x86 AVX2/FMA implementations when the compiler supports them |
| `USE_LIBBACKTRACE` | `ON` | Enable libbacktrace for stack traces |
| `BACKTRACE_ON_SEGFAULT` | `ON` | Install a SIGSEGV traceback handler |
| `AETHERMIND_SANITIZER` | `none` | Project-target instrumentation: `none`, `asan-ubsan`, or `tsan` |
| `AETHERMIND_VALIDATION_REQUIRED_ASSETS` | `OFF` | Require checked model fixtures and a CLI for the validation gates |
| `AETHERMIND_TEST_MODELS_DIR` | source `.models` | Absolute root of externally prepared test models; tests only |
| `AETHERMIND_TEST_NUMERICAL_DIR` | test-model root `/inference-numerical` | Absolute root of the checked offline HF numerical bundle; tests only |
| `AETHERMIND_INFER_BINARY` | empty | Absolute external CLI for CTest; an in-build CLI takes precedence |
| `USE_ALLOC_ALIGNMENT` | `ON` | Apply `KALLOC_ALIGNMENT` to allocations |
| `KALLOC_ALIGNMENT` | `64` | Canonical positive decimal power of two, representable as target `size_t`; checked when alignment is enabled |

The `AetherMind` CMake target propagates its C++20 requirement to C++ consumers.
When configuring spdlog itself, the project defaults `SPDLOG_USE_STD_FORMAT` to
`ON`, unless external fmt was selected. Explicit settings and an existing
`spdlog::spdlog` target supplied by a parent project are respected. The format
backend and its required definitions come from the spdlog target.

Older build directories may cache `SPDLOG_USE_STD_FORMAT=OFF`. This now selects
fmt consistently instead of being overridden by a directory-wide macro. To
retain the previous effective `std::format` backend, reconfigure with
`-DSPDLOG_USE_STD_FORMAT=ON`. Rebuild the affected targets after reconfiguration.

With `USE_ALLOC_ALIGNMENT=ON`, invalid values such as `0`, `24`, `08`, negative
numbers, or target-size overflow fail during configuration. Byte alignment `1`
is valid; the CPU allocator raises it to the platform's pointer-alignment floor.
Large valid alignments can still fail at allocation time. With alignment disabled,
`KALLOC_ALIGNMENT` is ignored.

## Run Inference

The CLI supports CPU, FP32 Llama Dense, synchronous greedy generation, and Token IDs I/O. Supply a local supported HF directory; model files are not included in the repository. The [fixture instructions](tests/fixtures/inference_admission/README.md) describe the pinned tiny FP32 example and test assets.

```bash
cmake -S . -B build-infer -DBUILD_INFERENCE_TOOL=ON \
  -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-infer --target aethermind_infer -j
./build-infer/apps/inference/aethermind_infer \
  --model .models/tiny-random-LlamaForCausalLM --tokens 7 \
  --context-tokens 8 --max-new-tokens 3
./build-infer/apps/inference/aethermind_infer --help
```

`--model`, `--tokens`, `--context-tokens`, and `--max-new-tokens` are required. Use separate flag/value arguments; unknown or duplicate flags, signed/whitespace-containing integers, empty token items, trailing garbage, and overflow are rejected. Quote paths containing spaces. Optional `--eos-token-id` sets an explicit stopping token; no EOS is inferred from model files. `--preset` selects `o1-plain` (default) or `o2-packed`; each is validated separately, with no silent fallback or performance promise.

Successful generation writes one JSON integer array and one final newline to stdout, containing only new tokens and including generated EOS. Diagnostics and library logs go to stderr. A standalone `--help` prints usage instead of JSON. Exit codes are 0 for success/help, 2 for argument/request errors, and 1 for loading, unsupported model/precision, compilation, resource, preparation, or execution failures. `--max-new-tokens 0` returns `[]` after input/model validation; capacity remains an explicit positive value bounded by the model context.

The tool does not depend on GTest or benchmark; spdlog remains a formal dependency. This build configuration does not promise offline dependency acquisition, installation/export, tokenizer support, or quantized execution.

## C ABI

`include/c_api.h` exposes the implemented v1 runtime/model/session API. Supply explicit positive KV geometry and capacity when creating Runtime, then load a model, create a session, and generate. Model loading validates the existing Runtime against all compiled K/V states before preparation. Models retain Runtime; sessions retain their model and Runtime, so external owner handles may be released first. Calls sharing a Runtime require external serialization.

Generation writes only complete successful results to a caller-owned buffer. Reserve at least `max_new_tokens` elements; a short buffer returns `AM_STATUS_OUT_OF_RANGE` and that upper bound without executing. Errors are optional independently owned diagnostics. Full null, version, error/OOM, and zero-generation rules are in the [C ABI contract](docs/api/runtime-c-abi.md).

The pinned-fixture smoke client can be compiled and linked with the C compiler, without GTest. Its geometry comes from the M0 recipe; it releases external Runtime/model handles before repeated generation:

```bash
cmake --build build --target AetherMind -j
cc -std=c11 -Iinclude -Itests/unit/c_api \
  tests/unit/c_api/test_c_api_runtime_consumer.c \
  tests/unit/c_api/test_c_api_runtime_smoke.c \
  -L"$PWD/build/src" -Wl,-rpath,"$PWD/build/src" -lAetherMind \
  -o /tmp/aethermind_c_api_smoke
/tmp/aethermind_c_api_smoke .models/inference-admission/f32-explicit o1-plain
/tmp/aethermind_c_api_smoke .models/inference-admission/f32-explicit o2-packed
```

## Testing

Tests use [GoogleTest](https://google.github.io/googletest/) and live under `tests/unit/`.

CTest discovers the existing GoogleTest cases and runs the Python fixture/CLI checks. After preparing the fixtures below, use the same configure, build, and test presets locally and in automation:

```bash
cmake --preset validation-release
cmake --build --preset validation-release
ctest --preset validation-release
```

This preset runs a bounded `validation` slice. Additional labels include `smoke`, `numerical`, `lifecycle`, `context_boundary`, `thread-smoke`, `model`, `cli`, `interposer`, and `isa-avx2`. For example, `ctest --test-dir build/validation/validation-release -L numerical --output-on-failure` selects numerical cases. Context-capacity tests do not establish long-context coverage.

The validation presets require checked model assets and build their own CLI. Missing assets fail the fixture gate. Ordinary test configurations explicitly disable cases whose model assets or CLI are unavailable; CTest reports them as not run. Sanitizer builds disable incompatible interposer/preload cases, which remain covered in the ordinary Linux/glibc build. Required validation cases fail if they attempt to skip. Configure and build before testing; a missing C++ test executable cannot pass the validation entrypoint.

New cases that use the malloc counting/failure helpers must update the explicit `interposer` classification in `cmake/AetherMindTests.cmake` and its classification contract.

The inference admission fixture tests require generated HF variants. Follow the [fixture preparation and validation instructions](tests/fixtures/inference_admission/README.md) before running them; missing fixtures fail the tests rather than being skipped. The generator and its Python tests use only the standard library.

The [independent HF numerical reference instructions](tests/fixtures/inference_numerical/README.md) describe pinned CPU FP32 golden generation, asset verification, and candidate review. Prepare that bundle before the numerical model tests. Only offline reference generation needs PyTorch/transformers; the checker uses standard-library Python and the C++ tests read serialized tensors. Run `ctest --preset validation-numerical` after configuring and building `validation-avx2` to include fixed Prefill/Decode, independent greedy Token IDs, and near-context-limit coverage for the listed MHA/GQA/MQA fixtures. Set the absolute `AETHERMIND_TEST_NUMERICAL_DIR` cache path to test an externally prepared bundle.

The Linux CLI/direct-Session integration suite additionally requires the standalone executable from Run Inference above. Set `AETHERMIND_INFER_BINARY` to its absolute path; missing executables or unsupported platforms fail this integration gate explicitly. This variable is consumed by tests only. The Python CLI tests also require the existing C toolchain for Linux/glibc failure injection. Commands and scoped results are recorded in the [M4 validation report](docs/tests/upper_level_runtime_assembly_m4_validation_2026-10-22.md).

```bash
# Run all unit tests after preparing fixtures and building the standalone CLI
AETHERMIND_INFER_BINARY="$PWD/build-infer/apps/inference/aethermind_infer" \
  ./build/tests/unit/aethermind_unit_tests

# Run only the CLI/direct-Session integration matrix
AETHERMIND_INFER_BINARY="$PWD/build-infer/apps/inference/aethermind_infer" \
  ./build/tests/unit/aethermind_unit_tests --gtest_filter='CliMatrix/InferenceCliIntegration.*'

# Run Python CLI contracts
python3 tests/unit/apps/inference/test_inference_cli.py \
  --infer "$PWD/build-infer/apps/inference/aethermind_infer" -v

# Run a single test suite
./build/tests/unit/aethermind_unit_tests --gtest_filter=TensorRandomNew.*

# Run a single test case (recommended quick loop)
./build/tests/unit/aethermind_unit_tests --gtest_filter=TensorRandomNew.UniformTensorDeterministicWithSameSeed

# Repeat a test for flakiness checks
./build/tests/unit/aethermind_unit_tests --gtest_filter=Device.* --gtest_repeat=10
```

### CPU ISA Validation

Optional AVX2/FMA implementations are compiled in isolated translation units. Test entry points, dispatch, and ordinary library/application code use baseline compiler options. `AETHERMIND_ENABLE_AVX2=OFF` excludes the optional implementations and their availability macros.

```bash
cmake --preset validation-baseline
cmake --build --preset validation-baseline
ctest --preset validation-baseline

cmake --preset validation-avx2
cmake --build --preset validation-avx2
ctest --preset validation-avx2
```

The baseline preset runs the production/lifecycle slice with reference and packed-identity kernels. The AVX2 preset requires both compiled candidates and OS-usable AVX2/FMA hardware, then runs the direct candidate tests. Missing binaries, empty ISA discovery, unavailable hardware, and required skips cannot pass this gate. Pure metadata and backend-policy tests remain runnable without AVX2.

`fixture.cpu_isa_build` checks `compile_commands.json` for target-wide ISA uplift and per-file isolation; ISA implementation files also stay outside Unity and LTO merging. Update the explicit source manifest and its negative tests when adding an ISA implementation. Backend feature policy controls descriptor eligibility; internal dot-product dispatch independently detects host features. The baseline build excludes those optional implementations at compile time. Physical non-AVX hardware execution remains a separate platform qualification.

### Sanitizer Builds

```bash
# AddressSanitizer + UndefinedBehaviorSanitizer, including leak detection
cmake --preset validation-asan-ubsan
cmake --build --preset validation-asan-ubsan
ctest --preset validation-asan-ubsan

# Separate ThreadSanitizer build
cmake --preset validation-tsan
cmake --build --preset validation-tsan
ctest --preset validation-tsan
```

Sanitizer flags apply to AetherMind's targets, including the C++ tests and CLI, and do not propagate to third-party targets. ASAN/UBSAN and TSAN are mutually exclusive. The presets save JUnit results in their build directories. Passing a TSAN slice of the synchronous engine and thread primitives does not establish concurrent Session support.

On Linux hosts where TSAN exits before `main` with `unexpected memory mapping`, a supported process-local workaround is `setarch "$(uname -m)" -R ctest --preset validation-tsan`. This requires permission to change the process personality; the command does not change system-wide ASLR settings. Report startup failures separately from executed test results.

## Benchmarks

```bash
# Run all benchmarks
./build/tests/benchmark/aethermind_benchmark

# Filter a specific benchmark
./build/tests/benchmark/aethermind_benchmark --benchmark_filter=BM_Malloc_Churn
```

## Code Style

Code formatting is governed by [`.clang-format`](.clang-format) (LLVM-based, 4-space indent, `ColumnLimit: 0`).

```bash
# Check formatting of changed C/C++ files in include/, src/, apps/, and tests/ (staged + unstaged + untracked)
{
  git diff --cached --name-only --diff-filter=d
  git diff --name-only --diff-filter=d
  git ls-files --others --exclude-standard
} | grep -E '\.(c|cc|cpp|cxx|h|hpp|hxx)$' | grep -E '^(include/|src/|apps/|tests/)' | sort -u | xargs -r clang-format -n --Werror

# Auto-format
{
  git diff --cached --name-only --diff-filter=d
  git diff --name-only --diff-filter=d
  git ls-files --others --exclude-standard
} | grep -E '\.(c|cc|cpp|cxx|h|hpp|hxx)$' | grep -E '^(include/|src/|apps/|tests/)' | sort -u | xargs -r clang-format -i
```

If `build/compile_commands.json` is available, `clang-tidy` can be run against it.

## Contributing

AetherMind follows a disciplined development workflow. Before contributing:

1. Read [AGENTS.md](AGENTS.md) — the repository-wide engineering guide covering build, test, coding style, and review conventions.
2. Read the [PRD](docs/products/aethermind_prd.md) for the current product scope and acceptance criteria.
3. Make minimal, style-consistent changes; build the narrowest affected target first.
4. Run focused tests (`--gtest_filter=Suite.Case`) before widening scope.
5. For performance-sensitive changes, run the focused benchmark target.
6. Format changed files with `clang-format` before submitting.

Key guidelines:

- **C++20** with RAII, const-correctness, explicit ownership, and no raw owning pointers.
- Prefer early returns and small, focused functions; avoid hidden O(N²) in hot paths.
- Tests use PascalCase names; death-test function names end with `Death`; seed RNGs explicitly; use `EXPECT_FLOAT_EQ` for floats. See the [test writing guide](docs/guides/test_writing_guidelines.md).
- Graph node retrieval APIs return `std::vector<GraphNodeId>` with a `Find` prefix; ID-based access is preferred over raw pointers.

## Execution Binding API

`BoundValue` represents an unbound, read-only, or writable tensor binding. Use `ReadView()` for a read-only descriptor returned by value and `TryGetWritableView()` for a borrowed writable descriptor. These replace the former `readable`, `writable`, and `has_writable` fields; C++ consumers must migrate and rebuild. See the [public API reference](docs/api/public-api.md) for backing-storage lifetimes and empty-state behavior.

## Alignment API

Use `IsValidAlignment(alignment)` and `AlignOffsetUp(offset, alignment)` from `aethermind/base/alignment.h` for alignment validation and checked offset alignment. A valid alignment is a nonzero power of two. `AlignOffsetUp` reports invalid alignment, overflow, and diagnostic allocation failure through `StatusOr<size_t>`. The former `IsValidWorkspaceAlignment()` and `AlignWorkspaceOffset()` APIs have been removed without forwarding wrappers; update callers and rebuild. See the [public API reference](docs/api/public-api.md).

## Documentation

| Document | Description |
|----------|-------------|
| [AGENTS.md](AGENTS.md) | Repository-wide engineering guide |
| [docs/products/aethermind_prd.md](docs/products/aethermind_prd.md) | Current product requirements and acceptance criteria |
| [docs/guides/cpp_coding_style_guidelines.md](docs/guides/cpp_coding_style_guidelines.md) | C++ coding style |
| [docs/guides/cpp_comment_guidelines.md](docs/guides/cpp_comment_guidelines.md) | Comment guidelines |
| [docs/guides/test_writing_guidelines.md](docs/guides/test_writing_guidelines.md) | Test writing guidelines |
| [docs/designs/](docs/designs/) | Architecture and module design documents |

## Product Scope and Evolution

- **Current product contract**: CPU-only Llama-family inference runtime with INT8/INT4 quantization, static KV cache, and synchronous greedy generation.
- **Long-term directions, not committed**: GPU offloading, continuous batching, HTTP API, PagedAttention, and speculative decoding. See the [PRD appendix](docs/products/aethermind_prd.md).

## FAQ

**Q: Does AetherMind include a tokenizer?**
No. The current product contract uses token IDs as the data boundary. Tokenizer integration is outside the current scope.

**Q: Is GPU inference supported?**
No. GPU/CUDA backends are a long-term direction, not a current product commitment.

**Q: Which models are supported?**
Llama-family dense models loaded from HuggingFace format (`config.json` + `*.safetensors`). MoE, encoder-decoder, and sliding-window attention models are explicitly rejected.

**Q: What sampling strategies are supported?**
Greedy sampling only (`argmax(logits)`). Temperature / top-k / top-p are outside the current product scope.

**Q: How is determinism guaranteed?**
Reference CPU kernels provide numerical stability and deterministic output for the same platform and kernel precision. Bit-identical cross-platform output is a target of the reference kernels only.

**Q: Where is the C ABI?**
The C ABI is a current product target under design (draft v1.0). The existing `include/c_api.h` exposes object lifecycle, error handling, and traceback primitives. The full `am_session_generate` contract is defined in the PRD and will be frozen before the v1.0 release.

## License

AetherMind is licensed under the [Apache License 2.0](LICENSE).
