# AetherMind

> A phased LLM inference engine — Phase 1 delivers a desktop/server CPU local inference runtime for the Llama family of dense models.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/)
[![CMake](https://img.shields.io/badge/CMake-%E2%89%A53.28-green.svg)](https://cmake.org/)

## Overview

AetherMind is an inference engine built in phases. The current effort focuses on **Phase 1**: a self-contained, production-grade CPU runtime that loads HuggingFace-format Llama-family dense models and performs greedy token generation with a statically pre-allocated KV cache. The runtime is single-process, synchronous, and token-ID based (no tokenizer in Phase 1).

Future phases (GPU offloading, serving, distributed execution) are direction-only and documented in the [PRD appendix](docs/products/aethermind_prd.md).

### Phase 1 Scope

| Included | Excluded |
|----------|----------|
| CPU backend (x86_64 AVX2/FMA, ARM64 NEON planned) | GPU / CUDA backend |
| Llama-family dense models | MoE, encoder-decoder |
| Greedy sampling | temperature / top-k / top-p |
| Static pre-allocated KV cache | PagedAttention, continuous batching |
| INT8/INT4 weight-only quantization | FP8 |
| Token IDs I/O (no tokenizer) | HTTP/gRPC serving |

## Key Features

- **Model loading**: HuggingFace `config.json` + `*.safetensors` loader with weight repacking into CPU-optimized layouts.
- **Graph IR**: A model-graph intermediate representation with optimization passes (constant folding, dead-code elimination, SiLU-Mul fusion).
- **Operator layer**: Reference and SIMD-optimized kernels (RMSNorm with AVX2, embedding lookup, dot product) with static dispatch.
- **Execution engine**: Synchronous Prefill/Decode executor with a static KV-cache manager and per-layer runner.
- **`ammalloc` allocator**: A high-concurrency user-space memory allocator (ThreadCache → CentralCache → PageCache → PageAllocator) for nanosecond-latency allocations.
- **Engineering rigor**: TSAN/ASAN-friendly, deterministic CPU reference kernels, GoogleTest unit suite, and Google Benchmark targets.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      API Layer                                   │
│   C ABI (draft v1.0)        C++ Headers (include/aethermind/)    │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                    Runtime Layer                                 │
│  Runtime · Executor (Prefill/Decode) · KV Cache Manager · Loader │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                    Operator Layer                                 │
│  Graph IR · Operators · Kernels (Reference + SIMD) · Dispatch     │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                HAL — CPU Backend (x86_64 / ARM64)                 │
│  ammalloc · Thread Pool · CPU Feature Detection                  │
└─────────────────────────────────────────────────────────────────┘
```

A detailed architecture diagram and design rationale are in the [PRD](docs/products/aethermind_prd.md).

## Project Structure

```
AetherMind/
├── include/aethermind/      # Public headers
│   ├── base/                # Tensor, status, mmap, shape/stride
│   ├── dtypes/              # half, bfloat16, float8, complex
│   ├── shape_inference/     # Symbolic shape & constraint solver
│   ├── model/               # Model loader, instance, weights
│   │   ├── graph/           # Graph IR, passes, const evaluator
│   │   └── formats/hf/      # HuggingFace safetensors/config
│   ├── operators/           # Operator registry & definitions
│   ├── backend/             # Kernel selector/registry, CPU backend
│   │   └── cpu/kernels/     # RMSNorm (AVX2), embedding, dot product
│   ├── execution/           # Executor, execution plan, KV cache
│   ├── runtime/             # Runtime context, builder, workspace
│   └── memory/              # Allocator abstractions (CPU/CUDA/CANN)
├── src/                     # Implementation (mirrors include/)
├── ammalloc/                # High-performance memory allocator subsystem
│   ├── include/ammalloc/    # ThreadCache, CentralCache, PageCache
│   └── src/                 # Allocator implementation
├── tests/
│   ├── unit/                # GoogleTest unit tests
│   └── benchmark/           # Google Benchmark targets
├── cmake/                   # CMake helper modules
├── 3rdparty/                # Third-party code (libbacktrace)
├── docs/                    # Architecture, design, and guide documents
│   ├── products/            # PRD
│   ├── designs/              # Architecture & module designs
│   ├── guides/               # Coding style & test writing guides
│   └── reviews/              # Design and code reviews
└── tools/                   # Auxiliary tools
```

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
| [Google Benchmark](https://github.com/google/benchmark) | v1.9.5 | Performance benchmarks |

## Build

```bash
# Default configuration (tests and benchmarks ON)
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Full build
cmake --build build -j

# Build a specific target (recommended for faster iteration)
cmake --build build --target AetherMind -j            # Core shared library
cmake --build build --target ammalloc -j               # Allocator static library
cmake --build build --target aethermind_unit_tests -j  # Unit tests
cmake --build build --target aethermind_benchmark -j  # Benchmarks
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | `ON` | Build unit test targets |
| `BUILD_BENCHMARKS` | `ON` | Build benchmark targets |
| `USE_LIBBACKTRACE` | `ON` | Enable libbacktrace for stack traces |
| `BACKTRACE_ON_SEGFAULT` | `ON` | Install a SIGSEGV traceback handler |
| `ENABLE_TSAN` | `OFF` | Enable ThreadSanitizer |
| `USE_ALLOC_ALIGNMENT` | `ON` | Apply `KALLOC_ALIGNMENT` to allocations |
| `KALLOC_ALIGNMENT` | `64` | Memory alignment in bytes |

## Testing

Tests use [GoogleTest](https://google.github.io/googletest/) and live under `tests/unit/`.

```bash
# Run all unit tests
./build/tests/unit/aethermind_unit_tests

# Run a single test suite
./build/tests/unit/aethermind_unit_tests --gtest_filter=TensorRandomNew.*

# Run a single test case (recommended quick loop)
./build/tests/unit/aethermind_unit_tests --gtest_filter=TensorRandomNew.UniformTensorDeterministicWithSameSeed

# Repeat a test for flakiness checks
./build/tests/unit/aethermind_unit_tests --gtest_filter=Device.* --gtest_repeat=10
```

### Sanitizer Builds

```bash
# ThreadSanitizer variant
cmake -S . -B build-tsan -DENABLE_TSAN=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF
cmake --build build-tsan --target aethermind_unit_tests -j
./build-tsan/tests/unit/aethermind_unit_tests
```

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
# Check formatting of changed files (staged + unstaged + untracked)
{
  git diff --cached --name-only --diff-filter=d
  git diff --name-only --diff-filter=d
  git ls-files --others --exclude-standard
} | grep -E '\.(c|cc|cpp|cxx|h|hpp|hxx)$' | grep -E '^(include/|src/|ammalloc/|tests/)' | sort -u | xargs -r clang-format -n --Werror

# Auto-format
{
  git diff --cached --name-only --diff-filter=d
  git diff --name-only --diff-filter=d
  git ls-files --others --exclude-standard
} | grep -E '\.(c|cc|cpp|cxx|h|hpp|hxx)$' | grep -E '^(include/|src/|ammalloc/|tests/)' | sort -u | xargs -r clang-format -i
```

If `build/compile_commands.json` is available, `clang-tidy` can be run against it.

## Contributing

AetherMind follows a disciplined development workflow. Before contributing:

1. Read [AGENTS.md](AGENTS.md) — the repository-wide engineering guide covering build, test, coding style, and review conventions.
2. Read the [PRD](docs/products/aethermind_prd.md) for Phase 1 scope and acceptance criteria.
3. For allocator work, also read [ammalloc/AGENTS.md](ammalloc/AGENTS.md) for module-specific constraints.
4. Make minimal, style-consistent changes; build the narrowest affected target first.
5. Run focused tests (`--gtest_filter=Suite.Case`) before widening scope.
6. For performance-sensitive changes, run the focused benchmark target.
7. Format changed files with `clang-format` before submitting.

Key guidelines:

- **C++20** with RAII, const-correctness, explicit ownership, and no raw owning pointers.
- Prefer early returns and small, focused functions; avoid hidden O(N²) in hot paths.
- Tests use PascalCase names; death-test function names end with `Death`; seed RNGs explicitly; use `EXPECT_FLOAT_EQ` for floats. See the [test writing guide](docs/guides/test_writing_guidelines.md).
- Graph node retrieval APIs return `std::vector<GraphNodeId>` with a `Find` prefix; ID-based access is preferred over raw pointers.

## Documentation

| Document | Description |
|----------|-------------|
| [AGENTS.md](AGENTS.md) | Repository-wide engineering guide |
| [docs/products/aethermind_prd.md](docs/products/aethermind_prd.md) | Phase 1 product requirements & acceptance criteria |
| [docs/guides/cpp_coding_style_guidelines.md](docs/guides/cpp_coding_style_guidelines.md) | C++ coding style |
| [docs/guides/cpp_comment_guidelines.md](docs/guides/cpp_comment_guidelines.md) | Comment guidelines |
| [docs/guides/test_writing_guidelines.md](docs/guides/test_writing_guidelines.md) | Test writing guidelines |
| [docs/designs/](docs/designs/) | Architecture and module design documents |

## Roadmap

- **Phase 1 (current)**: CPU-only Llama-family inference runtime with INT8/INT4 quantization, static KV cache, and synchronous greedy generation.
- **Phase 2+ (direction only, not committed)**: GPU offloading, continuous batching, HTTP API, PagedAttention, speculative decoding. See the [PRD appendix](docs/products/aethermind_prd.md) for the long-term direction.

## FAQ

**Q: Does AetherMind include a tokenizer?**
No. Phase 1 uses token IDs as the data boundary. Tokenizer integration is deferred to a later phase.

**Q: Is GPU inference supported?**
Not in Phase 1. GPU/CUDA backends are a Phase 2 direction only.

**Q: Which models are supported?**
Llama-family dense models loaded from HuggingFace format (`config.json` + `*.safetensors`). MoE, encoder-decoder, and sliding-window attention models are explicitly rejected.

**Q: What sampling strategies are supported?**
Greedy sampling only (`argmax(logits)`). Temperature / top-k / top-p are out of Phase 1 scope.

**Q: How is determinism guaranteed?**
Reference CPU kernels provide numerical stability and deterministic output for the same platform and kernel precision. Bit-identical cross-platform output is a target of the reference kernels only.

**Q: Where is the C ABI?**
The C ABI is a Phase 1 target currently under design (draft v1.0). The existing `include/c_api.h` exposes object lifecycle, error handling, and traceback primitives. The full `am_session_generate` contract is defined in the PRD and will be frozen before the v1.0 release.

## License

AetherMind is licensed under the [Apache License 2.0](LICENSE).
