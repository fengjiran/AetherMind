# amstring M6 validation report (2026-04-28)

## 1. Scope

This report records the current M6 validation status for `amstring`:

- differential tests against `std::basic_string`
- regression matrix execution
- sanitizer execution status
- benchmark baseline capture

M7 optimization work was not started.

## 2. Differential tests

### Added coverage

- `tests/unit/amstring/test_basic_string_differential.cpp`
- deterministic differential matrix for:
  - constructors
  - assignment
  - append
  - resize
  - reserve
  - shrink_to_fit
  - push/pop/clear
- fixed-seed random operation sequences
- multi-`CharT` coverage:
  - `char`
  - `char8_t`
  - `char16_t`
  - `char32_t`
  - `wchar_t`

### Commands

```bash
cmake --build build --target aethermind_unit_tests -j
./build/tests/unit/aethermind_unit_tests --gtest_filter='*Differential*'
./build/tests/unit/aethermind_unit_tests --gtest_filter='*BasicString*:*GenericLayoutPolicy*:*DefaultLayoutPolicy*'
```

### Results

- `*Differential*`: `25/25` passed
- full amstring-related filter: `540/540` passed

Conclusion: wrapper-level visible semantics remain aligned with `std::basic_string` for the covered M6 operation set.

## 3. Sanitizer status

### Commands

```bash
cmake -S . -B build-asan-ubsan -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer -g' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'

cmake --build build-asan-ubsan --target aethermind_unit_tests -j
./build-asan-ubsan/tests/unit/aethermind_unit_tests \
  --gtest_filter='*BasicString*:*GenericLayoutPolicy*:*DefaultLayoutPolicy*'
```

### Results

- initial sanitizer build blocker resolved by adding an out-of-line default implementation for:
  - `aethermind::MemoryReportingInfoBase::reportOutOfMemory(...)`
- ASan/UBSan test binary now builds and runs
- sanitizer run still reports **pre-existing non-amstring leaks** from old runtime/registry static initialization paths, including stacks through:
  - `src/function.cpp`
  - `src/container/string.cpp`
  - `include/object_allocator.h`
  - `include/utils/qualified_name.h`

### Current assessment

- amstring-specific differential tests run under the sanitizer binary
- M6 sanitizer **pass** is currently blocked by unrelated existing leaks outside `amstring`

## 4. Benchmark baseline

### Added coverage

- `tests/benchmark/benchmark_amstring.cpp`
- generic `aethermind::string` vs `std::string` baseline for:
  - construct
  - copy
  - append with reserve
  - append with growth
  - assign
  - shrink_to_fit

### Commands

```bash
cmake --build build --target aethermind_benchmark -j
./build/tests/benchmark/aethermind_benchmark \
  --benchmark_filter='BM_(AmString|StdString)_' \
  --benchmark_min_time=0.02s \
  --benchmark_repetitions=3 \
  --benchmark_report_aggregates_only=true \
  --benchmark_out=/tmp/amstring_m6_baseline_20260428.json \
  --benchmark_out_format=json
```

### Environment

- build: `Debug`
- CPU summary from benchmark runtime:
  - `16 X 3686.4 MHz CPU s`
- warning emitted by benchmark runtime:
  - `Library was built as DEBUG. Timings may be affected.`

### Selected baseline results

| Benchmark | am mean CPU | std mean CPU |
|---|---:|---:|
| `Construct/8` | `28.2 ns` | `30.4 ns` |
| `Construct/16` | `27.2 ns` | `48.2 ns` |
| `Copy/64` | `78.2 ns` | `59.8 ns` |
| `AppendReserved/64/32` | `253 ns` | `128 ns` |
| `AppendGrow/64/64` | `192 ns` | `124 ns` |
| `Assign/256` | `102 ns` | `33.5 ns` |
| `ShrinkToFit/256` | `234 ns` | `137 ns` |

This is a baseline capture only. No tuning or optimization conclusions are drawn at M6.

## 5. Current M6 status

Completed:

- differential test suite added
- deterministic + random-sequence regression coverage added
- narrowest relevant unit tests executed
- full amstring-related unit tests executed
- benchmark baseline captured

Blocked from full M6 exit:

- sanitizer pass is not yet clean because of pre-existing non-amstring leaks

## 6. Next step

To fully close M6, resolve or explicitly suppress the unrelated sanitizer findings in the legacy runtime/registry path, then rerun the sanitizer validation commands above.
