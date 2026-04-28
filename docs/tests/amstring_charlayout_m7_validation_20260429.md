# amstring CharLayoutPolicy M7 validation report (2026-04-29)

## 1. Scope

This report records the first validation pass after switching `DefaultLayoutPolicy<char>` to `CharLayoutPolicy`:

- `BasicString<char>` differential tests against `std::basic_string<char>`
- `amstring` benchmark baseline with the CharLayoutPolicy selector path
- comparison against the M6 GenericLayoutPolicy baseline recorded in `docs/tests/amstring_m6_validation_20260428.md`

## 2. Differential test

### Command

```bash
./build/tests/unit/aethermind_unit_tests --gtest_filter='BasicStringDifferentialTest/0.*'
```

### Result

- `BasicStringDifferentialTest/0.*`: `5/5` passed

Conclusion: `BasicString<char>` remains aligned with `std::basic_string<char>` for the current differential coverage after selector switch.

## 3. Benchmark command

The default `build/` directory currently has `BUILD_BENCHMARKS=OFF`, so benchmark validation used a dedicated benchmark build directory.

```bash
cmake -S . -B build-benchmark -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=ON
cmake --build build-benchmark --target aethermind_benchmark -j
./build-benchmark/tests/benchmark/aethermind_benchmark \
  --benchmark_filter='BM_(AmString|StdString)_' \
  --benchmark_min_time=0.02s \
  --benchmark_repetitions=3 \
  --benchmark_report_aggregates_only=true \
  --benchmark_out=/tmp/amstring_charlayout_baseline_20260429.json \
  --benchmark_out_format=json
```

## 4. Environment notes

- Build type: Debug
- Benchmark warning: `Library was built as DEBUG. Timings may be affected.`
- Current run CPU summary: `20 X 2918.4 MHz CPU s`
- M6 baseline CPU summary: `16 X 3686.4 MHz CPU s`

Because the CPU/runtime environment differs from M6, the comparison below should be treated as a directional first-pass signal, not a strict performance regression gate.

## 5. Selected benchmark comparison

CPU mean values are shown in nanoseconds.

| Benchmark | M6 Generic baseline | CharLayoutPolicy run | Delta |
|---|---:|---:|---:|
| `Construct/8` | `28.2 ns` | `34.9 ns` | `+23.9%` |
| `Construct/16` | `27.2 ns` | `34.8 ns` | `+27.8%` |
| `Copy/64` | `78.2 ns` | `80.7 ns` | `+3.2%` |
| `AppendReserved/64/32` | `253 ns` | `219 ns` | `-13.5%` |
| `AppendGrow/64/64` | `192 ns` | `201 ns` | `+4.9%` |
| `Assign/256` | `102 ns` | `118 ns` | `+15.3%` |
| `ShrinkToFit/256` | `234 ns` | `250 ns` | `+6.6%` |

## 6. Current assessment

Completed:

- differential test for `BasicString<char>` passed
- CharLayoutPolicy benchmark run completed
- benchmark JSON captured at `/tmp/amstring_charlayout_baseline_20260429.json`

Initial performance observation:

- `AppendReserved/64/32` improved in this run.
- small construction and assign paths are slower than the M6 baseline in this Debug/environment-mismatched run.
- No tuning conclusion should be drawn until a same-environment Release benchmark is captured.

## 7. Recommended next step

Run a same-environment Release benchmark for both GenericLayoutPolicy and CharLayoutPolicy, or add a compile-time selector toggle so both layouts can be benchmarked back-to-back in one binary.
