// benchmark_construct.cpp - Benchmarks for string construction
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#include <benchmark/benchmark.h>
#include <string>

namespace {

void BM_StdStringDefaultConstruct(benchmark::State& state) {
    for (auto _ : state) {
        std::string s;
        benchmark::DoNotOptimize(s);
    }
}
BENCHMARK(BM_StdStringDefaultConstruct);

void BM_StdStringSmallLiteral(benchmark::State& state) {
    for (auto _ : state) {
        std::string s("hello");
        benchmark::DoNotOptimize(s);
    }
}
BENCHMARK(BM_StdStringSmallLiteral);

void BM_StdStringHeapLiteral(benchmark::State& state) {
    const std::string long_str(256, 'x');
    for (auto _ : state) {
        std::string s(long_str);
        benchmark::DoNotOptimize(s);
    }
}
BENCHMARK(BM_StdStringHeapLiteral);

}// namespace