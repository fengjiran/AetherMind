// benchmark_vs_std.cpp - Benchmarks comparing amstring vs std::string
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#include <benchmark/benchmark.h>
#include <string>

namespace {

void BM_StdStringAppendSmall(benchmark::State& state) {
    for (auto _ : state) {
        std::string s;
        s.reserve(64);
        for (int i = 0; i < 10; ++i) {
            s.append("abc");
        }
        benchmark::DoNotOptimize(s);
    }
}
BENCHMARK(BM_StdStringAppendSmall);

void BM_StdStringCopySmall(benchmark::State& state) {
    std::string src("hello world");
    for (auto _ : state) {
        std::string dst(src);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_StdStringCopySmall);

}// namespace