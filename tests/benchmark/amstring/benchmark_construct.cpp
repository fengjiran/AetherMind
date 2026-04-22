#include <amstring/string.hpp>

#include <benchmark/benchmark.h>

namespace aethermind {
namespace bench {

// Milestone 12 benchmarks for amstring
// TODO: Implement actual benchmarks in Milestone 12
// This is a placeholder for M0 to verify benchmark framework compiles

static void BM_ConstructPlaceholder(benchmark::State& state) {
    for (auto _: state) {
        benchmark::DoNotOptimize(true);
    }
}

BENCHMARK(BM_ConstructPlaceholder);

static void BM_CopyPlaceholder(benchmark::State& state) {
    for (auto _: state) {
        benchmark::DoNotOptimize(true);
    }
}

BENCHMARK(BM_CopyPlaceholder);

static void BM_AppendPlaceholder(benchmark::State& state) {
    for (auto _: state) {
        benchmark::DoNotOptimize(true);
    }
}

BENCHMARK(BM_AppendPlaceholder);

}// namespace bench
}// namespace aethermind