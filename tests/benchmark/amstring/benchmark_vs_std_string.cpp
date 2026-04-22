#include <amstring/string.hpp>

#include <benchmark/benchmark.h>

namespace aethermind {
namespace bench {

// Benchmark comparing aethermind::string with std::string
// TODO: Implement actual benchmarks in Milestone 12
// This is a placeholder for M0 to verify benchmark framework compiles

static void BM_VsStdStringPlaceholder(benchmark::State& state) {
    for (auto _: state) {
        benchmark::DoNotOptimize(true);
    }
}

BENCHMARK(BM_VsStdStringPlaceholder);

}// namespace bench
}// namespace aethermind