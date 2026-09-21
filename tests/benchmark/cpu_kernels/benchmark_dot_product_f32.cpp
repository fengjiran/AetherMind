#include <benchmark/benchmark.h>

#include "aethermind/backend/cpu/kernels/common/dot_product.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

std::vector<float> MakeValues(std::size_t n, std::size_t seed) {
    std::vector<float> values(n);
    for (std::size_t i = 0; i < n; ++i) {
        values[i] = static_cast<float>((i + seed) & 0xFF) * 0.1F;
    }
    return values;
}

void BM_CPUKernel_DotProductF32(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    std::vector<float> a = MakeValues(n, 0);
    std::vector<float> b = MakeValues(n, 255);

    for (auto _: state) {
        float result = aethermind::cpu::DotProductF32(a.data(), b.data(), n);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
    state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(n) * static_cast<std::int64_t>(sizeof(float) * 2));
}

void BM_CPUKernel_DotProductF32MultiTarget(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto num_targets = static_cast<std::size_t>(state.range(1));
    std::vector<float> lhs = MakeValues(n, 0);
    std::vector<float> rhs(num_targets * n);
    for (std::size_t t = 0; t < num_targets; ++t) {
        for (std::size_t k = 0; k < n; ++k) {
            rhs[t * n + k] = static_cast<float>((k + t + 7) & 0xFF) * 0.1F;
        }
    }
    std::vector<float> out(num_targets);

    for (auto _: state) {
        aethermind::cpu::DotProductF32MultiTarget(lhs.data(), rhs.data(), n, num_targets, n, out.data());
        benchmark::DoNotOptimize(out);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n) * static_cast<std::int64_t>(num_targets));
    state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(n) * static_cast<std::int64_t>(num_targets) * static_cast<std::int64_t>(sizeof(float) * 2));
}

BENCHMARK(BM_CPUKernel_DotProductF32)
        ->Arg(4)
        ->Arg(12)
        ->Arg(20)
        ->Arg(36)
        ->Arg(68)
        ->Arg(128)
        ->Arg(1024)
        ->Arg(8192)
        ->Arg(65536)
        ->Arg(1 << 20)
        ->ArgName("N");

// LLM decode projection shapes: K x vocab-like target counts, plus non-block
// target counts (7) covering the leftover-target path.
BENCHMARK(BM_CPUKernel_DotProductF32MultiTarget)
        ->Args({32, 1})
        ->Args({32, 4})
        ->Args({32, 8})
        ->Args({4096, 1})
        ->Args({4096, 4})
        ->Args({4096, 7})
        ->Args({4096, 8})
        ->Args({11008, 4})
        ->Args({32000, 4})
        ->ArgNames({"N", "Targets"});

} // namespace