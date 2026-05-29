#include <benchmark/benchmark.h>

#include "aethermind/backend/cpu/kernels/cpu_dot_product_avx2.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

void BM_DotProductAvx2Unroll(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    std::vector<float> a(n);
    std::vector<float> b(n);

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i & 0xFF) * 0.1F;
        b[i] = static_cast<float>(255 - (i & 0xFF)) * 0.1F;
    }

    for (auto _: state) {
        float result = aethermind::DotProductAvx2Unroll(a.data(), b.data(), n);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
    state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(n) * static_cast<std::int64_t>(sizeof(float) * 2));
}

BENCHMARK(BM_DotProductAvx2Unroll)
        ->Arg(128)
        ->Arg(1024)
        ->Arg(8192)
        ->Arg(65536)
        ->Arg(1 << 20)
        ->ArgName("N");

}// namespace
