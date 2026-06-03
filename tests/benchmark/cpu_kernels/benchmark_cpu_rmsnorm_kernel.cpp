#include "aethermind/backend/cpu/kernels/rmsnorm/cpu_rmsnorm_kernel.h"
#include "aethermind/backend/cpu/kernels/rmsnorm/cpu_rmsnorm_kernel_reference.h"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <vector>

namespace {

constexpr double kRmsNormFlopsPerElement = 4.0;

void SetRmsNormThroughputCounters(benchmark::State& state, std::int64_t seq_len, std::int64_t hidden) {
    const auto elements = state.iterations() * (seq_len * hidden);
    state.SetItemsProcessed(elements);
    state.SetBytesProcessed(elements * static_cast<std::int64_t>(sizeof(float) * 4));

    const double flops = static_cast<double>(elements) * kRmsNormFlopsPerElement;
    state.counters["GFLOP/s"] = benchmark::Counter(flops, benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
}

void BM_CPUKernel_RmsNorm(benchmark::State& state) {
    const auto seq_len = state.range(0);
    const auto hidden = state.range(1);
    const auto numel = static_cast<std::size_t>(seq_len * hidden);

    std::vector<float> input(numel);
    std::vector<float> weight(static_cast<std::size_t>(hidden));
    std::vector<float> output(numel);

    for (std::size_t i = 0; i < numel; ++i) {
        input[i] = static_cast<float>((i & 0xFF) + 1) * 0.01F;
    }
    for (std::int64_t i = 0; i < hidden; ++i) {
        weight[static_cast<std::size_t>(i)] = 1.0F;
    }

    const aethermind::CpuRmsNormKernelArgs args{
            .input_ = input.data(),
            .weight_ = weight.data(),
            .output_ = output.data(),
            .seq_len_ = seq_len,
            .hidden_size_ = hidden,
            .input_row_stride_ = hidden,
            .input_col_stride_ = 1,
            .weight_stride_ = 1,
            .output_row_stride_ = hidden,
            .output_col_stride_ = 1,
            .epsilon_ = 1.0e-5F,
    };

    for (auto _: state) {
        bool ok = aethermind::CpuRmsNormKernel(args).ok();
        benchmark::DoNotOptimize(ok);
        benchmark::DoNotOptimize(output.data());
    }

    SetRmsNormThroughputCounters(state, seq_len, hidden);
}

void BM_CPUKernel_ReferenceRmsNorm(benchmark::State& state) {
    const auto seq_len = state.range(0);
    const auto hidden = state.range(1);
    const auto numel = static_cast<std::size_t>(seq_len * hidden);

    std::vector<float> input(numel);
    std::vector<float> weight(static_cast<std::size_t>(hidden));
    std::vector<float> output(numel);

    for (std::size_t i = 0; i < numel; ++i) {
        input[i] = static_cast<float>((i & 0xFF) + 1) * 0.01F;
    }
    for (std::int64_t i = 0; i < hidden; ++i) {
        weight[static_cast<std::size_t>(i)] = 1.0F;
    }

    for (auto _: state) {
        constexpr float kEpsilon = 1.0e-5F;
        aethermind::ReferenceRmsNorm(aethermind::CpuRmsNormKernelArgs{
                .input_ = input.data(),
                .weight_ = weight.data(),
                .output_ = output.data(),
                .seq_len_ = seq_len,
                .hidden_size_ = hidden,
                .input_row_stride_ = hidden,
                .input_col_stride_ = 1,
                .weight_stride_ = 1,
                .output_row_stride_ = hidden,
                .output_col_stride_ = 1,
                .epsilon_ = kEpsilon,
        });
        benchmark::DoNotOptimize(output.data());
    }

    SetRmsNormThroughputCounters(state, seq_len, hidden);
}

BENCHMARK(BM_CPUKernel_ReferenceRmsNorm)
        ->Args({1, 4096})
        ->Args({1, 8192})
        ->Args({16, 4096})
        ->Args({128, 4096})
        ->Args({128, 8192})
        ->ArgNames({"seq_len", "hidden"});

BENCHMARK(BM_CPUKernel_RmsNorm)
        ->Args({1, 4096})
        ->Args({1, 8192})
        ->Args({16, 4096})
        ->Args({128, 4096})
        ->Args({128, 8192})
        ->ArgNames({"seq_len", "hidden"});

}// namespace
