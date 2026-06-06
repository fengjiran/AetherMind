#include "aethermind/backend/cpu/kernels/rmsnorm/cpu_rmsnorm_kernel.h"
#include "backend/cpu/kernels/rmsnorm/rmsnorm_internal.h"

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

    const aethermind::RmsNormArgs args{
            .input = input.data(),
            .weight = weight.data(),
            .output = output.data(),
            .seq_len = seq_len,
            .hidden_size = hidden,
            .input_row_stride = hidden,
            .input_col_stride = 1,
            .weight_stride = 1,
            .output_row_stride = hidden,
            .output_col_stride = 1,
            .eps = 1.0e-5F,
            .dtype = aethermind::DataType::Float32(),
    };

    for (auto _: state) {
        bool ok = aethermind::LaunchRmsNorm(args).ok();
        benchmark::DoNotOptimize(ok);
        benchmark::DoNotOptimize(output.data());
    }

    SetRmsNormThroughputCounters(state, seq_len, hidden);
}

void BM_CPUKernel_RmsNormScalar(benchmark::State& state) {
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
        constexpr float kEps = 1.0e-5F;
        const aethermind::Status status = aethermind::cpu::RmsNormKernel_CPU_FP32_Scalar(aethermind::cpu::RmsNormFp32KernelArgs{
                .input = input.data(),
                .weight = weight.data(),
                .output = output.data(),
                .seq_len = seq_len,
                .hidden_size = hidden,
                .input_row_stride = hidden,
                .input_col_stride = 1,
                .weight_stride = 1,
                .output_row_stride = hidden,
                .output_col_stride = 1,
                .eps = kEps,
        });
        benchmark::DoNotOptimize(status.ok());
        benchmark::DoNotOptimize(output.data());
    }

    SetRmsNormThroughputCounters(state, seq_len, hidden);
}

BENCHMARK(BM_CPUKernel_RmsNormScalar)
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
