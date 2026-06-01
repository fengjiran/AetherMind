#include "aethermind/backend/cpu/kernels/cpu_rmsnorm_kernel.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_invocation.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/operators/op_type.h"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <span>
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

void ReferenceRmsNorm(const float* input,
                      const float* weight,
                      float* output,
                      int64_t seq_len,
                      int64_t hidden_size,
                      float epsilon) noexcept {
    for (int64_t s = 0; s < seq_len; ++s) {
        const float* row_in = input + s * hidden_size;
        float* row_out = output + s * hidden_size;

        double mean_square = 0.0F;
        for (int64_t i = 0; i < hidden_size; ++i) {
            mean_square += row_in[i] * row_in[i];
        }
        mean_square /= static_cast<float>(hidden_size);

        const float inv_rms = 1.0F / std::sqrt(mean_square + epsilon);
        for (int64_t i = 0; i < hidden_size; ++i) {
            row_out[i] = row_in[i] * inv_rms * weight[i];
        }
    }
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

    const std::int64_t io_shape[2] = {seq_len, hidden};
    const std::int64_t io_strides[2] = {hidden, 1};
    const std::int64_t w_shape[1] = {hidden};
    const std::int64_t w_strides[1] = {1};

    const aethermind::CpuRmsNormParams params{
            .Input = aethermind::TensorView{input.data(), aethermind::DataType::Float32(), io_shape, io_strides},
            .Weight = aethermind::TensorView{weight.data(), aethermind::DataType::Float32(), w_shape, w_strides},
            .Output = aethermind::MutableTensorView{output.data(), aethermind::DataType::Float32(), io_shape, io_strides},
    };

    constexpr aethermind::CpuRmsNormAttrs attrs{.Epsilon = 1.0e-5F};
    const aethermind::KernelInvocation invocation{
            .op_type = aethermind::OpType::kRmsNorm,
            .selector = {.device_type = aethermind::DeviceType::kCPU},
    };
    const aethermind::KernelContext context{
            .device = aethermind::Device::CPU(),
            .packed_params = &params,
            .attrs = std::as_bytes(std::span{&attrs, std::size_t{1}}),
    };

    for (auto _: state) {
        auto status = aethermind::CpuRmsNormKernel(invocation, context, aethermind::WorkspaceBinding{});
        benchmark::DoNotOptimize(status);
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
        ReferenceRmsNorm(input.data(), weight.data(), output.data(), seq_len, hidden, kEpsilon);
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
