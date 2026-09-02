#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/execution/kernel_invoker.h"
#include "aethermind/operators/op_params.h"
#include "backend/cpu/kernels/rmsnorm/rmsnorm_internal.h"

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace {

using namespace aethermind;

constexpr double kRmsNormFlopsPerElement = 4.0;

KernelSelector MakeRmsNormSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<CpuFeaturePolicy> MakeScalarCpuFeaturePolicy() {
    const auto capabilities = cpu::DetectCpuCapabilities();
    if (!capabilities.ok()) {
        return capabilities.status();
    }
    return CpuFeaturePolicy{
            .disabled_features = capabilities->usable_features,
    };
}

bool IsAvx2FmaRmsNormKernel(const ResolvedKernel& kernel) noexcept {
    return kernel.name != nullptr &&
           std::string_view{kernel.name} ==
                   std::string_view{"cpu::rmsnorm_f32_avx2_fma"};
}

void SetRmsNormThroughputCounters(benchmark::State& state,
                                  int64_t row_count,
                                  int64_t hidden_size) {
    const int64_t elements = state.iterations() * row_count * hidden_size;
    state.SetItemsProcessed(elements);
    state.SetBytesProcessed(elements * static_cast<int64_t>(sizeof(float) * 4));

    const double flops = static_cast<double>(elements) * kRmsNormFlopsPerElement;
    state.counters["GFLOP/s"] = benchmark::Counter(
            flops, benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
}

enum class RmsNormBenchmarkMode {
    /// Params prepared once outside the loop; the loop runs only kernel.fn.
    kPrepared,
    /// Legacy per-call path: stack params + builder + kernel each iteration.
    kLegacyBuildAndInvoke,
    /// Measures binding specialization cost: builder + validation per call.
    kBindingSpecialization,
};

void BenchmarkRmsNormFp32(benchmark::State& state,
                          RmsNormBenchmarkMode mode,
                          bool require_avx2_fma) {
    const int64_t row_count = state.range(0);
    const int64_t hidden_size = state.range(1);
    const size_t element_count = static_cast<size_t>(row_count * hidden_size);

    std::vector<float> input(element_count);
    std::vector<float> weight(static_cast<size_t>(hidden_size));
    std::vector<float> output(element_count);
    for (size_t index = 0; index < element_count; ++index) {
        input[index] = static_cast<float>((index & 0xFFU) + 1U) * 0.01F;
    }
    for (float& value: weight) {
        value = 1.0F;
    }

    const auto prepare = [&]() -> StatusOr<ResolvedKernel> {
        if (require_avx2_fma) {
            CpuBackend backend;
            return backend.PrepareKernel(
                    OpType::kRmsNorm, MakeRmsNormSelector(),
                    OpParams{RmsNormParams{.eps = 1.0e-5F}});
        }
        const auto policy = MakeScalarCpuFeaturePolicy();
        if (!policy.ok()) {
            return policy.status();
        }
        CpuBackend backend(*policy);
        return backend.PrepareKernel(
                OpType::kRmsNorm, MakeRmsNormSelector(),
                OpParams{RmsNormParams{.eps = 1.0e-5F}});
    };
    const StatusOr<ResolvedKernel> resolved = prepare();
    if (!resolved.ok()) {
        state.SkipWithError(resolved.status().ToString().c_str());
        return;
    }
    if (require_avx2_fma && !IsAvx2FmaRmsNormKernel(*resolved)) {
        state.SkipWithError("AVX2+FMA RMSNorm kernel is unavailable");
        return;
    }

    const int64_t io_shape[2] = {row_count, hidden_size};
    const int64_t io_strides[2] = {hidden_size, 1};
    const int64_t weight_shape[1] = {hidden_size};
    const int64_t weight_strides[1] = {1};
    const std::array<TensorView, 2> inputs{
            TensorView{input.data(), DataType::Float32(), io_shape, io_strides},
            TensorView{weight.data(), DataType::Float32(), weight_shape, weight_strides},
    };
    const std::array<MutableTensorView, 1> outputs{
            MutableTensorView{output.data(), DataType::Float32(), io_shape, io_strides},
    };

    // Buffer matching the BindingTable params arena slot.
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> params_storage{};
    const KernelParamsBuildContext build_context{
            .inputs = inputs,
            .outputs = outputs,
            .attrs = resolved->attrs,
    };
    if (mode == RmsNormBenchmarkMode::kPrepared) {
        const Status build_status =
                resolved->params_builder(build_context, params_storage.data());
        if (!build_status.ok()) {
            state.SkipWithError(build_status.ToString().c_str());
            return;
        }
    }

    KernelContext context{
            .kernel_params = params_storage.data(),
            .attrs = resolved->attrs,
    };
    state.SetLabel(resolved->name);

    for (auto _: state) {
        Status status = Status::Ok();
        switch (mode) {
            case RmsNormBenchmarkMode::kPrepared:
                status = resolved->fn(context);
                break;
            case RmsNormBenchmarkMode::kLegacyBuildAndInvoke:
                status = BuildAndInvokeKernelForTesting(*resolved, context, inputs, outputs);
                break;
            case RmsNormBenchmarkMode::kBindingSpecialization:
                status = resolved->params_builder(build_context,
                                                  params_storage.data());
                break;
        }
        if (!status.ok()) {
            state.SkipWithError(status.ToString().c_str());
            break;
        }
        benchmark::DoNotOptimize(output.data());
    }

    SetRmsNormThroughputCounters(state, row_count, hidden_size);
}

void BM_RmsNormPreparedScalar(benchmark::State& state) {
    BenchmarkRmsNormFp32(state, RmsNormBenchmarkMode::kPrepared, false);
}

void BM_RmsNormPreparedAvx2Fma(benchmark::State& state) {
    BenchmarkRmsNormFp32(state, RmsNormBenchmarkMode::kPrepared, true);
}

void BM_RmsNormLegacyBuildAndInvoke(benchmark::State& state) {
    BenchmarkRmsNormFp32(state, RmsNormBenchmarkMode::kLegacyBuildAndInvoke, false);
}

void BM_RmsNormBindingSpecialization(benchmark::State& state) {
    BenchmarkRmsNormFp32(state, RmsNormBenchmarkMode::kBindingSpecialization, false);
}

BENCHMARK(BM_RmsNormPreparedScalar)
        ->Args({1, 4096})
        ->Args({1, 8192})
        ->Args({1, 11008})
        ->Args({16, 4096})
        ->Args({128, 4096})
        ->Args({128, 8192})
        ->ArgNames({"row_count", "hidden_size"});

BENCHMARK(BM_RmsNormPreparedAvx2Fma)
        ->Args({1, 4096})
        ->Args({1, 8192})
        ->Args({1, 11008})
        ->Args({16, 4096})
        ->Args({128, 4096})
        ->Args({128, 8192})
        ->ArgNames({"row_count", "hidden_size"});

BENCHMARK(BM_RmsNormLegacyBuildAndInvoke)
        ->Args({1, 4096})
        ->Args({1, 8192})
        ->Args({128, 4096})
        ->ArgNames({"row_count", "hidden_size"});

BENCHMARK(BM_RmsNormBindingSpecialization)
        ->Args({1, 4096})
        ->Args({1, 8192})
        ->Args({128, 4096})
        ->ArgNames({"row_count", "hidden_size"});

} // namespace
