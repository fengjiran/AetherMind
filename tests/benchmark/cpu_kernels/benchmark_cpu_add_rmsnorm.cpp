#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/operators/op_params.h"

#include <array>
#include <benchmark/benchmark.h>
#include <cmath>
#include <vector>

namespace {

using namespace aethermind;

enum class Mode { kUnfused,
                  kFusedReference,
                  kFusedSelected };

struct PreparedCall {
    ResolvedKernel resolved{};
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> params{};

    Status Run() const noexcept {
        return resolved.fn({.kernel_params = params.data(), .attrs = resolved.attrs});
    }
};

Status Prepare(PreparedCall& call, const CpuBackend& backend, OpType op,
               const KernelSelector& selector, const OpParams& op_params,
               std::span<const TensorView> inputs, std::span<const MutableTensorView> outputs,
               Allocator& allocator, std::unique_ptr<PackedWeight>& artifact) {
    AM_ASSIGN_OR_RETURN(call.resolved, backend.PrepareKernel(op, selector, op_params));
    KernelParamsBuildContext context{.inputs = inputs, .outputs = outputs, .attrs = call.resolved.attrs};
    if (selector.weight_format == WeightFormat::kPacked) {
        AM_ASSIGN_OR_RETURN(artifact, backend.PackWeights(
                                              op, inputs.last(1), selector,
                                              call.resolved.expected_packing_recipe, allocator));
        const auto& storage = artifact->storage();
        context.inputs = inputs.first(inputs.size() - 1);
        context.packed_weight = PackedWeightView{
                .data = storage.data(),
                .nbytes = storage.nbytes(),
                .logical_dtype = artifact->logical_dtype(),
                .logical_shape = artifact->logical_shape(),
                .recipe_layout = artifact->recipe().layout,
                .recipe_alignment = artifact->recipe().alignment,
                .alignment = storage.alignment(),
        };
    }
    return call.resolved.params_builder(context, call.params.data());
}

void BenchmarkAddRmsNorm(benchmark::State& state, Mode mode) {
    const int64_t rows = state.range(0);
    const int64_t hidden = state.range(1);
    const bool packed = state.range(2) != 0;
    const auto dtype = DataType::Float32();
    constexpr float epsilon = 1.0e-5F;
    const std::array<int64_t, 2> shape{rows, hidden};
    const std::array<int64_t, 2> strides{hidden, 1};
    const std::array<int64_t, 1> weight_shape{hidden};
    const std::array<int64_t, 1> weight_strides{1};
    std::vector<float> input(rows * hidden), residual(input.size()), normalized(input.size()),
            new_residual(input.size()), weight(hidden);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i % 61) / 17.0F;
        residual[i] = -static_cast<float>(i % 37) / 13.0F;
    }
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = 0.5F + static_cast<float>(i % 17) / 16.0F;
    }

    const TensorView x(input.data(), dtype, shape, strides);
    const TensorView r(residual.data(), dtype, shape, strides);
    const TensorView w(weight.data(), dtype, weight_shape, weight_strides);
    const TensorView sum(new_residual.data(), dtype, shape, strides);
    const MutableTensorView y(normalized.data(), dtype, shape, strides);
    const MutableTensorView sum_out(new_residual.data(), dtype, shape, strides);
    const KernelSelector plain{.device_type = DeviceType::kCPU,
                               .act_dtype = dtype,
                               .output_dtype = dtype,
                               .weight_dtype = dtype,
                               .weight_format = WeightFormat::kPlain,
                               .phase = rows == 1 ? ExecPhase::kDecode : ExecPhase::kPrefill};
    auto weighted = plain;
    if (packed) {
        weighted.weight_format = WeightFormat::kPacked;
    }
    const CpuBackend backend(mode == Mode::kFusedReference
                                     ? CpuFeaturePolicy{.disabled_features = CpuFeatureSet::From({CpuFeature::kAvx2})}
                                     : CpuFeaturePolicy{});
    CPUAllocator allocator(Device::CPU());
    std::unique_ptr<PackedWeight> artifact;
    PreparedCall main, add;
    const auto setup = [&]() -> Status {
        if (mode == Mode::kUnfused) {
            AM_RETURN_IF_ERROR(Prepare(add, backend, OpType::kAdd, plain, AddParams{},
                                       std::array{x, r}, std::array{sum_out}, allocator, artifact));
            return Prepare(main, backend, OpType::kRmsNorm, weighted, RmsNormParams{.eps = epsilon},
                           std::array{sum, w}, std::array{y}, allocator, artifact);
        }
        return Prepare(main, backend, OpType::kAddRmsNorm, weighted, AddRmsNormParams{.eps = epsilon},
                       std::array{x, r, w}, std::array{y, sum_out}, allocator, artifact);
    };
    const auto run = [&]() -> Status {
        if (mode == Mode::kUnfused) {
            AM_RETURN_IF_ERROR(add.Run());
        }
        return main.Run();
    };
    if (const auto status = setup(); !status.ok()) {
        state.SkipWithError(status.ToString());
        return;
    }
    if (const auto status = run(); !status.ok()) {
        state.SkipWithError(status.ToString());
        return;
    }
    for (int64_t row = 0; row < rows; ++row) {
        double sum_sq = 0.0;
        for (int64_t col = 0; col < hidden; ++col) {
            const size_t i = row * hidden + col;
            const float expected_sum = input[i] + residual[i];
            if (new_residual[i] != expected_sum) {
                state.SkipWithError("AddRmsNorm residual differs from FP32 Add");
                return;
            }
            sum_sq += static_cast<double>(expected_sum) * expected_sum;
        }
        const double norm = std::sqrt(sum_sq / hidden + epsilon);
        for (int64_t col = 0; col < hidden; ++col) {
            const size_t i = row * hidden + col;
            const float expected = static_cast<float>(new_residual[i] / norm * weight[col]);
            if (!(std::abs(normalized[i] - expected) <= 2.0e-6F)) {
                state.SkipWithError("AddRmsNorm normalized output differs from oracle");
                return;
            }
        }
    }
    state.SetLabel(mode == Mode::kUnfused ? std::string(add.resolved.name) + " + " + std::string(main.resolved.name)
                                          : std::string(main.resolved.name));
    for (auto _: state) {
        if (const auto status = run(); !status.ok()) {
            state.SkipWithError(status.ToString());
            break;
        }
        benchmark::ClobberMemory();
    }
}

void Shapes(benchmark::Benchmark* bench) {
    for (int64_t packed: {0, 1}) {
        bench->Args({1, 5, packed})
                ->Args({1, 16, packed})
                ->Args({1, 513, packed})
                ->Args({1, 4096, packed})
                ->Args({32, 4096, packed})
                ->Args({256, 4096, packed});
    }
}

BENCHMARK_CAPTURE(BenchmarkAddRmsNorm, Unfused, Mode::kUnfused)->Apply(Shapes);
BENCHMARK_CAPTURE(BenchmarkAddRmsNorm, FusedReference, Mode::kFusedReference)->Apply(Shapes);
BENCHMARK_CAPTURE(BenchmarkAddRmsNorm, FusedSelected, Mode::kFusedSelected)->Apply(Shapes);

} // namespace
