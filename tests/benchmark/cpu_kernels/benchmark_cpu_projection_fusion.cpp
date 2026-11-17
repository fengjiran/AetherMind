#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/memory/cpu_allocator.h"

#include <array>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {
using namespace aethermind;

struct PreparedProjection {
    ResolvedKernel kernel;
    std::unique_ptr<PackedWeight> weight;
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> params{};

    Status Run() const noexcept {
        return kernel.fn({.kernel_params = params.data(), .attrs = kernel.attrs});
    }
};

// These dyadic values are exact in F32/F16/BF16, so the independent oracle
// need not share any widening helper with the implementation under test.
float ProjectionValue(size_t i) {
    return static_cast<float>(static_cast<int>((i * 17 + 3) % 31) - 15) * 0.0625F;
}

Status PrepareProjection(CpuBackend& backend, Allocator& allocator, OpType op,
                         const KernelSelector& selector, const OpParams& attrs,
                         std::span<const TensorView> weights, const TensorView& input,
                         std::span<const MutableTensorView> outputs, PreparedProjection& result) {
    AM_ASSIGN_OR_RETURN(result.kernel, backend.PrepareKernel(op, selector, attrs));
    AM_ASSIGN_OR_RETURN(result.weight, backend.PackWeights(op, weights, selector,
                                                           result.kernel.expected_packing_recipe, allocator));
    const PackedWeightView packed{
            .data = result.weight->storage().data(),
            .nbytes = result.weight->storage().nbytes(),
            .logical_dtype = result.weight->logical_dtype(),
            .logical_shape = result.weight->logical_shape(),
            .recipe_layout = result.weight->recipe().layout,
            .recipe_alignment = result.weight->recipe().alignment,
            .alignment = result.weight->storage().alignment(),
    };
    return result.kernel.params_builder({.inputs = std::span<const TensorView>(&input, 1),
                                         .outputs = outputs,
                                         .attrs = result.kernel.attrs,
                                         .packed_weight = packed},
                                        result.params.data());
}

void BenchmarkProjectionFusion(benchmark::State& state, bool fused) {
    const bool qkv = state.range(0) == 0;
    const int precision = static_cast<int>(state.range(1));
    const int64_t m = state.range(2);
    const int64_t k = state.range(3);
    const int64_t n = state.range(4);
    const DataType dtype = precision == 0 ? DataType::Float32() : precision == 1 ? DataType::Float(16)
                                                                                 : DataType::BFloat(16);
    const size_t count = qkv ? 3U : 2U;
    const std::array<int64_t, 3> widths{n, qkv ? n / 4 : n, n / 4};
    std::vector<float> input(static_cast<size_t>(m * k));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = ProjectionValue(i + 5);
    }
    std::array<std::vector<std::byte>, 3> weights;
    std::array<std::vector<float>, 3> output;
    std::array<std::array<int64_t, 2>, 3> weight_shapes;
    std::array<std::array<int64_t, 2>, 3> weight_strides;
    std::array<std::array<int64_t, 2>, 3> output_shapes;
    std::array<std::array<int64_t, 2>, 3> output_strides;
    std::vector<TensorView> weight_views;
    std::vector<MutableTensorView> output_views;
    for (size_t component = 0; component < count; ++component) {
        weights[component].resize(static_cast<size_t>(widths[component] * k) * dtype.nbytes());
        output[component].resize(static_cast<size_t>(m * widths[component]));
        for (int64_t i = 0; i < widths[component] * k; ++i) {
            const float value = ProjectionValue(static_cast<size_t>(i) + component * 7);
            std::byte* destination = weights[component].data() + static_cast<size_t>(i) * dtype.nbytes();
            if (dtype == DataType::Float32()) {
                std::memcpy(destination, &value, sizeof(value));
            } else {
                const uint16_t bits = dtype == DataType::Float(16) ? Half(value).bits() : BFloat16(value).x;
                std::memcpy(destination, &bits, sizeof(bits));
            }
        }
        weight_shapes[component] = {widths[component], k};
        weight_strides[component] = {k, 1};
        output_shapes[component] = {m, widths[component]};
        output_strides[component] = {widths[component], 1};
        weight_views.emplace_back(weights[component].data(), dtype, weight_shapes[component], weight_strides[component]);
        output_views.emplace_back(output[component].data(), DataType::Float32(), output_shapes[component], output_strides[component]);
    }
    const std::array<int64_t, 2> input_shape{m, k}, input_strides{k, 1};
    const TensorView input_view(input.data(), DataType::Float32(), input_shape, input_strides);
    const KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .output_dtype = DataType::Float32(),
            .weight_dtype = dtype,
            .weight_format = WeightFormat::kPacked,
            .phase = m == 1 ? ExecPhase::kDecode : ExecPhase::kPrefill};
    CpuBackend backend;
    CPUAllocator allocator(Device::CPU());
    std::array<PreparedProjection, 3> prepared;
    const size_t calls = fused ? 1 : count;
    if (fused) {
        const OpType op = qkv ? OpType::kQkvLinear : OpType::kGateUpLinear;
        const OpParams params = qkv ? OpParams{QkvLinearParams{n, n / 4, n / 4, false}}
                                    : OpParams{GateUpLinearParams{n, n, false}};
        const Status status = PrepareProjection(backend, allocator, op, selector, params,
                                                weight_views, input_view, output_views, prepared[0]);
        if (!status.ok()) {
            state.SkipWithError(status.ToString());
            return;
        }
    } else {
        for (size_t component = 0; component < count; ++component) {
            const Status status = PrepareProjection(backend, allocator, OpType::kLinear, selector, OpParams{LinearParams{}},
                                                    std::span<const TensorView>(&weight_views[component], 1), input_view,
                                                    std::span<const MutableTensorView>(&output_views[component], 1), prepared[component]);
            if (!status.ok()) {
                state.SkipWithError(status.ToString());
                return;
            }
        }
    }
    const auto run = [&]() -> Status {
        for (size_t i = 0; i < calls; ++i) {
            AM_RETURN_IF_ERROR(prepared[i].Run());
        }
        return Status::Ok();
    };
    if (const Status status = run(); !status.ok()) {
        state.SkipWithError(status.ToString());
        return;
    }
    for (size_t c = 0; c < count; ++c) {
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < widths[c]; ++col) {
                double expected = 0.0;
                for (int64_t inner = 0; inner < k; ++inner) {
                    expected += static_cast<double>(input[static_cast<size_t>(row * k + inner)]) *
                                ProjectionValue(static_cast<size_t>(col * k + inner) + c * 7);
                }
                const float actual = output[c][static_cast<size_t>(row * widths[c] + col)];
                if (!std::isfinite(actual) || std::fabs(actual - expected) > 1.0e-4 + 1.0e-4 * std::fabs(expected)) {
                    state.SkipWithError("Projection fusion disagrees with independent FP64 oracle");
                    return;
                }
            }
        }
    }
    state.SetLabel(std::string(fused ? "fused=" : "unfused=") + std::string(prepared[0].kernel.name) +
                   " recipe=" + std::string(ToString(prepared[0].kernel.expected_packing_recipe.layout)));
    for (auto _: state) {
        if (const Status status = run(); !status.ok()) {
            state.SkipWithError(status.ToString());
            break;
        }
        benchmark::ClobberMemory();
    }
}

void BM_ProjectionFusedHot(benchmark::State& state) { BenchmarkProjectionFusion(state, true); }
void BM_ProjectionUnfusedHot(benchmark::State& state) { BenchmarkProjectionFusion(state, false); }

void ProjectionCases(benchmark::Benchmark* benchmark) {
    for (const int kind: {0, 1}) {
        for (const int precision: {0, 1, 2}) {
            for (const int64_t m: {1, 8, 32}) {
                benchmark->Args({kind, precision, m, 512, 512});
            }
        }
    }
    benchmark->ArgNames({"Kind", "Precision", "M", "K", "N"});
}
BENCHMARK(BM_ProjectionFusedHot)->Apply(ProjectionCases);
BENCHMARK(BM_ProjectionUnfusedHot)->Apply(ProjectionCases);

} // namespace
