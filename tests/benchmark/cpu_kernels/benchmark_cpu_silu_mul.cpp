#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/operators/op_params.h"

#include <array>
#include <benchmark/benchmark.h>
#include <cmath>
#include <limits>
#include <vector>

namespace {

using namespace aethermind;

enum class Mode { kUnfused,
                  kReference,
                  kSelected };

struct Call {
    ResolvedKernel kernel{};
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> params{};

    Status Prepare(const CpuBackend& backend, OpType op, const KernelSelector& selector,
                   const OpParams& op_params, std::span<const TensorView> inputs,
                   std::span<const MutableTensorView> outputs) {
        AM_ASSIGN_OR_RETURN(kernel, backend.PrepareKernel(op, selector, op_params));
        return kernel.params_builder({.inputs = inputs, .outputs = outputs, .attrs = kernel.attrs}, params.data());
    }

    Status Run() const noexcept {
        return kernel.fn({.kernel_params = params.data(), .attrs = kernel.attrs});
    }
};

template<typename Element>
void BenchmarkSiluMulTyped(benchmark::State& state, Mode mode) {
    const int64_t rows = state.range(0);
    const int64_t hidden = state.range(1);
    const bool extreme = state.range(2) != 0;
    const auto dtype = DataType::Make<Element>();
    const std::array<int64_t, 2> shape{rows, hidden};
    const std::array<int64_t, 2> strides{hidden, 1};
    std::vector<Element> gate(rows * hidden), up(gate.size()), output(gate.size());
    std::vector<Element> scratch(mode == Mode::kUnfused ? gate.size() : 0);
    for (size_t i = 0; i < gate.size(); ++i) {
        float x = static_cast<float>(static_cast<int>(i % 257) - 128) / 32.0F;
        if (extreme) {
            x += (i % 2 == 0 ? 32.0F : -32.0F);
        }
        gate[i] = x;
        up[i] = static_cast<float>(static_cast<int>((i * 5) % 83) - 41) / 16.0F;
    }
    const TensorView g(gate.data(), dtype, shape, strides);
    const TensorView u(up.data(), dtype, shape, strides);
    const MutableTensorView y(output.data(), dtype, shape, strides);
    const KernelSelector selector{.device_type = DeviceType::kCPU,
                                  .act_dtype = dtype,
                                  .output_dtype = dtype,
                                  .weight_dtype = dtype,
                                  .weight_format = WeightFormat::kPlain,
                                  .phase = rows == 1 ? ExecPhase::kDecode : ExecPhase::kPrefill};
    const CpuBackend backend(mode == Mode::kReference
                                     ? CpuFeaturePolicy{.disabled_features = CpuFeatureSet::From({CpuFeature::kAvx2})}
                                     : CpuFeaturePolicy{});
    Call silu, product;
    const auto setup = [&]() -> Status {
        if (mode == Mode::kUnfused) {
            const MutableTensorView temporary(scratch.data(), dtype, shape, strides);
            AM_RETURN_IF_ERROR(silu.Prepare(backend, OpType::kSilu, selector, SiluParams{},
                                            std::array{g}, std::array{temporary}));
            return product.Prepare(backend, OpType::kElementwiseMul, selector, ElementwiseMulParams{},
                                   std::array{TensorView(scratch.data(), dtype, shape, strides), u},
                                   std::array{y});
        }
        return product.Prepare(backend, OpType::kSiluMul, selector, SiluMulParams{}, std::array{g, u}, std::array{y});
    };
    const auto run = [&]() -> Status {
        if (mode == Mode::kUnfused) {
            AM_RETURN_IF_ERROR(silu.Run());
        }
        return product.Run();
    };
    const auto status = setup();
    if (!status.ok()) {
        state.SkipWithError(status.ToString());
        return;
    }
    if (const auto executed = run(); !executed.ok()) {
        state.SkipWithError(executed.ToString());
        return;
    }
    for (size_t i = 0; i < gate.size(); ++i) {
        constexpr double relative_error = std::is_same_v<Element, float>
                                                  ? 6 * std::numeric_limits<float>::epsilon()
                                                  : 4.0e-3;
        const double x = static_cast<float>(gate[i]);
        const double expected = x * static_cast<float>(up[i]) / (1.0 + std::exp(-x));
        if (!(std::abs(static_cast<float>(output[i]) - expected) <= std::abs(expected) * relative_error)) {
            state.SkipWithError("Prepared output differs from the FP64 oracle");
            return;
        }
    }
    state.SetLabel(mode == Mode::kUnfused ? std::string(silu.kernel.name) + " + " + std::string(product.kernel.name)
                                          : std::string(product.kernel.name));
    for (auto _: state) {
        if (const auto executed = run(); !executed.ok()) {
            state.SkipWithError(executed.ToString());
            return;
        }
        benchmark::ClobberMemory();
    }
}

void BenchmarkSiluMul(benchmark::State& state, Mode mode) {
    BenchmarkSiluMulTyped<float>(state, mode);
}

void BenchmarkBf16SiluMul(benchmark::State& state, Mode mode) {
    BenchmarkSiluMulTyped<BFloat16>(state, mode);
}

void Shapes(benchmark::Benchmark* bench) {
    bench->Args({1, 16, 0})
            ->Args({1, 2819, 0})
            ->Args({1, 11008, 0})
            ->Args({32, 2816, 0})
            ->Args({128, 11008, 0})
            ->Args({1, 2819, 1});
}

BENCHMARK_CAPTURE(BenchmarkSiluMul, Unfused, Mode::kUnfused)->Apply(Shapes);
BENCHMARK_CAPTURE(BenchmarkSiluMul, Reference, Mode::kReference)->Apply(Shapes);
BENCHMARK_CAPTURE(BenchmarkSiluMul, Selected, Mode::kSelected)->Apply(Shapes);
BENCHMARK_CAPTURE(BenchmarkBf16SiluMul, Reference, Mode::kReference)->Apply(Shapes);
BENCHMARK_CAPTURE(BenchmarkBf16SiluMul, Selected, Mode::kSelected)->Apply(Shapes);

} // namespace
