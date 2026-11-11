#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/operators/op_params.h"
#include "backend/cpu/kernels/gemm/gemm_internal.h"

#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace aethermind;

constexpr size_t kStreamingWeightTargetBytes = 256U * 1024U * 1024U;
constexpr size_t kMaxStreamingWeightReplicas = 8;

enum class LinearBenchmarkMode {
    /// Reuses one prepared invocation and one weight buffer.
    kPreparedHot,
    /// Rotates prepared invocations whose total weight working set targets 256 MiB.
    kPreparedStreaming,
    /// Measures only cold-path params construction; it never enters the kernel.
    kBindingSpecialization,
};

struct PreparedLinearInvocation {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> params{};
};

KernelSelector MakePlainLinearSelector(DataType weight_dtype = DataType::Float32()) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = weight_dtype,
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PreparePlainLinearKernel(DataType weight_dtype = DataType::Float32()) {
    CpuBackend backend;
    return backend.PrepareKernel(
            OpType::kLinear, MakePlainLinearSelector(weight_dtype), OpParams{LinearParams{}});
}

float DeterministicValue(size_t index) {
    constexpr size_t kPeriod = 251;
    return static_cast<float>(static_cast<int64_t>(index % kPeriod) -
                              static_cast<int64_t>(kPeriod / 2U)) *
           0.0078125F;
}

void FillDeterministic(std::vector<float>& values) {
    for (size_t index = 0; index < values.size(); ++index) {
        values[index] = DeterministicValue(index);
    }
}

Status BuildLinearPreparedParams(const ResolvedKernel& kernel,
                                 const float* input,
                                 const void* weight,
                                 DataType weight_dtype,
                                 float* output,
                                 int64_t m,
                                 int64_t k,
                                 int64_t n,
                                 PreparedLinearInvocation& prepared) noexcept {
    const std::array<int64_t, 2> input_shape = {m, k};
    const std::array<int64_t, 2> input_strides = {k, 1};
    const std::array<int64_t, 2> weight_shape = {n, k};
    const std::array<int64_t, 2> weight_strides = {k, 1};
    const std::array<int64_t, 2> output_shape = {m, n};
    const std::array<int64_t, 2> output_strides = {n, 1};
    const std::array<TensorView, 2> inputs{
            TensorView{input, DataType::Float32(), input_shape, input_strides},
            TensorView{weight, weight_dtype, weight_shape, weight_strides},
    };
    const std::array<MutableTensorView, 1> outputs{
            MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    };
    return kernel.params_builder(KernelParamsBuildContext{
                                         .inputs = inputs,
                                         .outputs = outputs,
                                         .attrs = kernel.attrs,
                                 },
                                 prepared.params.data());
}

Status InvokePreparedLinear(const ResolvedKernel& kernel,
                            const PreparedLinearInvocation& prepared) noexcept {
    return kernel.fn(KernelContext{
            .kernel_params = prepared.params.data(),
            .attrs = kernel.attrs,
    });
}

Status ValidateLinearAgainstGemmReference(const ResolvedKernel& kernel,
                                          const PreparedLinearInvocation& prepared,
                                          const float* input,
                                          const float* weight,
                                          float* output,
                                          int64_t m,
                                          int64_t k,
                                          int64_t n,
                                          float error_tolerance) {
    const auto output_elements = static_cast<size_t>(m * n);
    std::vector<float> expected(output_elements,
                                std::numeric_limits<float>::quiet_NaN());
    AM_RETURN_IF_ERROR(cpu::detail::RunGemmF32Reference(cpu::detail::GemmF32Args{
            .lhs = input,
            .rhs = weight,
            .output = expected.data(),
            .m = m,
            .n = n,
            .k = k,
            .lhs_m_stride = k,
            .lhs_k_stride = 1,
            .rhs_k_stride = 1,
            .rhs_n_stride = k,
            .output_m_stride = n,
            .output_n_stride = 1,
    }));

    std::fill_n(output, output_elements, std::numeric_limits<float>::quiet_NaN());
    AM_RETURN_IF_ERROR(InvokePreparedLinear(kernel, prepared));
    for (size_t index = 0; index < output_elements; ++index) {
        const float actual = output[index];
        const float reference = expected[index];
        if (!std::isfinite(actual) || !std::isfinite(reference)) {
            return Status::Internal(
                    "Linear benchmark correctness guard produced NaN or Inf");
        }

        const float absolute_error = std::fabs(actual - reference);
        const float relative_error =
                absolute_error / std::max(std::fabs(reference), 1.0e-6F);
        if (absolute_error > error_tolerance && relative_error > error_tolerance) {
            return Status::Internal(
                    "Linear benchmark correctness guard disagrees with GEMM reference");
        }
    }
    return Status::Ok();
}

size_t StreamingReplicaCount(size_t weight_bytes) {
    if (weight_bytes == 0) {
        return 1;
    }
    const size_t required =
            (kStreamingWeightTargetBytes + weight_bytes - 1U) / weight_bytes;
    return std::clamp(required, size_t{1}, kMaxStreamingWeightReplicas);
}

std::string BenchmarkLabel(const ResolvedKernel& kernel,
                           LinearBenchmarkMode mode) {
    const char* const mode_name = [&] {
        switch (mode) {
            case LinearBenchmarkMode::kPreparedHot:
                return "hot";
            case LinearBenchmarkMode::kPreparedStreaming:
                return "streaming";
            case LinearBenchmarkMode::kBindingSpecialization:
                return "binding";
        }
        return "unknown";
    }();
    return std::string{"kernel="} +
           (kernel.name.empty() ? std::string{"<unnamed>"} : std::string{kernel.name}) +
           " cache_mode=" + mode_name;
}

void SetLinearCounters(benchmark::State& state,
                       int64_t m,
                       int64_t k,
                       int64_t n,
                       size_t weight_replica_count,
                       size_t weight_working_set_bytes,
                       bool includes_compute,
                       size_t weight_element_size = sizeof(float)) {
    const int64_t output_elements = m * n;
    const int64_t logical_bytes =
            static_cast<int64_t>(sizeof(float)) * (m * k + output_elements) +
            static_cast<int64_t>(weight_element_size) * n * k;
    state.counters["M"] = static_cast<double>(m);
    state.counters["K"] = static_cast<double>(k);
    state.counters["N"] = static_cast<double>(n);
    state.counters["weight replicas"] =
            static_cast<double>(weight_replica_count);
    state.counters["weight working set MiB"] =
            static_cast<double>(weight_working_set_bytes) / (1024.0 * 1024.0);
    if (!includes_compute) {
        return;
    }

    const int64_t output_count = state.iterations() * output_elements;
    state.SetItemsProcessed(output_count);
    state.SetBytesProcessed(state.iterations() * logical_bytes);
    const double flops = static_cast<double>(state.iterations()) * 2.0 *
                         static_cast<double>(m) * static_cast<double>(n) *
                         static_cast<double>(k);
    const double bytes = static_cast<double>(state.iterations()) *
                         static_cast<double>(logical_bytes);
    state.counters["GFLOP/s"] = benchmark::Counter(
            flops, benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
    state.counters["logical GB/s"] = benchmark::Counter(
            bytes, benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
}

void BenchmarkLinearPreparedPath(benchmark::State& state, LinearBenchmarkMode mode) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);
    const auto input_elements = static_cast<size_t>(m * k);
    const auto weight_elements = static_cast<size_t>(n * k);
    const auto output_elements = static_cast<size_t>(m * n);
    const size_t weight_bytes = weight_elements * sizeof(float);
    const size_t replica_count = mode == LinearBenchmarkMode::kPreparedStreaming
                                         ? StreamingReplicaCount(weight_bytes)
                                         : 1;

    std::vector<float> input(input_elements);
    std::vector<float> weights(weight_elements * replica_count);
    std::vector<float> output(output_elements);
    FillDeterministic(input);
    FillDeterministic(weights);

    const StatusOr<ResolvedKernel> resolved = PreparePlainLinearKernel();
    if (!resolved.ok()) {
        state.SkipWithError(resolved.status().ToString());
        return;
    }

    if (resolved->params_builder == nullptr || resolved->fn == nullptr) {
        state.SkipWithError(
                "Prepared Linear benchmark requires params_builder and kernel function");
        return;
    }

    std::vector<PreparedLinearInvocation> prepared(replica_count);
    for (size_t replica = 0; replica < replica_count; ++replica) {
        const Status status = BuildLinearPreparedParams(
                *resolved, input.data(), weights.data() + replica * weight_elements,
                DataType::Float32(), output.data(), m, k, n, prepared[replica]);
        if (!status.ok()) {
            state.SkipWithError(status.ToString());
            return;
        }
    }

    constexpr float kErrorTolerance = 1.0e-5F;
    const Status correctness = ValidateLinearAgainstGemmReference(
            *resolved, prepared.front(), input.data(),
            weights.data(), output.data(), m, k, n, kErrorTolerance);
    if (!correctness.ok()) {
        state.SkipWithError(correctness.ToString());
        return;
    }

    state.SetLabel(BenchmarkLabel(*resolved, mode));
    size_t iteration = 0;
    for (auto _: state) {
        const size_t replica =
                mode == LinearBenchmarkMode::kPreparedStreaming
                        ? iteration % replica_count
                        : 0;
        Status status = Status::Ok();
        if (mode == LinearBenchmarkMode::kBindingSpecialization) {
            status = BuildLinearPreparedParams(
                    *resolved, input.data(), weights.data() + replica * weight_elements,
                    DataType::Float32(), output.data(), m, k, n, prepared[replica]);
        } else {
            status = InvokePreparedLinear(*resolved, prepared[replica]);
        }

        if (!status.ok()) {
            state.SkipWithError(status.ToString());
            break;
        }
        benchmark::DoNotOptimize(output.data());
        ++iteration;
    }

    SetLinearCounters(state, m, k, n, replica_count,
                      weight_bytes * replica_count,
                      mode != LinearBenchmarkMode::kBindingSpecialization);
}

void BM_LinearPreparedHot(benchmark::State& state) {
    BenchmarkLinearPreparedPath(state, LinearBenchmarkMode::kPreparedHot);
}

void BM_LinearPreparedStreaming(benchmark::State& state) {
    BenchmarkLinearPreparedPath(state, LinearBenchmarkMode::kPreparedStreaming);
}

void BM_LinearBindingSpecialization(benchmark::State& state) {
    BenchmarkLinearPreparedPath(state, LinearBenchmarkMode::kBindingSpecialization);
}

uint16_t NarrowToBits(float value, DataType dtype) {
    if (dtype == DataType::Float(16)) {
        return Half(value).bits();
    }
    return BFloat16(value).x;
}

float WidenFromBits(uint16_t bits, DataType dtype) {
    const int fraction_bits = dtype == DataType::Float(16) ? 10 : 7;
    const int bias = fraction_bits == 10 ? 15 : 127;
    const unsigned exponent_mask = fraction_bits == 10 ? 31U : 255U;
    const unsigned exponent = (bits >> fraction_bits) & exponent_mask;
    const unsigned fraction = bits & ((1U << fraction_bits) - 1);
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16;
    if (exponent == exponent_mask) {
        return std::bit_cast<float>(sign | 0x7f800000U |
                                    (fraction == 0 ? 0U : (fraction << (23 - fraction_bits)) | 0x00400000U));
    }
    const double significand = exponent == 0 ? fraction : (1U << fraction_bits) + fraction;
    const float value = static_cast<float>(std::ldexp(significand,
                                                      static_cast<int>(exponent == 0 ? 1 : exponent) - bias - fraction_bits));
    return std::bit_cast<float>(sign | std::bit_cast<uint32_t>(value));
}

// Independent FP64 oracle with the frozen gamma(2K) bound plus an absolute
// floor; bit-exact widening is gated separately by the kernel unit tests.
Status ValidateLinearW16AgainstDoubleOracle(const ResolvedKernel& kernel,
                                            const PreparedLinearInvocation& prepared,
                                            const float* input,
                                            const std::byte* weight,
                                            float* output,
                                            int64_t m,
                                            int64_t k,
                                            int64_t n,
                                            DataType dtype) {
    const double unit_roundoff =
            static_cast<double>(std::numeric_limits<float>::epsilon()) / 2.0;
    const double gamma = (2.0 * static_cast<double>(k) * unit_roundoff) /
                         (1.0 - 2.0 * static_cast<double>(k) * unit_roundoff);
    const auto output_elements = static_cast<size_t>(m * n);
    std::vector<float> expected(output_elements);
    std::vector<double> bounds(output_elements);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            double sum = 0.0;
            double absolute_products = 0.0;
            for (int64_t t = 0; t < k; ++t) {
                uint16_t bits = 0;
                std::memcpy(&bits,
                            weight + static_cast<size_t>(col * k + t) * sizeof(bits),
                            sizeof(bits));
                const double product = static_cast<double>(input[row * k + t]) *
                                       static_cast<double>(WidenFromBits(bits, dtype));
                sum += product;
                absolute_products += std::abs(product);
            }
            const size_t index = static_cast<size_t>(row * n + col);
            expected[index] = static_cast<float>(sum);
            bounds[index] = gamma * absolute_products + 1.0e-6;
        }
    }

    std::fill_n(output, output_elements, std::numeric_limits<float>::quiet_NaN());
    AM_RETURN_IF_ERROR(InvokePreparedLinear(kernel, prepared));
    for (size_t index = 0; index < output_elements; ++index) {
        if (!std::isfinite(output[index])) {
            return Status::Internal(
                    "W16 Linear benchmark correctness guard produced NaN or Inf");
        }
        if (std::abs(static_cast<double>(output[index]) -
                     static_cast<double>(expected[index])) > bounds[index]) {
            return Status::Internal(
                    "W16 Linear benchmark correctness guard exceeds the double-oracle bound");
        }
    }
    return Status::Ok();
}

void BenchmarkLinearW16PreparedPath(benchmark::State& state,
                                    LinearBenchmarkMode mode,
                                    DataType weight_dtype) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);
    const auto input_elements = static_cast<size_t>(m * k);
    const auto weight_elements = static_cast<size_t>(n * k);
    const auto output_elements = static_cast<size_t>(m * n);
    const size_t weight_bytes = weight_elements * sizeof(uint16_t);
    const size_t replica_count = mode == LinearBenchmarkMode::kPreparedStreaming
                                         ? StreamingReplicaCount(weight_bytes)
                                         : 1;

    std::vector<float> input(input_elements);
    std::vector<std::byte> weights(weight_bytes * replica_count);
    std::vector<float> output(output_elements);
    FillDeterministic(input);
    for (size_t index = 0; index < weight_elements * replica_count; ++index) {
        const uint16_t bits = NarrowToBits(DeterministicValue(index), weight_dtype);
        std::memcpy(weights.data() + index * sizeof(bits), &bits, sizeof(bits));
    }

    const StatusOr<ResolvedKernel> resolved = PreparePlainLinearKernel(weight_dtype);
    if (!resolved.ok()) {
        state.SkipWithError(resolved.status().ToString());
        return;
    }
    if (resolved->params_builder == nullptr || resolved->fn == nullptr) {
        state.SkipWithError(
                "Prepared W16 Linear benchmark requires params_builder and kernel function");
        return;
    }

    std::vector<PreparedLinearInvocation> prepared(replica_count);
    for (size_t replica = 0; replica < replica_count; ++replica) {
        const Status status = BuildLinearPreparedParams(
                *resolved, input.data(), weights.data() + replica * weight_bytes,
                weight_dtype, output.data(), m, k, n, prepared[replica]);
        if (!status.ok()) {
            state.SkipWithError(status.ToString());
            return;
        }
    }

    const Status correctness = ValidateLinearW16AgainstDoubleOracle(
            *resolved, prepared.front(), input.data(), weights.data(),
            output.data(), m, k, n, weight_dtype);
    if (!correctness.ok()) {
        state.SkipWithError(correctness.ToString());
        return;
    }

    state.SetLabel(BenchmarkLabel(*resolved, mode));
    size_t iteration = 0;
    for (auto _: state) {
        const size_t replica =
                mode == LinearBenchmarkMode::kPreparedStreaming
                        ? iteration % replica_count
                        : 0;
        Status status = Status::Ok();
        if (mode == LinearBenchmarkMode::kBindingSpecialization) {
            status = BuildLinearPreparedParams(
                    *resolved, input.data(), weights.data() + replica * weight_bytes,
                    weight_dtype, output.data(), m, k, n, prepared[replica]);
        } else {
            status = InvokePreparedLinear(*resolved, prepared[replica]);
        }
        if (!status.ok()) {
            state.SkipWithError(status.ToString());
            break;
        }
        benchmark::DoNotOptimize(output.data());
        ++iteration;
    }

    SetLinearCounters(state, m, k, n, replica_count,
                      weight_bytes * replica_count,
                      mode != LinearBenchmarkMode::kBindingSpecialization,
                      sizeof(uint16_t));
}

void BM_LinearW16PreparedHot(benchmark::State& state, DataType dtype) {
    BenchmarkLinearW16PreparedPath(state, LinearBenchmarkMode::kPreparedHot, dtype);
}

void BM_LinearW16PreparedStreaming(benchmark::State& state, DataType dtype) {
    BenchmarkLinearW16PreparedPath(state, LinearBenchmarkMode::kPreparedStreaming, dtype);
}

void BM_LinearW16BindingSpecialization(benchmark::State& state, DataType dtype) {
    BenchmarkLinearW16PreparedPath(state, LinearBenchmarkMode::kBindingSpecialization, dtype);
}

BENCHMARK(BM_LinearPreparedHot)
        // Decode projections.
        ->Args({1, 4096, 4096})
        ->Args({1, 4096, 6144})
        ->Args({1, 4096, 11008})
        ->Args({1, 4096, 22016})
        ->Args({1, 11008, 4096})
        ->Args({1, 4096, 32000})
        // Prefill projections.
        ->Args({16, 4096, 4096})
        ->Args({64, 4096, 4096})
        ->Args({16, 4096, 22016})
        ->Args({16, 11008, 4096})
        // Small/tail shapes used to guard future tile-specific paths.
        ->Args({1, 33, 31})
        ->Args({2, 32, 33})
        ->ArgNames({"M", "K", "N"});

BENCHMARK(BM_LinearPreparedStreaming)
        ->Args({1, 4096, 4096})
        ->Args({1, 4096, 6144})
        ->Args({1, 4096, 11008})
        ->Args({1, 4096, 22016})
        ->Args({1, 11008, 4096})
        ->Args({1, 4096, 32000})
        ->Args({16, 4096, 4096})
        ->Args({64, 4096, 4096})
        ->Args({16, 4096, 22016})
        ->Args({16, 11008, 4096})
        ->ArgNames({"M", "K", "N"});

BENCHMARK(BM_LinearBindingSpecialization)
        ->Args({1, 4096, 4096})
        ->Args({1, 4096, 6144})
        ->Args({1, 4096, 11008})
        ->Args({1, 4096, 22016})
        ->Args({1, 11008, 4096})
        ->Args({1, 4096, 32000})
        ->Args({16, 4096, 4096})
        ->Args({64, 4096, 4096})
        ->Args({16, 4096, 22016})
        ->Args({16, 11008, 4096})
        ->Args({1, 33, 31})
        ->Args({2, 32, 33})
        ->ArgNames({"M", "K", "N"});

BENCHMARK_CAPTURE(BM_LinearW16PreparedHot, f16, DataType::Float(16))
        // Decode projections.
        ->Args({1, 4096, 4096})
        ->Args({1, 4096, 11008})
        ->Args({1, 11008, 4096})
        ->Args({1, 4096, 32000})
        // Prefill projections.
        ->Args({16, 4096, 4096})
        ->Args({64, 4096, 4096})
        // Small/tail shapes used to guard future tile-specific paths.
        ->Args({1, 33, 31})
        ->Args({2, 32, 33})
        ->ArgNames({"M", "K", "N"});

BENCHMARK_CAPTURE(BM_LinearW16PreparedHot, bf16, DataType::BFloat(16))
        ->Args({1, 4096, 4096})
        ->Args({1, 4096, 11008})
        ->Args({1, 11008, 4096})
        ->Args({1, 4096, 32000})
        ->Args({16, 4096, 4096})
        ->Args({64, 4096, 4096})
        ->Args({1, 33, 31})
        ->Args({2, 32, 33})
        ->ArgNames({"M", "K", "N"});

BENCHMARK_CAPTURE(BM_LinearW16PreparedStreaming, f16, DataType::Float(16))
        ->Args({1, 4096, 4096})
        ->Args({1, 4096, 11008})
        ->Args({1, 11008, 4096})
        ->Args({1, 4096, 32000})
        ->Args({16, 4096, 4096})
        ->Args({64, 4096, 4096})
        ->ArgNames({"M", "K", "N"});

BENCHMARK_CAPTURE(BM_LinearW16PreparedStreaming, bf16, DataType::BFloat(16))
        ->Args({1, 4096, 4096})
        ->Args({1, 4096, 11008})
        ->Args({1, 11008, 4096})
        ->Args({1, 4096, 32000})
        ->Args({16, 4096, 4096})
        ->Args({64, 4096, 4096})
        ->ArgNames({"M", "K", "N"});

BENCHMARK_CAPTURE(BM_LinearW16BindingSpecialization, f16, DataType::Float(16))
        ->Args({1, 4096, 4096})
        ->Args({1, 4096, 11008})
        ->Args({1, 11008, 4096})
        ->Args({1, 4096, 32000})
        ->Args({1, 33, 31})
        ->Args({2, 32, 33})
        ->ArgNames({"M", "K", "N"});

BENCHMARK_CAPTURE(BM_LinearW16BindingSpecialization, bf16, DataType::BFloat(16))
        ->Args({1, 4096, 4096})
        ->Args({1, 4096, 11008})
        ->Args({1, 11008, 4096})
        ->Args({1, 4096, 32000})
        ->Args({1, 33, 31})
        ->Args({2, 32, 33})
        ->ArgNames({"M", "K", "N"});

} // namespace
