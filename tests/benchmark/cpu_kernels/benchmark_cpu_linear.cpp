#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/memory/cpu_allocator.h"
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
#include <memory>
#include <optional>
#include <span>
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
            .output_dtype = DataType::Float32(),
            .weight_dtype = weight_dtype,
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PrepareLinearKernel(DataType weight_dtype = DataType::Float32(), WeightFormat format = WeightFormat::kPlain) {
    CpuBackend backend;
    auto selector = MakePlainLinearSelector(weight_dtype);
    selector.weight_format = format;
    return backend.PrepareKernel(OpType::kLinear, selector, OpParams{LinearParams{}});
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
                                 const void* input,
                                 const void* weight,
                                 DataType weight_dtype,
                                 void* output,
                                 int64_t m,
                                 int64_t k,
                                 int64_t n,
                                 PreparedLinearInvocation& prepared,
                                 const PackedWeight* artifact = nullptr,
                                 DataType act_dtype = DataType::Float32(),
                                 DataType output_dtype = DataType::Float32()) noexcept {
    const std::array<int64_t, 2> input_shape = {m, k};
    const std::array<int64_t, 2> input_strides = {k, 1};
    const std::array<int64_t, 2> weight_shape = {n, k};
    const std::array<int64_t, 2> weight_strides = {k, 1};
    const std::array<int64_t, 2> output_shape = {m, n};
    const std::array<int64_t, 2> output_strides = {n, 1};
    const std::array<TensorView, 2> inputs{
            TensorView{input, act_dtype, input_shape, input_strides},
            TensorView{weight, weight_dtype, weight_shape, weight_strides},
    };
    const std::array<MutableTensorView, 1> outputs{
            MutableTensorView{output, output_dtype, output_shape, output_strides},
    };
    std::optional<PackedWeightView> packed_view;
    if (artifact != nullptr) {
        packed_view = PackedWeightView{
                .data = artifact->storage().data(),
                .nbytes = artifact->storage().nbytes(),
                .logical_dtype = artifact->logical_dtype(),
                .logical_shape = artifact->logical_shape(),
                .recipe_layout = artifact->recipe().layout,
                .recipe_alignment = artifact->recipe().alignment,
                .alignment = artifact->storage().alignment(),
        };
    }
    return kernel.params_builder(KernelParamsBuildContext{
                                         .inputs = artifact == nullptr ? std::span<const TensorView>(inputs) : std::span<const TensorView>(inputs.data(), 1),
                                         .outputs = outputs,
                                         .attrs = kernel.attrs,
                                         .packed_weight = packed_view,
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
    AM_RETURN_IF_ERROR(cpu::detail::RunGemmF32Reference(cpu::detail::GemmArgs{
            .lhs = reinterpret_cast<const std::byte*>(input),
            .rhs = reinterpret_cast<const std::byte*>(weight),
            .output = reinterpret_cast<std::byte*>(expected.data()),
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
                       size_t weight_element_size = sizeof(float),
                       size_t input_element_size = sizeof(float),
                       size_t output_element_size = sizeof(float)) {
    const int64_t output_elements = m * n;
    const int64_t logical_bytes =
            static_cast<int64_t>(input_element_size) * m * k +
            static_cast<int64_t>(output_element_size) * output_elements +
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

void BenchmarkLinearPreparedPath(benchmark::State& state, LinearBenchmarkMode mode, bool packed_identity = false) {
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

    const StatusOr<ResolvedKernel> resolved = PrepareLinearKernel(DataType::Float32(), packed_identity ? WeightFormat::kPacked : WeightFormat::kPlain);
    if (!resolved.ok()) {
        state.SkipWithError(resolved.status().ToString());
        return;
    }

    if (resolved->params_builder == nullptr || resolved->fn == nullptr) {
        state.SkipWithError(
                "Prepared Linear benchmark requires params_builder and kernel function");
        return;
    }

    CpuBackend pack_backend;
    CPUAllocator allocator(Device::CPU());
    std::vector<std::unique_ptr<PackedWeight>> artifacts(replica_count);
    std::vector<PreparedLinearInvocation> prepared(replica_count);
    for (size_t replica = 0; replica < replica_count; ++replica) {
        if (packed_identity) {
            const std::array<int64_t, 2> shape{n, k};
            const std::array<int64_t, 2> strides{k, 1};
            const std::array<TensorView, 1> components{TensorView(weights.data() + replica * weight_elements, DataType::Float32(), shape, strides)};
            auto selector = MakePlainLinearSelector();
            selector.weight_format = WeightFormat::kPacked;
            auto artifact = pack_backend.PackWeights(OpType::kLinear, components, selector,
                                                     resolved->expected_packing_recipe, allocator);
            if (!artifact.ok()) {
                state.SkipWithError(artifact.status().ToString());
                return;
            }
            artifacts[replica] = std::move(*artifact);
        }
        const Status status = BuildLinearPreparedParams(
                *resolved, input.data(), weights.data() + replica * weight_elements,
                DataType::Float32(), output.data(), m, k, n, prepared[replica], artifacts[replica].get());
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
                    DataType::Float32(), output.data(), m, k, n, prepared[replica], artifacts[replica].get());
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
                                            void* output,
                                            int64_t m,
                                            int64_t k,
                                            int64_t n,
                                            DataType dtype,
                                            DataType output_dtype = DataType::Float32()) {
    const double unit_roundoff =
            static_cast<double>(std::numeric_limits<float>::epsilon()) / 2.0;
    const double gamma = (2.0 * static_cast<double>(k) * unit_roundoff) /
                         (1.0 - 2.0 * static_cast<double>(k) * unit_roundoff);
    const auto output_elements = static_cast<size_t>(m * n);
    // Cold oracle storage is independent of the implementation and is never
    // passed to a kernel. Decode each weight once, outside the measured loop.
    std::vector<float> decoded_weights(static_cast<size_t>(n * k));
    for (size_t i = 0; i < decoded_weights.size(); ++i) {
        uint16_t bits;
        std::memcpy(&bits, weight + i * sizeof(bits), sizeof(bits));
        decoded_weights[i] = WidenFromBits(bits, dtype);
    }
    std::vector<float> expected(output_elements);
    std::vector<double> bounds(output_elements);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            double sum = 0.0;
            double absolute_products = 0.0;
            for (int64_t t = 0; t < k; ++t) {
                const double product = static_cast<double>(input[row * k + t]) *
                                       static_cast<double>(decoded_weights[static_cast<size_t>(col * k + t)]);
                sum += product;
                absolute_products += std::abs(product);
            }
            const size_t index = static_cast<size_t>(row * n + col);
            expected[index] = static_cast<float>(sum);
            bounds[index] = gamma * absolute_products + 1.0e-6;
        }
    }

    std::memset(output, 0xff, output_elements * static_cast<size_t>(output_dtype.nbytes()));
    AM_RETURN_IF_ERROR(InvokePreparedLinear(kernel, prepared));
    for (size_t index = 0; index < output_elements; ++index) {
        float actual;
        double lower = static_cast<double>(expected[index]) - bounds[index];
        double upper = static_cast<double>(expected[index]) + bounds[index];
        if (output_dtype == DataType::Float32()) {
            actual = static_cast<const float*>(output)[index];
        } else {
            actual = WidenFromBits(static_cast<const uint16_t*>(output)[index], output_dtype);
            lower = WidenFromBits(NarrowToBits(static_cast<float>(lower), output_dtype), output_dtype);
            upper = WidenFromBits(NarrowToBits(static_cast<float>(upper), output_dtype), output_dtype);
        }
        if (!std::isfinite(actual)) {
            return Status::Internal(
                    "W16 Linear benchmark correctness guard produced NaN or Inf");
        }
        if (actual < lower || actual > upper) {
            return Status::Internal(
                    "W16 Linear benchmark correctness guard exceeds the double-oracle bound");
        }
    }
    return Status::Ok();
}

void BM_LinearA16PreparedHot(benchmark::State& state, DataType dtype, bool fp32_output) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);
    const DataType output_dtype = fp32_output ? DataType::Float32() : dtype;
    std::vector<float> decoded_input(static_cast<size_t>(m * k));
    std::vector<uint16_t> input(decoded_input.size());
    std::vector<uint16_t> weight(static_cast<size_t>(n * k));
    for (size_t index = 0; index < input.size(); ++index) {
        input[index] = NarrowToBits(DeterministicValue(index), dtype);
        decoded_input[index] = WidenFromBits(input[index], dtype);
    }
    for (size_t index = 0; index < weight.size(); ++index) {
        weight[index] = NarrowToBits(DeterministicValue(index + 13), dtype);
    }
    const auto output_elements = static_cast<size_t>(m * n);
    std::vector<float> output32(fp32_output ? output_elements : 0);
    std::vector<uint16_t> output16(fp32_output ? 0 : output_elements);
    void* output = fp32_output ? static_cast<void*>(output32.data()) : output16.data();
    auto selector = MakePlainLinearSelector(dtype);
    selector.act_dtype = dtype;
    selector.output_dtype = output_dtype;
    selector.phase = m == 1 ? ExecPhase::kDecode : ExecPhase::kPrefill;
    CpuBackend backend;
    const auto kernel = backend.PrepareKernel(
            OpType::kLinear, selector, OpParams{LinearParams{.output_dtype = output_dtype}});
    if (!kernel.ok()) {
        state.SkipWithError(kernel.status().ToString());
        return;
    }
    PreparedLinearInvocation prepared;
    const Status built = BuildLinearPreparedParams(
            *kernel, input.data(), weight.data(), dtype, output, m, k, n, prepared,
            nullptr, dtype, output_dtype);
    if (!built.ok()) {
        state.SkipWithError(built.ToString());
        return;
    }
    const Status correctness = ValidateLinearW16AgainstDoubleOracle(
            *kernel, prepared, decoded_input.data(), reinterpret_cast<const std::byte*>(weight.data()),
            output, m, k, n, dtype, output_dtype);
    if (!correctness.ok()) {
        state.SkipWithError(correctness.ToString());
        return;
    }
    state.SetLabel(BenchmarkLabel(*kernel, LinearBenchmarkMode::kPreparedHot));
    for (auto _: state) {
        const Status status = InvokePreparedLinear(*kernel, prepared);
        if (!status.ok()) {
            state.SkipWithError(status.ToString());
            break;
        }
        benchmark::DoNotOptimize(output);
    }
    SetLinearCounters(state, m, k, n, 1, weight.size() * sizeof(uint16_t),
                      true, sizeof(uint16_t), sizeof(uint16_t), output_dtype.nbytes());
}

void A16Cases(benchmark::internal::Benchmark* benchmark) {
    benchmark->Args({1, 4096, 4096})
            ->Args({8, 512, 512})
            ->Args({9, 513, 513})
            ->Args({16, 4096, 4096})
            ->Args({64, 4096, 4096})
            ->Args({49, 513, 513})
            ->Args({128, 16, 32000})
            ->ArgNames({"M", "K", "N"});
}

BENCHMARK_CAPTURE(BM_LinearA16PreparedHot, f16, DataType::Float(16), false)->Apply(A16Cases);
BENCHMARK_CAPTURE(BM_LinearA16PreparedHot, bf16, DataType::BFloat(16), false)->Apply(A16Cases);
BENCHMARK_CAPTURE(BM_LinearA16PreparedHot, f16_out32, DataType::Float(16), true)->Apply(A16Cases);
BENCHMARK_CAPTURE(BM_LinearA16PreparedHot, bf16_out32, DataType::BFloat(16), true)->Apply(A16Cases);

void BenchmarkLinearW16PreparedPath(benchmark::State& state,
                                    LinearBenchmarkMode mode,
                                    DataType weight_dtype, bool packed_identity = false) {
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

    const StatusOr<ResolvedKernel> resolved = PrepareLinearKernel(weight_dtype, packed_identity ? WeightFormat::kPacked : WeightFormat::kPlain);
    if (!resolved.ok()) {
        state.SkipWithError(resolved.status().ToString());
        return;
    }
    if (resolved->params_builder == nullptr || resolved->fn == nullptr) {
        state.SkipWithError(
                "Prepared W16 Linear benchmark requires params_builder and kernel function");
        return;
    }

    CpuBackend pack_backend;
    CPUAllocator allocator(Device::CPU());
    std::vector<std::unique_ptr<PackedWeight>> artifacts(replica_count);
    std::vector<PreparedLinearInvocation> prepared(replica_count);
    for (size_t replica = 0; replica < replica_count; ++replica) {
        if (packed_identity) {
            const std::array<int64_t, 2> shape{n, k};
            const std::array<int64_t, 2> strides{k, 1};
            const std::array<TensorView, 1> components{TensorView(weights.data() + replica * weight_bytes, weight_dtype, shape, strides)};
            auto selector = MakePlainLinearSelector(weight_dtype);
            selector.weight_format = WeightFormat::kPacked;
            auto artifact = pack_backend.PackWeights(OpType::kLinear, components, selector,
                                                     resolved->expected_packing_recipe, allocator);
            if (!artifact.ok()) {
                state.SkipWithError(artifact.status().ToString());
                return;
            }
            artifacts[replica] = std::move(*artifact);
        }
        const Status status = BuildLinearPreparedParams(
                *resolved, input.data(), weights.data() + replica * weight_bytes,
                weight_dtype, output.data(), m, k, n, prepared[replica], artifacts[replica].get());
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

    state.SetLabel(BenchmarkLabel(*resolved, mode) + " recipe=" + std::string(ToString(resolved->expected_packing_recipe.layout)));
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
                    weight_dtype, output.data(), m, k, n, prepared[replica], artifacts[replica].get());
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

void BM_LinearW16PackedIdentityHot(benchmark::State& state, DataType dtype) {
    BenchmarkLinearW16PreparedPath(state, LinearBenchmarkMode::kPreparedHot, dtype, true);
}
void BM_LinearW16PackedIdentityStreaming(benchmark::State& state, DataType dtype) {
    BenchmarkLinearW16PreparedPath(state, LinearBenchmarkMode::kPreparedStreaming, dtype, true);
}


void BM_LinearPackedIdentityHot(benchmark::State& state) {
    BenchmarkLinearPreparedPath(state, LinearBenchmarkMode::kPreparedHot, true);
}
void BM_LinearPackedIdentityStreaming(benchmark::State& state) {
    BenchmarkLinearPreparedPath(state, LinearBenchmarkMode::kPreparedStreaming, true);
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
        ->Args({2, 512, 512})
        ->Args({4, 512, 512})
        ->Args({8, 512, 512})
        ->Args({32, 512, 512})
        ->Args({128, 512, 512})
        ->Args({512, 512, 512})
        ->Args({1, 1, 3})
        ->Args({1, 7, 5})
        ->Args({8, 7, 19})
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
        ->Args({48, 512, 512})
        ->Args({53, 521, 517})
        ->Args({128, 1536, 2048})
        ->Args({512, 16, 4096})
        // Small/tail shapes used to guard future tile-specific paths.
        ->Args({1, 33, 31})
        ->Args({2, 32, 33})
        ->Args({2, 512, 512})
        ->Args({4, 512, 512})
        ->Args({8, 512, 512})
        ->Args({32, 512, 512})
        ->Args({128, 512, 512})
        ->Args({512, 512, 512})
        ->Args({1, 1, 3})
        ->Args({1, 7, 5})
        ->Args({8, 7, 19})
        ->ArgNames({"M", "K", "N"});

BENCHMARK_CAPTURE(BM_LinearW16PreparedHot, bf16, DataType::BFloat(16))
        ->Args({1, 4096, 4096})
        ->Args({1, 4096, 11008})
        ->Args({1, 11008, 4096})
        ->Args({1, 4096, 32000})
        ->Args({16, 4096, 4096})
        ->Args({64, 4096, 4096})
        ->Args({48, 512, 512})
        ->Args({53, 521, 517})
        ->Args({128, 1536, 2048})
        ->Args({512, 16, 4096})
        ->Args({1, 33, 31})
        ->Args({2, 32, 33})
        ->Args({2, 512, 512})
        ->Args({4, 512, 512})
        ->Args({8, 512, 512})
        ->Args({32, 512, 512})
        ->Args({128, 512, 512})
        ->Args({512, 512, 512})
        ->Args({1, 1, 3})
        ->Args({1, 7, 5})
        ->Args({8, 7, 19})
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

BENCHMARK_CAPTURE(BM_LinearW16PackedIdentityHot, f16, DataType::Float(16))
        ->Args({1, 4096, 4096})
        ->Args({16, 4096, 4096})
        ->Args({64, 4096, 4096})
        ->Args({1, 33, 31})
        ->Args({1, 1, 3})
        ->Args({1, 7, 5})
        ->Args({8, 7, 19})
        ->ArgNames({"M", "K", "N"});

BENCHMARK_CAPTURE(BM_LinearW16PackedIdentityHot, bf16, DataType::BFloat(16))
        ->Args({1, 4096, 4096})
        ->Args({16, 4096, 4096})
        ->Args({64, 4096, 4096})
        ->Args({1, 33, 31})
        ->Args({1, 1, 3})
        ->Args({1, 7, 5})
        ->Args({8, 7, 19})
        ->ArgNames({"M", "K", "N"});

BENCHMARK_CAPTURE(BM_LinearW16PackedIdentityStreaming, f16, DataType::Float(16))
        ->Args({1, 4096, 4096})
        ->Args({16, 4096, 4096})
        ->Args({64, 4096, 4096})
        ->Args({1, 33, 31})
        ->ArgNames({"M", "K", "N"});

BENCHMARK_CAPTURE(BM_LinearW16PackedIdentityStreaming, bf16, DataType::BFloat(16))
        ->Args({1, 4096, 4096})
        ->Args({16, 4096, 4096})
        ->Args({64, 4096, 4096})
        ->Args({1, 33, 31})
        ->ArgNames({"M", "K", "N"});

BENCHMARK(BM_LinearPackedIdentityHot)
        ->Args({1, 4096, 4096})
        ->Args({16, 4096, 4096})
        ->Args({64, 4096, 4096})
        ->Args({1, 1, 3})
        ->Args({1, 7, 5})
        ->Args({8, 7, 19})
        ->ArgNames({"M", "K", "N"});

BENCHMARK(BM_LinearPackedIdentityStreaming)
        ->Args({1, 4096, 4096})
        ->Args({16, 4096, 4096})
        ->Args({64, 4096, 4096})
        ->ArgNames({"M", "K", "N"});

} // namespace
