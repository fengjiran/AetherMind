#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/operators/op_params.h"
#include "backend/cpu/kernels/gemm/gemm_internal.h"

#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

enum class LinearKernelSelection {
    kDefault,
    kScalarFallback,
};

struct PreparedLinearInvocation {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> params{};
};

KernelSelector MakePlainLinearSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PreparePlainLinearKernel(LinearKernelSelection selection) {
    CpuFeaturePolicy policy;
    if (selection == LinearKernelSelection::kScalarFallback) {
        policy.disabled_features = CpuFeatureSet::From({CpuFeature::kAvx2});
    }
    CpuBackend backend(policy);
    return backend.PrepareKernel(
            OpType::kLinear, MakePlainLinearSelector(), OpParams{LinearParams{}});
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
                                 const float* weight,
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
            TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
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
           (kernel.name == nullptr ? "<unnamed>" : kernel.name) +
           " cache_mode=" + mode_name;
}

void SetLinearCounters(benchmark::State& state,
                       int64_t m,
                       int64_t k,
                       int64_t n,
                       size_t weight_replica_count,
                       size_t weight_working_set_bytes,
                       bool includes_compute) {
    const int64_t output_elements = m * n;
    const int64_t logical_bytes =
            static_cast<int64_t>(sizeof(float)) * (m * k + n * k + output_elements);
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

void BenchmarkLinearPreparedPath(benchmark::State& state,
                                 LinearBenchmarkMode mode,
                                 LinearKernelSelection selection = LinearKernelSelection::kDefault) {
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

    const StatusOr<ResolvedKernel> resolved = PreparePlainLinearKernel(selection);
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
                output.data(), m, k, n, prepared[replica]);
        if (!status.ok()) {
            state.SkipWithError(status.ToString());
            return;
        }
    }

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
    constexpr float kErrorTolerance = 2.0e-4F;
#elif defined(AETHERMIND_ENABLE_GEMM_SCALAR_CANDIDATE)
    constexpr float kErrorTolerance = 1.0e-4F;
#else
    constexpr float kErrorTolerance = 1.0e-5F;
#endif
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
                    output.data(), m, k, n, prepared[replica]);
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

#if defined(AETHERMIND_ENABLE_GEMM_SCALAR_CANDIDATE) && !defined(GEMM_HAS_AVX2_FMA_KERNEL)
void BM_LinearPreparedScalarCandidateHot(benchmark::State& state) {
    BenchmarkLinearPreparedPath(state, LinearBenchmarkMode::kPreparedHot);
}

void BM_LinearPreparedScalarCandidateStreaming(benchmark::State& state) {
    BenchmarkLinearPreparedPath(state, LinearBenchmarkMode::kPreparedStreaming);
}
#endif

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
void BM_LinearPreparedAvx2CandidateHot(benchmark::State& state) {
    BenchmarkLinearPreparedPath(state, LinearBenchmarkMode::kPreparedHot);
}

void BM_LinearPreparedAvx2CandidateStreaming(benchmark::State& state) {
    BenchmarkLinearPreparedPath(state, LinearBenchmarkMode::kPreparedStreaming);
}

void BM_LinearPreparedScalarFallbackHot(benchmark::State& state) {
    BenchmarkLinearPreparedPath(
            state, LinearBenchmarkMode::kPreparedHot, LinearKernelSelection::kScalarFallback);
}

void BM_LinearPreparedScalarFallbackStreaming(benchmark::State& state) {
    BenchmarkLinearPreparedPath(
            state, LinearBenchmarkMode::kPreparedStreaming, LinearKernelSelection::kScalarFallback);
}
#endif

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

#if defined(AETHERMIND_ENABLE_GEMM_SCALAR_CANDIDATE) && !defined(GEMM_HAS_AVX2_FMA_KERNEL)
BENCHMARK(BM_LinearPreparedScalarCandidateHot)
        // This resolves the opt-in descriptor and invokes ResolvedKernel::fn.
        ->Args({1, 4096, 4096})
        ->Args({1, 4096, 6144})
        ->Args({1, 4096, 11008})
        ->Args({1, 4096, 22016})
        ->Args({1, 11008, 4096})
        ->Args({1, 4096, 32000})
        ->Args({1, 33, 31})
        ->Args({1, 32, 33})
        ->ArgNames({"M", "K", "N"});

BENCHMARK(BM_LinearPreparedScalarCandidateStreaming)
        ->Args({1, 4096, 4096})
        ->Args({1, 4096, 6144})
        ->Args({1, 4096, 11008})
        ->Args({1, 4096, 22016})
        ->Args({1, 11008, 4096})
        ->Args({1, 4096, 32000})
        ->ArgNames({"M", "K", "N"});
#endif

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
using LinearShape = std::array<int64_t, 3>;

constexpr std::array<LinearShape, 6> kAvx2DecodeShapes{{
        {1, 4096, 4096},
        {1, 4096, 6144},
        {1, 4096, 11008},
        {1, 4096, 22016},
        {1, 11008, 4096},
        {1, 4096, 32000},
}};

benchmark::Benchmark* RegisterAvx2DecodeShapes(benchmark::Benchmark* benchmark) {
    for (const auto& shape: kAvx2DecodeShapes) {
        benchmark->Args({shape[0], shape[1], shape[2]});
    }
    return benchmark->ArgNames({"M", "K", "N"});
}

const auto* const kLinearPreparedAvx2CandidateHot = RegisterAvx2DecodeShapes(
        benchmark::RegisterBenchmark("BM_LinearPreparedAvx2CandidateHot",
                                     &BM_LinearPreparedAvx2CandidateHot));
const auto* const kLinearPreparedAvx2CandidateStreaming = RegisterAvx2DecodeShapes(
        benchmark::RegisterBenchmark("BM_LinearPreparedAvx2CandidateStreaming",
                                     &BM_LinearPreparedAvx2CandidateStreaming));
const auto* const kLinearPreparedScalarFallbackHot = RegisterAvx2DecodeShapes(
        benchmark::RegisterBenchmark("BM_LinearPreparedScalarFallbackHot",
                                     &BM_LinearPreparedScalarFallbackHot));
const auto* const kLinearPreparedScalarFallbackStreaming = RegisterAvx2DecodeShapes(
        benchmark::RegisterBenchmark("BM_LinearPreparedScalarFallbackStreaming",
                                     &BM_LinearPreparedScalarFallbackStreaming));
#endif

} // namespace
