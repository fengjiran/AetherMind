#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/backend/kernel_context.h"
#include "backend/cpu/cpu_backend_internal.h"
#include "backend/cpu/kernels/gemm/gemm_internal.h"

#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace aethermind;

constexpr size_t kStreamingTargetBytes = 256U * 1024U * 1024U;
constexpr size_t kMaxStreamingReplicas = 8;

struct PreparedInvocation {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> params{};
};

enum class CacheMode {
    kHot,
    kStreaming,
};

float DeterministicValue(size_t index) {
    constexpr size_t kPeriod = 251;
    return static_cast<float>(static_cast<int64_t>(index % kPeriod) -
                              static_cast<int64_t>(kPeriod / 2U)) *
           0.0078125F;
}

void Fill(std::vector<float>& values, size_t offset = 0) {
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = DeterministicValue(i + offset);
    }
}

size_t StreamingReplicas(size_t bytes) {
    if (bytes == 0) return 1;
    const size_t count = (kStreamingTargetBytes + bytes - 1U) / bytes;
    return std::clamp(count, size_t{1}, kMaxStreamingReplicas);
}

StatusOr<const KernelDef*> FindCandidateDescriptor(
        KernelRegistry& registry) {
    AM_RETURN_IF_ERROR(registry.Freeze());
    AM_ASSIGN_OR_RETURN(const auto descriptors,
                        registry.FindByOpType(OpType::kLinear));
    const auto candidate = std::find_if(
            descriptors.begin(), descriptors.end(), [](const KernelDef* d) {
                return d->name == "cpu::linear_f32_packed_bpanel_candidate";
            });
    if (candidate == descriptors.end()) {
        return Status::NotFound("Linear packed-B candidate descriptor is missing");
    }
    return *candidate;
}

Status BuildPackedParams(const ResolvedKernel& kernel,
                         const TensorView& input,
                         const MutableTensorView& output,
                         const PackedWeights& weights,
                         PreparedInvocation& prepared) noexcept {
    const PackedWeightView packed{
            .data = weights.storage().data(),
            .nbytes = weights.storage().nbytes(),
            .logical_dtype = weights.logical_dtype(),
            .logical_shape = weights.logical_shape(),
            .recipe_layout = weights.recipe().layout,
            .recipe_alignment = weights.recipe().alignment,
            .alignment = weights.storage().alignment(),
    };
    const std::array<TensorView, 1> inputs{input};
    const std::array<MutableTensorView, 1> outputs{output};
    return kernel.params_builder(
            KernelParamsBuildContext{
                    .inputs = inputs,
                    .outputs = outputs,
                    .attrs = kernel.attrs,
                    .packed_weight = packed,
            },
            prepared.params.data());
}

Status RunPrepared(const ResolvedKernel& kernel,
                   const PreparedInvocation& prepared) noexcept {
    return kernel.fn(KernelContext{
            .device_type = DeviceType::kCPU,
            .kernel_params = prepared.params.data(),
            .attrs = kernel.attrs,
    });
}

Status BuildPlainParams(const ResolvedKernel& kernel,
                        const TensorView& input,
                        const TensorView& weight,
                        const MutableTensorView& output,
                        PreparedInvocation& prepared) noexcept {
    const std::array<TensorView, 2> inputs{input, weight};
    const std::array<MutableTensorView, 1> outputs{output};
    return kernel.params_builder(
            KernelParamsBuildContext{
                    .inputs = inputs,
                    .outputs = outputs,
                    .attrs = kernel.attrs,
            },
            prepared.params.data());
}

Status CheckAgainstReference(const ResolvedKernel& kernel,
                             const PreparedInvocation& prepared,
                             const std::vector<float>& input,
                             const std::vector<float>& weights,
                             std::vector<float>& output,
                             int64_t m,
                             int64_t k,
                             int64_t n) {
    std::vector<float> expected(static_cast<size_t>(m * n),
                                std::numeric_limits<float>::quiet_NaN());
    AM_RETURN_IF_ERROR(cpu::detail::RunGemmF32Reference(cpu::detail::GemmF32Args{
            .lhs = input.data(),
            .rhs = weights.data(),
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
    AM_RETURN_IF_ERROR(RunPrepared(kernel, prepared));
    for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i]) || !std::isfinite(expected[i])) {
            return Status::Internal("packed Linear benchmark produced NaN or Inf");
        }
        const float abs_error = std::fabs(output[i] - expected[i]);
        const float rel_error = abs_error / std::max(std::fabs(expected[i]), 1.0e-6F);
        if (abs_error > 2.0e-4F && rel_error > 2.0e-4F) {
            return Status::Internal("packed Linear benchmark disagrees with reference");
        }
    }
    return Status::Ok();
}

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
void BenchmarkPackedLinear(benchmark::State& state, CacheMode mode) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);
    const size_t weight_elements = static_cast<size_t>(n * k);
    const size_t logical_weight_bytes = weight_elements * sizeof(float);
    const size_t replica_count = mode == CacheMode::kStreaming
                                         ? StreamingReplicas(logical_weight_bytes)
                                         : 1U;
    const auto capabilities = cpu::DetectCpuCapabilities();
    if (!capabilities.ok() ||
        !capabilities->effective_features.Contains(CpuFeature::kAvx2) ||
        !capabilities->effective_features.Contains(CpuFeature::kFma)) {
        state.SkipWithError("packed Linear candidate requires effective AVX2+FMA");
        return;
    }

    KernelRegistry& global_registry = KernelRegistry::Global();
    const auto registered = FindCandidateDescriptor(global_registry);
    if (!registered.ok()) {
        state.SkipWithError(registered.status().ToString());
        return;
    }
    KernelRegistry candidate_registry;
    const Status registration = candidate_registry.Register(**registered);
    if (!registration.ok()) {
        state.SkipWithError(registration.ToString());
        return;
    }
    if (const Status freeze = candidate_registry.Freeze(); !freeze.ok()) {
        state.SkipWithError(freeze.ToString());
        return;
    }
    const KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .phase = ExecPhase::kBoth,
    };
    const auto resolved_descriptor = cpu::internal::ResolveEligibleDescriptor(
            candidate_registry, OpType::kLinear, selector,
            capabilities->effective_features);
    if (!resolved_descriptor.ok()) {
        state.SkipWithError(resolved_descriptor.status().ToString());
        return;
    }
    const auto recipe = cpu::internal::ResolvePackingRecipeFromRegistry(
            candidate_registry, OpType::kLinear, selector,
            capabilities->effective_features);
    if (!recipe.ok()) {
        state.SkipWithError(recipe.status().ToString());
        return;
    }
    ResolvedKernel kernel{
            .op_type = OpType::kLinear,
            .fn = (*resolved_descriptor)->kernel_func,
            .attrs = {},
            .name = (*resolved_descriptor)->name.c_str(),
            .params_builder = (*resolved_descriptor)->params_builder,
            .params_size = (*resolved_descriptor)->params_size,
            .workspace_requirement = {},
            .expected_packing_recipe = *recipe,
    };
    CpuBackend pack_backend(*capabilities);

    std::vector<float> input(static_cast<size_t>(m * k));
    std::vector<float> output(static_cast<size_t>(m * n),
                              std::numeric_limits<float>::quiet_NaN());
    Fill(input);
    const std::array<int64_t, 2> input_shape{m, k};
    const std::array<int64_t, 2> input_strides{k, 1};
    const std::array<int64_t, 2> weight_shape{n, k};
    const std::array<int64_t, 2> weight_strides{k, 1};
    const std::array<int64_t, 2> output_shape{m, n};
    const std::array<int64_t, 2> output_strides{n, 1};
    const TensorView input_view(
            input.data(), DataType::Float32(), input_shape, input_strides);
    const MutableTensorView output_view(
            output.data(), DataType::Float32(), output_shape, output_strides);

    CpuWeightPrepacker prepacker;
    std::vector<std::unique_ptr<PackedWeights>> artifacts;
    std::vector<PreparedInvocation> prepared(replica_count);
    std::vector<float> first_logical_weights;
    std::chrono::nanoseconds pack_time{};
    artifacts.reserve(replica_count);
    for (size_t replica = 0; replica < replica_count; ++replica) {
        std::vector<float> logical_weights(weight_elements);
        Fill(logical_weights, replica * 104729U);
        if (replica == 0) first_logical_weights = logical_weights;
        const TensorView logical_view(
                logical_weights.data(), DataType::Float32(),
                weight_shape, weight_strides);
        const std::array<TensorView, 1> components{logical_view};
        const auto pack_begin = std::chrono::steady_clock::now();
        auto packed = prepacker.Pack(
                OpType::kLinear, components, selector, *recipe);
        pack_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - pack_begin);
        if (!packed.ok()) {
            state.SkipWithError(packed.status().ToString());
            return;
        }
        if ((*packed)->storage().nbytes() !=
            *cpu::CpuBPanelF32V1PackedByteSize(n, k)) {
            state.SkipWithError("packed Linear artifact size disagrees with recipe");
            return;
        }
        artifacts.push_back(std::move(*packed));
        const Status build = BuildPackedParams(
                kernel, input_view, output_view, *artifacts.back(), prepared[replica]);
        if (!build.ok()) {
            state.SkipWithError(build.ToString());
            return;
        }
    }

    const Status correctness = CheckAgainstReference(
            kernel, prepared.front(), input, first_logical_weights,
            output, m, k, n);
    if (!correctness.ok()) {
        state.SkipWithError(correctness.ToString());
        return;
    }
    const KernelSelector plain_selector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
    const auto plain_kernel = pack_backend.PrepareKernel(
            OpType::kLinear, plain_selector, OpParams{LinearParams{}});
    if (!plain_kernel.ok()) {
        state.SkipWithError(plain_kernel.status().ToString());
        return;
    }
    const TensorView first_weight_view(
            first_logical_weights.data(), DataType::Float32(),
            weight_shape, weight_strides);
    PreparedInvocation plain_prepared;
    const Status plain_build = BuildPlainParams(
            *plain_kernel, input_view, first_weight_view, output_view,
            plain_prepared);
    if (!plain_build.ok()) {
        state.SkipWithError(plain_build.ToString());
        return;
    }
    const Status plain_correctness = CheckAgainstReference(
            *plain_kernel, plain_prepared, input, first_logical_weights,
            output, m, k, n);
    if (!plain_correctness.ok()) {
        state.SkipWithError(plain_correctness.ToString());
        return;
    }

    // A short interleaved setup sample supplies only a local break-even
    // estimate. The benchmark repetitions/raw data remain the performance gate.
    constexpr size_t kBreakEvenSamples = 8;
    std::array<double, kBreakEvenSamples> plain_ns{};
    std::array<double, kBreakEvenSamples> packed_ns{};
    for (size_t sample = 0; sample < kBreakEvenSamples; ++sample) {
        const auto plain_begin = std::chrono::steady_clock::now();
        const Status plain_status = RunPrepared(*plain_kernel, plain_prepared);
        plain_ns[sample] = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - plain_begin)
                        .count());
        if (!plain_status.ok()) {
            state.SkipWithError(plain_status.ToString());
            return;
        }
        const auto packed_begin = std::chrono::steady_clock::now();
        const Status packed_status = RunPrepared(kernel, prepared.front());
        packed_ns[sample] = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - packed_begin)
                        .count());
        if (!packed_status.ok()) {
            state.SkipWithError(packed_status.ToString());
            return;
        }
    }
    std::sort(plain_ns.begin(), plain_ns.end());
    std::sort(packed_ns.begin(), packed_ns.end());
    const double plain_median_ns = (plain_ns[3] + plain_ns[4]) * 0.5;
    const double packed_median_ns = (packed_ns[3] + packed_ns[4]) * 0.5;
    const double pack_ns_per_artifact = static_cast<double>(pack_time.count()) /
                                        static_cast<double>(replica_count);
    const double break_even_calls = plain_median_ns > packed_median_ns
                                            ? pack_ns_per_artifact /
                                                      (plain_median_ns - packed_median_ns)
                                            : -1.0;

    const double packed_bytes =
            static_cast<double>(artifacts.front()->storage().nbytes());
    const double size_amplification = packed_bytes /
                                      static_cast<double>(logical_weight_bytes);
    const double avg_pack_ns = pack_ns_per_artifact;
    state.SetLabel(std::string{"kernel="} + kernel.name +
                   (mode == CacheMode::kHot ? " cache_mode=hot" : " cache_mode=streaming"));
    state.counters["weight replicas"] = static_cast<double>(replica_count);
    state.counters["weight working set MiB"] =
            packed_bytes * static_cast<double>(replica_count) / (1024.0 * 1024.0);
    state.counters["packed size amplification"] = size_amplification;
    state.counters["packing ns per artifact"] = avg_pack_ns;
    state.counters["hot break-even estimate calls"] = break_even_calls;
    size_t iteration = 0;
    for (auto _: state) {
        const size_t replica = mode == CacheMode::kStreaming
                                       ? iteration % replica_count
                                       : 0U;
        const Status run = RunPrepared(kernel, prepared[replica]);
        if (!run.ok()) {
            state.SkipWithError(run.ToString());
            break;
        }
        benchmark::DoNotOptimize(output.data());
        ++iteration;
    }
    const int64_t logical_bytes = static_cast<int64_t>(sizeof(float)) *
                                  (m * k + n * k + m * n);
    state.SetItemsProcessed(state.iterations() * m * n);
    state.SetBytesProcessed(state.iterations() * logical_bytes);
    const double flops = static_cast<double>(state.iterations()) *
                         2.0 * static_cast<double>(m) * n * k;
    state.counters["GFLOP/s"] = benchmark::Counter(
            flops, benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
}

void BM_LinearPackedBpanelHot(benchmark::State& state) {
    BenchmarkPackedLinear(state, CacheMode::kHot);
}

void BM_LinearPackedBpanelStreaming(benchmark::State& state) {
    BenchmarkPackedLinear(state, CacheMode::kStreaming);
}

void BM_WeightPackingCpuBpanel(benchmark::State& state) {
    const int64_t n = state.range(0);
    const int64_t k = state.range(1);
    const size_t elements = static_cast<size_t>(n * k);
    const size_t logical_bytes = elements * sizeof(float);
    std::vector<float> logical(elements);
    Fill(logical);
    const std::array<int64_t, 2> shape{n, k};
    const std::array<int64_t, 2> strides{k, 1};
    const TensorView view(logical.data(), DataType::Float32(), shape, strides);
    const std::array<TensorView, 1> components{view};
    const KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .phase = ExecPhase::kBoth,
    };
    CpuWeightPrepacker prepacker;
    const PackingRecipe recipe = cpu::CpuBPanelF32V1Avx2Recipe();
    auto checked = prepacker.Pack(OpType::kLinear, components, selector, recipe);
    if (!checked.ok()) {
        state.SkipWithError(checked.status().ToString());
        return;
    }
    if ((*checked)->storage().nbytes() != *cpu::CpuBPanelF32V1PackedByteSize(n, k)) {
        state.SkipWithError("packed size amplification correctness guard failed");
        return;
    }
    const size_t packed_bytes = (*checked)->storage().nbytes();
    state.SetLabel(std::string{"recipe="} + recipe.layout + " mode=cold-packing");
    state.counters["size amplification"] =
            static_cast<double>(packed_bytes) / static_cast<double>(logical_bytes);
    state.counters["logical bytes"] = static_cast<double>(logical_bytes);
    state.counters["packed bytes"] = static_cast<double>(packed_bytes);
    for (auto _: state) {
        auto packed = prepacker.Pack(OpType::kLinear, components, selector, recipe);
        if (!packed.ok()) {
            state.SkipWithError(packed.status().ToString());
            break;
        }
        benchmark::DoNotOptimize((*packed)->storage().data());
    }
    state.SetBytesProcessed(state.iterations() *
                            static_cast<int64_t>(logical_bytes + packed_bytes));
    state.counters["pack GB/s"] = benchmark::Counter(
            static_cast<double>(state.iterations()) *
                    static_cast<double>(logical_bytes + packed_bytes),
            benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
}

BENCHMARK(BM_LinearPackedBpanelHot)
        ->Args({1, 4096, 4096})
        ->Args({1, 4096, 11008})
        ->Args({16, 4096, 4096})
        ->Args({64, 4096, 4096})
        ->ArgNames({"M", "K", "N"});

BENCHMARK(BM_LinearPackedBpanelStreaming)
        ->Args({1, 4096, 4096})
        ->Args({1, 4096, 11008})
        ->Args({16, 4096, 4096})
        ->Args({64, 4096, 4096})
        ->ArgNames({"M", "K", "N"});

BENCHMARK(BM_WeightPackingCpuBpanel)
        ->Args({4096, 4096})
        ->Args({4096, 768})
        ->Args({11008, 4096})
        ->ArgNames({"N", "K"});

#endif

} // namespace
