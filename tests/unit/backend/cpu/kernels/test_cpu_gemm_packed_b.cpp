#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/backend/kernel_context.h"
#include "backend/cpu/cpu_backend_internal.h"
#include "backend/cpu/kernels/gemm/gemm_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <string_view>
#include <vector>

namespace {

using namespace aethermind;

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)

bool CanExecutePackedAvx2() {
    const auto capabilities = cpu::DetectCpuCapabilities();
    return capabilities.ok() &&
           capabilities->effective_features.Contains(CpuFeature::kAvx2) &&
           capabilities->effective_features.Contains(CpuFeature::kFma);
}

float TestValue(size_t index) {
    return static_cast<float>(static_cast<int64_t>((index * 13U + 7U) % 31U) - 15) *
           0.0625F;
}

void Fill(std::vector<float>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = TestValue(i);
    }
}

std::unique_ptr<PackedWeights> PackBpanel(int64_t n,
                                          int64_t k,
                                          std::vector<float>& weights,
                                          const KernelSelector& selector) {
    const int64_t shape[] = {n, k};
    const int64_t strides[] = {k, 1};
    const TensorView weight_view(
            weights.data(), DataType::Float32(), shape, strides);
    const std::array<TensorView, 1> components{weight_view};
    CpuWeightPrepacker prepacker;
    auto packed = prepacker.Pack(
            OpType::kLinear, components, selector,
            cpu::CpuBPanelF32V1Avx2Recipe());
    if (!packed.ok()) {
        ADD_FAILURE() << packed.status().ToString();
        return nullptr;
    }
    return std::move(*packed);
}

void ExpectPackedDriverMatchesReference(int64_t m,
                                        int64_t n,
                                        int64_t k,
                                        int64_t logical_n,
                                        int64_t weight_n_offset,
                                        bool blocked) {
    const int64_t output_n = n;
    const int64_t output_stride = output_n + 3;
    std::vector<float> lhs(static_cast<size_t>(m * k));
    std::vector<float> weights(static_cast<size_t>(logical_n * k));
    std::vector<float> packed_output(static_cast<size_t>(m * output_stride), -7.0F);
    std::vector<float> reference_output(static_cast<size_t>(m * output_stride), 13.0F);
    Fill(lhs);
    Fill(weights);

    KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .phase = ExecPhase::kBoth,
    };
    auto artifact = PackBpanel(logical_n, k, weights, selector);
    ASSERT_NE(artifact, nullptr);
    const auto* const packed_data = static_cast<const float*>(artifact->storage().data());
    const int64_t n_blocks = logical_n / cpu::kCpuBPanelF32V1NR +
                             (logical_n % cpu::kCpuBPanelF32V1NR != 0);
    const cpu::detail::PackedGemmF32Args packed_args{
            .gemm = cpu::detail::GemmF32Args{
                    .lhs = lhs.data(),
                    .output = packed_output.data(),
                    .m = m,
                    .n = output_n,
                    .k = k,
                    .lhs_m_stride = k,
                    .lhs_k_stride = 1,
                    .output_m_stride = output_stride,
                    .output_n_stride = 1,
            },
            .packed_b = packed_data,
            .packed_nbytes = artifact->storage().nbytes(),
            .logical_n = logical_n,
            .logical_k = k,
            .weight_n_offset = weight_n_offset,
            .n_blocks = n_blocks,
    };
    const cpu::detail::GemmF32Args reference_args{
            .lhs = lhs.data(),
            .rhs = weights.data() + weight_n_offset * k,
            .output = reference_output.data(),
            .m = m,
            .n = output_n,
            .k = k,
            .lhs_m_stride = k,
            .lhs_k_stride = 1,
            .rhs_k_stride = 1,
            .rhs_n_stride = k,
            .output_m_stride = output_stride,
            .output_n_stride = 1,
    };

    const Status packed_status = blocked
                                         ? cpu::detail::RunGemmF32PackedBBlocked(packed_args)
                                         : cpu::detail::RunGemmF32PackedBScan(packed_args);
    ASSERT_TRUE(packed_status.ok()) << packed_status.ToString();
    ASSERT_TRUE(cpu::detail::RunGemmF32Reference(reference_args).ok());
    const float absolute_tolerance =
            2.0e-4F + 2.0e-5F * std::sqrt(static_cast<float>(k));
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < output_n; ++col) {
            const size_t index = static_cast<size_t>(row * output_stride + col);
            const float absolute_error = std::fabs(packed_output[index] -
                                                   reference_output[index]);
            const float relative_error = absolute_error /
                                         std::max(std::fabs(reference_output[index]), 1.0e-6F);
            EXPECT_TRUE(absolute_error <= absolute_tolerance || relative_error <= 2.0e-4F)
                    << "row=" << row << " col=" << col
                    << " actual=" << packed_output[index]
                    << " expected=" << reference_output[index];
        }
        for (int64_t col = output_n; col < output_stride; ++col) {
            EXPECT_EQ(packed_output[static_cast<size_t>(row * output_stride + col)],
                      -7.0F)
                    << "output guard row=" << row << " col=" << col;
        }
    }
}

TEST(CpuGemmPackedB, ScanSupportsUnalignedOutputSliceKPanelsAndNtail) {
    if (!CanExecutePackedAvx2()) {
        GTEST_SKIP() << "AVX2+FMA GEMM candidate is unavailable on this host";
    }
    ExpectPackedDriverMatchesReference(1, 30, 513, 37, 3, false);
}

TEST(CpuGemmPackedB, BlockedHandlesMAndNTailsWithoutOverwritingEdges) {
    if (!CanExecutePackedAvx2()) {
        GTEST_SKIP() << "AVX2+FMA GEMM candidate is unavailable on this host";
    }
    ExpectPackedDriverMatchesReference(11, 19, 33, 35, 16, true);
}

TEST(CpuGemmPackedB, RegisteredCandidateDescriptorPacksAndExecutes) {
    if (!CanExecutePackedAvx2()) {
        GTEST_SKIP() << "AVX2+FMA GEMM candidate is unavailable on this host";
    }
    const auto capabilities = cpu::DetectCpuCapabilities();
    ASSERT_TRUE(capabilities.ok()) << capabilities.status().ToString();

    KernelRegistry& global = KernelRegistry::Global();
    ASSERT_TRUE(global.Freeze().ok());
    const auto registered = global.FindByOpType(OpType::kLinear);
    ASSERT_TRUE(registered.ok()) << registered.status().ToString();
    const auto candidate = std::find_if(
            registered->begin(), registered->end(), [](const KernelDef* descriptor) {
                return descriptor->name == "cpu::linear_f32_packed_bpanel_candidate";
            });
    ASSERT_NE(candidate, registered->end());

    KernelRegistry isolated_registry;
    ASSERT_TRUE(isolated_registry.Register(**candidate).ok());
    ASSERT_TRUE(isolated_registry.Freeze().ok());
    const KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .phase = ExecPhase::kBoth,
    };
    const auto resolved_descriptor = cpu::internal::ResolveEligibleDescriptor(
            isolated_registry, OpType::kLinear, selector,
            capabilities->effective_features);
    ASSERT_TRUE(resolved_descriptor.ok()) << resolved_descriptor.status().ToString();
    ASSERT_EQ((*resolved_descriptor)->name,
              "cpu::linear_f32_packed_bpanel_candidate");

    const auto candidate_recipe = cpu::internal::ResolvePackingRecipeFromRegistry(
            isolated_registry, OpType::kLinear, selector,
            capabilities->effective_features);
    ASSERT_TRUE(candidate_recipe.ok()) << candidate_recipe.status().ToString();
    EXPECT_EQ(*candidate_recipe, (*resolved_descriptor)->packing_recipe);

    constexpr int64_t m = 9;
    constexpr int64_t n = 32;
    constexpr int64_t k = 33;
    const int64_t input_shape[] = {m, k};
    const int64_t input_strides[] = {k, 1};
    const int64_t weight_shape[] = {n, k};
    const int64_t weight_strides[] = {k, 1};
    const int64_t output_shape[] = {m, n};
    const int64_t output_stride = n + 3;
    const int64_t output_strides[] = {output_stride, 1};
    std::vector<float> input(static_cast<size_t>(m * k));
    std::vector<float> weights(static_cast<size_t>(n * k));
    std::vector<float> output(static_cast<size_t>(m * output_stride), -9.0F);
    std::vector<float> expected(static_cast<size_t>(m * output_stride), 17.0F);
    Fill(input);
    Fill(weights);
    const TensorView input_view(
            input.data(), DataType::Float32(), input_shape, input_strides);
    const TensorView weight_view(
            weights.data(), DataType::Float32(), weight_shape, weight_strides);
    const std::array<TensorView, 1> components{weight_view};
    CpuWeightPrepacker prepacker;
    auto packed = prepacker.Pack(
            OpType::kLinear, components, selector, *candidate_recipe);
    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    ASSERT_EQ((*packed)->recipe(), *candidate_recipe);
    const PackedWeightView packed_view{
            .data = (*packed)->storage().data(),
            .nbytes = (*packed)->storage().nbytes(),
            .logical_dtype = (*packed)->logical_dtype(),
            .logical_shape = (*packed)->logical_shape(),
            .recipe_layout = (*packed)->recipe().layout,
            .recipe_alignment = (*packed)->recipe().alignment,
            .alignment = (*packed)->storage().alignment(),
    };
    const std::array<MutableTensorView, 1> outputs{
            MutableTensorView(output.data(), DataType::Float32(),
                              output_shape, output_strides)};
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> params{};
    ASSERT_TRUE((*resolved_descriptor)->params_builder(KernelParamsBuildContext{
                                                               .inputs = std::span<const TensorView>(&input_view, 1),
                                                               .outputs = outputs,
                                                               .packed_weight = packed_view,
                                                       },
                                                       params.data())
                        .ok());
    ASSERT_TRUE((*resolved_descriptor)->kernel_func(KernelContext{
                                                            .device_type = DeviceType::kCPU,
                                                            .kernel_params = params.data(),
                                                    })
                        .ok());

    const cpu::detail::GemmF32Args reference{
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
            .output_m_stride = output_stride,
            .output_n_stride = 1,
    };
    ASSERT_TRUE(cpu::detail::RunGemmF32Reference(reference).ok());
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            const size_t index = static_cast<size_t>(row * output_stride + col);
            EXPECT_NEAR(output[index], expected[index], 2.0e-3F)
                    << "row=" << row << " col=" << col;
        }
        for (int64_t col = n; col < output_stride; ++col) {
            EXPECT_EQ(output[static_cast<size_t>(row * output_stride + col)], -9.0F)
                    << "output guard row=" << row << " col=" << col;
        }
    }
}

#endif

} // namespace
