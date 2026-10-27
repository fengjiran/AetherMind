#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/backend/cpu/cpu_packed_weight_layout.h"
#include "aethermind/backend/kernel_registry.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/op_params.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <tuple>

namespace {

using namespace aethermind;

KernelSelector CpuFloat32Selector(WeightFormat format, ExecPhase phase) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = format,
            .phase = phase,
    };
}

class CpuIsaFallbackPolicy : public ::testing::TestWithParam<std::tuple<CpuFeature, ExecPhase>> {};

TEST_P(CpuIsaFallbackPolicy, RmsNormResolvesReferenceWhenRequiredFeatureIsDisabled) {
    const auto [disabled, phase] = GetParam();
    CpuBackend backend(CpuFeaturePolicy{
            .disabled_features = CpuFeatureSet::From({disabled}),
    });

    const auto resolved = backend.PrepareKernel(
            OpType::kRmsNorm, CpuFloat32Selector(WeightFormat::kPlain, phase),
            OpParams{RmsNormParams{.eps = 1.0e-5F}});

    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
    EXPECT_FALSE(backend.cpu_capabilities().effective_features.Contains(disabled));
    EXPECT_EQ(resolved->name, "cpu::rmsnorm_f32_reference");
    EXPECT_NE(resolved->fn, nullptr);
}

TEST_P(CpuIsaFallbackPolicy, PackedLinearKeepsIdentityRecipeWhenRequiredFeatureIsDisabled) {
    const auto [disabled, phase] = GetParam();
    CpuBackend backend(CpuFeaturePolicy{
            .disabled_features = CpuFeatureSet::From({disabled}),
    });
    const KernelSelector selector = CpuFloat32Selector(WeightFormat::kPacked, phase);

    const auto recipe = backend.GetPackingRecipe(OpType::kLinear, selector);
    ASSERT_TRUE(recipe.ok()) << recipe.status().ToString();
    const auto resolved = backend.PrepareKernel(OpType::kLinear, selector, OpParams{LinearParams{}});

    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
    EXPECT_FALSE(backend.cpu_capabilities().effective_features.Contains(disabled));
    EXPECT_EQ(*recipe, CpuIdentityPackingRecipe());
    EXPECT_EQ(resolved->expected_packing_recipe, *recipe);
    EXPECT_EQ(resolved->name, "cpu::linear_f32_packed_identity_reference");
    EXPECT_NE(resolved->fn, nullptr);
}

INSTANTIATE_TEST_SUITE_P(
        FeaturePolicies, CpuIsaFallbackPolicy,
        ::testing::Combine(::testing::Values(CpuFeature::kAvx2, CpuFeature::kFma),
                           ::testing::Values(ExecPhase::kPrefill, ExecPhase::kDecode)),
        [](const ::testing::TestParamInfo<CpuIsaFallbackPolicy::ParamType>& info) {
            const CpuFeature feature = std::get<0>(info.param);
            const ExecPhase phase = std::get<1>(info.param);
            return std::string(feature == CpuFeature::kAvx2 ? "DisableAvx2" : "DisableFma") +
                   (phase == ExecPhase::kPrefill ? "Prefill" : "Decode");
        });

TEST(CpuIsaFallback, OptionalDescriptorsMatchCompiledKernelAvailability) {
    const CpuBackend backend;
    const KernelRegistry* const registry = backend.TryGetKernelRegistryForDebug();
    ASSERT_NE(registry, nullptr);
    const auto has_avx2_fma_candidate = [&](OpType op, WeightFormat format) {
        const auto candidates = registry->FindCandidates(
                op, CpuFloat32Selector(format, ExecPhase::kBoth));
        EXPECT_TRUE(candidates.ok()) << candidates.status().ToString();
        if (!candidates.ok()) {
            return false;
        }
        return std::ranges::any_of(*candidates, [](const KernelDef* candidate) {
            return candidate->cpu_requirements.Contains(CpuFeature::kAvx2) &&
                   candidate->cpu_requirements.Contains(CpuFeature::kFma);
        });
    };

#if defined(RMSNORM_HAS_AVX2_FMA_KERNEL)
    EXPECT_TRUE(has_avx2_fma_candidate(OpType::kRmsNorm, WeightFormat::kPlain));
#else
    EXPECT_FALSE(has_avx2_fma_candidate(OpType::kRmsNorm, WeightFormat::kPlain));
#endif
#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
    EXPECT_TRUE(has_avx2_fma_candidate(OpType::kLinear, WeightFormat::kPacked));
#else
    EXPECT_FALSE(has_avx2_fma_candidate(OpType::kLinear, WeightFormat::kPacked));
#endif
}

TEST(CpuIsaFallback, FactoryCannotEnableAFeatureAbsentFromTheHardwareArchitecture) {
    const auto detected = cpu::DetectCpuCapabilities();
    ASSERT_TRUE(detected.ok()) << detected.status().ToString();
    // Architecture feature sets are disjoint. This requirement cannot become
    // available through an OS permission change between the two detections.
    const CpuFeature unavailable = detected->architecture == CpuArchitecture::kAArch64
                                           ? CpuFeature::kAvx2
                                           : CpuFeature::kNeon;
    ASSERT_FALSE(detected->hardware_features.Contains(unavailable));
    const CpuBackendFactory factory(CpuFeaturePolicy{
            .required_features = CpuFeatureSet::From({unavailable}),
    });

    const auto backend = factory.Create();

    ASSERT_FALSE(backend.ok());
    EXPECT_EQ(backend.status().code(), StatusCode::kFailedPrecondition);
}

} // namespace
