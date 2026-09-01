#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_registry.h"

#include "aethermind/dtypes/data_type.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

Status ScalarKernel(const KernelContext&) noexcept {
    return Status::Ok();
}

Status FeatureKernel(const KernelContext&) noexcept {
    return Status::Ok();
}

Status DecodeOnlyKernel(const KernelContext&) noexcept {
    return Status::Ok();
}

KernelSelector MakeSelector(ExecPhase phase = ExecPhase::kBoth) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = phase,
    };
}

TEST(KernelRegistryCandidates, BothPhaseMatchesDecodeRequest) {
    KernelRegistry registry;
    ASSERT_TRUE(registry.Register(KernelDescriptor{
                                          .op_type = OpType::kRmsNorm,
                                          .selector = MakeSelector(ExecPhase::kBoth),
                                          .kernel_func = &ScalarKernel,
                                          .name = "scalar",
                                          .priority = 1,
                                  })
                        .ok());
    ASSERT_TRUE(registry.Freeze().ok());

    const auto candidates = registry.FindCandidates(
            OpType::kRmsNorm, MakeSelector(ExecPhase::kDecode));

    ASSERT_TRUE(candidates.ok());
    ASSERT_EQ(candidates->size(), 1U);
    EXPECT_EQ((*candidates)[0]->kernel_func, &ScalarKernel);
}

TEST(KernelRegistryCandidates, ReturnsAllStructuralVariantsRegardlessOfRequirements) {
    KernelRegistry registry;
    ASSERT_TRUE(registry.Register(KernelDescriptor{
                                          .op_type = OpType::kRmsNorm,
                                          .selector = MakeSelector(),
                                          .kernel_func = &ScalarKernel,
                                          .name = "scalar",
                                          .priority = 1,
                                  })
                        .ok());
    ASSERT_TRUE(registry.Register(KernelDescriptor{
                                          .op_type = OpType::kRmsNorm,
                                          .selector = MakeSelector(),
                                          .cpu_requirements = CpuFeatureSet::From(
                                                  {CpuFeature::kAvx2}),
                                          .kernel_func = &FeatureKernel,
                                          .name = "avx2",
                                          .priority = 10,
                                  })
                        .ok());
    ASSERT_TRUE(registry.Freeze().ok());

    const auto candidates = registry.FindCandidates(
            OpType::kRmsNorm, MakeSelector(ExecPhase::kPrefill));

    ASSERT_TRUE(candidates.ok());
    ASSERT_EQ(candidates->size(), 2U);
    EXPECT_EQ((*candidates)[0]->kernel_func, &ScalarKernel);
    EXPECT_EQ((*candidates)[1]->kernel_func, &FeatureKernel);
}

TEST(KernelRegistryCandidates, IncompatiblePhaseReturnsNoCandidates) {
    KernelRegistry registry;
    ASSERT_TRUE(registry.Register(KernelDescriptor{
                                          .op_type = OpType::kRmsNorm,
                                          .selector = MakeSelector(ExecPhase::kDecode),
                                          .kernel_func = &DecodeOnlyKernel,
                                          .name = "decode-only",
                                          .priority = 5,
                                  })
                        .ok());
    ASSERT_TRUE(registry.Freeze().ok());

    const auto candidates = registry.FindCandidates(
            OpType::kRmsNorm, MakeSelector(ExecPhase::kPrefill));

    ASSERT_TRUE(candidates.ok());
    EXPECT_TRUE(candidates->empty());
}

} // namespace
