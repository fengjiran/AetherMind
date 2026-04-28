#include "../../include/aethermind/execution/workspace_types.h"
#include "aethermind/backend/kernel_registry.h"
#include "aethermind/backend/op_kernel_context.h"

#include "data_type.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

Status ScalarKernel(const KernelInvocation&,
                    const OpKernelContext&,
                    const WorkspaceBinding&) noexcept {
    return Status::Ok();
}

Status Avx2Kernel(const KernelInvocation&,
                  const OpKernelContext&,
                  const WorkspaceBinding&) noexcept {
    return Status::Ok();
}

Status DecodeOnlyKernel(const KernelInvocation&,
                        const OpKernelContext&,
                        const WorkspaceBinding&) noexcept {
    return Status::Ok();
}

KernelSelector MakeSelector(ExecPhase phase = ExecPhase::kBoth,
                            IsaLevel isa = IsaLevel::kScalar) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = isa,
            .phase = phase,
    };
}

TEST(KernelRegistryResolve, BothPhaseMatchesDecodeRequest) {
    KernelRegistry registry;
    ASSERT_TRUE(registry.Register(KernelDescriptor{
                                          .op_type = OpType::kRMSNorm,
                                          .selector = MakeSelector(ExecPhase::kBoth),
                                          .kernel_func = &ScalarKernel,
                                          .name = "scalar",
                                          .priority = 1,
                                  })
                        .ok());
    ASSERT_TRUE(registry.Freeze().ok());

    const StatusOr<const KernelDescriptor*> descriptor =
            registry.Resolve(OpType::kRMSNorm, MakeSelector(ExecPhase::kDecode));

    ASSERT_TRUE(descriptor.ok());
    EXPECT_EQ((*descriptor)->kernel_func, &ScalarKernel);
}

TEST(KernelRegistryResolve, IsaCompatibilityPrefersHigherPriorityMatch) {
    KernelRegistry registry;
    ASSERT_TRUE(registry.Register(KernelDescriptor{
                                          .op_type = OpType::kRMSNorm,
                                          .selector = MakeSelector(ExecPhase::kBoth, IsaLevel::kScalar),
                                          .kernel_func = &ScalarKernel,
                                          .name = "scalar",
                                          .priority = 1,
                                  })
                        .ok());
    ASSERT_TRUE(registry.Register(KernelDescriptor{
                                          .op_type = OpType::kRMSNorm,
                                          .selector = MakeSelector(ExecPhase::kBoth, IsaLevel::kAVX2),
                                          .kernel_func = &Avx2Kernel,
                                          .name = "avx2",
                                          .priority = 10,
                                  })
                        .ok());
    ASSERT_TRUE(registry.Freeze().ok());

    const StatusOr<const KernelDescriptor*> descriptor =
            registry.Resolve(OpType::kRMSNorm, MakeSelector(ExecPhase::kPrefill, IsaLevel::kAVX512));

    ASSERT_TRUE(descriptor.ok());
    EXPECT_EQ((*descriptor)->kernel_func, &Avx2Kernel);
}

TEST(KernelRegistryResolve, IncompatibleIsaReturnsNotFound) {
    KernelRegistry registry;
    ASSERT_TRUE(registry.Register(KernelDescriptor{
                                          .op_type = OpType::kRMSNorm,
                                          .selector = MakeSelector(ExecPhase::kBoth, IsaLevel::kAVX512),
                                          .kernel_func = &Avx2Kernel,
                                          .name = "avx512-only",
                                          .priority = 10,
                                  })
                        .ok());
    ASSERT_TRUE(registry.Freeze().ok());

    const StatusOr<const KernelDescriptor*> descriptor =
            registry.Resolve(OpType::kRMSNorm, MakeSelector(ExecPhase::kPrefill, IsaLevel::kAVX2));

    EXPECT_EQ(descriptor.status().code(), StatusCode::kNotFound);
}

TEST(KernelRegistryResolve, ExactPhaseBeatsMissingPhaseMatch) {
    KernelRegistry registry;
    ASSERT_TRUE(registry.Register(KernelDescriptor{
                                          .op_type = OpType::kRMSNorm,
                                          .selector = MakeSelector(ExecPhase::kDecode),
                                          .kernel_func = &DecodeOnlyKernel,
                                          .name = "decode-only",
                                          .priority = 5,
                                  })
                        .ok());
    ASSERT_TRUE(registry.Freeze().ok());

    const StatusOr<const KernelDescriptor*> descriptor =
            registry.Resolve(OpType::kRMSNorm, MakeSelector(ExecPhase::kPrefill));

    EXPECT_EQ(descriptor.status().code(), StatusCode::kNotFound);
}

}// namespace
