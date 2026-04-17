#include "aethermind/backend/kernel_registry.h"
#include "aethermind/backend/op_kernel_context.h"
#include "aethermind/backend/workspace_types.h"

#include "data_type.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

Status FakeKernel(const KernelInvocation&,
                  const OpKernelContext&,
                  const WorkspaceBinding&) noexcept {
    return Status::Ok();
}

KernelDescriptor MakeTestKernelDescriptor() {
    return KernelDescriptor{
            .op_type = OpType::kRMSNorm,
            .selector = KernelSelector{
                    .device_type = DeviceType::kCPU,
                    .activation_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
                    .weight_format = WeightFormat::kPlain,
                    .isa = IsaLevel::kScalar,
                    .phase = ExecPhase::kBoth,
            },
            .kernel_func = &FakeKernel,
            .name = "test::op",
            .priority = 1,
    };
}

KernelSelector MakeMissingSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };
}

TEST(KernelRegistry, RegisterAndLookupReturnsKernel) {
    KernelRegistry registry;
    const KernelDescriptor descriptor = MakeTestKernelDescriptor();

    ASSERT_TRUE(registry.Register(descriptor).ok());
    ASSERT_TRUE(registry.Freeze().ok());

    const StatusOr<const KernelDescriptor*> resolved =
            registry.Resolve(OpType::kRMSNorm, descriptor.selector);
    ASSERT_TRUE(resolved.ok());
    EXPECT_EQ((*resolved)->kernel_func, &FakeKernel);
}

TEST(KernelRegistry, LookupMissingKeyReturnsNullptr) {
    KernelRegistry registry;
    ASSERT_TRUE(registry.Freeze().ok());

    const StatusOr<const KernelDescriptor*> resolved =
            registry.Resolve(OpType::kLinear, MakeMissingSelector());
    EXPECT_EQ(resolved.status().code(), StatusCode::kNotFound);
}

TEST(KernelRegistry, ResolveBeforeFreezeFails) {
    KernelRegistry registry;
    const KernelDescriptor descriptor = MakeTestKernelDescriptor();

    ASSERT_TRUE(registry.Register(descriptor).ok());

    const StatusOr<const KernelDescriptor*> resolved =
            registry.Resolve(OpType::kRMSNorm, descriptor.selector);
    EXPECT_EQ(resolved.status().code(), StatusCode::kFailedPrecondition);
}

TEST(KernelRegistry, RegisterAfterFreezeFails) {
    KernelRegistry registry;
    const KernelDescriptor descriptor = MakeTestKernelDescriptor();

    ASSERT_TRUE(registry.Register(descriptor).ok());
    ASSERT_TRUE(registry.Freeze().ok());
    EXPECT_TRUE(registry.frozen());

    KernelDescriptor extra = descriptor;
    extra.op_type = OpType::kLinear;
    extra.name = "test::other_op";

    const Status status = registry.Register(extra);
    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

TEST(KernelRegistry, DuplicateRegistrationFails) {
    KernelRegistry registry;
    const KernelDescriptor descriptor = MakeTestKernelDescriptor();

    ASSERT_TRUE(registry.Register(descriptor).ok());
    EXPECT_EQ(registry.Register(descriptor).code(), StatusCode::kAlreadyExists);
}

}// namespace
