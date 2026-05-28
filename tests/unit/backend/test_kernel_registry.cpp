#include "aethermind/runtime/workspace.h"
#include "aethermind/backend/kernel_registry.h"
#include "aethermind/backend/kernel_context.h"

#include "data_type.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

Status FakeKernel(const KernelInvocation&,
                  const KernelContext&,
                  const WorkspaceBinding&) noexcept {
    return Status::Ok();
}

KernelDescriptor MakeTestKernelDescriptor() {
    return KernelDescriptor{
            .op_type = OpType::kRmsNorm,
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
    registry.Freeze();

    const StatusOr<const KernelDescriptor*> resolved =
            registry.Resolve(OpType::kRmsNorm, descriptor.selector);
    ASSERT_TRUE(resolved.ok());
    EXPECT_EQ((*resolved)->kernel_func, &FakeKernel);
}

TEST(KernelRegistry, RequestBasedResolveReturnsKernel) {
    KernelRegistry registry;
    const KernelDescriptor descriptor = MakeTestKernelDescriptor();

    ASSERT_TRUE(registry.Register(descriptor).ok());
    registry.Freeze();

    const StatusOr<const KernelDescriptor*> resolved =
            registry.Resolve(OpType::kRmsNorm, descriptor.selector);

    ASSERT_TRUE(resolved.ok());
    EXPECT_EQ((*resolved)->kernel_func, &FakeKernel);
}

TEST(KernelRegistry, LookupMissingKeyReturnsNullptr) {
    KernelRegistry registry;
    registry.Freeze();

    const StatusOr<const KernelDescriptor*> resolved =
            registry.Resolve(OpType::kLinear, MakeMissingSelector());
    EXPECT_EQ(resolved.status().code(), StatusCode::kNotFound);
}

TEST(KernelRegistry, RequestRejectsUnknownOpType) {
    KernelRegistry registry;
    registry.Freeze();

    const StatusOr<const KernelDescriptor*> resolved =
            registry.Resolve(OpType::kUnknown, MakeMissingSelector());

    ASSERT_FALSE(resolved.ok());
    EXPECT_EQ(resolved.status().code(), StatusCode::kInvalidArgument);
}

TEST(KernelRegistry, ResolveBeforeFreezeFails) {
    KernelRegistry registry;
    const KernelDescriptor descriptor = MakeTestKernelDescriptor();

    ASSERT_TRUE(registry.Register(descriptor).ok());

    const StatusOr<const KernelDescriptor*> resolved =
            registry.Resolve(OpType::kRmsNorm, descriptor.selector);
    EXPECT_EQ(resolved.status().code(), StatusCode::kFailedPrecondition);
}

TEST(KernelRegistry, RegisterAfterFreezeFails) {
    KernelRegistry registry;
    const KernelDescriptor descriptor = MakeTestKernelDescriptor();

    ASSERT_TRUE(registry.Register(descriptor).ok());
    registry.Freeze();
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

TEST(KernelRegistry, FindByOpTypeReturnsMatchingDescriptors) {
    KernelRegistry registry;
    const KernelDescriptor rms = MakeTestKernelDescriptor();

    KernelDescriptor linear = rms;
    linear.op_type = OpType::kLinear;
    linear.name = "test::linear";

    ASSERT_TRUE(registry.Register(rms).ok());
    ASSERT_TRUE(registry.Register(linear).ok());
    registry.Freeze();

    const auto rms_kernels = registry.FindByOpType(OpType::kRmsNorm);
    ASSERT_EQ(rms_kernels.size(), 1u);
    EXPECT_EQ(rms_kernels[0]->op_type, OpType::kRmsNorm);

    const auto linear_kernels = registry.FindByOpType(OpType::kLinear);
    ASSERT_EQ(linear_kernels.size(), 1u);
    EXPECT_EQ(linear_kernels[0]->op_type, OpType::kLinear);

    const auto none = registry.FindByOpType(OpType::kSoftmax);
    EXPECT_TRUE(none.empty());
}

TEST(KernelRegistry, DebugDumpContainsRegisteredEntries) {
    KernelRegistry registry;
    ASSERT_TRUE(registry.Register(MakeTestKernelDescriptor()).ok());
    registry.Freeze();

    const std::string dump = registry.DebugDump();
    EXPECT_FALSE(dump.empty());
    EXPECT_NE(dump.find("RmsNorm"), std::string::npos);
    EXPECT_NE(dump.find("test::op"), std::string::npos);
}

}// namespace
