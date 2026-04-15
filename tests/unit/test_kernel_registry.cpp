#include "aethermind/backend/kernel_registry.h"

#include "data_type.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

Status FakeKernel() noexcept {
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

    const KernelDescriptor* resolved = nullptr;
    ASSERT_TRUE(registry.Resolve(OpType::kRMSNorm, descriptor.selector, &resolved).ok());
    ASSERT_NE(resolved, nullptr);
    EXPECT_EQ(resolved->kernel_func, &FakeKernel);
}

TEST(KernelRegistry, LookupMissingKeyReturnsNullptr) {
    KernelRegistry registry;
    const KernelDescriptor* resolved = nullptr;

    const Status status = registry.Resolve(OpType::kLinear, MakeMissingSelector(), &resolved);
    EXPECT_EQ(status.code(), StatusCode::kNotFound);
    EXPECT_EQ(resolved, nullptr);
}

TEST(KernelRegistry, DuplicateRegistrationFails) {
    KernelRegistry registry;
    const KernelDescriptor descriptor = MakeTestKernelDescriptor();

    ASSERT_TRUE(registry.Register(descriptor).ok());
    EXPECT_EQ(registry.Register(descriptor).code(), StatusCode::kAlreadyExists);
}

}// namespace
