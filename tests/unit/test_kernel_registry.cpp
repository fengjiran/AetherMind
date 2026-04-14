#include "aethermind/backend/kernel_registry.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

Status FakeKernel() noexcept {
    return Status::Ok();
}

KernelKey MakeTestKernelKey() {
    return KernelKey{
            .device_type = DeviceType::kCPU,
            .op_name = OperatorName("test::op", ""),
    };
}

TEST(KernelRegistry, RegisterAndLookupReturnsKernel) {
    KernelRegistry registry;
    const KernelKey key = MakeTestKernelKey();

    registry.Register(key, &FakeKernel);

    EXPECT_EQ(registry.Find(key), &FakeKernel);
}

TEST(KernelRegistry, LookupMissingKeyReturnsNullptr) {
    KernelRegistry registry;
    const KernelKey key{
            .device_type = DeviceType::kCPU,
            .op_name = OperatorName("test::missing", ""),
    };

    EXPECT_EQ(registry.Find(key), nullptr);
}

TEST(KernelRegistry, DuplicateRegistrationFails) {
    KernelRegistry registry;
    const KernelKey key = MakeTestKernelKey();

    registry.Register(key, &FakeKernel);
    EXPECT_DEATH(registry.Register(key, &FakeKernel), "Duplicate kernel registration");
}

}// namespace
