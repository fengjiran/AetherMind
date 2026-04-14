#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/dispatcher_bridge.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(CpuResolveKernel, RegisteredKeyReturnsKernel) {
    CpuBackend backend;

    const KernelKey key = MakeKernelKey(
            DeviceType::kCPU,
            OperatorName("test::fake_cpu_kernel", ""));

    EXPECT_NE(backend.ResolveKernel(key), nullptr);
}

TEST(CpuResolveKernel, MissingKeyReturnsNullptr) {
    CpuBackend backend;

    const KernelKey key = MakeKernelKey(
            DeviceType::kCPU,
            OperatorName("test::missing", ""));

    EXPECT_EQ(backend.ResolveKernel(key), nullptr);
}

TEST(CpuResolveKernel, DebugRegistryIsExposedForInspection) {
    CpuBackend backend;
    EXPECT_NE(backend.TryGetKernelRegistryForDebug(), nullptr);
}

TEST(CpuResolveKernel, RegisteredKernelCanBeInvoked) {
    CpuBackend backend;

    const KernelKey key = MakeKernelKey(
            DeviceType::kCPU,
            OperatorName("test::fake_cpu_kernel", ""));

    const KernelFn fn = backend.ResolveKernel(key);
    ASSERT_NE(fn, nullptr);

    const Status status = fn();
    EXPECT_TRUE(status.ok());
}

}// namespace
