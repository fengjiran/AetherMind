#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/runtime/runtime_builder.h"
#include "device.h"

#include <gtest/gtest.h>

namespace aethermind {
namespace {

TEST(CpuBackend, DeviceTypeIsCPU) {
    CpuBackend backend;
    EXPECT_EQ(backend.device_type(), DeviceType::kCPU);
}

TEST(CpuBackend, CapabilitiesExposeCPUType) {
    CpuBackend backend;
    const auto& caps = backend.capabilities();
    EXPECT_EQ(caps.device_type, DeviceType::kCPU);
}

TEST(CpuBackend, ResolveKernelReturnsNullptr) {
    CpuBackend backend;
    KernelKey key;
    EXPECT_EQ(backend.ResolveKernel(key), nullptr);
}

TEST(CpuBackend, TryGetKernelRegistryForDebugReturnsNullptr) {
    CpuBackend backend;
    EXPECT_EQ(backend.TryGetKernelRegistryForDebug(), nullptr);
}

TEST(CpuBackendFactory, CreatesValidBackend) {
    CpuBackendFactory factory;
    EXPECT_EQ(factory.device_type(), DeviceType::kCPU);

    auto backend = factory.Create();
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->device_type(), DeviceType::kCPU);
}

TEST(CpuBackend, RuntimeBuilderDefaultProvidesCpuBackend) {
    RuntimeBuilder builder;
    RuntimeContext context = builder.Build();

    auto backend_or = context.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(backend_or.ok());
    ASSERT_NE(backend_or.value(), nullptr);
    EXPECT_EQ(backend_or.value()->device_type(), DeviceType::kCPU);
}

}// namespace
}// namespace aethermind
