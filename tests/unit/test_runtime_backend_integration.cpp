#include "aethermind/backend/backend.h"
#include "aethermind/runtime/runtime_builder.h"
#include "aethermind/runtime/runtime_context.h"
#include "device.h"
#include <gtest/gtest.h>

namespace {

using namespace aethermind;

TEST(RuntimeBackendIntegration, BuildCreatesBackendRegistry) {
    RuntimeBuilder builder;
    RuntimeContext context = builder.Build();

    auto backend_or = context.GetBackend(DeviceType::kCPU);
    EXPECT_TRUE(backend_or.ok());
}

TEST(RuntimeBackendIntegration, GetBackendCPUWorks) {
    RuntimeBuilder builder;
    RuntimeContext context = builder.Build();

    auto backend_or = context.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(backend_or.ok());
    Backend* backend = backend_or.value();
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->device_type(), DeviceType::kCPU);
}

TEST(RuntimeBackendIntegration, GetBackendReturnsCachedInstance) {
    RuntimeBuilder builder;
    RuntimeContext context = builder.Build();

    auto backend1_or = context.GetBackend(DeviceType::kCPU);
    auto backend2_or = context.GetBackend(DeviceType::kCPU);

    ASSERT_TRUE(backend1_or.ok());
    ASSERT_TRUE(backend2_or.ok());
    EXPECT_EQ(backend1_or.value(), backend2_or.value());
}

TEST(RuntimeBackendIntegration, DefaultCpuFactoryIsRegistered) {
    RuntimeBuilder builder;
    RuntimeContext context = builder.Build();

    auto backend_or = context.GetBackend(DeviceType::kCPU);

    EXPECT_TRUE(backend_or.ok());
    if (backend_or.ok()) {
        EXPECT_EQ(backend_or.value()->device_type(), DeviceType::kCPU);
    }
}

TEST(RuntimeBackendIntegration, GetBackendForUnregisteredDeviceFails) {
    RuntimeOptions options;
    options.backend.enable_cpu = true;
    options.backend.enable_cuda = false;

    RuntimeBuilder builder;
    builder.WithOptions(options);
    RuntimeContext context = builder.Build();

    auto backend_or = context.GetBackend(DeviceType::kCUDA);

    EXPECT_FALSE(backend_or.ok());
    EXPECT_EQ(backend_or.status().code(), StatusCode::kNotFound);
}

class MockBackend : public Backend {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    const BackendCapabilities& capabilities() const noexcept override {
        static BackendCapabilities caps;
        return caps;
    }
    KernelFunc ResolveKernel(const KernelKey&) const noexcept override { return nullptr; }
    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override { return nullptr; }
};

class MockBackendFactory : public BackendFactory {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    std::unique_ptr<Backend> Create() const override {
        return std::make_unique<MockBackend>();
    }
};

TEST(RuntimeBackendIntegration, CustomCpuFactoryOverridesDefault) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU, std::make_unique<MockBackendFactory>());

    RuntimeContext context = builder.Build();
    auto backend_or = context.GetBackend(DeviceType::kCPU);

    ASSERT_TRUE(backend_or.ok());
    Backend* backend = backend_or.value();
    ASSERT_NE(backend, nullptr);
    EXPECT_NE(dynamic_cast<MockBackend*>(backend), nullptr);
}

}// namespace
