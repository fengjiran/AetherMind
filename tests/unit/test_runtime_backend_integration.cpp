#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_invocation.h"
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/backend/op_kernel_context.h"
#include "aethermind/backend/workspace_types.h"
#include "aethermind/execution/kv_cache_manager.h"
#include "aethermind/operators/op_type.h"
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
    KernelFunc ResolveKernel(OpType, const KernelSelector&) const noexcept override { return nullptr; }
    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override { return nullptr; }
    StatusOr<ResolvedKernel> ResolveKernelInfo(OpType op_type, const KernelSelector& selector) const noexcept override {
        const KernelFunc fn = ResolveKernel(op_type, selector);
        if (fn == nullptr) {
            return Status::NotFound(
                    "No matching kernel registered: op_type=" +
                    std::string(ToString(op_type)) +
                    ", selector=" + ToString(selector));
        }

        return ResolvedKernel{
                .op_type = op_type,
                .fn = fn,
                .attrs = {},
                .debug_name = nullptr,
        };
    }
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

TEST(RuntimeBackendIntegration, KVCacheManagerIsAbsentByDefault) {
    RuntimeBuilder builder;
    RuntimeContext context = builder.Build();

    EXPECT_EQ(context.GetKVCacheManager(), nullptr);
}

TEST(RuntimeBackendIntegration, KVCacheManagerCanBeBuiltFromRuntimeOptions) {
    RuntimeOptions options;
    options.kv_cache.enable_manager = true;
    options.kv_cache.num_layers = 2;
    options.kv_cache.num_kv_heads = 4;
    options.kv_cache.max_tokens = 32;
    options.kv_cache.head_dim = 16;
    options.kv_cache.kv_dtype = DataType(DLDataTypeCode::kFloat, 16, 1);
    options.kv_cache.alignment = 64;

    RuntimeBuilder builder;
    builder.WithOptions(options);
    RuntimeContext context = builder.Build();

    KVCacheManager* manager = context.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);
    EXPECT_TRUE(manager->is_initialized());
    EXPECT_EQ(manager->capacity_tokens(), 32U);

    const StatusOr<KVCacheView> view = manager->ReserveForSession(8, 8);
    ASSERT_TRUE(view.ok());
    EXPECT_EQ(view->current_pos(), 8U);
    EXPECT_EQ(view->token_capacity(), 16U);
}

}// namespace
