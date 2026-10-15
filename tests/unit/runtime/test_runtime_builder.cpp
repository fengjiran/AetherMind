#include "aethermind/runtime/runtime_builder.h"

#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "aethermind/dtypes/data_type.h"
#include "aethermind/memory/allocator.h"
#include "inference/test_malloc_interposer.h"

#include <gtest/gtest.h>

#include <memory>
#include <utility>

namespace aethermind {
namespace {

constexpr size_t kNumLayers = 2;
constexpr size_t kNumKvHeads = 3;
constexpr size_t kMaxTokens = 16;
constexpr size_t kHeadDim = 8;

RuntimeOptions MakeKvOptions() {
    RuntimeOptions options;
    options.kv_cache.enable_manager = true;
    options.kv_cache.num_layers = kNumLayers;
    options.kv_cache.num_kv_heads = kNumKvHeads;
    options.kv_cache.max_tokens = kMaxTokens;
    options.kv_cache.head_dim = kHeadDim;
    options.kv_cache.kv_dtype = DataType::Float32();
    options.kv_cache.alignment = 64;
    return options;
}

class StubAllocator final : public Allocator {
public:
    explicit StubAllocator(Device device) : device_(device) {}

    AM_NODISCARD Device device() const noexcept override {
        return device_;
    }

protected:
    AM_NODISCARD StatusOr<MemoryHandle> AllocateStorage(
            const AllocationRequest&) override {
        return Status::Unimplemented("stub allocator provides no storage");
    }

private:
    Device device_;
};

class StubAllocatorProvider final : public AllocatorProvider {
public:
    std::unique_ptr<Allocator> CreateAllocator(Device device) override {
        return std::make_unique<StubAllocator>(device);
    }
};

class StubBackendFactory final : public BackendFactory {
public:
    AM_NODISCARD DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    AM_NODISCARD StatusOr<std::unique_ptr<Backend>> Create() const override {
        return Status::Unimplemented("stub backend factory");
    }
};

/// @brief Builds a Runtime whose registries stay empty unless a pending
///        registration is transferred, so the only registry node insert inside
///        TryBuild is the one performed by that transfer.
RuntimeOptions MakeEmptyRuntimeOptions() {
    RuntimeOptions options;
    options.allocator.enable_cpu = false;
    options.backend.enable_cpu = false;
    options.kv_cache.enable_manager = false;
    return options;
}

} // namespace

TEST(RuntimeBuilderTryBuild, InitializesKvCacheManagerWhenEnabled) {
    RuntimeBuilder builder;
    builder.WithOptions(MakeKvOptions());

    auto result = builder.TryBuild();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    Runtime runtime = std::move(result).value();

    const KVCacheManager* const manager = runtime.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);
    EXPECT_TRUE(manager->is_initialized());
    EXPECT_EQ(manager->capacity_tokens(), kMaxTokens);
    // Both planes are backed, so the reported total covers key and value.
    EXPECT_GT(manager->total_bytes(), 0U);
    EXPECT_TRUE(runtime.HasAllocatorProvider(DeviceType::kCPU));
}

TEST(RuntimeBuilderTryBuild, LeavesKvCacheManagerNullWhenDisabled) {
    RuntimeOptions options;
    options.kv_cache.enable_manager = false;
    RuntimeBuilder builder;
    builder.WithOptions(options);

    auto result = builder.TryBuild();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    Runtime runtime = std::move(result).value();

    EXPECT_EQ(runtime.GetKVCacheManager(), nullptr);
    EXPECT_TRUE(runtime.HasAllocatorProvider(DeviceType::kCPU));
}

TEST(RuntimeBuilderTryBuild, ReportsInvalidKvGeometryInsteadOfAborting) {
    RuntimeOptions options = MakeKvOptions();
    options.kv_cache.head_dim = 0;
    RuntimeBuilder builder;
    builder.WithOptions(options);

    const auto result = builder.TryBuild();
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(RuntimeBuilderTryBuild, ReportsInvalidKvAlignmentInsteadOfAborting) {
    RuntimeOptions options = MakeKvOptions();
    options.kv_cache.alignment = 0;
    RuntimeBuilder builder;
    builder.WithOptions(options);

    const auto result = builder.TryBuild();
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(RuntimeBuilderTryBuild, FailureStillConsumesPendingRegistrations) {
    RuntimeOptions options = MakeKvOptions();
    options.kv_cache.head_dim = 0;
    RuntimeBuilder builder;
    builder.WithOptions(options);
    builder.RegisterCustomAllocatorProvider(
            DeviceType::kCUDA, std::make_unique<StubAllocatorProvider>());

    ASSERT_FALSE(builder.TryBuild().ok());

    // Build() documents that pending registrations are consumed by the call;
    // a failed TryBuild() must not leave them queued for a later build.
    options.kv_cache.head_dim = kHeadDim;
    builder.WithOptions(options);
    auto retry = builder.TryBuild();
    ASSERT_TRUE(retry.ok()) << retry.status().ToString();
    EXPECT_FALSE(retry->HasAllocatorProvider(DeviceType::kCUDA));
}

TEST(RuntimeBuilderTryBuild, SucceedingBuildMatchesTryBuild) {
    RuntimeBuilder builder;
    builder.WithOptions(MakeKvOptions());

    Runtime runtime = builder.Build();
    const KVCacheManager* const manager = runtime.GetKVCacheManager();
    ASSERT_NE(manager, nullptr);
    EXPECT_TRUE(manager->is_initialized());
    EXPECT_EQ(manager->capacity_tokens(), kMaxTokens);
}

TEST(RuntimeBuilderTryBuild, AllocatorProviderTransferOutOfMemoryStaysReusable) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    RuntimeBuilder builder;
    builder.WithOptions(MakeEmptyRuntimeOptions());
    builder.RegisterCustomAllocatorProvider(
            DeviceType::kCUDA, std::make_unique<StubAllocatorProvider>());

    {
        test::ScopedMallocFailure failure(0, 0,
                                          false,
                                          test::MallocFailureTarget::kMalloc);
        const auto first = builder.TryBuild();
        const test::MallocFailureResult injected = failure.Stop();
        ASSERT_TRUE(injected.allocation_failed)
                << "injection missed the allocator provider transfer";
        ASSERT_FALSE(first.ok());
        EXPECT_EQ(first.status().code(), StatusCode::kResourceExhausted);
    }

    // The failed transfer already surrendered the provider to the discarded
    // registry, so it cannot come back. What must hold is that the pending
    // list keeps no moved-from entry: a retry has to succeed instead of
    // tripping SetProvider's non-null check.
    const auto retry = builder.TryBuild();
    ASSERT_TRUE(retry.ok()) << retry.status().ToString();
    EXPECT_FALSE(retry->HasAllocatorProvider(DeviceType::kCUDA));
}

TEST(RuntimeBuilderTryBuild, BackendFactoryTransferOutOfMemoryStaysReusable) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    RuntimeBuilder builder;
    builder.WithOptions(MakeEmptyRuntimeOptions());
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<StubBackendFactory>());

    {
        test::ScopedMallocFailure failure(0, 0,
                                          false,
                                          test::MallocFailureTarget::kMalloc);
        const auto first = builder.TryBuild();
        const test::MallocFailureResult injected = failure.Stop();
        ASSERT_TRUE(injected.allocation_failed)
                << "injection missed the backend factory transfer";
        ASSERT_FALSE(first.ok());
        EXPECT_EQ(first.status().code(), StatusCode::kResourceExhausted);
    }

    auto retry = builder.TryBuild();
    ASSERT_TRUE(retry.ok()) << retry.status().ToString();
    const auto backend = retry->GetBackend(DeviceType::kCPU);
    EXPECT_FALSE(backend.ok());
}

} // namespace aethermind
