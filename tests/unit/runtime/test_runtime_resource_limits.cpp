#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace {

using namespace aethermind;

// Two K/V planes, two layers, three heads, sixteen token rows. Each row is
// padded from eight Float32 elements to the requested 64-byte alignment.
constexpr size_t kRequiredKvBytes = 2 * 2 * 3 * 16 * 64;

RuntimeOptions MakeOptions() {
    RuntimeOptions options;
    options.backend.enable_cpu = false;
    options.kv_cache = {.enable_manager = true,
                        .num_layers = 2,
                        .num_kv_heads = 3,
                        .max_tokens = 16,
                        .head_dim = 8,
                        .kv_dtype = DataType::Float32(),
                        .alignment = 64};
    return options;
}

struct AllocationObservation {
    size_t allocator_creations = 0;
    size_t storage_calls = 0;
    bool fail_storage = false;
};

void ReleaseBuffer(void* context, void*) noexcept {
    delete static_cast<Buffer*>(context);
}

class ObservedAllocator final : public Allocator {
public:
    explicit ObservedAllocator(std::shared_ptr<AllocationObservation> observation)
        : observation_(std::move(observation)), backing_(Device::CPU()) {}

    Device device() const noexcept override {
        return Device::CPU();
    }

protected:
    StatusOr<MemoryHandle> AllocateStorage(const AllocationRequest& request) override {
        ++observation_->storage_calls;
        if (observation_->fail_storage) {
            return Status::ResourceExhausted({});
        }
        AM_ASSIGN_OR_RETURN(auto buffer, backing_.TryAllocate(request));
        auto owner = std::make_unique<Buffer>(std::move(buffer));
        void* const data = owner->mutable_data();
        const size_t alignment = owner->alignment();
        // The release context retains the allocation independently of this
        // observing allocator and its CPU allocator member.
        return MemoryHandle(data, owner.release(), &ReleaseBuffer,
                            Device::CPU(), alignment);
    }

private:
    std::shared_ptr<AllocationObservation> observation_;
    CPUAllocator backing_;
};

class ObservedAllocatorProvider final : public AllocatorProvider {
public:
    explicit ObservedAllocatorProvider(std::shared_ptr<AllocationObservation> observation)
        : observation_(std::move(observation)) {}

    std::unique_ptr<Allocator> CreateAllocator(Device device) override {
        EXPECT_EQ(device, Device::CPU());
        ++observation_->allocator_creations;
        return std::make_unique<ObservedAllocator>(observation_);
    }

private:
    std::shared_ptr<AllocationObservation> observation_;
};

void ObserveCpuAllocator(RuntimeBuilder& builder,
                         const std::shared_ptr<AllocationObservation>& observation) {
    builder.RegisterCustomAllocatorProvider(
            DeviceType::kCPU, std::make_unique<ObservedAllocatorProvider>(observation));
}

TEST(RuntimeResourceLimits, RejectsKvPayloadOverLimitBeforeCreatingAllocator) {
    auto options = MakeOptions();
    options.kv_cache.max_bytes = kRequiredKvBytes - 1;
    const auto observation = std::make_shared<AllocationObservation>();
    RuntimeBuilder builder;
    builder.WithOptions(options);
    ObserveCpuAllocator(builder, observation);

    const auto result = builder.TryBuild();

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kResourceExhausted);
    EXPECT_NE(result.status().message().find("KV cache payload"), std::string::npos);
    EXPECT_NE(result.status().message().find("required=" + std::to_string(kRequiredKvBytes)), std::string::npos);
    EXPECT_NE(result.status().message().find("limit=" + std::to_string(kRequiredKvBytes - 1)), std::string::npos);
    EXPECT_EQ(observation->allocator_creations, 0U);
    EXPECT_EQ(observation->storage_calls, 0U);
}

TEST(RuntimeResourceLimits, AcceptsKvPayloadAtAndBelowLimit) {
    for (const size_t limit: {kRequiredKvBytes, kRequiredKvBytes + 1, size_t{0}}) {
        SCOPED_TRACE(limit);
        auto options = MakeOptions();
        options.kv_cache.max_bytes = limit;
        const auto observation = std::make_shared<AllocationObservation>();
        RuntimeBuilder builder;
        builder.WithOptions(options);
        ObserveCpuAllocator(builder, observation);

        const auto result = builder.TryBuild();

        ASSERT_TRUE(result.ok()) << result.status().ToString();
        ASSERT_NE(result->GetKVCacheManager(), nullptr);
        EXPECT_EQ(result->GetKVCacheManager()->total_bytes(), kRequiredKvBytes);
        EXPECT_EQ(observation->allocator_creations, 1U);
        EXPECT_EQ(observation->storage_calls, 2U);
    }
}

TEST(RuntimeResourceLimits, DisabledKvDoesNotPlanOrCreateAllocator) {
    RuntimeOptions options;
    options.backend.enable_cpu = false;
    options.kv_cache.max_bytes = 1;
    const auto observation = std::make_shared<AllocationObservation>();
    RuntimeBuilder builder;
    builder.WithOptions(options);
    ObserveCpuAllocator(builder, observation);

    const auto result = builder.TryBuild();

    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->GetKVCacheManager(), nullptr);
    EXPECT_EQ(observation->allocator_creations, 0U);
    EXPECT_EQ(observation->storage_calls, 0U);
}

TEST(RuntimeResourceLimits, RejectsInvalidKvGeometryBeforeBudgetComparison) {
    for (const bool invalid_alignment: {false, true}) {
        SCOPED_TRACE(invalid_alignment);
        auto options = MakeOptions();
        options.kv_cache.max_bytes = 1;
        if (invalid_alignment) {
            options.kv_cache.alignment = 3;
        } else {
            options.kv_cache.head_dim = 0;
        }
        const auto observation = std::make_shared<AllocationObservation>();
        RuntimeBuilder builder;
        builder.WithOptions(options);
        ObserveCpuAllocator(builder, observation);

        const auto result = builder.TryBuild();

        ASSERT_FALSE(result.ok());
        EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
        EXPECT_EQ(observation->allocator_creations, 0U);
        EXPECT_EQ(observation->storage_calls, 0U);
    }
}

TEST(RuntimeResourceLimits, RejectsKvByteOverflowBeforeCreatingAllocator) {
    auto options = MakeOptions();
    options.kv_cache.max_bytes = 1;
    options.kv_cache.num_layers = std::numeric_limits<size_t>::max();
    const auto observation = std::make_shared<AllocationObservation>();
    RuntimeBuilder builder;
    builder.WithOptions(options);
    ObserveCpuAllocator(builder, observation);

    const auto result = builder.TryBuild();

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kOverflow);
    EXPECT_EQ(observation->allocator_creations, 0U);
    EXPECT_EQ(observation->storage_calls, 0U);
}

TEST(RuntimeResourceLimits, BudgetFailureConsumesRegistrationsAndBuilderCanRetry) {
    auto options = MakeOptions();
    options.kv_cache.max_bytes = kRequiredKvBytes - 1;
    const auto observation = std::make_shared<AllocationObservation>();
    RuntimeBuilder builder;
    builder.WithOptions(options);
    ObserveCpuAllocator(builder, observation);
    builder.RegisterCustomAllocatorProvider(
            DeviceType::kCUDA, std::make_unique<CPUAllocatorProvider>());

    const auto failure = builder.TryBuild();
    ASSERT_FALSE(failure.ok());
    EXPECT_EQ(failure.status().code(), StatusCode::kResourceExhausted);
    EXPECT_EQ(observation->allocator_creations, 0U);

    options.kv_cache.max_bytes = kRequiredKvBytes;
    builder.WithOptions(options);
    const auto retry = builder.TryBuild();

    ASSERT_TRUE(retry.ok()) << retry.status().ToString();
    EXPECT_FALSE(retry->HasAllocatorProvider(DeviceType::kCUDA));
    ASSERT_NE(retry->GetKVCacheManager(), nullptr);
    EXPECT_EQ(retry->GetKVCacheManager()->total_bytes(), kRequiredKvBytes);
    EXPECT_EQ(observation->allocator_creations, 0U);
    EXPECT_EQ(observation->storage_calls, 0U);
}

TEST(RuntimeResourceLimits, AllocatorFailureWithinBudgetCanRetry) {
    auto options = MakeOptions();
    options.kv_cache.max_bytes = kRequiredKvBytes;
    const auto observation = std::make_shared<AllocationObservation>();
    observation->fail_storage = true;
    RuntimeBuilder builder;
    builder.WithOptions(options);
    ObserveCpuAllocator(builder, observation);
    const auto failure = builder.TryBuild();
    ASSERT_FALSE(failure.ok());
    EXPECT_EQ(failure.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(failure.status().message().empty());
    EXPECT_EQ(observation->storage_calls, 1U);

    observation->fail_storage = false;
    ObserveCpuAllocator(builder, observation);
    const auto retry = builder.TryBuild();

    ASSERT_TRUE(retry.ok()) << retry.status().ToString();
    EXPECT_EQ(observation->allocator_creations, 2U);
    EXPECT_EQ(observation->storage_calls, 3U);
}

TEST(RuntimeResourceLimits, WorkspaceLimitIsOwnedWithoutWorkspaceManager) {
    RuntimeOptions options;
    options.backend.enable_cpu = false;
    options.workspace.default_workspace_limit_bytes = 1234;
    RuntimeBuilder builder;
    builder.WithOptions(options);
    const auto first = builder.TryBuild();
    ASSERT_TRUE(first.ok());

    options.workspace.default_workspace_limit_bytes = 4321;
    builder.WithOptions(options);
    const auto second = builder.TryBuild();
    ASSERT_TRUE(second.ok());
    EXPECT_EQ(first->workspace_limit_bytes(), 1234U);
    EXPECT_EQ(second->workspace_limit_bytes(), 4321U);
}

TEST(RuntimeResourceLimits, WorkspaceLimitSurvivesRuntimeMoves) {
    RuntimeOptions options;
    options.backend.enable_cpu = false;
    options.workspace.default_workspace_limit_bytes = 1234;
    RuntimeBuilder builder;
    builder.WithOptions(options);
    auto result = builder.TryBuild();
    ASSERT_TRUE(result.ok());
    Runtime first = std::move(*result);
    Runtime second = std::move(first);
    EXPECT_EQ(second.workspace_limit_bytes(), 1234U);

    options.workspace.default_workspace_limit_bytes = 0;
    builder.WithOptions(options);
    auto other = builder.TryBuild();
    ASSERT_TRUE(other.ok());
    EXPECT_EQ(other->workspace_limit_bytes(), 0U);
    *other = std::move(second);
    EXPECT_EQ(other->workspace_limit_bytes(), 1234U);
}

} // namespace
