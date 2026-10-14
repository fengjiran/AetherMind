#include "aethermind/memory/allocator.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/memory/memory_handle.h"
#include "aethermind/runtime/runtime.h"
#include "aethermind/runtime/runtime_builder.h"
#include "inference/test_malloc_interposer.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <utility>

namespace {
using namespace aethermind;

struct AlignedCpuBufferCase {
    const char* name;
    size_t nbytes;
    size_t alignment;
};

class AlignedCpuBuffer : public ::testing::TestWithParam<AlignedCpuBufferCase> {
protected:
    CPUAllocator allocator_{Device::CPU()};
};

TEST_P(AlignedCpuBuffer, PreservesRequestedStorageAndMetadata) {
    const auto& request = GetParam();

    auto allocation = allocator_.TryAllocate(
            {.nbytes = request.nbytes, .alignment = request.alignment});

    ASSERT_TRUE(allocation.ok()) << allocation.status().ToString();
    Buffer& buffer = *allocation;
    ASSERT_TRUE(buffer.is_initialized());
    ASSERT_NE(buffer.mutable_data(), nullptr);
    EXPECT_EQ(buffer.nbytes(), request.nbytes);
    EXPECT_EQ(buffer.device(), Device::CPU());
    EXPECT_EQ(buffer.alignment(), request.alignment);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer.data()) % request.alignment, 0U);
    EXPECT_EQ(buffer.use_count(), 1U);

    std::memset(buffer.mutable_data(), 0xA5, request.nbytes);
    const auto* const bytes = static_cast<const unsigned char*>(buffer.data());
    EXPECT_EQ(bytes[0], 0xA5U);
    EXPECT_EQ(bytes[request.nbytes - 1], 0xA5U);
}

INSTANTIATE_TEST_SUITE_P(
        ValidRequests,
        AlignedCpuBuffer,
        ::testing::Values(
                AlignedCpuBufferCase{"PointerAlignment", 129, sizeof(void*)},
                AlignedCpuBufferCase{"CacheLineAlignment", 257, 64},
                AlignedCpuBufferCase{"PageAlignment", 8193, 4096}),
        [](const ::testing::TestParamInfo<AlignedCpuBufferCase>& info) { return info.param.name; });

TEST_F(AlignedCpuBuffer, ZeroByteAllocationKeepsAValidOwningBuffer) {
    auto allocation = allocator_.TryAllocate({.nbytes = 0, .alignment = 64});

    ASSERT_TRUE(allocation.ok()) << allocation.status().ToString();
    ASSERT_TRUE(allocation->is_initialized());
    EXPECT_EQ(allocation->nbytes(), 0U);
    EXPECT_NE(allocation->data(), nullptr);
    EXPECT_EQ(allocation->device(), Device::CPU());
    EXPECT_EQ(allocation->alignment(), 64U);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(allocation->data()) % 64, 0U);
    EXPECT_EQ(allocation->use_count(), 1U);
}

TEST_F(AlignedCpuBuffer, RejectsNonPowerOfTwoAlignmentBeforeAllocating) {
    for (const size_t alignment: std::array<size_t, 3>{3, 6, 24}) {
        SCOPED_TRACE(alignment);
        test::BeginMallocCallCounting();
        const auto allocation = allocator_.TryAllocate(
                {.nbytes = 128, .alignment = alignment});
        const auto counts = test::EndMallocCallCounting();

        EXPECT_FALSE(allocation.ok());
        EXPECT_EQ(allocation.status().code(), StatusCode::kInvalidArgument);
        // Validation precedes storage, so a rejected request touches neither
        // the system allocator nor the ledger.
        EXPECT_EQ(counts.posix_memalign_calls, 0U);
        EXPECT_EQ(allocator_.allocation_state()->Snapshot().total_allocations, 0U);
    }
}

TEST_F(AlignedCpuBuffer, PromotesAlignmentBelowPointerSize) {
    // A non-zero alignment is a floor, and posix_memalign cannot express one
    // weaker than sizeof(void*), so the allocator promotes instead of rejecting.
    for (const size_t alignment: std::array<size_t, 3>{1, 2, 4}) {
        if (alignment >= sizeof(void*)) {
            continue;
        }
        SCOPED_TRACE(alignment);
        const auto allocation = allocator_.TryAllocate(
                {.nbytes = 128, .alignment = alignment});

        ASSERT_TRUE(allocation.ok()) << allocation.status().ToString();
        EXPECT_GE(allocation->alignment(), sizeof(void*));
        EXPECT_TRUE(std::has_single_bit(allocation->alignment()));
        EXPECT_EQ(reinterpret_cast<uintptr_t>(allocation->data()) %
                          allocation->alignment(),
                  0U);
    }
}

TEST_F(AlignedCpuBuffer, ZeroAlignmentUsesTheDefaultPolicy) {
    const auto allocation = allocator_.TryAllocate({.nbytes = 128});

    ASSERT_TRUE(allocation.ok()) << allocation.status().ToString();
    EXPECT_GE(allocation->alignment(), sizeof(void*));
    EXPECT_TRUE(std::has_single_bit(allocation->alignment()));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(allocation->data()) %
                      allocation->alignment(),
              0U);
}

TEST_F(AlignedCpuBuffer, SystemAllocationFailureReturnsStatusAndCanRetry) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    static_assert(noexcept(allocator_.TryAllocate({.nbytes = 128, .alignment = 64})));

    test::ScopedMallocFailure failure(0, 0, false,
                                      test::MallocFailureTarget::kPosixMemalign);
    const auto allocation = allocator_.TryAllocate({.nbytes = 128, .alignment = 64});
    const auto result = failure.Stop();

    EXPECT_TRUE(result.allocation_failed);
    EXPECT_FALSE(allocation.ok());
    EXPECT_EQ(allocation.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(allocation.status().message().empty());
    EXPECT_EQ(allocator_.allocation_state()->Snapshot().system_allocation_failures, 1U);
    auto retry = allocator_.TryAllocate({.nbytes = 128, .alignment = 64});
    ASSERT_TRUE(retry.ok()) << retry.status().ToString();
    EXPECT_TRUE(retry->is_initialized());
    EXPECT_EQ(retry->nbytes(), 128U);
    EXPECT_EQ(retry->alignment(), 64U);
}

TEST_F(AlignedCpuBuffer, MetadataFailureReleasesPayloadAndCanRetry) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }

    test::ScopedMallocFailure failure(0, 0, true);
    const auto allocation = allocator_.TryAllocate({.nbytes = 128, .alignment = 64});
    const auto result = failure.Stop();

    EXPECT_TRUE(result.allocation_failed);
    EXPECT_FALSE(allocation.ok());
    EXPECT_EQ(allocation.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(allocation.status().message().empty());
    EXPECT_EQ(result.aligned_allocation_calls, 1U);
    EXPECT_TRUE(result.last_aligned_allocation_released);
    // A failed allocation must not leave capacity charged.
    const auto stats = allocator_.allocation_state()->Snapshot();
    EXPECT_EQ(stats.live_bytes, 0U);
    EXPECT_EQ(stats.live_allocations, 0U);
    auto retry = allocator_.TryAllocate({.nbytes = 128, .alignment = 64});
    ASSERT_TRUE(retry.ok()) << retry.status().ToString();
    EXPECT_TRUE(retry->is_initialized());
    EXPECT_EQ(retry->nbytes(), 128U);
}

TEST_F(AlignedCpuBuffer, SharedBufferRemainsWritableAfterRuntimeDestruction) {
    Buffer surviving_copy;
    const void* raw_ptr = nullptr;
    {
        auto runtime = RuntimeBuilder().Build();
        auto allocation = runtime.GetAllocator(Device::CPU()).TryAllocate({.nbytes = 128, .alignment = 64});
        ASSERT_TRUE(allocation.ok()) << allocation.status().ToString();
        Buffer original = std::move(*allocation);
        raw_ptr = original.data();
        ASSERT_NE(raw_ptr, nullptr);
        std::memset(original.mutable_data(), 0x5A, original.nbytes());

        surviving_copy = original;
        EXPECT_EQ(original.use_count(), 2U);
        EXPECT_EQ(surviving_copy.data(), raw_ptr);
    }

    ASSERT_TRUE(surviving_copy.is_initialized());
    EXPECT_EQ(surviving_copy.data(), raw_ptr);
    EXPECT_EQ(surviving_copy.use_count(), 1U);
    EXPECT_EQ(static_cast<const unsigned char*>(surviving_copy.data())[127], 0x5AU);
    static_cast<unsigned char*>(surviving_copy.mutable_data())[127] = 0xA5;
    EXPECT_EQ(static_cast<const unsigned char*>(surviving_copy.data())[127], 0xA5U);
    surviving_copy = Buffer();
}

TEST(CPUAllocator, BasicAllocation) {
    Device cpu_dev = Device::CPU();
    CPUAllocator allocator(cpu_dev);

    constexpr size_t kSize = 1024;
    Buffer buffer = allocator.Allocate(kSize);

    ASSERT_TRUE(buffer.is_initialized());
    EXPECT_EQ(buffer.nbytes(), kSize);
    EXPECT_EQ(buffer.device(), cpu_dev);
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_GE(buffer.alignment(), 16);
    EXPECT_EQ(buffer.use_count(), 1);
}

TEST(CPUAllocator, ZeroByteAllocation) {
    Device cpu_dev = Device::CPU();
    CPUAllocator allocator(cpu_dev);

    Buffer buffer = allocator.Allocate(0);

    ASSERT_TRUE(buffer.is_initialized());
    EXPECT_EQ(buffer.nbytes(), 0);
    EXPECT_NE(buffer.data(), nullptr);
}

TEST(CPUAllocator, SystemAllocationFailureThrowsBadAllocAndCanRetry) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    CPUAllocator allocator(Device::CPU());
    bool caught_bad_alloc = false;
    test::ScopedMallocFailure failure(0, 0, false,
                                      test::MallocFailureTarget::kPosixMemalign);
    try {
        static_cast<void>(allocator.Allocate(128));
    } catch (const std::bad_alloc&) {
        caught_bad_alloc = true;
    }
    const auto result = failure.Stop();

    EXPECT_TRUE(result.allocation_failed);
    EXPECT_TRUE(caught_bad_alloc);
    Buffer retry = allocator.Allocate(128);
    EXPECT_TRUE(retry.is_initialized());
    EXPECT_EQ(retry.nbytes(), 128U);
}

TEST(CPUAllocator, ProviderAndRegistry) {
    auto ctx = RuntimeBuilder().Build();

    Device cpu_dev = Device::CPU();
    Allocator& alloc1 = ctx.GetAllocator(cpu_dev);
    EXPECT_EQ(alloc1.device(), cpu_dev);

    Allocator& alloc2 = ctx.GetAllocator(cpu_dev);
    EXPECT_EQ(&alloc1, &alloc2);

    Buffer buf = alloc1.Allocate(64);
    EXPECT_TRUE(buf.is_initialized());
}

TEST(CPUAllocator, BufferReleaseAfterContextDestruction) {
    Buffer buffer;
    void* raw_ptr = nullptr;

    {
        auto ctx = RuntimeBuilder().Build();
        Allocator& alloc = ctx.GetAllocator(Device::CPU());
        buffer = alloc.Allocate(128);
        raw_ptr = buffer.mutable_data();

        ASSERT_NE(raw_ptr, nullptr);
        EXPECT_EQ(buffer.use_count(), 1);
    }

    ASSERT_TRUE(buffer.is_initialized());
    EXPECT_EQ(buffer.mutable_data(), raw_ptr);
    EXPECT_EQ(buffer.use_count(), 1);

    buffer = Buffer();
}

TEST(CPUAllocator, MultipleBuffersSharingImpl) {
    Device cpu_dev = Device::CPU();
    CPUAllocator allocator(cpu_dev);

    Buffer buf1 = allocator.Allocate(256);
    {
        Buffer buf2 = buf1;
        EXPECT_EQ(buf1.use_count(), 2);
        EXPECT_EQ(buf2.use_count(), 2);
        EXPECT_EQ(buf1.data(), buf2.data());
    }
    EXPECT_EQ(buf1.use_count(), 1);
}

TEST(CPUAllocator, RegistryNegativePath) {
    AllocatorRegistry registry;
    EXPECT_DEATH(registry.GetAllocator(Device::CPU()), "No allocator provider registered");

    registry.RegisterProvider(DeviceType::kCPU, std::make_unique<CPUAllocatorProvider>());
    EXPECT_NO_THROW(registry.GetAllocator(Device::CPU()));
}

TEST(CPUAllocator, DefaultRegistration) {
    auto ctx = RuntimeBuilder().Build();
    Device cpu_dev = Device::CPU();
    EXPECT_NO_THROW(ctx.GetAllocator(cpu_dev));
}

TEST(AllocationAccounting, AllocateAndTryAllocateChargeTheSameLedger) {
    auto state = AllocationState::Create();
    CPUAllocator allocator(Device::CPU(), {}, state);

    {
        Buffer legacy = allocator.Allocate(2048);
        ASSERT_TRUE(legacy.is_initialized());
        const auto stats = state->Snapshot();
        EXPECT_EQ(stats.live_bytes, 2048U);
        EXPECT_EQ(stats.live_allocations, 1U);
        EXPECT_EQ(stats.total_allocations, 1U);
    }

    const auto after = state->Snapshot();
    EXPECT_EQ(after.live_bytes, 0U);
    EXPECT_EQ(after.total_releases, 1U);
}

TEST(AllocationAccounting, ZeroByteBufferChargesTheByteItReserves) {
    auto state = AllocationState::Create();
    CPUAllocator allocator(Device::CPU(), {}, state);

    auto allocation = allocator.TryAllocate({.nbytes = 0, .alignment = 64});
    ASSERT_TRUE(allocation.ok()) << allocation.status().ToString();
    EXPECT_EQ(allocation->nbytes(), 0U);
    EXPECT_EQ(state->Snapshot().live_bytes, 1U);
}

TEST(AllocationAccounting, SharedBufferRefundsExactlyOnce) {
    auto state = AllocationState::Create();
    CPUAllocator allocator(Device::CPU(), {}, state);

    {
        auto allocation = allocator.TryAllocate({.nbytes = 256, .alignment = 64});
        ASSERT_TRUE(allocation.ok()) << allocation.status().ToString();
        Buffer copy = *allocation;
        EXPECT_EQ(copy.data(), allocation->data());

        const auto during = state->Snapshot();
        EXPECT_EQ(during.live_bytes, 256U);
        EXPECT_EQ(during.live_allocations, 1U);
    }

    const auto after = state->Snapshot();
    EXPECT_EQ(after.live_bytes, 0U);
    EXPECT_EQ(after.live_allocations, 0U);
    EXPECT_EQ(after.total_allocations, 1U);
    EXPECT_EQ(after.total_releases, 1U);
}

TEST(AllocationAccounting, BorrowedStorageIsNotCharged) {
    auto state = AllocationState::Create();
    CPUAllocator allocator(Device::CPU(), {}, state);

    alignas(64) std::array<std::byte, 128> stack_storage{};
    {
        // A null deleter means borrowed memory: it never came from the
        // allocator, so the ledger must neither charge nor release it.
        Buffer borrowed{stack_storage.size(),
                        MemoryHandle(stack_storage.data(), nullptr, nullptr,
                                     Device::CPU(), 64)};
        ASSERT_TRUE(borrowed.is_initialized());

        const auto during = state->Snapshot();
        EXPECT_EQ(during.live_bytes, 0U);
        EXPECT_EQ(during.live_allocations, 0U);
    }

    const auto after = state->Snapshot();
    EXPECT_EQ(after.total_allocations, 0U);
    EXPECT_EQ(after.total_releases, 0U);
}

TEST(AllocationAccounting, TagCountersTrackEachStorageClassSeparately) {
    auto state = AllocationState::Create();
    CPUAllocator allocator(Device::CPU(), {}, state);

    auto kv = allocator.TryAllocate(
            {.nbytes = 1024, .tag = AllocationTag::KVCache()});
    ASSERT_TRUE(kv.ok()) << kv.status().ToString();
    auto packed = allocator.TryAllocate(
            {.nbytes = 256, .tag = AllocationTag::PackedWeight()});
    ASSERT_TRUE(packed.ok()) << packed.status().ToString();

    const auto kv_stats = state->TagSnapshot(AllocationTag::KVCache());
    EXPECT_EQ(kv_stats.live_bytes, 1024U);
    EXPECT_EQ(kv_stats.live_allocations, 1U);
    EXPECT_EQ(kv_stats.peak_live_bytes, 1024U);
    EXPECT_EQ(kv_stats.total_allocations, 1U);

    const auto packed_stats = state->TagSnapshot(AllocationTag::PackedWeight());
    EXPECT_EQ(packed_stats.live_bytes, 256U);
    EXPECT_EQ(packed_stats.total_allocations, 1U);

    // An unused tag reports zeros rather than being absent.
    EXPECT_EQ(state->TagSnapshot(AllocationTag::Workspace()).live_bytes, 0U);
    EXPECT_EQ(state->Snapshot().live_bytes, 1280U);
}

TEST(AllocationAccounting, PeakSurvivesRelease) {
    auto state = AllocationState::Create();
    CPUAllocator allocator(Device::CPU(), {}, state);

    {
        auto first = allocator.TryAllocate({.nbytes = 4096});
        ASSERT_TRUE(first.ok()) << first.status().ToString();
        auto second = allocator.TryAllocate({.nbytes = 4096});
        ASSERT_TRUE(second.ok()) << second.status().ToString();
        EXPECT_EQ(state->Snapshot().peak_live_bytes, 8192U);
    }

    const auto after = state->Snapshot();
    EXPECT_EQ(after.live_bytes, 0U);
    EXPECT_EQ(after.peak_live_bytes, 8192U);
}

TEST(AllocationAccounting, BufferOutlivingItsAllocatorRefundsTheSharedState) {
    auto state = AllocationState::Create();
    Buffer survivor;
    {
        CPUAllocator allocator(Device::CPU(), {}, state);
        auto allocation = allocator.TryAllocate({.nbytes = 512, .alignment = 64});
        ASSERT_TRUE(allocation.ok()) << allocation.status().ToString();
        survivor = std::move(*allocation);
        EXPECT_EQ(state->Snapshot().live_bytes, 512U);
    }

    ASSERT_TRUE(survivor.is_initialized());
    EXPECT_EQ(state->Snapshot().live_bytes, 512U);
    survivor = Buffer();
    EXPECT_EQ(state->Snapshot().live_bytes, 0U);
    EXPECT_EQ(state->Snapshot().total_releases, 1U);
}

TEST(AllocationAccounting, LastBufferReleaseDestroysTheLedger) {
    std::weak_ptr<AllocationState> weak;
    Buffer survivor;
    {
        auto state = AllocationState::Create();
        weak = state;
        {
            CPUAllocator allocator(Device::CPU(), {}, state);
            auto allocation = allocator.TryAllocate({.nbytes = 128});
            ASSERT_TRUE(allocation.ok()) << allocation.status().ToString();
            survivor = std::move(*allocation);
        }
        // Both the allocator and the test's own strong reference are gone; the
        // release record must be the only owner left.
    }
    ASSERT_FALSE(weak.expired());

    survivor = Buffer();
    EXPECT_TRUE(weak.expired());
}

TEST(AllocationAccounting, ProviderReplacementKeepsAccountingOldBuffers) {
    AllocatorRegistry registry;
    registry.SetProvider(DeviceType::kCPU, std::make_unique<CPUAllocatorProvider>());
    const auto state = registry.allocation_state();

    Buffer survivor;
    {
        Allocator& first = registry.GetAllocator(Device::CPU());
        auto allocation = first.TryAllocate({.nbytes = 256, .alignment = 64});
        ASSERT_TRUE(allocation.ok()) << allocation.status().ToString();
        survivor = std::move(*allocation);
        EXPECT_EQ(state->Snapshot().live_bytes, 256U);
    }

    // Replacing the provider destroys the cached allocator instance that
    // produced the surviving Buffer.
    registry.SetProvider(DeviceType::kCPU, std::make_unique<CPUAllocatorProvider>());
    Allocator& second = registry.GetAllocator(Device::CPU());
    EXPECT_EQ(second.allocation_state(), state);

    EXPECT_EQ(state->Snapshot().live_bytes, 256U);
    survivor = Buffer();
    EXPECT_EQ(state->Snapshot().live_bytes, 0U);
    EXPECT_EQ(state->Snapshot().total_releases, 1U);
}

TEST(AllocationAccounting, CustomTagIdsStayOutOfTheReservedRange) {
    static_assert(AllocationTag::General().id() < AllocationTag::kCustomBase);
    static_assert(AllocationTag::Activation().id() < AllocationTag::kCustomBase);
    static_assert(AllocationTag::KVCache().id() < AllocationTag::kCustomBase);
    static_assert(AllocationTag::PackedWeight().id() < AllocationTag::kCustomBase);
    static_assert(AllocationTag::Workspace().id() < AllocationTag::kCustomBase);

    auto state = AllocationState::Create();
    CPUAllocator allocator(Device::CPU(), {}, state);
    const AllocationTag custom{AllocationTag::kCustomBase};

    auto allocation = allocator.TryAllocate({.nbytes = 64, .tag = custom});
    ASSERT_TRUE(allocation.ok()) << allocation.status().ToString();
    EXPECT_EQ(state->TagSnapshot(custom).live_bytes, 64U);
    EXPECT_EQ(state->TagSnapshot(AllocationTag::General()).live_bytes, 0U);
}

} // namespace
