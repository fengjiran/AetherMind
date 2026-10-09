//
// Batch 4: Buffer tests
//

#include "aethermind/memory/buffer.h"
#include "inference/test_malloc_interposer.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <type_traits>

namespace {

using namespace aethermind;

namespace detail {

inline void free_buffer(void*, void* ptr) noexcept {
    std::free(ptr);
}

inline Buffer make_test_buffer(size_t nbytes, size_t alignment = 64) {
    void* ptr = nullptr;
    int rc = posix_memalign(&ptr, alignment, nbytes == 0 ? 1 : nbytes);
    if (rc != 0 || ptr == nullptr) {
        return {};
    }
    return {nbytes, MemoryHandle(ptr, nullptr, &free_buffer, Device::CPU(), alignment)};
}

void CountDeletion(void* ctx, void*) noexcept {
    ++*static_cast<size_t*>(ctx);
}

void ExpectEmptyObservers(Buffer& buffer) {
    EXPECT_FALSE(buffer.is_initialized());
    EXPECT_EQ(buffer.nbytes(), 0U);
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.mutable_data(), nullptr);
    EXPECT_EQ(buffer.device().type(), kUndefined);
    EXPECT_EQ(buffer.alignment(), 0U);
    EXPECT_EQ(buffer.use_count(), 0U);
    EXPECT_FALSE(buffer.unique());
    EXPECT_EQ(buffer.impl(), nullptr);
}

} // namespace detail

TEST(Buffer, EmptyBuffer) {
    Buffer b;
    detail::ExpectEmptyObservers(b);
}

TEST(Buffer, CpuBufferBasic) {
    Buffer b = detail::make_test_buffer(1024, 64);

    EXPECT_TRUE(b.is_initialized());
    EXPECT_TRUE(b.nbytes() == 1024);
    EXPECT_TRUE(b.device().is_cpu());
    EXPECT_TRUE(b.alignment() == 64);
    EXPECT_TRUE(b.data() != nullptr);
    EXPECT_TRUE(b.mutable_data() != nullptr);
}

TEST(Buffer, ZeroSizedBuffer) {
    Buffer b = detail::make_test_buffer(0, 64);

    EXPECT_TRUE(b.is_initialized());
    EXPECT_TRUE(b.nbytes() == 0);
    EXPECT_NE(b.data(), nullptr);
    EXPECT_EQ(b.use_count(), 1U);
    EXPECT_TRUE(b.unique());
    EXPECT_NE(b.impl(), nullptr);
}

TEST(Buffer, SharedOwnership) {
    Buffer b1 = detail::make_test_buffer(256, 64);
    Buffer b2 = b1;

    EXPECT_TRUE(b1.is_initialized());
    EXPECT_TRUE(b2.is_initialized());
    EXPECT_TRUE(b1.data() == b2.data());

    b1 = Buffer();
    EXPECT_TRUE(!b1.is_initialized());
    EXPECT_TRUE(b2.is_initialized());

    b2 = Buffer();
    EXPECT_TRUE(!b2.is_initialized());
}

TEST(Buffer, MoveSemantics) {
    Buffer b1 = detail::make_test_buffer(128, 64);
    const void* original_ptr = b1.data();

    Buffer b2 = std::move(b1);

    detail::ExpectEmptyObservers(b1);
    EXPECT_TRUE(b2.is_initialized());
    EXPECT_TRUE(b2.data() == original_ptr);
}

TEST(Buffer, MoveAssignmentLeavesEmptyObservers) {
    Buffer original = detail::make_test_buffer(128);
    const void* original_ptr = original.data();
    ASSERT_NE(original_ptr, nullptr);
    Buffer destination = detail::make_test_buffer(64);

    destination = std::move(original);

    detail::ExpectEmptyObservers(original);
    EXPECT_EQ(destination.data(), original_ptr);
    EXPECT_EQ(destination.nbytes(), 128U);
    EXPECT_TRUE(destination.unique());
}

TEST(Buffer, DeleterRunsOnceAfterLastSharedOwner) {
    unsigned char data = 0;
    size_t deletions = 0;
    {
        Buffer first(1, MemoryHandle(&data, &deletions, &detail::CountDeletion, Device::CPU()));
        Buffer second = first;
        Buffer third;
        third = second;
        EXPECT_EQ(first.use_count(), 3U);

        Buffer moved = std::move(third);
        first = Buffer();
        second = Buffer();
        EXPECT_EQ(deletions, 0U);
        EXPECT_TRUE(moved.unique());
    }
    EXPECT_EQ(deletions, 1U);
}

TEST(Buffer, AssignmentReleasesPreviousStorageExactlyOnce) {
    unsigned char old_data = 0;
    unsigned char new_data = 0;
    size_t old_deletions = 0;
    size_t new_deletions = 0;
    Buffer destination(1, MemoryHandle(&old_data, &old_deletions, &detail::CountDeletion, Device::CPU()));
    Buffer source(1, MemoryHandle(&new_data, &new_deletions, &detail::CountDeletion, Device::CPU()));

    destination = source;

    EXPECT_EQ(old_deletions, 1U);
    EXPECT_EQ(new_deletions, 0U);
    destination = std::move(source);
    EXPECT_EQ(new_deletions, 0U);
    destination = Buffer();
    EXPECT_EQ(old_deletions, 1U);
    EXPECT_EQ(new_deletions, 1U);
}

TEST(Buffer, CopiesObserveSharedMutation) {
    Buffer first = detail::make_test_buffer(1);
    ASSERT_NE(first.mutable_data(), nullptr);
    const Buffer second = first;

    *static_cast<unsigned char*>(first.mutable_data()) = 42;

    EXPECT_EQ(*static_cast<const unsigned char*>(second.data()), 42);
    EXPECT_FALSE(first.unique());
    EXPECT_EQ(second.use_count(), 2U);
}

TEST(Buffer, UniqueCountsImplementationReferencesForBorrowedMemory) {
    unsigned char data = 0;
    Buffer first(1, MemoryHandle(&data, nullptr, nullptr, Device::CPU()));
    Buffer separate(1, MemoryHandle(&data, nullptr, &NoOpMemoryDeleter, Device::CPU()));

    EXPECT_EQ(first.data(), separate.data());
    EXPECT_TRUE(first.unique());
    EXPECT_TRUE(separate.unique());
    EXPECT_EQ(first.use_count(), 1U);
    EXPECT_NE(first.impl(), separate.impl());
}

TEST(Buffer, UnknownAlignmentAcceptsUnalignedCpuPointer) {
    alignas(64) unsigned char storage[64]{};
    Buffer buffer(1, MemoryHandle(storage + 1, nullptr, &NoOpMemoryDeleter, Device::CPU()));

    EXPECT_TRUE(buffer.is_initialized());
    EXPECT_EQ(buffer.data(), storage + 1);
    EXPECT_EQ(buffer.alignment(), 0U);
}

TEST(Buffer, NonCpuMetadataDoesNotApplyCpuPointerAlignmentCheck) {
    // This borrowed pointer is only an opaque metadata token; no device access occurs.
    alignas(64) unsigned char storage[64]{};
    for (const Device device: {Device::CUDA(), Device::CANN()}) {
        Buffer buffer(1, MemoryHandle(storage + 1, nullptr, &NoOpMemoryDeleter, device, 64));

        EXPECT_TRUE(buffer.is_initialized());
        EXPECT_EQ(buffer.device(), device);
        EXPECT_EQ(buffer.alignment(), 64U);
    }
}

TEST(Buffer, ZeroSizeAcceptsEmptyHandle) {
    Buffer buffer(0, MemoryHandle{});

    EXPECT_FALSE(buffer.is_initialized());
    EXPECT_EQ(buffer.nbytes(), 0U);
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_FALSE(buffer.unique());
}

TEST(Buffer, PreCreatedImplementationSharesStorage) {
    unsigned char data = 0;
    auto impl = make_object<BufferImpl>(1, MemoryHandle(&data, nullptr, &NoOpMemoryDeleter, Device::CPU()));

    Buffer buffer(impl);

    EXPECT_EQ(buffer.impl(), impl.get());
    EXPECT_EQ(buffer.data(), &data);
    EXPECT_EQ(buffer.use_count(), 2U);
}

TEST(Buffer, PublicOwnershipInterfacesHaveRequiredTypes) {
    static_assert(std::is_same_v<decltype(std::declval<const Buffer&>().impl()), const BufferImpl*>);
    static_assert(std::is_nothrow_invocable_v<memory_deleter_fn, void*, void*>);
    static_assert(!std::is_convertible_v<void (*)(void*, void*), memory_deleter_fn>);
    static_assert(!std::is_nothrow_constructible_v<Buffer, size_t, MemoryHandle>);
}

TEST(Buffer, MetadataAllocationFailurePropagatesAndReleasesHandle) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    Buffer empty;
    detail::ExpectEmptyObservers(empty);
    unsigned char data = 0;
    size_t deletions = 0;
    bool caught_bad_alloc = false;

    test::ScopedMallocFailure failure(sizeof(BufferImpl));
    try {
        Buffer buffer(1, MemoryHandle(&data, &deletions, &detail::CountDeletion, Device::CPU()));
    } catch (const std::bad_alloc&) {
        caught_bad_alloc = true;
    }
    const auto result = failure.Stop();

    EXPECT_TRUE(result.allocation_failed);
    EXPECT_TRUE(caught_bad_alloc);
    EXPECT_EQ(deletions, 1U);
    Buffer subsequent(1, MemoryHandle(&data, nullptr, &NoOpMemoryDeleter, Device::CPU()));
    EXPECT_TRUE(subsequent.is_initialized());
}

TEST(Buffer, NonzeroSizeWithEmptyHandleDeath) {
    EXPECT_DEATH((void) Buffer(1, MemoryHandle{}), "Check failed");
}

TEST(Buffer, UndefinedDeviceWithNonemptyHandleDeath) {
    unsigned char data = 0;
    EXPECT_DEATH((void) Buffer(1, MemoryHandle(&data, nullptr, nullptr, Device(kUndefined))), "Check failed");
}

TEST(Buffer, UnknownDeviceWithNonemptyHandleDeath) {
    unsigned char data = 0;
    EXPECT_DEATH((void) Buffer(1, MemoryHandle(&data, nullptr, nullptr, Device(static_cast<DeviceType>(255)))), "Check failed");
}

TEST(Buffer, NonPowerOfTwoAlignmentDeath) {
    unsigned char data = 0;
    EXPECT_DEATH((void) Buffer(1, MemoryHandle(&data, nullptr, nullptr, Device::CPU(), 3)), "Check failed");
}

TEST(Buffer, FalseCpuAlignmentClaimDeath) {
    alignas(64) unsigned char storage[64]{};
    EXPECT_DEATH((void) Buffer(1, MemoryHandle(storage + 1, nullptr, nullptr, Device::CPU(), 64)), "Check failed");
}

TEST(Buffer, AlignmentTracking) {
    Buffer b = detail::make_test_buffer(512, 128);
    EXPECT_TRUE(b.alignment() == 128);
}

TEST(Buffer, DataPtrAccess) {
    Buffer b = detail::make_test_buffer(64, 64);

    const void* const_data = b.data();
    void* mutable_data = b.mutable_data();

    EXPECT_TRUE(const_data != nullptr);
    EXPECT_TRUE(mutable_data != nullptr);

    std::memset(mutable_data, 0xFF, 64);

    const unsigned char* bytes = static_cast<const unsigned char*>(const_data);
    for (size_t i = 0; i < 64; ++i) {
        EXPECT_TRUE(bytes[i] == 0xFF);
    }
}

TEST(Buffer, DifferentSizes) {
    for (size_t size: {1, 16, 64, 256, 1024, 4096}) {
        Buffer b = detail::make_test_buffer(size, 64);
        EXPECT_TRUE(b.is_initialized());
        EXPECT_TRUE(b.nbytes() == size);
    }
}

} // namespace
