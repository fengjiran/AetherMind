#include "aethermind/memory/allocator.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/runtime/runtime.h"
#include "aethermind/runtime/runtime_builder.h"
#include "inference/test_malloc_interposer.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <new>
#include <utility>

namespace {
using namespace aethermind;

struct AlignedCpuBufferCase {
    const char* name;
    size_t nbytes;
    size_t alignment;
};

class AlignedCpuBuffer : public ::testing::TestWithParam<AlignedCpuBufferCase> {};

TEST_P(AlignedCpuBuffer, PreservesRequestedStorageAndMetadata) {
    const auto& request = GetParam();

    auto allocation = TryAllocateAlignedCpuBuffer(request.nbytes, request.alignment);

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
    auto allocation = TryAllocateAlignedCpuBuffer(0, 64);

    ASSERT_TRUE(allocation.ok()) << allocation.status().ToString();
    ASSERT_TRUE(allocation->is_initialized());
    EXPECT_EQ(allocation->nbytes(), 0U);
    EXPECT_NE(allocation->data(), nullptr);
    EXPECT_EQ(allocation->device(), Device::CPU());
    EXPECT_EQ(allocation->alignment(), 64U);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(allocation->data()) % 64, 0U);
    EXPECT_EQ(allocation->use_count(), 1U);
}

TEST_F(AlignedCpuBuffer, RejectsInvalidExplicitAlignment) {
    for (const size_t alignment: std::array<size_t, 3>{0, 3, 24}) {
        SCOPED_TRACE(alignment);
        const auto allocation = TryAllocateAlignedCpuBuffer(128, alignment);

        EXPECT_FALSE(allocation.ok());
        EXPECT_EQ(allocation.status().code(), StatusCode::kInvalidArgument);
    }
    for (const size_t alignment: std::array<size_t, 3>{1, 2, 4}) {
        if (alignment >= sizeof(void*)) {
            continue;
        }
        SCOPED_TRACE(alignment);
        const auto allocation = TryAllocateAlignedCpuBuffer(128, alignment);

        EXPECT_FALSE(allocation.ok());
        EXPECT_EQ(allocation.status().code(), StatusCode::kInvalidArgument);
    }
}

TEST_F(AlignedCpuBuffer, SystemAllocationFailureReturnsStatusAndCanRetry) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    static_assert(noexcept(TryAllocateAlignedCpuBuffer(128, 64)));

    test::ScopedMallocFailure failure(0, 0, false,
                                      test::MallocFailureTarget::kPosixMemalign);
    const auto allocation = TryAllocateAlignedCpuBuffer(128, 64);
    const auto result = failure.Stop();

    EXPECT_TRUE(result.allocation_failed);
    EXPECT_FALSE(allocation.ok());
    EXPECT_EQ(allocation.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(allocation.status().message().empty());
    auto retry = TryAllocateAlignedCpuBuffer(128, 64);
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
    const auto allocation = TryAllocateAlignedCpuBuffer(128, 64);
    const auto result = failure.Stop();

    EXPECT_TRUE(result.allocation_failed);
    EXPECT_FALSE(allocation.ok());
    EXPECT_EQ(allocation.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(allocation.status().message().empty());
    EXPECT_EQ(result.aligned_allocation_calls, 1U);
    EXPECT_TRUE(result.last_aligned_allocation_released);
    auto retry = TryAllocateAlignedCpuBuffer(128, 64);
    ASSERT_TRUE(retry.ok()) << retry.status().ToString();
    EXPECT_TRUE(retry->is_initialized());
    EXPECT_EQ(retry->nbytes(), 128U);
}

TEST_F(AlignedCpuBuffer, InvalidAlignmentDiagnosticFailureReturnsAnEmptyStatus) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }

    test::ScopedMallocFailure failure(0);
    const auto allocation = TryAllocateAlignedCpuBuffer(128, 0);
    const auto result = failure.Stop();

    EXPECT_TRUE(result.allocation_failed);
    EXPECT_FALSE(allocation.ok());
    EXPECT_EQ(allocation.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(allocation.status().message().empty());
    EXPECT_EQ(result.aligned_allocation_calls, 0U);
}

TEST_F(AlignedCpuBuffer, SharedBufferRemainsWritableAfterRuntimeDestruction) {
    Buffer surviving_copy;
    const void* raw_ptr = nullptr;
    {
        auto runtime = RuntimeBuilder().Build();
        auto allocation = TryAllocateAlignedCpuBuffer(128, 64);
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

class FakeAllocator : public Allocator {
public:
    Buffer Allocate(size_t) override { return {}; }
    [[nodiscard]] Device device() const noexcept override { return Device(kCUDA, 0); }
};

class FakeProvider : public AllocatorProvider {
public:
    std::unique_ptr<Allocator> CreateAllocator(Device) override {
        return std::make_unique<FakeAllocator>();
    }
};

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

} // namespace
