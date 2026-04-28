#include "aethermind/memory/allocator.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/runtime/runtime_builder.h"
#include "aethermind/runtime/runtime_context.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

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

}// namespace
