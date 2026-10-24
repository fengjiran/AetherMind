#include "memory/memory_junk_fill.h"

#include "aethermind/memory/buffer.h"
#include "aethermind/memory/cpu_allocator.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

namespace {

using aethermind::memory::detail::FillMemoryJunk;

constexpr uint64_t kOriginalPattern = 0x7fedbeef7fedbeefULL;
constexpr auto kOriginalBytes = std::bit_cast<std::array<unsigned char, sizeof(kOriginalPattern)>>(kOriginalPattern);

static_assert(noexcept(FillMemoryJunk(nullptr, 0)));

TEST(MemoryJunkFill, NullDestinationIsNoOp) {
    for (const size_t nbytes: {size_t{0}, size_t{1}, size_t{8}, size_t{31}}) {
        FillMemoryJunk(nullptr, nbytes);
    }
}

TEST(MemoryJunkFill, ZeroLengthPreservesStorage) {
    std::array<unsigned char, 16> storage;
    storage.fill(0xA5);
    const auto original = storage;

    FillMemoryJunk(storage.data() + 1, 0);

    EXPECT_EQ(storage, original);
}

TEST(MemoryJunkFill, UnalignedRegionsPreserveBoundariesAndOriginalPattern) {
    for (size_t offset = 0; offset < sizeof(kOriginalPattern); ++offset) {
        for (size_t length = 0; length <= 4 * sizeof(kOriginalPattern) + 1; ++length) {
            SCOPED_TRACE(offset);
            SCOPED_TRACE(length);
            alignas(uint64_t) std::array<unsigned char, 64> storage;
            storage.fill(0xA5);
            const size_t begin = sizeof(kOriginalPattern) + offset;

            FillMemoryJunk(storage.data() + begin, length);

            for (size_t i = 0; i < storage.size(); ++i) {
                const unsigned char expected = i >= begin && i < begin + length
                                                       ? kOriginalBytes[(i - begin) % kOriginalBytes.size()]
                                                       : 0xA5;
                EXPECT_EQ(storage[i], expected) << "byte " << i;
            }
        }
    }
}

TEST(MemoryJunkFill, DefaultAllocatorPolicyFillsOddSizedStorageWithOriginalPattern) {
    EXPECT_TRUE(aethermind::CPUAllocatorPolicy{}.junk_fill);
    aethermind::CPUAllocator allocator(aethermind::Device::CPU());
    for (const size_t length: {size_t{1}, size_t{7}, size_t{8}, size_t{9}, size_t{31}}) {
        SCOPED_TRACE(length);
        auto allocation = allocator.TryAllocate({.nbytes = length, .alignment = 64});
        ASSERT_TRUE(allocation.ok()) << allocation.status().ToString();
        const auto* bytes = static_cast<const unsigned char*>(allocation->data());
        for (size_t i = 0; i < length; ++i) {
            EXPECT_EQ(bytes[i], kOriginalBytes[i % kOriginalBytes.size()]);
        }
    }
}

} // namespace
