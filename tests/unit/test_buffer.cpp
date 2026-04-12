//
// Batch 4: Buffer tests
//

#include "aethermind/memory/buffer.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <cstring>

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

}// namespace detail

namespace {

TEST(Buffer, EmptyBuffer) {
    Buffer b;
    EXPECT_TRUE(!b.is_initialized());
    EXPECT_TRUE(b.nbytes() == 0);
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
    
    EXPECT_TRUE(!b1.is_initialized());
    EXPECT_TRUE(b2.is_initialized());
    EXPECT_TRUE(b2.data() == original_ptr);
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
    for (size_t size : {1, 16, 64, 256, 1024, 4096}) {
        Buffer b = detail::make_test_buffer(size, 64);
        EXPECT_TRUE(b.is_initialized());
        EXPECT_TRUE(b.nbytes() == size);
    }
}

}// namespace