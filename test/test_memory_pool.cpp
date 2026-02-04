//
// Created by richard on 2/4/26.
//
#include <gtest/gtest.h>

import ammalloc;
import ammemory_pool;

namespace {
using namespace aethermind;

TEST(SizeClassTest, IndexAndSizeMapping) {
    // 1. 验证 8 字节对齐区间 [1, 128]
    EXPECT_EQ(SizeClass::Index(1), 0);
    EXPECT_EQ(SizeClass::Size(0), 8);
    EXPECT_EQ(SizeClass::Index(8), 0);
    EXPECT_EQ(SizeClass::Index(9), 1);
    EXPECT_EQ(SizeClass::Size(1), 16);
    EXPECT_EQ(SizeClass::Index(128), 15);
    EXPECT_EQ(SizeClass::Size(15), 128);

    // 2. 验证 16 字节对齐区间 [129, 1024]
    // 129 -> 对齐到 144 -> Index 16
    EXPECT_EQ(SizeClass::Index(129), 16);
    EXPECT_EQ(SizeClass::Size(16), 160);// 128 + 32

    // 3. 验证互逆性 (RoundTrip)
    // Size(Index(s)) 应该 >= s 且是对齐后的值
    for (size_t size = 1; size <= MagicConstants::MAX_TC_SIZE; size += 7) {
        size_t idx = SizeClass::Index(size);
        size_t aligned_size = SizeClass::Size(idx);
        EXPECT_GE(aligned_size, size);
        EXPECT_EQ(idx, SizeClass::Index(aligned_size));
    }
}

}// namespace