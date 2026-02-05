//
// Created by richard on 2/4/26.
//
#include <gtest/gtest.h>

// import ammalloc;
import ammemory_pool;
import ammalloc_config;

namespace {
using namespace aethermind;

TEST(ConfigTest, ParseSize) {
    // 基础测试
    EXPECT_EQ(ParseSize("100"), 100);
    EXPECT_EQ(ParseSize("1024"), 1024);

    // 单位测试 (大小写不敏感)
    EXPECT_EQ(ParseSize("1k"), 1024);
    EXPECT_EQ(ParseSize("1K"), 1024);
    EXPECT_EQ(ParseSize("1kb"), 1024);// 'b' 被忽略，只看首字母
    EXPECT_EQ(ParseSize("1M"), 1024 * 1024);
    EXPECT_EQ(ParseSize("2G"), 2ULL * 1024 * 1024 * 1024);

    // 空格测试
    EXPECT_EQ(ParseSize("  64"), 64);
    EXPECT_EQ(ParseSize("64 KB"), 64 * 1024);
    EXPECT_EQ(ParseSize("  10  mb  "), 10 * 1024 * 1024);

    // 边界与异常测试
    EXPECT_EQ(ParseSize(nullptr), 0);
    EXPECT_EQ(ParseSize(""), 0);
    EXPECT_EQ(ParseSize("abc"), 0); // 无效数字
    EXPECT_EQ(ParseSize("10x"), 10);// 未知单位，当作 Bytes

    // 溢出测试 (64位系统下)
    // 1. 正常边界测试
    // 10000 TB = 10 PB，远未溢出
    EXPECT_EQ(ParseSize("10000 TB"), 10000ULL * 1024 * 1024 * 1024 * 1024);

    // 2. 真正的溢出测试
    // 需要超过 16777216 TB (即 16 EB)
    // 这里使用 2000万 TB (20 EB)，肯定溢出
    EXPECT_EQ(ParseSize("20000000 TB"), std::numeric_limits<size_t>::max());
}

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
    for (size_t size = 1; size <= SizeConfig::MAX_TC_SIZE; size += 7) {
        size_t idx = SizeClass::Index(size);
        size_t aligned_size = SizeClass::Size(idx);
        EXPECT_GE(aligned_size, size);
        EXPECT_EQ(idx, SizeClass::Index(aligned_size));
    }
}

TEST(SizeClassTest, BatchStrategy) {
    // 小对象：批量数应为 512
    EXPECT_EQ(SizeClass::CalculateBatchSize(8), 512);
    // 大对象：批量数应为 2
    EXPECT_EQ(SizeClass::CalculateBatchSize(32 * 1024), 2);

    // 验证 GetMovePageNum 至少能支撑 8 次 Batch
    size_t size = 8;
    size_t batch = SizeClass::CalculateBatchSize(size);
    size_t pages = SizeClass::GetMovePageNum(size);
    size_t total_bytes = pages * SystemConfig::PAGE_SIZE;
    EXPECT_GE(total_bytes, batch * 8 * size);
}

}// namespace