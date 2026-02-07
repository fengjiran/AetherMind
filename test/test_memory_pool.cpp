//
// Created by richard on 2/4/26.
//
#include "ammalloc/ammalloc.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

TEST(ConfigTest, ParseSize) {
    // 基础测试
    EXPECT_EQ(details::ParseSize("100"), 100);
    EXPECT_EQ(details::ParseSize("1024"), 1024);

    // 单位测试 (大小写不敏感)
    EXPECT_EQ(details::ParseSize("1k"), 1024);
    EXPECT_EQ(details::ParseSize("1K"), 1024);
    EXPECT_EQ(details::ParseSize("1kb"), 1024);// 'b' 被忽略，只看首字母
    EXPECT_EQ(details::ParseSize("1M"), 1024 * 1024);
    EXPECT_EQ(details::ParseSize("2G"), 2ULL * 1024 * 1024 * 1024);

    // 空格测试
    EXPECT_EQ(details::ParseSize("  64"), 64);
    EXPECT_EQ(details::ParseSize("64 KB"), 64 * 1024);
    EXPECT_EQ(details::ParseSize("  10  mb  "), 10 * 1024 * 1024);

    // 边界与异常测试
    EXPECT_EQ(details::ParseSize(nullptr), 0);
    EXPECT_EQ(details::ParseSize(""), 0);
    EXPECT_EQ(details::ParseSize("abc"), 0); // 无效数字
    EXPECT_EQ(details::ParseSize("10x"), 10);// 未知单位，当作 Bytes

    // 溢出测试 (64位系统下)
    // 1. 正常边界测试
    // 10000 TB = 10 PB，远未溢出
    EXPECT_EQ(details::ParseSize("10000 TB"), 10000ULL * 1024 * 1024 * 1024 * 1024);

    // 2. 真正的溢出测试
    // 需要超过 16777216 TB (即 16 EB)
    // 这里使用 2000万 TB (20 EB)，肯定溢出
    EXPECT_EQ(details::ParseSize("20000000 TB"), std::numeric_limits<size_t>::max());
}

TEST(ConfigUtilsTest, ParseBool) {
    // 1. 基础 Truthy 测试
    EXPECT_TRUE(details::ParseBool("1"));
    EXPECT_TRUE(details::ParseBool("true"));
    EXPECT_TRUE(details::ParseBool("on"));
    EXPECT_TRUE(details::ParseBool("yes"));

    // 2. 大小写混合测试
    EXPECT_TRUE(details::ParseBool("True"));
    EXPECT_TRUE(details::ParseBool("TRUE"));
    EXPECT_TRUE(details::ParseBool("On"));
    EXPECT_TRUE(details::ParseBool("Yes"));
    EXPECT_TRUE(details::ParseBool("tRuE"));

    // 3. 空格测试
    EXPECT_TRUE(details::ParseBool(" 1 "));
    EXPECT_TRUE(details::ParseBool("  true"));
    EXPECT_TRUE(details::ParseBool("on  "));

    // 4. Falsy 测试
    EXPECT_FALSE(details::ParseBool("0"));
    EXPECT_FALSE(details::ParseBool("false"));
    EXPECT_FALSE(details::ParseBool("off"));
    EXPECT_FALSE(details::ParseBool("no"));
    EXPECT_FALSE(details::ParseBool("random_string"));
    EXPECT_FALSE(details::ParseBool(""));
    EXPECT_FALSE(details::ParseBool(nullptr));

    // 5. 边界干扰测试
    EXPECT_FALSE(details::ParseBool("true_value"));// 前缀匹配不应算作 true
    EXPECT_FALSE(details::ParseBool("10"));        // 包含1但不完全是1
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

TEST(PageAllocatorTest, AllocSmall) {
    size_t page_num = 1;
    void* ptr = PageAllocator::SystemAlloc(page_num);
    // 1. 验证指针非空
    EXPECT_TRUE(ptr != nullptr);

    // 2. 验证读写权限 (防止只分配了虚拟地址但不可写)
    // 写入 Pattern
    int* int_ptr = static_cast<int*>(ptr);
    *int_ptr = 0xDEADBEEF;
    EXPECT_EQ(*int_ptr, 0xDEADBEEF);

    // 填充整个页，确保没有 Segfaul
    std::memset(ptr, 0xAB, page_num * SystemConfig::PAGE_SIZE);
    for (size_t i = 0; i < page_num * SystemConfig::PAGE_SIZE; ++i) {
        EXPECT_EQ(static_cast<unsigned char*>(ptr)[i], 0xAB);
    }

    PageAllocator::SystemFree(ptr, page_num);
}

TEST(PageAllocatorTest, AllocHugeAlignment) {
    // 1. 计算触发 HugePage 逻辑的阈值
    // 代码逻辑是: size >= HUGE_PAGE_SIZE / 2
    size_t huge_size = SystemConfig::HUGE_PAGE_SIZE;
    size_t page_num = huge_size >> SystemConfig::PAGE_SHIFT;// 申请 2MB (通常 512 页)
    void* ptr = PageAllocator::SystemAlloc(page_num);
    EXPECT_TRUE(ptr != nullptr);

    // 2. 【核心】验证对齐
    auto addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t alignment = SystemConfig::HUGE_PAGE_SIZE;

    // 地址必须能被 2MB 整除
    EXPECT_EQ(addr % alignment, 0)
            << "Pointer " << ptr << " is NOT aligned to " << alignment;

    // 3. 验证首尾读写 (确保 Trim 逻辑没有把需要的内存切掉)
    char* char_ptr = static_cast<char*>(ptr);
    size_t total_bytes = page_num * SystemConfig::PAGE_SIZE;
    // 写头部
    char_ptr[0] = 'A';
    // 写尾部 (最后一个字节)
    char_ptr[total_bytes - 1] = 'Z';

    EXPECT_EQ(char_ptr[0], 'A');
    EXPECT_EQ(char_ptr[total_bytes - 1], 'Z');

    PageAllocator::SystemFree(ptr, page_num);
}

TEST(PageAllocatorTest, MultipleAllocations) {
    std::vector<std::pair<void*, size_t>> allocations;

    // 混合分配：1页, 10页, 512页(大页)
    std::vector<size_t> sizes = {1, 10, 128, 512, 600};

    for (size_t pages: sizes) {
        void* ptr = PageAllocator::SystemAlloc(pages);
        EXPECT_TRUE(ptr != nullptr);

        // 简单写入验证
        static_cast<char*>(ptr)[0] = 0xFF;

        // 如果是大内存，顺便验证对齐
        size_t bytes = pages << SystemConfig::PAGE_SHIFT;
        if (bytes >= (SystemConfig::HUGE_PAGE_SIZE >> 1)) {
            EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % SystemConfig::HUGE_PAGE_SIZE, 0);
        }

        allocations.emplace_back(ptr, pages);
    }

    // 释放所有
    for (auto& [fst, snd]: allocations) {
        PageAllocator::SystemFree(fst, snd);
    }
}

TEST(PageAllocatorTest, InvalidArgs) {
    // 1. 申请 0 页
    // mmap 申请 0 大小通常会失败，PageAllocator 应该返回 nullptr 或处理
    // 根据实现，size=0 会传入 mmap，导致失败返回 nullptr
    void* ptr = PageAllocator::SystemAlloc(0);
    EXPECT_TRUE(ptr == nullptr);

    // 2. 释放 nullptr (不应崩溃)
    PageAllocator::SystemFree(nullptr, 100);

    // 3. 释放页数为 0 (不应崩溃)
    char dummy;
    PageAllocator::SystemFree(&dummy, 0);
}

TEST(PageAllocatorTest, AllocWithPopulateConfig) {
    // 设置环境变量 (Linux/macOS)
    // 注意：setenv 不是线程安全的，最好在 main 开始前设置，或者独立跑这个测试
    setenv("AM_USE_MAP_POPULATE", "1", 1);

    // 重新初始化 Config (如果 Config 是懒加载单例)
    // 这一步依赖于 RuntimeConfig 实现是否支持重置，
    // 或者我们假设这是程序启动后的第一次调用。
    void* ptr = PageAllocator::SystemAlloc(10); // 应该触发 MAP_POPULATE
    EXPECT_TRUE(ptr != nullptr);

    PageAllocator::SystemFree(ptr, 10);

    // 清理环境
    unsetenv("AM_USE_MAP_POPULATE");
}

}// namespace