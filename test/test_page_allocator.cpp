//
// Created by richard on 2/9/26.
//
#include "ammalloc/page_allocator.h"

#include <cstring>
#include <gtest/gtest.h>
#include <sys/mman.h>

namespace {
using namespace aethermind;

class PageAllocatorTest : public ::testing::Test {
public:
    void SetUp() override {
        PageAllocator::ResetStats();
        PageAllocator::ReleaseHugePageCache();
        g_mock_huge_alloc_fail.store(false, std::memory_order_relaxed);
    }

    void TearDown() override {
        PageAllocator::ReleaseHugePageCache();
    }

    // 辅助函数：验证指针有效性
    bool IsValidPtr(void* ptr) {
        return ptr != nullptr && ptr != MAP_FAILED;
    }

    // 辅助函数：模拟大页分配失败（修改mmap返回值，仅测试用）
    static void MockHugePageAllocFail() {
        // 可通过全局标志/环境变量控制AllocHugePage返回nullptr
        g_mock_huge_alloc_fail.store(true, std::memory_order_relaxed);
    }

    static void ResetMock() {
        g_mock_huge_alloc_fail.store(false, std::memory_order_relaxed);
    }
};

// ========== 测试用例1：普通页分配/释放（无缓存） ==========
TEST_F(PageAllocatorTest, NormalPageAllocFree) {
    size_t page_num = 1;
    void* ptr = PageAllocator::SystemAlloc(page_num);
    // 1. 验证指针非空
    EXPECT_TRUE(IsValidPtr(ptr));

    const auto& stats = PageAllocator::GetStats();
    EXPECT_EQ(stats.normal_alloc_count.load(), 1);
    EXPECT_EQ(stats.normal_alloc_success.load(), 1);
    EXPECT_EQ(stats.normal_alloc_bytes.load(), SystemConfig::PAGE_SIZE * page_num);
    EXPECT_EQ(stats.huge_alloc_count.load(), 0);// 无大页请求

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
    EXPECT_EQ(stats.free_count.load(), 1);
    EXPECT_EQ(stats.free_bytes.load(), SystemConfig::PAGE_SIZE * page_num);

    EXPECT_EQ(stats.huge_cache_hit_count.load(), 0);
    EXPECT_EQ(stats.huge_cache_miss_count.load(), 0);
}

// ========== 测试用例2：大页分配/释放（缓存未命中） ==========
TEST_F(PageAllocatorTest, HugePageAllocFree_MissCache) {
    size_t page_num = SystemConfig::HUGE_PAGE_SIZE / SystemConfig::PAGE_SIZE;
    void* ptr = PageAllocator::SystemAlloc(page_num);
    EXPECT_TRUE(IsValidPtr(ptr));

    const auto& stats = PageAllocator::GetStats();
    EXPECT_EQ(stats.huge_alloc_count.load(), 1);
    EXPECT_EQ(stats.huge_alloc_success.load(), 1);
    EXPECT_EQ(stats.huge_alloc_bytes.load(), SystemConfig::HUGE_PAGE_SIZE);
    EXPECT_EQ(stats.huge_cache_miss_count.load(), 1);
    EXPECT_EQ(stats.huge_cache_hit_count.load(), 0);

    PageAllocator::SystemFree(ptr, page_num);
    EXPECT_EQ(stats.free_count.load(), 1);
    EXPECT_EQ(stats.free_bytes.load(), SystemConfig::HUGE_PAGE_SIZE);
}

// ========== 测试用例3：大页分配（缓存命中） ==========
TEST_F(PageAllocatorTest, HugePageAlloc_HitCache) {
    size_t page_num = SystemConfig::HUGE_PAGE_SIZE / SystemConfig::PAGE_SIZE;
    void* ptr1 = PageAllocator::SystemAlloc(page_num);
    EXPECT_TRUE(IsValidPtr(ptr1));
    PageAllocator::SystemFree(ptr1, page_num);

    void* ptr2 = PageAllocator::SystemAlloc(page_num);
    EXPECT_TRUE(IsValidPtr(ptr2));

    const auto& stats = PageAllocator::GetStats();
    EXPECT_EQ(stats.huge_cache_hit_count.load(), 1);
    EXPECT_EQ(stats.huge_cache_miss_count.load(), 1);// 第一次未命中
    EXPECT_EQ(stats.huge_alloc_count.load(), 1);     // 缓存命中不触发新分配

    PageAllocator::SystemFree(ptr2, page_num);
}

// ========== 测试用例4：大页分配失败→降级到普通页 ==========
TEST_F(PageAllocatorTest, HugePageAllocFail_FallbackToNormal) {
    MockHugePageAllocFail();

    size_t page_num = SystemConfig::HUGE_PAGE_SIZE / SystemConfig::PAGE_SIZE;
    void* ptr = PageAllocator::SystemAlloc(page_num);
    EXPECT_TRUE(IsValidPtr(ptr));

    const auto& stats = PageAllocator::GetStats();
    EXPECT_EQ(stats.huge_alloc_count.load(), 1);
    EXPECT_EQ(stats.huge_alloc_success.load(), 0);           // 大页分配失败
    EXPECT_EQ(stats.huge_fallback_to_normal_count.load(), 1);// 降级次数
    EXPECT_EQ(stats.normal_alloc_count.load(), 1);           // 普通页分配（降级）
    EXPECT_EQ(stats.normal_alloc_success.load(), 1);

    ResetMock();
    PageAllocator::SystemFree(ptr, page_num);
}

TEST_F(PageAllocatorTest, AllocHugeAlignment) {
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

TEST_F(PageAllocatorTest, MultipleAllocations) {
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

TEST_F(PageAllocatorTest, InvalidArgs) {
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

TEST_F(PageAllocatorTest, AllocWithPopulateConfig) {
    // 设置环境变量 (Linux/macOS)
    // 注意：setenv 不是线程安全的，最好在 main 开始前设置，或者独立跑这个测试
    setenv("AM_USE_MAP_POPULATE", "1", 1);

    // 重新初始化 Config (如果 Config 是懒加载单例)
    // 这一步依赖于 RuntimeConfig 实现是否支持重置，
    // 或者我们假设这是程序启动后的第一次调用。
    void* ptr = PageAllocator::SystemAlloc(10);// 应该触发 MAP_POPULATE
    EXPECT_TRUE(ptr != nullptr);

    PageAllocator::SystemFree(ptr, 10);

    // 清理环境
    unsetenv("AM_USE_MAP_POPULATE");
}

}// namespace