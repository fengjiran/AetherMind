//
// Created by richard on 2/12/26.
//
#include "ammalloc/page_cache.h"

#include <atomic>
#include <cstring>
#include <gtest/gtest.h>

namespace aethermind {

class PageCacheTest : public ::testing::Test {
protected:
    PageCache& cache_ = PageCache::GetInstance();

    void SetUp() override {
        cache_.Reset();
    }

    void TearDown() override {
        cache_.Reset();
    }

    AM_NODISCARD bool IsBucketEmpty(size_t page_num) const {
        return cache_.span_lists_[page_num].empty();
    }

    AM_NODISCARD size_t GetBucketSize(size_t page_num) const {
        size_t cnt = 0;
        auto* cur = cache_.span_lists_[page_num].begin();
        while (cur != cache_.span_lists_[page_num].end()) {
            ++cnt;
            cur = cur->next;
        }
        return cnt;
    }
};

}// namespace aethermind

namespace {
using namespace aethermind;

// 测试点 1: 超大内存分配 (> 128页)
// 预期：不经过桶，直接向 PageAllocator 申请，释放时直接还给系统
TEST_F(PageCacheTest, OversizedAllocation) {
    size_t huge_pages = PageConfig::MAX_PAGE_NUM + 10;
    auto* span = cache_.AllocSpan(huge_pages, 0);
    EXPECT_TRUE(span != nullptr);
    EXPECT_EQ(span->page_num, huge_pages);
    EXPECT_EQ(span->obj_size, 0);
    EXPECT_TRUE(span->is_used);

    EXPECT_EQ(PageMap::GetSpan(span->GetStartAddr()), span);
    void* last_page_ptr = static_cast<char*>(span->GetStartAddr()) + (huge_pages - 1) * SystemConfig::PAGE_SIZE;
    EXPECT_EQ(PageMap::GetSpan(last_page_ptr), span);

    cache_.ReleaseSpan(span);
    EXPECT_TRUE(PageMap::GetSpan(last_page_ptr) == nullptr);
}

// 测试点 2: 系统进货与切分 (Refill & Split)
TEST_F(PageCacheTest, RefillAndSplit) {
    // 1. 申请 1 页
    // 预期：PageCache 发现没内存 -> 申请 128 页 -> 切出 1 页给用户 -> 剩 127 页挂在 bucket[127]
    Span* span1 = cache_.AllocSpan(1, 8);
    ASSERT_NE(span1, nullptr);
    EXPECT_EQ(span1->page_num, 1);

    // 2. 再申请 10 页
    // 预期：从 bucket[127] 拿出 -> 切出 10 页 -> 剩 117 页挂在 bucket[117]
    Span* span2 = cache_.AllocSpan(10, 16);
    ASSERT_NE(span2, nullptr);
    EXPECT_EQ(span2->page_num, 10);

    // 验证：127 号桶空了，117 号桶有了
    // 注意：这取决于单例之前的状态，如果之前是空的，逻辑成立。
    // 如果之前有残留数据，可能直接从 bucket[10] 拿。
    // 假设这是冷启动环境：
    EXPECT_TRUE(IsBucketEmpty(127));
    EXPECT_FALSE(IsBucketEmpty(117));

    // 清理
    cache_.ReleaseSpan(span1);
    cache_.ReleaseSpan(span2);
}

}// namespace