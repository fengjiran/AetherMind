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

    EXPECT_EQ(PageMap::GetSpan(span->GetStartAddr()), span);

    cache_.ReleaseSpan(span);
}

}// namespace