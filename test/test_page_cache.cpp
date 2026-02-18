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

// 测试点 3: 合并逻辑 (Coalescing)
// 这是一个复杂的场景，验证左右合并
TEST_F(PageCacheTest, MergeLogic) {
    // 策略：为了确保地址连续，我们一次性申请一大块，然后手动切分释放来模拟碎片

    // 1. 申请 64 页 (占据半个最大块)
    Span* spanA = cache_.AllocSpan(64, 0);
    ASSERT_NE(spanA, nullptr);

    // 2. 申请 32 页
    Span* spanB = cache_.AllocSpan(32, 0);
    ASSERT_NE(spanB, nullptr);

    // 3. 申请 32 页
    Span* spanC = cache_.AllocSpan(32, 0);
    ASSERT_NE(spanC, nullptr);

    // 此时我们假设 spanA, spanB, spanC 在物理上是连续的
    // (这取决于 AllocSpan 的切分逻辑，通常是连续切下来的)
    // 验证连续性：
    bool is_continuous = (spanA->start_page_idx + spanA->page_num == spanB->start_page_idx) &&
                         (spanB->start_page_idx + spanB->page_num == spanC->start_page_idx);

    EXPECT_TRUE(is_continuous);
    // 如果不连续（比如中间夹杂了其他线程分配的），这个测试可能无法验证合并。
    // 但在单线程测试环境下，大概率是连续的。
    if (!is_continuous) {
        // 尝试调整顺序，可能是反向切分的
        // 这里的断言依赖于具体的切分实现（Head Split 还是 Tail Split）
        // 你的实现是 Head Split (small_span 拿走 start_page_idx)，所以地址是递增的。
        // 但 AllocSpan 遍历桶是从小到大还是从大到小？
        // 你的实现：for (i = page_num + 1; ...) 找第一个非空。
        // 进货时：span 放入 128 桶。
        // 第一次 Alloc(64)：找到 128，切出 64(A)，剩 64 放回 bucket[64]。
        // 第二次 Alloc(32)：找到 64，切出 32(B)，剩 32 放回 bucket[32]。
        // 第三次 Alloc(32)：找到 32，直接拿走(C)。
        // 所以顺序应该是 A -> B -> C 连续。
    }

    // 4. 释放 A (64页) -> 进入 bucket[64]
    cache_.ReleaseSpan(spanA);

    // 5. 释放 C (32页) -> 进入 bucket[32]
    cache_.ReleaseSpan(spanC);

    // 此时 B 夹在中间，A 和 C 不能合并。

    // 6. 释放 B (32页) -> 触发合并！
    // B 左边是 A (空闲)，右边是 C (空闲)。
    // 应该发生：A + B + C = 128 页。
    cache_.ReleaseSpan(spanB);

    // 验证：
    // 理论上 bucket[128] 应该增加了一个 Span。
    // 我们可以尝试申请一个 128 页的 Span，如果能申请到且不用系统调用（很难验证系统调用），说明合并成功。

    Span* spanFull = cache_.AllocSpan(128, 0);
    ASSERT_NE(spanFull, nullptr);

    // 验证拿到的这个大块，是不是原来的 A 的起始地址
    // 如果合并成功，spanFull->start_page_idx 应该等于 spanA->start_page_idx
    // (前提是这期间没有其他干扰)
    EXPECT_EQ(spanFull->start_page_idx, spanA->start_page_idx);

    cache_.ReleaseSpan(spanFull);
}

// 测试点 4: PageMap 映射一致性
TEST_F(PageCacheTest, PageMapConsistency) {
    size_t pages = 4;
    Span* span = cache_.AllocSpan(pages, 0);

    // 验证 span 覆盖的每一个页号，在 PageMap 中都指向该 span
    for (size_t i = 0; i < pages; ++i) {
        void* addr = (char*) span->GetStartAddr() + i * SystemConfig::PAGE_SIZE;
        EXPECT_EQ(PageMap::GetSpan(addr), span);
    }

    cache_.ReleaseSpan(span);
    // 释放后，PageMap 指向的 Span 对象已被 delete，
    // 但 SetSpan(span) 在 ReleaseSpan 内部被调用，
    // 此时 PageMap 指向的是已经在 FreeList 中的 span（虽然指针值没变，但状态变了）。
    // 验证 is_used 状态
    Span* freed_span = PageMap::GetSpan(span->GetStartAddr());
    EXPECT_FALSE(freed_span->is_used);
}

// 测试点 5: 压力测试 (随机分配释放)
TEST_F(PageCacheTest, RandomStress) {
    std::vector<Span*> spans;
    srand(42);

    for (int i = 0; i < 1000; ++i) {
        // 随机申请 1 ~ 20 页
        size_t k = (rand() % 20) + 1;
        Span* s = cache_.AllocSpan(k, 0);
        spans.push_back(s);
    }

    // 随机释放
    std::random_shuffle(spans.begin(), spans.end());
    for (auto* s: spans) {
        cache_.ReleaseSpan(s);
    }

    // 最终检查：所有内存应该都归还了，不应有泄漏
    // (这需要配合 ASan 检查)
}

}// namespace