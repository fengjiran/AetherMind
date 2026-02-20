//
// Created by richard on 2/19/26.
//
#include "ammalloc/central_cache.h"
#include "ammalloc/page_cache.h"
#include "ammalloc/thread_cache.h"

#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <vector>

namespace aethermind {

class ThreadCacheTest : public ::testing::Test {
protected:
    PageCache& page_cache_ = PageCache::GetInstance();
    CentralCache& central_cache_ = CentralCache::GetInstance();

    void SetUp() override {
        page_cache_.Reset();
    }

    void TearDown() override {
        central_cache_.Reset();
        page_cache_.Reset();
    }
};

}// namespace aethermind

namespace {
using namespace aethermind;

// 测试点 1: 基本的 Allocate 操作
TEST_F(ThreadCacheTest, BasicAllocate) {
    thread_local ThreadCache cache;

    void* ptr = cache.Allocate(16);
    EXPECT_NE(ptr, nullptr);

    // 释放回 CentralCache
    void* head = ptr;
    static_cast<FreeBlock*>(ptr)->next = nullptr;
    central_cache_.ReleaseListToSpans(head, 16);
}

// 测试点 2: Allocate(0) 返回有效指针（8字节块）
TEST_F(ThreadCacheTest, AllocateZero) {
    thread_local ThreadCache cache;

    void* ptr = cache.Allocate(0);
    EXPECT_NE(ptr, nullptr);

    // 释放
    void* head = ptr;
    static_cast<FreeBlock*>(ptr)->next = nullptr;
    central_cache_.ReleaseListToSpans(head, 8);
}

// 测试点 3: 基本的 Deallocate 操作
TEST_F(ThreadCacheTest, BasicDeallocate) {
    thread_local ThreadCache cache;

    void* ptr = cache.Allocate(32);
    EXPECT_NE(ptr, nullptr);

    cache.Deallocate(ptr, 32);

    // ReleaseAll 清理
    cache.ReleaseAll();
}

// 测试点 4: 多次分配和释放
TEST_F(ThreadCacheTest, MultipleAllocateDeallocate) {
    thread_local ThreadCache cache;
    constexpr int num_allocs = 100;
    std::vector<void*> ptrs;

    for (int i = 0; i < num_allocs; ++i) {
        void* ptr = cache.Allocate(64);
        EXPECT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    for (void* ptr: ptrs) {
        cache.Deallocate(ptr, 64);
    }

    cache.ReleaseAll();
}

// 测试点 5: 不同 Size Class 的分配
TEST_F(ThreadCacheTest, DifferentSizeClasses) {
    thread_local ThreadCache cache;
    std::vector<size_t> sizes = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

    for (size_t size: sizes) {
        void* ptr = cache.Allocate(size);
        EXPECT_NE(ptr, nullptr) << "Failed for size " << size;
        cache.Deallocate(ptr, size);
    }

    cache.ReleaseAll();
}

// 测试点 6: ReleaseAll 功能
TEST_F(ThreadCacheTest, ReleaseAll) {
    thread_local ThreadCache cache;

    // 分配一些对象
    for (int i = 0; i < 50; ++i) {
        void* ptr = cache.Allocate(128);
        EXPECT_NE(ptr, nullptr);
        // 不释放，直接 ReleaseAll
    }

    cache.ReleaseAll();

    // 再次分配应该正常工作
    void* ptr = cache.Allocate(128);
    EXPECT_NE(ptr, nullptr);
    cache.Deallocate(ptr, 128);
    cache.ReleaseAll();
}

// 测试点 7: 慢启动策略测试
TEST_F(ThreadCacheTest, SlowStartStrategy) {
    thread_local ThreadCache cache;
    size_t size = 256;

    // 第一次分配会触发 FetchFromCentralCache
    void* ptr1 = cache.Allocate(size);
    EXPECT_NE(ptr1, nullptr);

    // 释放，触发慢启动增长 max_size
    cache.Deallocate(ptr1, size);

    // 再次分配和释放，观察 max_size 增长
    for (int i = 0; i < 10; ++i) {
        void* ptr = cache.Allocate(size);
        EXPECT_NE(ptr, nullptr);
        cache.Deallocate(ptr, size);
    }

    cache.ReleaseAll();
}

// 测试点 8: 触发 ReleaseTooLongList
TEST_F(ThreadCacheTest, TriggerReleaseTooLongList) {
    thread_local ThreadCache cache;
    size_t size = 512;
    size_t batch_size = SizeClass::CalculateBatchSize(size);

    // 分配足够多的对象
    std::vector<void*> ptrs;
    for (size_t i = 0; i < batch_size * 4; ++i) {
        void* ptr = cache.Allocate(size);
        EXPECT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    // 释放所有对象，触发 ReleaseTooLongList
    for (void* ptr: ptrs) {
        cache.Deallocate(ptr, size);
    }

    cache.ReleaseAll();
}

// 测试点 9: 压力测试
TEST_F(ThreadCacheTest, StressTest) {
    thread_local ThreadCache cache;
    std::vector<std::pair<void*, size_t>> allocated;
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<> size_dis(8, 1024);

    // 随机分配
    for (int i = 0; i < 100; ++i) {
        size_t size = size_dis(g);
        size = SizeClass::RoundUp(size);
        void* ptr = cache.Allocate(size);
        if (ptr) {
            allocated.emplace_back(ptr, size);
        }
    }

    // 随机释放
    std::shuffle(allocated.begin(), allocated.end(), g);
    for (auto& [ptr, size]: allocated) {
        cache.Deallocate(ptr, size);
    }

    cache.ReleaseAll();
}

// 测试点 10: 多线程分配（每个线程有自己的 ThreadCache）
TEST_F(ThreadCacheTest, MultiThreadedAllocation) {
    constexpr int num_threads = 4;
    constexpr int allocations_per_thread = 100;

    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&success_count]() {
            thread_local ThreadCache cache;
            for (int i = 0; i < allocations_per_thread; ++i) {
                void* ptr = cache.Allocate(64);
                if (ptr) {
                    success_count.fetch_add(1);
                    cache.Deallocate(ptr, 64);
                }
            }
            cache.ReleaseAll();
        });
    }

    for (auto& t: threads) {
        t.join();
    }

    EXPECT_EQ(success_count.load(), num_threads * allocations_per_thread);
}

// 测试点 11: 多线程不同大小分配
TEST_F(ThreadCacheTest, MultiThreadedDifferentSizes) {
    constexpr int num_threads = 4;
    constexpr int allocations_per_thread = 50;

    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&success_count, t]() {
            thread_local ThreadCache cache;
            std::vector<size_t> sizes = {8, 16, 32, 64, 128, 256, 512, 1024};
            size_t size = sizes[t % sizes.size()];

            for (int i = 0; i < allocations_per_thread; ++i) {
                void* ptr = cache.Allocate(size);
                if (ptr) {
                    success_count.fetch_add(1);
                    cache.Deallocate(ptr, size);
                }
            }
            cache.ReleaseAll();
        });
    }

    for (auto& t: threads) {
        t.join();
    }

    EXPECT_EQ(success_count.load(), num_threads * allocations_per_thread);
}

// 测试点 12: 小对象分配 (8 字节)
TEST_F(ThreadCacheTest, SmallObjectAllocation) {
    thread_local ThreadCache cache;

    for (int i = 0; i < 100; ++i) {
        void* ptr = cache.Allocate(8);
        EXPECT_NE(ptr, nullptr);
        cache.Deallocate(ptr, 8);
    }

    cache.ReleaseAll();
}

// 测试点 13: 边界大小分配
TEST_F(ThreadCacheTest, BoundarySizeAllocation) {
    thread_local ThreadCache cache;
    size_t max_size = SizeConfig::MAX_TC_SIZE;

    void* ptr = cache.Allocate(max_size);
    EXPECT_NE(ptr, nullptr);
    cache.Deallocate(ptr, max_size);

    cache.ReleaseAll();
}

// 测试点 14: 重复分配释放同一大小
TEST_F(ThreadCacheTest, RepeatedAllocateDeallocate) {
    thread_local ThreadCache cache;
    size_t size = 128;

    for (int round = 0; round < 10; ++round) {
        std::vector<void*> ptrs;
        for (int i = 0; i < 20; ++i) {
            void* ptr = cache.Allocate(size);
            EXPECT_NE(ptr, nullptr);
            ptrs.push_back(ptr);
        }
        for (void* ptr: ptrs) {
            cache.Deallocate(ptr, size);
        }
    }

    cache.ReleaseAll();
}

// 测试点 15: 验证 FetchFromCentralCache 触发
TEST_F(ThreadCacheTest, FetchFromCentralCacheTrigger) {
    thread_local ThreadCache cache;
    size_t size = 256;

    // 连续分配超过 batch_size 的数量，触发多次 FetchFromCentralCache
    size_t batch_size = SizeClass::CalculateBatchSize(size);
    std::vector<void*> ptrs;

    for (size_t i = 0; i < batch_size * 3; ++i) {
        void* ptr = cache.Allocate(size);
        EXPECT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    for (void* ptr: ptrs) {
        cache.Deallocate(ptr, size);
    }

    cache.ReleaseAll();
}

}// namespace
