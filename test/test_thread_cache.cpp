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
        central_cache_.Reset();
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
    EXPECT_TRUE(ptr != nullptr);

    cache.Deallocate(ptr, 16);
    cache.ReleaseAll();
}

// 测试点 2: Allocate(0) 返回有效指针（8字节块）
TEST_F(ThreadCacheTest, AllocateZero) {
    thread_local ThreadCache cache;

    void* ptr = cache.Allocate(0);
    EXPECT_TRUE(ptr != nullptr);

    cache.Deallocate(ptr, 0);
    cache.ReleaseAll();
}

// 测试点 3: 基本的 Deallocate 操作
TEST_F(ThreadCacheTest, BasicDeallocate) {
    thread_local ThreadCache cache;

    void* ptr = cache.Allocate(32);
    EXPECT_TRUE(ptr != nullptr);

    cache.Deallocate(ptr, 32);

    // ReleaseAll 清理
    cache.ReleaseAll();
}

TEST_F(ThreadCacheTest, EdgeCases) {
    thread_local ThreadCache tc;

    // 1. size == 0 (应该被提升为最小的 8 字节桶)
    void* ptr_zero = tc.Allocate(0);
    EXPECT_TRUE(ptr_zero != nullptr);
    tc.Deallocate(ptr_zero, 0);

    // 2. size == MAX_TC_SIZE (256KB)
    size_t max_size = SizeConfig::MAX_TC_SIZE;
    void* ptr_max = tc.Allocate(max_size);
    EXPECT_TRUE(ptr_max != nullptr);

    // 写入首尾验证
    char* char_ptr = static_cast<char*>(ptr_max);
    char_ptr[0] = 'A';
    char_ptr[max_size - 1] = 'Z';
    EXPECT_EQ(char_ptr[0], 'A');
    EXPECT_EQ(char_ptr[max_size - 1], 'Z');

    tc.Deallocate(ptr_max, max_size);
    tc.ReleaseAll();
}

// 测试点 4: 多次分配和释放
TEST_F(ThreadCacheTest, MultipleAllocateDeallocate) {
    thread_local ThreadCache cache;
    constexpr int num_allocs = 100;
    std::vector<void*> ptrs;

    for (int i = 0; i < num_allocs; ++i) {
        void* ptr = cache.Allocate(64);
        EXPECT_TRUE(ptr != nullptr);
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
        EXPECT_TRUE(ptr != nullptr) << "Failed for size " << size;
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
        EXPECT_TRUE(ptr != nullptr);
        // 不释放，直接 ReleaseAll
    }

    cache.ReleaseAll();

    // 再次分配应该正常工作
    void* ptr = cache.Allocate(128);
    EXPECT_TRUE(ptr != nullptr);
    cache.Deallocate(ptr, 128);
    cache.ReleaseAll();
}

// 测试点 7: 慢启动策略测试
TEST_F(ThreadCacheTest, SlowStartAndScavenge) {
    thread_local ThreadCache tc;
    size_t size = 8;// 最小对象，batch_num 通常是 512
    size_t batch_num = SizeClass::CalculateBatchSize(size);

    std::vector<void*> ptrs;

    // 1. 持续分配，触发慢启动增长
    // 分配 1500 个对象，必然触发多次 FetchFromCentralCache
    for (size_t i = 0; i < 1500; ++i) {
        void* ptr = tc.Allocate(size);
        EXPECT_TRUE(ptr != nullptr);
        ptrs.push_back(ptr);
    }

    // 验证分配的指针互不相同
    std::sort(ptrs.begin(), ptrs.end());
    auto it = std::unique(ptrs.begin(), ptrs.end());
    EXPECT_EQ(it, ptrs.end()) << "Duplicate pointers allocated!";

    // 2. 持续释放，触发 ReleaseTooLongList
    // 当释放数量超过 limit (1024) 时，会触发批量归还
    for (void* ptr: ptrs) {
        tc.Deallocate(ptr, size);
    }

    // 3. 清理残留
    tc.ReleaseAll();
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
        EXPECT_TRUE(ptr != nullptr);
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

// 模拟单线程的随机分配/释放行为
void ThreadRoutine(int thread_id, size_t iterations) {
    thread_local ThreadCache tc;// 每个线程独享一个 ThreadCache 实例
    std::vector<void*> allocated_ptrs;
    allocated_ptrs.reserve(1000);

    std::mt19937 gen(thread_id);
    // 随机大小：1 字节 到 64KB
    std::uniform_int_distribution<size_t> size_dist(1, 32 * 1024);
    // 随机动作：70% 概率分配，30% 概率释放 (模拟内存增长期)
    std::uniform_int_distribution<int> action_dist(1, 100);

    for (size_t i = 0; i < iterations; ++i) {
        if (allocated_ptrs.empty() || action_dist(gen) <= 70) {
            // Allocate
            size_t size = size_dist(gen);
            void* ptr = tc.Allocate(size);
            if (ptr) {
                // 简单写入验证
                *static_cast<size_t*>(ptr) = size;
                allocated_ptrs.push_back(ptr);
            }
        } else {
            // Deallocate
            // 随机挑一个释放
            size_t idx = gen() % allocated_ptrs.size();
            void* ptr = allocated_ptrs[idx];
            size_t size = *static_cast<size_t*>(ptr);// 读出大小

            tc.Deallocate(ptr, SizeClass::RoundUp(size));

            // 移除并替换最后一个元素
            allocated_ptrs[idx] = allocated_ptrs.back();
            allocated_ptrs.pop_back();
        }
    }

    // 线程退出前，释放所有剩余内存
    for (void* ptr: allocated_ptrs) {
        size_t size = *static_cast<size_t*>(ptr);
        tc.Deallocate(ptr, SizeClass::RoundUp(size));
    }

    // 归还 ThreadCache 缓存到 CentralCache
    tc.ReleaseAll();
}

TEST_F(ThreadCacheTest, MultiThreadStress) {
    const int num_threads = std::thread::hardware_concurrency();
    // const int num_threads = 1;
    const size_t iterations_per_thread = 50000;// 每个线程 5 万次操作

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(ThreadRoutine, i, iterations_per_thread);
    }

    for (auto& t: threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    size_t total_ops = num_threads * iterations_per_thread;
    std::cout << " " << num_threads << " threads executed "
              << total_ops << " ops in " << diff.count() << " seconds.\n";
    std::cout << " " << (total_ops / diff.count() / 1000000.0)
              << " Million Ops/sec\n";
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

// 这是一个简单的对比测试，用于直观感受 ThreadCache 的威力
TEST_F(ThreadCacheTest, BenchmarkVsStdMalloc) {
    const size_t iterations = 1000000;// 100万次
    const size_t alloc_size = 32;     // 32 字节小对象

    // 1. 测试 std::malloc
    auto start_std = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        void* p = std::malloc(alloc_size);
        // benchmark::DoNotOptimize(p);// 防止被编译器优化掉 (需引入 benchmark 库，或用 volatile)
        std::free(p);
    }

    auto end_std = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_std = end_std - start_std;

    // 2. 测试 ThreadCache
    thread_local ThreadCache tc;
    auto start_tc = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        void* p = tc.Allocate(alloc_size);
        // benchmark::DoNotOptimize(p);
        tc.Deallocate(p, alloc_size);
    }
    tc.ReleaseAll();
    auto end_tc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_tc = end_tc - start_tc;

    std::cout << " Time: " << diff_std.count() << " s\n";
    std::cout << " Time: " << diff_tc.count() << " s\n";

    // 通常 ThreadCache 会比 std::malloc 快得多，因为全是无锁的数组/链表操作
    // EXPECT_LT(diff_tc.count(), diff_std.count());
}

}// namespace
