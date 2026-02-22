//
// Created by AetherMind Team on 2/16/26.
// Performance Benchmark Tests for ammalloc using Google Benchmark
//

#include "ammalloc/ammalloc.h"
#include "ammalloc/config.h"

#include <benchmark/benchmark.h>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

namespace {

using namespace aethermind;

using alloc_func_type = void* (*) (size_t);
using free_func_type = void (*)(void*);

template<size_t AllocSize, size_t WindowSize, alloc_func_type alloc_func, free_func_type free_func>
void BM_Malloc_Churn(benchmark::State& state) {
    static_assert((WindowSize & (WindowSize - 1)) == 0, "WindowSize must be a power of 2");

    std::array<void*, WindowSize> window{};
    size_t i = 0;
    for (auto _: state) {
        size_t idx = i & WindowSize - 1;
        void* old_ptr = window[idx];

        window[idx] = alloc_func(AllocSize);
        benchmark::DoNotOptimize(window[idx]);
        if (old_ptr) {
            free_func(old_ptr);
        }
        ++i;
    }

    for (void* ptr: window) {
        if (ptr) {
            free_func(ptr);
        }
    }
}

template<size_t AllocSize, size_t BatchSize, alloc_func_type alloc_func, free_func_type free_func>
void BM_Malloc_Deep_Churn(benchmark::State& state) {
    std::vector<void*> ptrs;
    ptrs.reserve(BatchSize);
    for (auto _: state) {
        // Allocate
        for (size_t i = 0; i < BatchSize; ++i) {
            void* p = alloc_func(AllocSize);
            benchmark::DoNotOptimize(p);
            ptrs.push_back(p);
        }

        // Free
        for (size_t i = 0; i < BatchSize; ++i) {
            free_func(ptrs[i]);
        }

        ptrs.clear();
    }
}

void BM_am_malloc_free_pair_random_size(benchmark::State& state) {
    constexpr size_t num_sizes = 8192;
    std::vector<size_t> sizes(num_sizes);
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(1, SizeConfig::MAX_TC_SIZE);
    for (size_t i = 0; i < num_sizes; ++i) {
        sizes[i] = dist(rng);
    }

    constexpr size_t window_size = 1024;
    std::array<void*, window_size> window{};
    size_t i = 0;

    for (auto _: state) {
        size_t w_idx = i & (window_size - 1);
        size_t s_idx = i & (num_sizes - 1);

        if (window[w_idx] != nullptr) {
            am_free(window[w_idx]);
        }

        window[w_idx] = am_malloc(sizes[s_idx]);
        benchmark::DoNotOptimize(window[w_idx]);
        ++i;
    }

    for (void* p: window) {
        if (p != nullptr) {
            am_free(p);
        }
    }
}

void BM_std_malloc_free_pair_random_size(benchmark::State& state) {
    constexpr size_t num_sizes = 8192;
    std::vector<size_t> sizes(num_sizes);
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(1, SizeConfig::MAX_TC_SIZE);
    for (size_t i = 0; i < num_sizes; ++i) {
        sizes[i] = dist(rng);
    }

    constexpr size_t window_size = 1024;
    std::array<void*, window_size> window{};
    size_t i = 0;

    for (auto _: state) {
        size_t w_idx = i & (window_size - 1);
        size_t s_idx = i & (num_sizes - 1);

        if (window[w_idx] != nullptr) {
            std::free(window[w_idx]);
        }

        window[w_idx] = std::malloc(sizes[s_idx]);
        benchmark::DoNotOptimize(window[w_idx]);
        ++i;
    }

    for (void* p: window) {
        if (p != nullptr) {
            std::free(p);
        }
    }
}

// 1. 测试极致 Fast Path (Window = 1)
// 每次循环都是 malloc -> free -> malloc -> free
// 测量的是纯 malloc+free 开销
BENCHMARK_TEMPLATE(BM_Malloc_Churn, 8, 1, am_malloc, am_free);
BENCHMARK_TEMPLATE(BM_Malloc_Churn, 8, 1, std::malloc, std::free);

BENCHMARK_TEMPLATE(BM_Malloc_Churn, 64, 1, am_malloc, am_free);
BENCHMARK_TEMPLATE(BM_Malloc_Churn, 64, 1, std::malloc, std::free);

// 2. 测试 ThreadCache 稳态吞吐 (Window = 256)
// 对象在 ThreadCache 内部循环，不触发 CentralCache
BENCHMARK_TEMPLATE(BM_Malloc_Churn, 8, 256, am_malloc, am_free);
BENCHMARK_TEMPLATE(BM_Malloc_Churn, 8, 256, std::malloc, std::free);

BENCHMARK_TEMPLATE(BM_Malloc_Churn, 64, 256, am_malloc, am_free);
BENCHMARK_TEMPLATE(BM_Malloc_Churn, 64, 256, std::malloc, std::free);

// 3. 测试系统综合搅动率 (Window = 1024)
// 强制触发 ThreadCache 溢出，测试 CentralCache 桶锁和批量搬运性能
BENCHMARK_TEMPLATE(BM_Malloc_Churn, 8, 1024, am_malloc, am_free);
BENCHMARK_TEMPLATE(BM_Malloc_Churn, 8, 1024, std::malloc, std::free);

BENCHMARK_TEMPLATE(BM_Malloc_Churn, 4096, 1024, am_malloc, am_free);
BENCHMARK_TEMPLATE(BM_Malloc_Churn, 4096, 1024, std::malloc, std::free);

BENCHMARK(BM_am_malloc_free_pair_random_size);
BENCHMARK(BM_std_malloc_free_pair_random_size);

// 注册测试：BatchSize 设置为 2000，远超 ThreadCache 的 max_size (512)
// 这将绝对触发 CentralCache 的慢速路径！
BENCHMARK_TEMPLATE(BM_Malloc_Deep_Churn, 8, 2000, am_malloc, am_free);
BENCHMARK_TEMPLATE(BM_Malloc_Deep_Churn, 8, 2000, std::malloc, std::free);

template<size_t Size, size_t NumThreads>
void BM_am_malloc_multithread(benchmark::State& state) {
    std::vector<std::thread> threads;
    threads.reserve(NumThreads);
    for (auto _: state) {
        state.PauseTiming();
        threads.clear();
        std::atomic<size_t> total_ops{0};
        for (size_t t = 0; t < NumThreads; ++t) {
            threads.emplace_back([&]() {
                std::vector<void*> local_ptrs;
                local_ptrs.reserve(1000);
                for (size_t i = 0; i < 1000; ++i) {
                    void* ptr = am_malloc(Size);
                    local_ptrs.push_back(ptr);
                }
                total_ops.fetch_add(1000, std::memory_order_relaxed);
                for (void* p: local_ptrs) {
                    am_free(p);
                }
            });
        }
        state.ResumeTiming();
        for (auto& t: threads) {
            t.join();
        }
    }
}

BENCHMARK_TEMPLATE(BM_am_malloc_multithread, 8, 2)->Iterations(10);
BENCHMARK_TEMPLATE(BM_am_malloc_multithread, 8, 4)->Iterations(10);
BENCHMARK_TEMPLATE(BM_am_malloc_multithread, 8, 8)->Iterations(10);
BENCHMARK_TEMPLATE(BM_am_malloc_multithread, 64, 2)->Iterations(10);
BENCHMARK_TEMPLATE(BM_am_malloc_multithread, 64, 4)->Iterations(10);
BENCHMARK_TEMPLATE(BM_am_malloc_multithread, 64, 8)->Iterations(10);

}// namespace
