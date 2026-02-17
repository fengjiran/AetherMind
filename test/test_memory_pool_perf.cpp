//
// Created by AetherMind Team on 2/16/26.
// Performance Benchmark Tests for ammalloc
//

#include "ammalloc/ammalloc.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <vector>

#ifdef NOOOO

namespace {

using namespace aethermind;
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::micro>;

// ===========================================================================
// Benchmark Utilities
// ===========================================================================

struct BenchmarkResult {
    double ops_per_second;
    double avg_latency_us;
    double min_latency_us;
    double max_latency_us;
    size_t total_ops;
};

template<typename F>
BenchmarkResult RunBenchmark(F&& func, size_t iterations, size_t warmup = 1000) {
    // Warmup
    for (size_t i = 0; i < warmup && i < iterations; ++i) {
        func();
    }

    std::vector<double> latencies;
    latencies.reserve(iterations);

    auto start = Clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        auto op_start = Clock::now();
        func();
        auto op_end = Clock::now();
        Duration op_duration = op_end - op_start;
        latencies.push_back(op_duration.count());
    }
    auto end = Clock::now();

    Duration total_duration = end - start;
    double total_seconds = total_duration.count() / 1'000'000.0;

    BenchmarkResult result;
    result.total_ops = iterations;
    result.ops_per_second = iterations / total_seconds;

    if (!latencies.empty()) {
        std::sort(latencies.begin(), latencies.end());
        result.min_latency_us = latencies.front();
        result.max_latency_us = latencies.back();

        double sum = 0;
        for (double l: latencies) {
            sum += l;
        }
        result.avg_latency_us = sum / latencies.size();
    }

    return result;
}

void PrintResult(const std::string& name, const BenchmarkResult& result) {
    std::printf("  %-30s: %10.0f ops/s, avg=%8.3fus, min=%8.3fus, max=%8.3fus\n",
                name.c_str(),
                result.ops_per_second,
                result.avg_latency_us,
                result.min_latency_us,
                result.max_latency_us);
}

// ===========================================================================
// Single-Threaded Benchmarks
// ===========================================================================

TEST(MemoryPoolPerf, SingleThread_SmallAlloc_8B) {
    constexpr size_t size = 8;
    constexpr size_t iterations = 100'000;
    std::vector<void*> ptrs;
    ptrs.reserve(iterations);

    // ammalloc
    auto am_result = RunBenchmark([&]() {
        void* ptr = am_malloc(size);
        ptrs.push_back(ptr);
    },
                                  iterations);

    for (void* p: ptrs) {
        am_free(p);
    }
    ptrs.clear();

    // system malloc
    auto sys_result = RunBenchmark([&]() {
        void* ptr = std::malloc(size);
        ptrs.push_back(ptr);
    },
                                   iterations);

    for (void* p: ptrs) {
        std::free(p);
    }

    std::printf("=== Single-Thread: 8B Allocation ===\n");
    PrintResult("ammalloc", am_result);
    PrintResult("system malloc", sys_result);
}

TEST(MemoryPoolPerf, SingleThread_SmallAlloc_64B) {
    constexpr size_t size = 64;
    constexpr size_t iterations = 100'000;
    std::vector<void*> ptrs;
    ptrs.reserve(iterations);

    auto am_result = RunBenchmark([&]() {
        void* ptr = am_malloc(size);
        ptrs.push_back(ptr);
    },
                                  iterations);

    for (void* p: ptrs) {
        am_free(p);
    }
    ptrs.clear();

    auto sys_result = RunBenchmark([&]() {
        void* ptr = std::malloc(size);
        ptrs.push_back(ptr);
    },
                                   iterations);

    for (void* p: ptrs) {
        std::free(p);
    }

    std::printf("=== Single-Thread: 64B Allocation ===\n");
    PrintResult("ammalloc", am_result);
    PrintResult("system malloc", sys_result);
}

TEST(MemoryPoolPerf, SingleThread_AllocFreePair_8B) {
    constexpr size_t size = 8;
    constexpr size_t iterations = 100'000;

    auto am_result = RunBenchmark([&]() {
        void* ptr = am_malloc(size);
        am_free(ptr);
    },
                                  iterations);

    auto sys_result = RunBenchmark([&]() {
        void* ptr = std::malloc(size);
        std::free(ptr);
    },
                                   iterations);

    std::printf("=== Single-Thread: 8B Alloc+Free Pair ===\n");
    PrintResult("ammalloc", am_result);
    PrintResult("system malloc", sys_result);
}

TEST(MemoryPoolPerf, SingleThread_MediumAlloc_512B) {
    constexpr size_t size = 512;
    constexpr size_t iterations = 50'000;
    std::vector<void*> ptrs;
    ptrs.reserve(iterations);

    auto am_result = RunBenchmark([&]() {
        void* ptr = am_malloc(size);
        ptrs.push_back(ptr);
    },
                                  iterations);

    for (void* p: ptrs) {
        am_free(p);
    }
    ptrs.clear();

    auto sys_result = RunBenchmark([&]() {
        void* ptr = std::malloc(size);
        ptrs.push_back(ptr);
    },
                                   iterations);

    for (void* p: ptrs) {
        std::free(p);
    }

    std::printf("=== Single-Thread: 512B Allocation ===\n");
    PrintResult("ammalloc", am_result);
    PrintResult("system malloc", sys_result);
}

TEST(MemoryPoolPerf, SingleThread_LargeAlloc_4KB) {
    constexpr size_t size = 4096;
    constexpr size_t iterations = 10'000;
    std::vector<void*> ptrs;
    ptrs.reserve(iterations);

    auto am_result = RunBenchmark([&]() {
        void* ptr = am_malloc(size);
        ptrs.push_back(ptr);
    },
                                  iterations);

    for (void* p: ptrs) {
        am_free(p);
    }
    ptrs.clear();

    auto sys_result = RunBenchmark([&]() {
        void* ptr = std::malloc(size);
        ptrs.push_back(ptr);
    },
                                   iterations);

    for (void* p: ptrs) {
        std::free(p);
    }

    std::printf("=== Single-Thread: 4KB Allocation ===\n");
    PrintResult("ammalloc", am_result);
    PrintResult("system malloc", sys_result);
}

TEST(MemoryPoolPerf, SingleThread_RandomSize) {
    constexpr size_t iterations = 50'000;
    std::vector<void*> ptrs;
    ptrs.reserve(iterations);

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(1, SizeConfig::MAX_TC_SIZE);

    std::vector<size_t> sizes;
    sizes.reserve(iterations);
    for (size_t i = 0; i < iterations; ++i) {
        sizes.push_back(dist(rng));
    }

    size_t idx = 0;
    auto am_result = RunBenchmark([&]() {
        void* ptr = am_malloc(sizes[idx++ % sizes.size()]);
        ptrs.push_back(ptr);
    },
                                  iterations);

    for (void* p: ptrs) {
        am_free(p);
    }
    ptrs.clear();

    idx = 0;
    auto sys_result = RunBenchmark([&]() {
        void* ptr = std::malloc(sizes[idx++ % sizes.size()]);
        ptrs.push_back(ptr);
    },
                                   iterations);

    for (void* p: ptrs) {
        std::free(p);
    }

    std::printf("=== Single-Thread: Random Size Allocation ===\n");
    PrintResult("ammalloc", am_result);
    PrintResult("system malloc", sys_result);
}

// ===========================================================================
// Multi-Threaded Benchmarks
// ===========================================================================

TEST(MemoryPoolPerf, MultiThread_2Threads_8B) {
    constexpr size_t size = 8;
    constexpr size_t iterations_per_thread = 50'000;
    constexpr size_t num_threads = 2;

    std::atomic<size_t> total_ops{0};
    auto start = Clock::now();

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            std::vector<void*> local_ptrs;
            local_ptrs.reserve(iterations_per_thread);
            for (size_t i = 0; i < iterations_per_thread; ++i) {
                void* ptr = am_malloc(size);
                local_ptrs.push_back(ptr);
            }
            total_ops.fetch_add(iterations_per_thread, std::memory_order_relaxed);
            for (void* p: local_ptrs) {
                am_free(p);
            }
        });
    }

    for (auto& thread: threads) {
        thread.join();
    }

    auto end = Clock::now();
    Duration total_duration = end - start;
    double total_seconds = total_duration.count() / 1'000'000.0;

    std::printf("=== Multi-Thread: 2 Threads, 8B Allocation ===\n");
    std::printf("  Total ops: %zu, Time: %.3fs, Ops/s: %.0f\n",
                total_ops.load(),
                total_seconds,
                total_ops.load() / total_seconds);
}

TEST(MemoryPoolPerf, MultiThread_4Threads_64B) {
    constexpr size_t size = 64;
    constexpr size_t iterations_per_thread = 25'000;
    constexpr size_t num_threads = 4;

    std::atomic<size_t> total_ops{0};
    auto start = Clock::now();

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            std::vector<void*> local_ptrs;
            local_ptrs.reserve(iterations_per_thread);
            for (size_t i = 0; i < iterations_per_thread; ++i) {
                void* ptr = am_malloc(size);
                local_ptrs.push_back(ptr);
            }
            total_ops.fetch_add(iterations_per_thread, std::memory_order_relaxed);
            for (void* p: local_ptrs) {
                am_free(p);
            }
        });
    }

    for (auto& thread: threads) {
        thread.join();
    }

    auto end = Clock::now();
    Duration total_duration = end - start;
    double total_seconds = total_duration.count() / 1'000'000.0;

    std::printf("=== Multi-Thread: 4 Threads, 64B Allocation ===\n");
    std::printf("  Total ops: %zu, Time: %.3fs, Ops/s: %.0f\n",
                total_ops.load(),
                total_seconds,
                total_ops.load() / total_seconds);
}

TEST(MemoryPoolPerf, MultiThread_8Threads_MixedSize) {
    constexpr size_t iterations_per_thread = 12'500;
    constexpr size_t num_threads = 8;

    std::atomic<size_t> total_ops{0};
    auto start = Clock::now();

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::vector<size_t> test_sizes = {8, 16, 32, 64, 128, 256, 512, 1024};

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<void*> local_ptrs;
            local_ptrs.reserve(iterations_per_thread);
            for (size_t i = 0; i < iterations_per_thread; ++i) {
                size_t size = test_sizes[(t + i) % test_sizes.size()];
                void* ptr = am_malloc(size);
                local_ptrs.push_back(ptr);
            }
            total_ops.fetch_add(iterations_per_thread, std::memory_order_relaxed);
            for (void* p: local_ptrs) {
                am_free(p);
            }
        });
    }

    for (auto& thread: threads) {
        thread.join();
    }

    auto end = Clock::now();
    Duration total_duration = end - start;
    double total_seconds = total_duration.count() / 1'000'000.0;

    std::printf("=== Multi-Thread: 8 Threads, Mixed Size Allocation ===\n");
    std::printf("  Total ops: %zu, Time: %.3fs, Ops/s: %.0f\n",
                total_ops.load(),
                total_seconds,
                total_ops.load() / total_seconds);
}

}// namespace
#endif