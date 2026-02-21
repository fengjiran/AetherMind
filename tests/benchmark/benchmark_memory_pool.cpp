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

void BM_am_malloc_8B(benchmark::State& state) {
    std::vector<void*> ptrs;
    ptrs.reserve(state.max_iterations);
    for (auto _: state) {
        void* ptr = am_malloc(8);
        ptrs.push_back(ptr);
    }
    for (void* p: ptrs) {
        am_free(p);
    }
}

void BM_std_malloc_8B(benchmark::State& state) {
    std::vector<void*> ptrs;
    ptrs.reserve(state.max_iterations);
    for (auto _: state) {
        void* ptr = std::malloc(8);
        ptrs.push_back(ptr);
    }
    for (void* p: ptrs) {
        std::free(p);
    }
}

void BM_am_malloc_64B(benchmark::State& state) {
    std::vector<void*> ptrs;
    ptrs.reserve(state.max_iterations);
    for (auto _: state) {
        void* ptr = am_malloc(64);
        ptrs.push_back(ptr);
    }
    for (void* p: ptrs) {
        am_free(p);
    }
}

void BM_std_malloc_64B(benchmark::State& state) {
    std::vector<void*> ptrs;
    ptrs.reserve(state.max_iterations);
    for (auto _: state) {
        void* ptr = std::malloc(64);
        ptrs.push_back(ptr);
    }
    for (void* p: ptrs) {
        std::free(p);
    }
}

void BM_am_malloc_512B(benchmark::State& state) {
    std::vector<void*> ptrs;
    ptrs.reserve(state.max_iterations);
    for (auto _: state) {
        void* ptr = am_malloc(512);
        ptrs.push_back(ptr);
    }
    for (void* p: ptrs) {
        am_free(p);
    }
}

void BM_std_malloc_512B(benchmark::State& state) {
    std::vector<void*> ptrs;
    ptrs.reserve(state.max_iterations);
    for (auto _: state) {
        void* ptr = std::malloc(512);
        ptrs.push_back(ptr);
    }
    for (void* p: ptrs) {
        std::free(p);
    }
}

void BM_am_malloc_4KB(benchmark::State& state) {
    std::vector<void*> ptrs;
    ptrs.reserve(state.max_iterations);
    for (auto _: state) {
        void* ptr = am_malloc(4096);
        ptrs.push_back(ptr);
    }
    for (void* p: ptrs) {
        am_free(p);
    }
}

void BM_std_malloc_4KB(benchmark::State& state) {
    std::vector<void*> ptrs;
    ptrs.reserve(state.max_iterations);
    for (auto _: state) {
        void* ptr = std::malloc(4096);
        ptrs.push_back(ptr);
    }
    for (void* p: ptrs) {
        std::free(p);
    }
}

void BM_am_malloc_free_pair_8B(benchmark::State& state) {
    for (auto _: state) {
        void* ptr = am_malloc(8);
        am_free(ptr);
    }
}

void BM_std_malloc_free_pair_8B(benchmark::State& state) {
    for (auto _: state) {
        void* ptr = std::malloc(8);
        std::free(ptr);
    }
}

void BM_am_malloc_free_pair_64B(benchmark::State& state) {
    for (auto _: state) {
        void* ptr = am_malloc(64);
        am_free(ptr);
    }
}

void BM_std_malloc_free_pair_64B(benchmark::State& state) {
    for (auto _: state) {
        void* ptr = std::malloc(64);
        std::free(ptr);
    }
}

void BM_am_malloc_random_size(benchmark::State& state) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(1, SizeConfig::MAX_TC_SIZE);
    std::vector<size_t> sizes;
    sizes.reserve(10000);
    for (size_t i = 0; i < 10000; ++i) {
        sizes.push_back(dist(rng));
    }

    std::vector<void*> ptrs;
    ptrs.reserve(state.max_iterations);
    size_t idx = 0;
    for (auto _: state) {
        void* ptr = am_malloc(sizes[idx++ % sizes.size()]);
        ptrs.push_back(ptr);
    }
    for (void* p: ptrs) {
        am_free(p);
    }
}

void BM_std_malloc_random_size(benchmark::State& state) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(1, SizeConfig::MAX_TC_SIZE);
    std::vector<size_t> sizes;
    sizes.reserve(10000);
    for (size_t i = 0; i < 10000; ++i) {
        sizes.push_back(dist(rng));
    }

    std::vector<void*> ptrs;
    ptrs.reserve(state.max_iterations);
    size_t idx = 0;
    for (auto _: state) {
        void* ptr = std::malloc(sizes[idx++ % sizes.size()]);
        ptrs.push_back(ptr);
    }
    for (void* p: ptrs) {
        std::free(p);
    }
}

BENCHMARK(BM_am_malloc_8B);
BENCHMARK(BM_std_malloc_8B);
BENCHMARK(BM_am_malloc_64B);
BENCHMARK(BM_std_malloc_64B);
BENCHMARK(BM_am_malloc_512B);
BENCHMARK(BM_std_malloc_512B);
BENCHMARK(BM_am_malloc_4KB);
BENCHMARK(BM_std_malloc_4KB);
BENCHMARK(BM_am_malloc_free_pair_8B);
BENCHMARK(BM_std_malloc_free_pair_8B);
BENCHMARK(BM_am_malloc_free_pair_64B);
BENCHMARK(BM_std_malloc_free_pair_64B);
BENCHMARK(BM_am_malloc_random_size);
BENCHMARK(BM_std_malloc_random_size);

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
