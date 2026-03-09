//
// Created by AetherMind Team on 3/8/26.
//

#include "ammalloc/size_class.h"

#include <benchmark/benchmark.h>

namespace {

using namespace aethermind;

void BM_SizeClass_Index_Small(benchmark::State& state) {
    for (auto _: state) {
        for (size_t size = 1; size <= 128; size += 8) {
            benchmark::DoNotOptimize(SizeClass::Index(size));
        }
    }
}
BENCHMARK(BM_SizeClass_Index_Small);

void BM_SizeClass_Index_Large(benchmark::State& state) {
    for (auto _: state) {
        for (size_t size = 129; size <= 32768; size += 1024) {
            benchmark::DoNotOptimize(SizeClass::Index(size));
        }
    }
}
BENCHMARK(BM_SizeClass_Index_Large);

void BM_SizeClass_Size(benchmark::State& state) {
    for (auto _: state) {
        for (size_t idx = 0; idx < SizeClass::kNumSizeClasses; ++idx) {
            benchmark::DoNotOptimize(SizeClass::Size(idx));
        }
    }
}
BENCHMARK(BM_SizeClass_Size);

void BM_SizeClass_RoundUp_Small(benchmark::State& state) {
    for (auto _: state) {
        for (size_t size = 1; size <= 128; size += 8) {
            benchmark::DoNotOptimize(SizeClass::RoundUp(size));
        }
    }
}
BENCHMARK(BM_SizeClass_RoundUp_Small);

void BM_SizeClass_RoundUp_Large(benchmark::State& state) {
    for (auto _: state) {
        for (size_t size = 129; size <= 32768; size += 1024) {
            benchmark::DoNotOptimize(SizeClass::RoundUp(size));
        }
    }
}
BENCHMARK(BM_SizeClass_RoundUp_Large);

void BM_SizeClass_CalculateBatchSize(benchmark::State& state) {
    for (auto _: state) {
        for (size_t idx = 0; idx < SizeClass::kNumSizeClasses; ++idx) {
            size_t size = SizeClass::Size(idx);
            benchmark::DoNotOptimize(SizeClass::CalculateBatchSize(size));
        }
    }
}
BENCHMARK(BM_SizeClass_CalculateBatchSize);

void BM_SizeClass_GetMovePageNum(benchmark::State& state) {
    for (auto _: state) {
        for (size_t idx = 0; idx < SizeClass::kNumSizeClasses; ++idx) {
            size_t size = SizeClass::Size(idx);
            benchmark::DoNotOptimize(SizeClass::GetMovePageNum(size));
        }
    }
}
BENCHMARK(BM_SizeClass_GetMovePageNum);

}// namespace
