//
// Created by richard on 11/28/25.
//

#include "container/string.h"

#include <benchmark/benchmark.h>
#include <string>

namespace {
using namespace aethermind;

void BM_StringCreation(benchmark::State& state) {
    for (auto _: state) {
        std::string str("Hello, World!");
    }
}

void BM_StringCopy(benchmark::State& state) {
    std::string str("Hello, World!");
    for (auto _: state) {
        std::string copy(str);
    }
}

BENCHMARK(BM_StringCreation);
BENCHMARK(BM_StringCopy);

}// namespace