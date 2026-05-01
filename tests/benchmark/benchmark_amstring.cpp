#include "amstring/string.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <string>

namespace {

std::string MakePayload(std::size_t size) {
    std::string payload;
    payload.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
        payload.push_back(static_cast<char>('a' + static_cast<char>(i % 23)));
    }
    return payload;
}

void BM_AmString_Construct(benchmark::State& state) {
    const std::string payload = MakePayload(static_cast<std::size_t>(state.range(0)));
    for (auto _: state) {
        aethermind::string value(payload.data(), payload.size());
        benchmark::DoNotOptimize(value.data());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size()));
}

void BM_StdString_Construct(benchmark::State& state) {
    const std::string payload = MakePayload(static_cast<std::size_t>(state.range(0)));
    for (auto _: state) {
        std::string value(payload.data(), payload.size());
        benchmark::DoNotOptimize(value.data());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size()));
}

void BM_AmString_Copy(benchmark::State& state) {
    const std::string payload = MakePayload(static_cast<std::size_t>(state.range(0)));
    const aethermind::string original(payload.data(), payload.size());
    for (auto _: state) {
        aethermind::string copy(original);
        benchmark::DoNotOptimize(copy.data());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size()));
}

void BM_StdString_Copy(benchmark::State& state) {
    const std::string payload = MakePayload(static_cast<std::size_t>(state.range(0)));
    const std::string original(payload.data(), payload.size());
    for (auto _: state) {
        std::string copy(original);
        benchmark::DoNotOptimize(copy.data());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size()));
}

void BM_AmString_AppendReserved(benchmark::State& state) {
    const std::string prefix = MakePayload(static_cast<std::size_t>(state.range(0)));
    const std::string suffix = MakePayload(static_cast<std::size_t>(state.range(1)));
    for (auto _: state) {
        aethermind::string value(prefix.data(), prefix.size());
        value.reserve(prefix.size() + suffix.size());
        value.append(suffix.data(), suffix.size());
        benchmark::DoNotOptimize(value.data());
    }
    state.SetBytesProcessed(state.iterations() *
                            static_cast<int64_t>(prefix.size() + suffix.size()));
}

void BM_StdString_AppendReserved(benchmark::State& state) {
    const std::string prefix = MakePayload(static_cast<std::size_t>(state.range(0)));
    const std::string suffix = MakePayload(static_cast<std::size_t>(state.range(1)));
    for (auto _: state) {
        std::string value(prefix.data(), prefix.size());
        value.reserve(prefix.size() + suffix.size());
        value.append(suffix.data(), suffix.size());
        benchmark::DoNotOptimize(value.data());
    }
    state.SetBytesProcessed(state.iterations() *
                            static_cast<int64_t>(prefix.size() + suffix.size()));
}

void BM_AmString_AppendGrow(benchmark::State& state) {
    const std::string prefix = MakePayload(static_cast<std::size_t>(state.range(0)));
    const std::string suffix = MakePayload(static_cast<std::size_t>(state.range(1)));
    for (auto _: state) {
        aethermind::string value(prefix.data(), prefix.size());
        value.append(suffix.data(), suffix.size());
        benchmark::DoNotOptimize(value.data());
    }
    state.SetBytesProcessed(state.iterations() *
                            static_cast<int64_t>(prefix.size() + suffix.size()));
}

void BM_StdString_AppendGrow(benchmark::State& state) {
    const std::string prefix = MakePayload(static_cast<std::size_t>(state.range(0)));
    const std::string suffix = MakePayload(static_cast<std::size_t>(state.range(1)));
    for (auto _: state) {
        std::string value(prefix.data(), prefix.size());
        value.append(suffix.data(), suffix.size());
        benchmark::DoNotOptimize(value.data());
    }
    state.SetBytesProcessed(state.iterations() *
                            static_cast<int64_t>(prefix.size() + suffix.size()));
}

void BM_AmString_Assign(benchmark::State& state) {
    const std::string payload = MakePayload(static_cast<std::size_t>(state.range(0)));
    aethermind::string value;
    for (auto _: state) {
        value.assign(payload.data(), payload.size());
        benchmark::DoNotOptimize(value.data());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size()));
}

void BM_StdString_Assign(benchmark::State& state) {
    const std::string payload = MakePayload(static_cast<std::size_t>(state.range(0)));
    std::string value;
    for (auto _: state) {
        value.assign(payload.data(), payload.size());
        benchmark::DoNotOptimize(value.data());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size()));
}

void BM_AmString_ShrinkToFit(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    const std::string payload = MakePayload(size);
    const std::size_t reserve_size = std::max<std::size_t>(size + 64, size * 2 + 1);
    for (auto _: state) {
        aethermind::string value(payload.data(), payload.size());
        value.reserve(reserve_size);
        value.shrink_to_fit();
        benchmark::DoNotOptimize(value.data());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size()));
}

void BM_StdString_ShrinkToFit(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    const std::string payload = MakePayload(size);
    const std::size_t reserve_size = std::max<std::size_t>(size + 64, size * 2 + 1);
    for (auto _: state) {
        std::string value(payload.data(), payload.size());
        value.reserve(reserve_size);
        value.shrink_to_fit();
        benchmark::DoNotOptimize(value.data());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size()));
}

}// namespace

BENCHMARK(BM_AmString_Construct)->Arg(8)->Arg(15)->Arg(16)->Arg(64)->Arg(256);
BENCHMARK(BM_StdString_Construct)->Arg(8)->Arg(15)->Arg(16)->Arg(64)->Arg(256);
BENCHMARK(BM_AmString_Copy)->Arg(8)->Arg(15)->Arg(16)->Arg(64)->Arg(256);
BENCHMARK(BM_StdString_Copy)->Arg(8)->Arg(15)->Arg(16)->Arg(64)->Arg(256);
BENCHMARK(BM_AmString_AppendReserved)->Args({8, 4})->Args({15, 1})->Args({16, 8})->Args({64, 32});
BENCHMARK(BM_StdString_AppendReserved)->Args({8, 4})->Args({15, 1})->Args({16, 8})->Args({64, 32});
BENCHMARK(BM_AmString_AppendGrow)->Args({8, 8})->Args({15, 8})->Args({16, 16})->Args({64, 64});
BENCHMARK(BM_StdString_AppendGrow)->Args({8, 8})->Args({15, 8})->Args({16, 16})->Args({64, 64});
BENCHMARK(BM_AmString_Assign)->Arg(8)->Arg(15)->Arg(16)->Arg(64)->Arg(256);
BENCHMARK(BM_StdString_Assign)->Arg(8)->Arg(15)->Arg(16)->Arg(64)->Arg(256);
BENCHMARK(BM_AmString_ShrinkToFit)->Arg(8)->Arg(15)->Arg(16)->Arg(64)->Arg(256);
BENCHMARK(BM_StdString_ShrinkToFit)->Arg(8)->Arg(15)->Arg(16)->Arg(64)->Arg(256);
