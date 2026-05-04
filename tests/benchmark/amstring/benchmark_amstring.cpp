#include "amstring/string.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <string>

namespace {

using namespace aethermind;

struct PositionKind {
    std::uint8_t value_;
};

struct FindScenario {
    std::uint8_t value_;
};

std::string MakePayload(std::size_t size) {
    std::string payload;
    payload.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
        payload.push_back(static_cast<char>('a' + static_cast<char>(i % 23)));
    }
    return payload;
}

PositionKind PositionKindFromRange(int64_t value) {
    if (value == 0) {
        return PositionKind{0};
    }
    if (value == 1) {
        return PositionKind{1};
    }
    return PositionKind{2};
}

FindScenario FindScenarioFromRange(int64_t value) {
    if (value == 0) {
        return FindScenario{0};
    }
    if (value == 1) {
        return FindScenario{1};
    }
    if (value == 2) {
        return FindScenario{2};
    }
    return FindScenario{3};
}

std::size_t PositionFor(std::size_t size, PositionKind position_kind) {
    if (position_kind.value_ == 0) {
        return 0;
    }
    if (position_kind.value_ == 1) {
        return size / 2;
    }
    return size;
}

std::size_t ReplacePositionFor(std::size_t size, std::size_t count, PositionKind position_kind) {
    const std::size_t last_valid = size - count;
    return std::min(PositionFor(size, position_kind), last_valid);
}

bool IsFindMiss(FindScenario scenario) {
    return scenario.value_ == 3;
}

std::size_t FindMatchPositionFor(std::size_t haystack_size, std::size_t needle_size, FindScenario scenario) {
    if (scenario.value_ == 0) {
        return 0;
    }

    const std::size_t last_valid = haystack_size - needle_size;
    if (scenario.value_ == 1) {
        return last_valid / 2;
    }
    return last_valid;
}

std::string MakeFindCharPayload(std::size_t size, FindScenario scenario, char target) {
    std::string payload = MakePayload(size);
    if (!IsFindMiss(scenario) && size > 0) {
        payload[FindMatchPositionFor(size, 1, scenario)] = target;
    }
    return payload;
}

std::string MakeFindCleanPayload(std::size_t size, std::size_t needle_size, FindScenario scenario, const std::string& needle) {
    std::string payload(size, 'a');
    if (!IsFindMiss(scenario)) {
        payload.replace(FindMatchPositionFor(size, needle_size, scenario), needle_size, needle);
    }
    return payload;
}

std::string MakeFindPartialPayload(std::size_t size, std::size_t needle_size, FindScenario scenario) {
    std::string payload(size, 'a');
    if (!IsFindMiss(scenario)) {
        payload[FindMatchPositionFor(size, needle_size, scenario) + needle_size - 1] = 'b';
    }
    return payload;
}

std::string MakePartialNeedle(std::size_t needle_size) {
    std::string needle(needle_size, 'a');
    needle.back() = 'b';
    return needle;
}

void BM_AmString_Construct(benchmark::State& state) {
    const std::string payload = MakePayload(static_cast<std::size_t>(state.range(0)));
    for (auto _: state) {
        amstring value(payload.data(), payload.size());
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
    const amstring original(payload.data(), payload.size());
    for (auto _: state) {
        amstring copy(original);
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
        amstring value(prefix.data(), prefix.size());
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
        amstring value(prefix.data(), prefix.size());
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
    amstring value;
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
    const auto size = static_cast<std::size_t>(state.range(0));
    const std::string payload = MakePayload(size);
    const std::size_t reserve_size = std::max<std::size_t>(size + 64, size * 2 + 1);
    for (auto _: state) {
        amstring value(payload.data(), payload.size());
        value.reserve(reserve_size);
        value.shrink_to_fit();
        benchmark::DoNotOptimize(value.data());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size()));
}

void BM_StdString_ShrinkToFit(benchmark::State& state) {
    const auto size = static_cast<std::size_t>(state.range(0));
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

void BM_AmString_ReserveFromEmpty(benchmark::State& state) {
    const auto capacity = static_cast<std::size_t>(state.range(0));
    for (auto _: state) {
        amstring value;
        value.reserve(capacity);
        benchmark::DoNotOptimize(value.data());
        benchmark::DoNotOptimize(value.capacity());
    }
    state.SetItemsProcessed(state.iterations());
}

void BM_StdString_ReserveFromEmpty(benchmark::State& state) {
    const auto capacity = static_cast<std::size_t>(state.range(0));
    for (auto _: state) {
        std::string value;
        value.reserve(capacity);
        benchmark::DoNotOptimize(value.data());
        benchmark::DoNotOptimize(value.capacity());
    }
    state.SetItemsProcessed(state.iterations());
}

void BM_AmString_GrowthStaircase(benchmark::State& state) {
    const auto target_size = static_cast<std::size_t>(state.range(0));
    const std::string payload = MakePayload(target_size);
    for (auto _: state) {
        amstring value;
        for (const char ch: payload) {
            value.push_back(ch);
        }
        benchmark::DoNotOptimize(value.data());
        benchmark::DoNotOptimize(value.capacity());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size()));
}

void BM_StdString_GrowthStaircase(benchmark::State& state) {
    const auto target_size = static_cast<std::size_t>(state.range(0));
    const std::string payload = MakePayload(target_size);
    for (auto _: state) {
        std::string value;
        for (const char ch: payload) {
            value.push_back(ch);
        }
        benchmark::DoNotOptimize(value.data());
        benchmark::DoNotOptimize(value.capacity());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size()));
}

void BM_AmString_Insert(benchmark::State& state) {
    const auto base_size = static_cast<std::size_t>(state.range(0));
    const auto insert_size = static_cast<std::size_t>(state.range(1));
    const PositionKind position_kind = PositionKindFromRange(state.range(2));
    const std::string payload = MakePayload(base_size);
    const std::string inserted = MakePayload(insert_size);
    const std::size_t position = PositionFor(base_size, position_kind);
    for (auto _: state) {
        amstring value(payload.data(), payload.size());
        value.insert(position, inserted.data(), inserted.size());
        benchmark::DoNotOptimize(value.data());
        benchmark::DoNotOptimize(value.size());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size() + inserted.size()));
}

void BM_StdString_Insert(benchmark::State& state) {
    const auto base_size = static_cast<std::size_t>(state.range(0));
    const auto insert_size = static_cast<std::size_t>(state.range(1));
    const PositionKind position_kind = PositionKindFromRange(state.range(2));
    const std::string payload = MakePayload(base_size);
    const std::string inserted = MakePayload(insert_size);
    const std::size_t position = PositionFor(base_size, position_kind);
    for (auto _: state) {
        std::string value(payload.data(), payload.size());
        value.insert(position, inserted.data(), inserted.size());
        benchmark::DoNotOptimize(value.data());
        benchmark::DoNotOptimize(value.size());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size() + inserted.size()));
}

void BM_AmString_Replace(benchmark::State& state) {
    const auto base_size = static_cast<std::size_t>(state.range(0));
    const auto erase_size = static_cast<std::size_t>(state.range(1));
    const auto replacement_size = static_cast<std::size_t>(state.range(2));
    const PositionKind position_kind = PositionKindFromRange(state.range(3));
    const std::string payload = MakePayload(base_size);
    const std::string replacement = MakePayload(replacement_size);
    const std::size_t position = ReplacePositionFor(base_size, erase_size, position_kind);
    for (auto _: state) {
        amstring value(payload.data(), payload.size());
        value.replace(position, erase_size, replacement.data(), replacement.size());
        benchmark::DoNotOptimize(value.data());
        benchmark::DoNotOptimize(value.size());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size() + replacement.size()));
}

void BM_StdString_Replace(benchmark::State& state) {
    const auto base_size = static_cast<std::size_t>(state.range(0));
    const auto erase_size = static_cast<std::size_t>(state.range(1));
    const auto replacement_size = static_cast<std::size_t>(state.range(2));
    const PositionKind position_kind = PositionKindFromRange(state.range(3));
    const std::string payload = MakePayload(base_size);
    const std::string replacement = MakePayload(replacement_size);
    const std::size_t position = ReplacePositionFor(base_size, erase_size, position_kind);
    for (auto _: state) {
        std::string value(payload.data(), payload.size());
        value.replace(position, erase_size, replacement.data(), replacement.size());
        benchmark::DoNotOptimize(value.data());
        benchmark::DoNotOptimize(value.size());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size() + replacement.size()));
}

void BM_AmString_FindChar(benchmark::State& state) {
    const auto haystack_size = static_cast<std::size_t>(state.range(0));
    const FindScenario scenario = FindScenarioFromRange(state.range(1));
    const std::string payload = MakeFindCharPayload(haystack_size, scenario, '~');
    const amstring value(payload.data(), payload.size());
    for (auto _: state) {
        auto result = value.find('~');
        benchmark::DoNotOptimize(result);
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size() + 1));
}

void BM_StdString_FindChar(benchmark::State& state) {
    const auto haystack_size = static_cast<std::size_t>(state.range(0));
    const FindScenario scenario = FindScenarioFromRange(state.range(1));
    const std::string value = MakeFindCharPayload(haystack_size, scenario, '~');
    for (auto _: state) {
        auto result = value.find('~');
        benchmark::DoNotOptimize(result);
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(value.size() + 1));
}

void BM_AmString_FindSubstringClean(benchmark::State& state) {
    const auto haystack_size = static_cast<std::size_t>(state.range(0));
    const auto needle_size = static_cast<std::size_t>(state.range(1));
    const FindScenario scenario = FindScenarioFromRange(state.range(2));
    const std::string needle(needle_size, 'b');
    const std::string payload = MakeFindCleanPayload(haystack_size, needle_size, scenario, needle);
    const amstring value(payload.data(), payload.size());
    for (auto _: state) {
        auto result = value.find(needle.data(), 0, needle.size());
        benchmark::DoNotOptimize(result);
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size() + needle.size()));
}

void BM_StdString_FindSubstringClean(benchmark::State& state) {
    const auto haystack_size = static_cast<std::size_t>(state.range(0));
    const auto needle_size = static_cast<std::size_t>(state.range(1));
    const FindScenario scenario = FindScenarioFromRange(state.range(2));
    const std::string needle(needle_size, 'b');
    const std::string value = MakeFindCleanPayload(haystack_size, needle_size, scenario, needle);
    for (auto _: state) {
        auto result = value.find(needle.data(), 0, needle.size());
        benchmark::DoNotOptimize(result);
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(value.size() + needle.size()));
}

void BM_AmString_FindSubstringPartial(benchmark::State& state) {
    const auto haystack_size = static_cast<std::size_t>(state.range(0));
    const auto needle_size = static_cast<std::size_t>(state.range(1));
    const FindScenario scenario = FindScenarioFromRange(state.range(2));
    const std::string needle = MakePartialNeedle(needle_size);
    const std::string payload = MakeFindPartialPayload(haystack_size, needle_size, scenario);
    const amstring value(payload.data(), payload.size());
    for (auto _: state) {
        auto result = value.find(needle.data(), 0, needle.size());
        benchmark::DoNotOptimize(result);
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size() + needle.size()));
}

void BM_StdString_FindSubstringPartial(benchmark::State& state) {
    const auto haystack_size = static_cast<std::size_t>(state.range(0));
    const auto needle_size = static_cast<std::size_t>(state.range(1));
    const FindScenario scenario = FindScenarioFromRange(state.range(2));
    const std::string needle = MakePartialNeedle(needle_size);
    const std::string value = MakeFindPartialPayload(haystack_size, needle_size, scenario);
    for (auto _: state) {
        auto result = value.find(needle.data(), 0, needle.size());
        benchmark::DoNotOptimize(result);
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(value.size() + needle.size()));
}

void BM_AmString_AppendLargePageBoundary(benchmark::State& state) {
    const auto base_size = static_cast<std::size_t>(state.range(0));
    const auto append_size = static_cast<std::size_t>(state.range(1));
    const std::string prefix = MakePayload(base_size);
    const std::string suffix = MakePayload(append_size);
    for (auto _: state) {
        amstring value(prefix.data(), prefix.size());
        value.append(suffix.data(), suffix.size());
        benchmark::DoNotOptimize(value.data());
        benchmark::DoNotOptimize(value.capacity());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(prefix.size() + suffix.size()));
}

void BM_StdString_AppendLargePageBoundary(benchmark::State& state) {
    const auto base_size = static_cast<std::size_t>(state.range(0));
    const auto append_size = static_cast<std::size_t>(state.range(1));
    const std::string prefix = MakePayload(base_size);
    const std::string suffix = MakePayload(append_size);
    for (auto _: state) {
        std::string value(prefix.data(), prefix.size());
        value.append(suffix.data(), suffix.size());
        benchmark::DoNotOptimize(value.data());
        benchmark::DoNotOptimize(value.capacity());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(prefix.size() + suffix.size()));
}

void BM_AmString_ShrinkToFitLargePage(benchmark::State& state) {
    const auto size = static_cast<std::size_t>(state.range(0));
    const auto reserve_size = static_cast<std::size_t>(state.range(1));
    const std::string payload = MakePayload(size);
    for (auto _: state) {
        amstring value(payload.data(), payload.size());
        value.reserve(reserve_size);
        value.shrink_to_fit();
        benchmark::DoNotOptimize(value.data());
        benchmark::DoNotOptimize(value.capacity());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size()));
}

void BM_StdString_ShrinkToFitLargePage(benchmark::State& state) {
    const auto size = static_cast<std::size_t>(state.range(0));
    const auto reserve_size = static_cast<std::size_t>(state.range(1));
    const std::string payload = MakePayload(size);
    for (auto _: state) {
        std::string value(payload.data(), payload.size());
        value.reserve(reserve_size);
        value.shrink_to_fit();
        benchmark::DoNotOptimize(value.data());
        benchmark::DoNotOptimize(value.capacity());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(payload.size()));
}

}// namespace

BENCHMARK(BM_AmString_Construct)
        ->Arg(8)
        ->Arg(15)
        ->Arg(16)
        ->Arg(22)
        ->Arg(23)
        ->Arg(24)
        ->Arg(64)
        ->Arg(256);
BENCHMARK(BM_StdString_Construct)
        ->Arg(8)
        ->Arg(15)
        ->Arg(16)
        ->Arg(22)
        ->Arg(23)
        ->Arg(24)
        ->Arg(64)
        ->Arg(256);
BENCHMARK(BM_AmString_Copy)
        ->Arg(8)
        ->Arg(15)
        ->Arg(16)
        ->Arg(22)
        ->Arg(23)
        ->Arg(24)
        ->Arg(64)
        ->Arg(256);
BENCHMARK(BM_StdString_Copy)
        ->Arg(8)
        ->Arg(15)
        ->Arg(16)
        ->Arg(22)
        ->Arg(23)
        ->Arg(24)
        ->Arg(64)
        ->Arg(256);
BENCHMARK(BM_AmString_AppendReserved)
        ->Args({8, 4})
        ->Args({15, 1})
        ->Args({16, 8})
        ->Args({22, 1})
        ->Args({23, 1})
        ->Args({22, 2})
        ->Args({24, 1})
        ->Args({64, 32});
BENCHMARK(BM_StdString_AppendReserved)
        ->Args({8, 4})
        ->Args({15, 1})
        ->Args({16, 8})
        ->Args({22, 1})
        ->Args({23, 1})
        ->Args({22, 2})
        ->Args({24, 1})
        ->Args({64, 32});
BENCHMARK(BM_AmString_AppendGrow)
        ->Args({8, 8})
        ->Args({15, 8})
        ->Args({16, 16})
        ->Args({22, 1})
        ->Args({23, 1})
        ->Args({22, 2})
        ->Args({24, 1})
        ->Args({64, 64});
BENCHMARK(BM_StdString_AppendGrow)
        ->Args({8, 8})
        ->Args({15, 8})
        ->Args({16, 16})
        ->Args({22, 1})
        ->Args({23, 1})
        ->Args({22, 2})
        ->Args({24, 1})
        ->Args({64, 64});
BENCHMARK(BM_AmString_Assign)
        ->Arg(8)
        ->Arg(15)
        ->Arg(16)
        ->Arg(22)
        ->Arg(23)
        ->Arg(24)
        ->Arg(64)
        ->Arg(256);
BENCHMARK(BM_StdString_Assign)
        ->Arg(8)
        ->Arg(15)
        ->Arg(16)
        ->Arg(22)
        ->Arg(23)
        ->Arg(24)
        ->Arg(64)
        ->Arg(256);
BENCHMARK(BM_AmString_ShrinkToFit)
        ->Arg(8)
        ->Arg(15)
        ->Arg(16)
        ->Arg(22)
        ->Arg(23)
        ->Arg(24)
        ->Arg(64)
        ->Arg(256);
BENCHMARK(BM_StdString_ShrinkToFit)
        ->Arg(8)
        ->Arg(15)
        ->Arg(16)
        ->Arg(22)
        ->Arg(23)
        ->Arg(24)
        ->Arg(64)
        ->Arg(256);
BENCHMARK(BM_AmString_ReserveFromEmpty)
        ->Arg(0)
        ->Arg(1)
        ->Arg(23)
        ->Arg(24)
        ->Arg(31)
        ->Arg(32)
        ->Arg(33)
        ->Arg(4094)
        ->Arg(4095)
        ->Arg(4096)
        ->Arg(8191)
        ->Arg(8192);
BENCHMARK(BM_StdString_ReserveFromEmpty)
        ->Arg(0)
        ->Arg(1)
        ->Arg(23)
        ->Arg(24)
        ->Arg(31)
        ->Arg(32)
        ->Arg(33)
        ->Arg(4094)
        ->Arg(4095)
        ->Arg(4096)
        ->Arg(8191)
        ->Arg(8192);
BENCHMARK(BM_AmString_GrowthStaircase)
        ->Arg(23)
        ->Arg(24)
        ->Arg(35)
        ->Arg(52)
        ->Arg(77)
        ->Arg(115)
        ->Arg(172)
        ->Arg(257)
        ->Arg(385)
        ->Arg(577)
        ->Arg(865)
        ->Arg(1297)
        ->Arg(1945)
        ->Arg(2917)
        ->Arg(8192)
        ->Arg(12288)
        ->Arg(16384);
BENCHMARK(BM_StdString_GrowthStaircase)
        ->Arg(23)
        ->Arg(24)
        ->Arg(35)
        ->Arg(52)
        ->Arg(77)
        ->Arg(115)
        ->Arg(172)
        ->Arg(257)
        ->Arg(385)
        ->Arg(577)
        ->Arg(865)
        ->Arg(1297)
        ->Arg(1945)
        ->Arg(2917)
        ->Arg(8192)
        ->Arg(12288)
        ->Arg(16384);
BENCHMARK(BM_AmString_Insert)
        ->Args({8, 1, 0})
        ->Args({23, 1, 2})
        ->Args({23, 1, 1})
        ->Args({24, 8, 1})
        ->Args({64, 8, 0})
        ->Args({64, 8, 1})
        ->Args({64, 8, 2})
        ->Args({256, 64, 1})
        ->Args({4096, 128, 1});
BENCHMARK(BM_StdString_Insert)
        ->Args({8, 1, 0})
        ->Args({23, 1, 2})
        ->Args({23, 1, 1})
        ->Args({24, 8, 1})
        ->Args({64, 8, 0})
        ->Args({64, 8, 1})
        ->Args({64, 8, 2})
        ->Args({256, 64, 1})
        ->Args({4096, 128, 1});
BENCHMARK(BM_AmString_Replace)
        ->Args({8, 1, 1, 0})
        ->Args({23, 1, 1, 1})
        ->Args({23, 1, 2, 1})
        ->Args({24, 1, 0, 1})
        ->Args({64, 8, 8, 0})
        ->Args({64, 8, 8, 1})
        ->Args({64, 8, 8, 2})
        ->Args({64, 8, 16, 1})
        ->Args({256, 64, 16, 1})
        ->Args({256, 16, 64, 1})
        ->Args({4096, 128, 256, 1});
BENCHMARK(BM_StdString_Replace)
        ->Args({8, 1, 1, 0})
        ->Args({23, 1, 1, 1})
        ->Args({23, 1, 2, 1})
        ->Args({24, 1, 0, 1})
        ->Args({64, 8, 8, 0})
        ->Args({64, 8, 8, 1})
        ->Args({64, 8, 8, 2})
        ->Args({64, 8, 16, 1})
        ->Args({256, 64, 16, 1})
        ->Args({256, 16, 64, 1})
        ->Args({4096, 128, 256, 1});
BENCHMARK(BM_AmString_FindChar)
        ->Args({0, 3})
        ->Args({8, 0})
        ->Args({8, 1})
        ->Args({8, 2})
        ->Args({8, 3})
        ->Args({22, 2})
        ->Args({23, 2})
        ->Args({24, 2})
        ->Args({64, 1})
        ->Args({64, 3})
        ->Args({256, 2})
        ->Args({256, 3})
        ->Args({4096, 2})
        ->Args({4096, 3})
        ->Args({65536, 2})
        ->Args({65536, 3});
BENCHMARK(BM_StdString_FindChar)
        ->Args({0, 3})
        ->Args({8, 0})
        ->Args({8, 1})
        ->Args({8, 2})
        ->Args({8, 3})
        ->Args({22, 2})
        ->Args({23, 2})
        ->Args({24, 2})
        ->Args({64, 1})
        ->Args({64, 3})
        ->Args({256, 2})
        ->Args({256, 3})
        ->Args({4096, 2})
        ->Args({4096, 3})
        ->Args({65536, 2})
        ->Args({65536, 3});
BENCHMARK(BM_AmString_FindSubstringClean)
        ->Args({8, 1, 0})
        ->Args({8, 1, 2})
        ->Args({8, 1, 3})
        ->Args({23, 2, 1})
        ->Args({23, 8, 2})
        ->Args({24, 2, 1})
        ->Args({24, 8, 3})
        ->Args({64, 8, 0})
        ->Args({64, 8, 1})
        ->Args({64, 8, 2})
        ->Args({64, 8, 3})
        ->Args({256, 16, 1})
        ->Args({256, 64, 3})
        ->Args({4096, 8, 2})
        ->Args({4096, 64, 3})
        ->Args({65536, 8, 2})
        ->Args({65536, 64, 3});
BENCHMARK(BM_StdString_FindSubstringClean)
        ->Args({8, 1, 0})
        ->Args({8, 1, 2})
        ->Args({8, 1, 3})
        ->Args({23, 2, 1})
        ->Args({23, 8, 2})
        ->Args({24, 2, 1})
        ->Args({24, 8, 3})
        ->Args({64, 8, 0})
        ->Args({64, 8, 1})
        ->Args({64, 8, 2})
        ->Args({64, 8, 3})
        ->Args({256, 16, 1})
        ->Args({256, 64, 3})
        ->Args({4096, 8, 2})
        ->Args({4096, 64, 3})
        ->Args({65536, 8, 2})
        ->Args({65536, 64, 3});
BENCHMARK(BM_AmString_FindSubstringPartial)
        ->Args({23, 4, 3})
        ->Args({24, 4, 3})
        ->Args({64, 4, 3})
        ->Args({64, 8, 2})
        ->Args({64, 8, 3})
        ->Args({256, 8, 3})
        ->Args({256, 16, 2})
        ->Args({256, 16, 3})
        ->Args({4096, 8, 3})
        ->Args({4096, 32, 2})
        ->Args({4096, 32, 3})
        ->Args({65536, 8, 3})
        ->Args({65536, 32, 2})
        ->Args({65536, 32, 3});
BENCHMARK(BM_StdString_FindSubstringPartial)
        ->Args({23, 4, 3})
        ->Args({24, 4, 3})
        ->Args({64, 4, 3})
        ->Args({64, 8, 2})
        ->Args({64, 8, 3})
        ->Args({256, 8, 3})
        ->Args({256, 16, 2})
        ->Args({256, 16, 3})
        ->Args({4096, 8, 3})
        ->Args({4096, 32, 2})
        ->Args({4096, 32, 3})
        ->Args({65536, 8, 3})
        ->Args({65536, 32, 2})
        ->Args({65536, 32, 3});
BENCHMARK(BM_AmString_AppendLargePageBoundary)
        ->Args({4094, 1})
        ->Args({4095, 1})
        ->Args({4096, 1})
        ->Args({8191, 1})
        ->Args({8192, 1})
        ->Args({12287, 1})
        ->Args({16383, 4096})
        ->Args({65535, 4096});
BENCHMARK(BM_StdString_AppendLargePageBoundary)
        ->Args({4094, 1})
        ->Args({4095, 1})
        ->Args({4096, 1})
        ->Args({8191, 1})
        ->Args({8192, 1})
        ->Args({12287, 1})
        ->Args({16383, 4096})
        ->Args({65535, 4096});
BENCHMARK(BM_AmString_ShrinkToFitLargePage)
        ->Args({4094, 8191})
        ->Args({4095, 8191})
        ->Args({4096, 8191})
        ->Args({8192, 24575})
        ->Args({12288, 32767})
        ->Args({65536, 98303});
BENCHMARK(BM_StdString_ShrinkToFitLargePage)
        ->Args({4094, 8191})
        ->Args({4095, 8191})
        ->Args({4096, 8191})
        ->Args({8192, 24575})
        ->Args({12288, 32767})
        ->Args({65536, 98303});
