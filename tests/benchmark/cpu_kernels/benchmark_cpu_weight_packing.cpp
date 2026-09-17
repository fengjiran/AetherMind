#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/base/tensor_view.h"

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace aethermind;

KernelSelector MakePackedLinearSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .phase = ExecPhase::kBoth,
    };
}

float DeterministicWeightValue(size_t index) {
    constexpr size_t kPeriod = 127;
    return static_cast<float>(static_cast<int64_t>(index % kPeriod) -
                              static_cast<int64_t>(kPeriod / 2U)) *
           0.015625F;
}

void SetPackingCounters(benchmark::State& state,
                        int64_t n,
                        int64_t k,
                        size_t logical_bytes,
                        size_t packed_bytes) {
    state.SetItemsProcessed(state.iterations() * n * k);
    state.SetBytesProcessed(state.iterations() *
                            static_cast<int64_t>(logical_bytes * 2U));
    state.counters["N"] = static_cast<double>(n);
    state.counters["K"] = static_cast<double>(k);
    state.counters["packed bytes"] = static_cast<double>(packed_bytes);
    state.counters["size amplification"] =
            static_cast<double>(packed_bytes) / static_cast<double>(logical_bytes);
    state.counters["effective GB/s"] = benchmark::Counter(
            static_cast<double>(state.iterations()) *
                    static_cast<double>(logical_bytes * 2U),
            benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
}

void BM_WeightPackingCpuIdentity(benchmark::State& state) {
    const int64_t n = state.range(0);
    const int64_t k = state.range(1);
    const size_t weight_elements = static_cast<size_t>(n * k);
    const size_t logical_bytes = weight_elements * sizeof(float);
    std::vector<float> weight(weight_elements);
    for (size_t index = 0; index < weight.size(); ++index) {
        weight[index] = DeterministicWeightValue(index);
    }

    const std::array<int64_t, 2> shape = {n, k};
    const std::array<int64_t, 2> strides = {k, 1};
    const TensorView logical_weight{
            weight.data(), DataType::Float32(), shape, strides};
    const KernelSelector selector = MakePackedLinearSelector();
    const CpuWeightPrepacker prepacker;

    // Validate the current identity recipe once before timing. This benchmark
    // is intentionally a cold pack measurement, not a packed Linear kernel
    // benchmark: no packed Linear descriptor exists yet.
    const auto checked_packed = prepacker.Pack(OpType::kLinear, logical_weight, selector);
    if (!checked_packed.ok()) {
        state.SkipWithError(checked_packed.status().ToString().c_str());
        return;
    }
    if ((*checked_packed)->storage().nbytes() != logical_bytes ||
        (logical_bytes > 0 &&
         std::memcmp((*checked_packed)->storage().data(), weight.data(), logical_bytes) != 0)) {
        state.SkipWithError("cpu_identity packing correctness guard failed");
        return;
    }

    state.SetLabel(std::string{"recipe="} + (*checked_packed)->recipe().layout +
                   " mode=cold-packing");
    for (auto _: state) {
        const auto packed = prepacker.Pack(OpType::kLinear, logical_weight, selector);
        if (!packed.ok()) {
            state.SkipWithError(packed.status().ToString().c_str());
            break;
        }
        benchmark::DoNotOptimize((*packed)->storage().data());
    }

    SetPackingCounters(state, n, k, logical_bytes, (*checked_packed)->storage().nbytes());
}

BENCHMARK(BM_WeightPackingCpuIdentity)
        ->Args({4096, 4096})
        ->Args({11008, 4096})
        ->Args({32000, 4096})
        ->ArgNames({"N", "K"});

} // namespace
