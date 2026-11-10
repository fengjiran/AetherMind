#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/dtypes/fp16_conversion.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/model/loaded_model.h"
#include "aethermind/model/weight/raw_weight_fp32_conversion.h"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace {
using namespace aethermind;

Fp16ConversionFuncs Functions(bool optimized) {
    if (!optimized) return {};
    const CpuBackend backend;
    return backend.GetHostFp16Conversions();
}

void BM_Fp16Batch(benchmark::State& state, bool widening) {
    const auto count = static_cast<size_t>(state.range(0));
    const bool optimized = state.range(1) != 0;
    const auto offset = static_cast<size_t>(state.range(2));
    const auto functions = Functions(optimized);
    const size_t input_width = widening ? 2 : 4;
    const size_t output_width = widening ? 4 : 2;
    std::vector<std::byte> source(offset + count * input_width);
    std::vector<std::byte> destination(offset + count * output_width);
    for (size_t index = 0; index < count; ++index) {
        const float value = static_cast<float>(static_cast<int>(index % 127) - 63) * 0.03125F;
        if (widening) {
            const uint16_t bits = Half(value).bits();
            std::memcpy(source.data() + offset + index * input_width, &bits, sizeof(bits));
        } else {
            std::memcpy(source.data() + offset + index * input_width, &value, sizeof(value));
        }
    }
    const auto input = std::span<const std::byte>(source).subspan(offset);
    auto output = std::span(destination).subspan(offset);
    for (auto _: state) {
        const auto status = widening ? ConvertFp16ToFp32(input, output, functions)
                                     : ConvertFp32ToFp16(input, output, functions);
        if (!status.ok()) {
            state.SkipWithError(status.ToString().c_str());
            break;
        }
        benchmark::ClobberMemory();
    }
    const bool selected = widening ? functions.to_fp32 != ConvertFp16ToFp32Scalar
                                   : functions.from_fp32 != ConvertFp32ToFp16Scalar;
    state.SetLabel(selected ? (widening ? "F16C" : "AVX2 integer") : "scalar");
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(count) * 6);
}

struct BenchmarkStorage final : RawStorage {
    explicit BenchmarkStorage(size_t count) : bytes(count * sizeof(uint16_t)) {
        for (size_t index = 0; index < count; ++index) {
            const uint16_t bits = static_cast<uint16_t>(0x2800U + index % 0x1000U);
            std::memcpy(bytes.data() + index * sizeof(bits), &bits, sizeof(bits));
        }
    }
    std::optional<RawStorageFootprint> footprint() const noexcept override {
        return RawStorageFootprint{RawStorageKind::kOwned, bytes.size()};
    }
    std::vector<std::byte> bytes;
};

void BM_Fp16WeightPreparation(benchmark::State& state) {
    const auto count = static_cast<size_t>(state.range(0));
    const auto functions = Functions(state.range(1) != 0);
    auto storage = std::make_shared<BenchmarkStorage>(count);
    RawWeightView view{.data = storage->bytes.data(),
                       .bytes = storage->bytes.size(),
                       .dtype = DataType::Float(16),
                       .shape = {static_cast<int64_t>(count)},
                       .storage = storage};
    ResolvedModelWeights weights;
    weights.embed_tokens = view;
    weights.final_norm = view;
    HfModelConfig config;
    config.weight_dtype_hint = DataType::Float(16);
    LoadedModel model(config, std::move(weights));
    CPUAllocator allocator(Device::CPU());
    for (auto _: state) {
        auto converted = ConvertRawModelWeightsToFp32(model, allocator, {.fp16_to_fp32 = functions.to_fp32});
        if (!converted.ok()) {
            state.SkipWithError(converted.status().ToString().c_str());
            break;
        }
        auto* data = (*converted)->GetResolvedWeights().embed_tokens.data;
        benchmark::DoNotOptimize(data);
        benchmark::ClobberMemory();
    }
    state.SetLabel(functions.to_fp32 == ConvertFp16ToFp32Scalar ? "scalar" : "F16C");
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(count) * 6);
}

void BatchArguments(benchmark::Benchmark* registration) {
    for (const int64_t count: {8, 31, 1024, 65536, 1048576}) {
        for (const int64_t optimized: {0, 1}) {
            for (const int64_t offset: {0, 1}) {
                registration->Args({count, optimized, offset});
            }
        }
    }
}

BENCHMARK_CAPTURE(BM_Fp16Batch, Widen, true)->Apply(BatchArguments)->ArgNames({"N", "Selected", "Offset"});
BENCHMARK_CAPTURE(BM_Fp16Batch, Narrow, false)->Apply(BatchArguments)->ArgNames({"N", "Selected", "Offset"});
BENCHMARK(BM_Fp16WeightPreparation)
        ->Args({1024, 0})
        ->Args({1024, 1})
        ->Args({65536, 0})
        ->Args({65536, 1})
        ->Args({1048576, 0})
        ->Args({1048576, 1})
        ->ArgNames({"N", "Selected"});

} // namespace
