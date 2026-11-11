#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/operators/op_params.h"

#include <array>
#include <benchmark/benchmark.h>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace aethermind;

constexpr int64_t kVocabSize = 32000;
constexpr int64_t kHiddenSize = 4096;

struct PreparedEmbeddingInvocation {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> params{};
};

StatusOr<ResolvedKernel> PrepareEmbeddingKernel(DataType weight_dtype, bool scalar = false) {
    CpuBackend backend = scalar ? CpuBackend(CpuCapabilities{}) : CpuBackend();
    return backend.PrepareKernel(
            OpType::kEmbedding,
            KernelSelector{
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .weight_dtype = weight_dtype,
                    .weight_format = WeightFormat::kPlain,
                    .phase = ExecPhase::kBoth,
            },
            OpParams{EmbeddingParams{.output_dtype = DataType::Float32()}});
}

float DeterministicValue(size_t index) {
    constexpr size_t kPeriod = 251;
    return static_cast<float>(static_cast<int64_t>(index % kPeriod) -
                              static_cast<int64_t>(kPeriod / 2U)) *
           0.0078125F;
}

uint16_t NarrowToBits(float value, DataType dtype) {
    if (dtype == DataType::Float(16)) {
        return Half(value).bits();
    }
    return BFloat16(value).x;
}

float WidenFromBits(uint16_t bits, DataType dtype) {
    const int fraction_bits = dtype == DataType::Float(16) ? 10 : 7;
    const int bias = fraction_bits == 10 ? 15 : 127;
    const unsigned exponent_mask = fraction_bits == 10 ? 31U : 255U;
    const unsigned exponent = (bits >> fraction_bits) & exponent_mask;
    const unsigned fraction = bits & ((1U << fraction_bits) - 1);
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16;
    if (exponent == exponent_mask) {
        return std::bit_cast<float>(sign | 0x7f800000U |
                                    (fraction == 0 ? 0U : (fraction << (23 - fraction_bits)) | 0x00400000U));
    }
    const double significand = exponent == 0 ? fraction : (1U << fraction_bits) + fraction;
    const float value = static_cast<float>(std::ldexp(significand,
                                                      static_cast<int>(exponent == 0 ? 1 : exponent) - bias - fraction_bits));
    return std::bit_cast<float>(sign | std::bit_cast<uint32_t>(value));
}

void FillWeight(std::vector<std::byte>& storage, DataType dtype) {
    const size_t elements = static_cast<size_t>(kVocabSize * kHiddenSize);
    if (dtype == DataType::Float32()) {
        for (size_t index = 0; index < elements; ++index) {
            const float value = DeterministicValue(index);
            std::memcpy(storage.data() + index * sizeof(value), &value, sizeof(value));
        }
        return;
    }
    for (size_t index = 0; index < elements; ++index) {
        const uint16_t bits = NarrowToBits(DeterministicValue(index), dtype);
        std::memcpy(storage.data() + index * sizeof(bits), &bits, sizeof(bits));
    }
}

Status BuildEmbeddingPreparedParams(const ResolvedKernel& kernel,
                                    const int64_t* tokens,
                                    int64_t token_count,
                                    const void* weight,
                                    DataType weight_dtype,
                                    float* output,
                                    PreparedEmbeddingInvocation& prepared) noexcept {
    const std::array<int64_t, 1> token_shape = {token_count};
    const std::array<int64_t, 1> token_strides = {1};
    const std::array<int64_t, 2> weight_shape = {kVocabSize, kHiddenSize};
    const std::array<int64_t, 2> weight_strides = {kHiddenSize, 1};
    const std::array<int64_t, 2> output_shape = {token_count, kHiddenSize};
    const std::array<int64_t, 2> output_strides = {kHiddenSize, 1};
    const std::array<TensorView, 2> inputs{
            TensorView{tokens, DataType::Int(64), token_shape, token_strides},
            TensorView{weight, weight_dtype, weight_shape, weight_strides},
    };
    const std::array<MutableTensorView, 1> outputs{
            MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    };
    return kernel.params_builder(KernelParamsBuildContext{
                                         .inputs = inputs,
                                         .outputs = outputs,
                                         .attrs = kernel.attrs,
                                 },
                                 prepared.params.data());
}

Status InvokePreparedEmbedding(const ResolvedKernel& kernel,
                               const PreparedEmbeddingInvocation& prepared) noexcept {
    return kernel.fn(KernelContext{
            .kernel_params = prepared.params.data(),
            .attrs = kernel.attrs,
    });
}

// Embedding widening is bit-exact, so the guard compares every gathered row
// against independently widened source bits rather than a tolerance.
Status ValidateEmbeddingAgainstWideningOracle(const ResolvedKernel& kernel,
                                              const PreparedEmbeddingInvocation& prepared,
                                              const int64_t* tokens,
                                              int64_t token_count,
                                              const std::byte* weight,
                                              DataType weight_dtype,
                                              float* output) {
    const size_t element_size = weight_dtype == DataType::Float32() ? sizeof(float) : sizeof(uint16_t);
    std::vector<float> expected(static_cast<size_t>(token_count * kHiddenSize));
    for (int64_t row = 0; row < token_count; ++row) {
        const size_t source = static_cast<size_t>(tokens[row]) * static_cast<size_t>(kHiddenSize);
        for (int64_t column = 0; column < kHiddenSize; ++column) {
            const size_t index = static_cast<size_t>(row * kHiddenSize + column);
            if (weight_dtype == DataType::Float32()) {
                float value = 0.0F;
                std::memcpy(&value, weight + (source + static_cast<size_t>(column)) * element_size, sizeof(value));
                expected[index] = value;
            } else {
                uint16_t bits = 0;
                std::memcpy(&bits, weight + (source + static_cast<size_t>(column)) * element_size, sizeof(bits));
                expected[index] = WidenFromBits(bits, weight_dtype);
            }
        }
    }

    AM_RETURN_IF_ERROR(InvokePreparedEmbedding(kernel, prepared));
    for (size_t index = 0; index < expected.size(); ++index) {
        if (output[index] != expected[index]) {
            return Status::Internal(
                    "Embedding benchmark correctness guard disagrees with the widening oracle");
        }
    }
    return Status::Ok();
}

void BenchmarkEmbeddingPreparedHot(benchmark::State& state, DataType weight_dtype, bool scalar = false) {
    const int64_t token_count = state.range(0);
    const size_t element_size = weight_dtype == DataType::Float32() ? sizeof(float) : sizeof(uint16_t);
    const size_t weight_bytes = static_cast<size_t>(kVocabSize * kHiddenSize) * element_size;

    std::vector<int64_t> tokens(static_cast<size_t>(token_count));
    for (size_t index = 0; index < tokens.size(); ++index) {
        tokens[index] = static_cast<int64_t>((index * 7U + 3U) % static_cast<size_t>(kVocabSize));
    }
    std::vector<std::byte> weight(weight_bytes);
    FillWeight(weight, weight_dtype);
    std::vector<float> output(static_cast<size_t>(token_count * kHiddenSize));

    const StatusOr<ResolvedKernel> resolved = PrepareEmbeddingKernel(weight_dtype, scalar);
    if (!resolved.ok()) {
        state.SkipWithError(resolved.status().ToString());
        return;
    }
    if (resolved->params_builder == nullptr || resolved->fn == nullptr) {
        state.SkipWithError(
                "Prepared Embedding benchmark requires params_builder and kernel function");
        return;
    }

    PreparedEmbeddingInvocation prepared;
    const Status build = BuildEmbeddingPreparedParams(
            *resolved, tokens.data(), token_count, weight.data(), weight_dtype,
            output.data(), prepared);
    if (!build.ok()) {
        state.SkipWithError(build.ToString());
        return;
    }
    const Status correctness = ValidateEmbeddingAgainstWideningOracle(
            *resolved, prepared, tokens.data(), token_count, weight.data(),
            weight_dtype, output.data());
    if (!correctness.ok()) {
        state.SkipWithError(correctness.ToString());
        return;
    }

    state.SetLabel(std::string{"kernel="} + std::string{resolved->name} + " cache_mode=hot");
    for (auto _: state) {
        const Status status = InvokePreparedEmbedding(*resolved, prepared);
        if (!status.ok()) {
            state.SkipWithError(status.ToString());
            break;
        }
        benchmark::DoNotOptimize(output.data());
    }

    state.counters["tokens"] = static_cast<double>(token_count);
    state.counters["weight MiB"] = static_cast<double>(weight_bytes) / (1024.0 * 1024.0);
    state.SetItemsProcessed(state.iterations() * token_count);
    const int64_t logical_bytes =
            token_count * kHiddenSize * static_cast<int64_t>(sizeof(float) + element_size) +
            token_count * static_cast<int64_t>(sizeof(int64_t));
    state.SetBytesProcessed(state.iterations() * logical_bytes);
}

void BM_EmbeddingPreparedHot(benchmark::State& state, DataType dtype) {
    BenchmarkEmbeddingPreparedHot(state, dtype);
}

void BM_EmbeddingPreparedScalar(benchmark::State& state, DataType dtype) {
    BenchmarkEmbeddingPreparedHot(state, dtype, true);
}

BENCHMARK_CAPTURE(BM_EmbeddingPreparedScalar, f16, DataType::Float(16))
        ->Args({1})
        ->Args({16})
        ->Args({64})
        ->ArgNames({"tokens"});
BENCHMARK_CAPTURE(BM_EmbeddingPreparedScalar, bf16, DataType::BFloat(16))
        ->Args({1})
        ->Args({16})
        ->Args({64})
        ->ArgNames({"tokens"});

BENCHMARK_CAPTURE(BM_EmbeddingPreparedHot, f32, DataType::Float32())
        ->Args({1})
        ->Args({16})
        ->Args({64})
        ->ArgNames({"tokens"});

BENCHMARK_CAPTURE(BM_EmbeddingPreparedHot, f16, DataType::Float(16))
        ->Args({1})
        ->Args({16})
        ->Args({64})
        ->ArgNames({"tokens"});

BENCHMARK_CAPTURE(BM_EmbeddingPreparedHot, bf16, DataType::BFloat(16))
        ->Args({1})
        ->Args({16})
        ->Args({64})
        ->ArgNames({"tokens"});

} // namespace
