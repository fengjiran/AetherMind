#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/operators/op_params.h"

#include <benchmark/benchmark.h>

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

using namespace aethermind;

void BenchmarkAttention(benchmark::State& state, DataType activation, DataType kv) {
    const int64_t rows = state.range(0);
    const int64_t context = state.range(1);
    const size_t sets = state.range(2);
    const int64_t group = state.range(3);
    const int64_t dim = state.range(4);
    const int64_t kv_heads = state.range(5);
    const int64_t q_heads = kv_heads * group;
    const int64_t width = q_heads * dim;
    const auto query_value = [](int64_t head, int64_t row) { return (head + row) % 2 == 0 ? 0.125F : 0.25F; };
    constexpr float kMin = 0.5F, kMax = 1.0F;
    const size_t act_bytes = activation.nbytes(), kv_bytes = kv.nbytes();
    const auto write = [](std::byte* dest, DataType dtype, float value) {
        if (dtype == DataType::Float32()) {
            std::memcpy(dest, &value, sizeof(value));
        } else {
            const uint16_t bits = dtype == DataType::Float(16) ? Half(value).bits() : BFloat16(value).x;
            std::memcpy(dest, &bits, sizeof(bits));
        }
    };
    const CpuBackend backend;
    const auto kernel = backend.PrepareKernel(
            OpType::kAttention,
            {.device_type = DeviceType::kCPU, .act_dtype = activation, .output_dtype = activation, .weight_dtype = activation, .state_dtype = kv, .weight_format = WeightFormat::kPlain, .phase = rows == 1 ? ExecPhase::kDecode : ExecPhase::kPrefill},
            AttentionParams{.num_q_heads = q_heads, .num_kv_heads = kv_heads, .head_dim = dim});
    if (!kernel.ok()) {
        state.SkipWithError(kernel.status().message().c_str());
        return;
    }

    std::vector<std::byte> query(rows * width * act_bytes), output(query.size());
    for (int64_t i = 0; i < rows * width; ++i) {
        write(query.data() + i * act_bytes, activation, query_value((i % width) / dim, i / width));
    }

    const size_t token_bytes = dim * kv_bytes;
    const size_t head_bytes = context * token_bytes;
    const size_t set_bytes = kv_heads * head_bytes;
    std::vector<std::byte> keys(sets * set_bytes), values(keys.size());
    std::vector<std::byte> key_rows(2 * token_bytes), value_rows(2 * token_bytes);
    for (size_t col = 0; col < dim; ++col) {
        write(key_rows.data() + col * kv_bytes, kv, 0.125F);
        write(key_rows.data() + token_bytes + col * kv_bytes, kv, 0.25F);
        write(value_rows.data() + col * kv_bytes, kv, kMin);
        write(value_rows.data() + token_bytes + col * kv_bytes, kv, kMax);
    }
    for (size_t offset = 0; offset < keys.size(); offset += token_bytes) {
        const size_t row = ((offset / token_bytes) % context) % 2;
        std::memcpy(keys.data() + offset, key_rows.data() + row * token_bytes, token_bytes);
        std::memcpy(values.data() + offset, value_rows.data() + row * token_bytes, token_bytes);
    }

    const std::array<int64_t, 2> shape{rows, width};
    const std::array<int64_t, 2> strides{width, 1};
    const std::array inputs{TensorView(query.data(), activation, shape, strides)};
    const std::array outputs{MutableTensorView(output.data(), activation, shape, strides)};
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> params{};
    const auto bind = kernel->params_builder({.inputs = inputs, .outputs = outputs, .attrs = kernel->attrs}, params.data());
    if (!bind.ok()) {
        state.SkipWithError(bind.message().c_str());
        return;
    }
    KVCacheReadBinding read{
            .storage = {.key_data = keys.data(), .value_data = values.data(), .dtype = kv, .num_kv_heads = static_cast<size_t>(kv_heads), .head_dim = static_cast<size_t>(dim), .token_capacity = static_cast<size_t>(context), .token_stride_bytes = token_bytes, .head_stride_bytes = head_bytes},
            .committed_end = static_cast<size_t>(context - rows),
            .visible_end = static_cast<size_t>(context),
            .query_begin = static_cast<size_t>(context - rows),
            .query_end = static_cast<size_t>(context)};
    const KernelContext call{.kernel_params = params.data(), .kv_read = &read, .attrs = kernel->attrs};
    const auto check = kernel->fn(call);
    if (!check.ok()) {
        state.SkipWithError(check.message().c_str());
        return;
    }
    // Two alternating logits let the analytic oracle check every causal row.
    for (int64_t row = 0; row < rows; ++row) {
        const size_t end = read.query_begin + row + 1;
        const size_t odd = end / 2, even = end - odd;
        for (int64_t col = 0; col < width; ++col) {
            const double odd_weight = std::exp(std::sqrt(static_cast<double>(dim)) * query_value(col / dim, row) / 8.0);
            const float expected = static_cast<float>((even * kMin + odd * odd_weight * kMax) / (even + odd * odd_weight));
            const auto* src = output.data() + (row * width + col) * act_bytes;
            float actual;
            if (activation == DataType::Float32()) {
                std::memcpy(&actual, src, sizeof(actual));
            } else {
                uint16_t bits;
                std::memcpy(&bits, src, sizeof(bits));
                actual = BFloat16(bits, BFloat16::from_bits());
            }
            if (!std::isfinite(actual) || std::abs(actual - expected) > (act_bytes == 2 ? 0.002F : 0.00005F)) {
                state.SkipWithError("Prepared Attention output failed the causal analytic guard");
                return;
            }
        }
    }

    // Even a low iteration limit must traverse the full rotating working set.
    while (state.KeepRunningBatch(static_cast<int64_t>(sets))) {
        for (size_t set = 0; set < sets; ++set) {
            read.storage.key_data = keys.data() + set * set_bytes;
            read.storage.value_data = values.data() + set * set_bytes;
            const auto status = kernel->fn(call);
            if (!status.ok()) {
                state.SkipWithError(status.message().c_str());
                return;
            }
            benchmark::ClobberMemory();
        }
    }
    state.SetLabel(std::string(kernel->name));
}

void Shapes(benchmark::Benchmark* benchmark) {
    // Shape tuple: query rows, visible context, rotating KV sets, Q:KV group, head_dim, KV heads.
    for (int64_t context: {1, 16, 64, 128, 1024, 4096}) {
        benchmark->Args({1, context, 1, 4, 128, 8});
    }
    benchmark->Args({1, 4096, 4, 4, 128, 8});
    benchmark->Args({1, 4096, 1, 2, 128, 8});
    benchmark->Args({1, 4096, 1, 1, 128, 8});
    benchmark->Args({1, 1024, 1, 3, 65, 8});
    benchmark->Args({1, 1024, 1, 2, 256, 8});
    for (int64_t rows: {2, 16, 31, 32, 128, 512}) {
        benchmark->Args({rows, rows, 1, 4, 128, 8});
    }
    // Full Prefill at long context with a reduced head count; separate from
    // the wide, nonzero-query-begin control below.
    benchmark->Args({2048, 2048, 1, 2, 128, 2});
    benchmark->Args({4096, 4096, 1, 2, 128, 2});
    benchmark->Args({64, 4096, 1, 4, 128, 8});
    benchmark->Args({129, 129, 1, 1, 65, 8});
    benchmark->Unit(benchmark::kMillisecond);
}

BENCHMARK_CAPTURE(BenchmarkAttention, F32, DataType::Float32(), DataType::Float32())->Apply(Shapes);
BENCHMARK_CAPTURE(BenchmarkAttention, KVF16, DataType::Float32(), DataType::Float(16))->Apply(Shapes);
BENCHMARK_CAPTURE(BenchmarkAttention, KVBF16, DataType::Float32(), DataType::BFloat(16))->Apply(Shapes);
BENCHMARK_CAPTURE(BenchmarkAttention, BF16A16, DataType::BFloat(16), DataType::BFloat(16))->Apply(Shapes);

} // namespace
