#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/operators/op_params.h"
#include "backend/cpu/kernels/attention/attention_internal.h"
#include "backend/cpu/kernels/kvcache_update/kvcache_update_internal.h"

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using namespace aethermind;

// Prepared-kernel baseline for the P3 KV16 reference path (step 10). Params are
// built once outside the loop; the loop runs only kernel.fn against a borrowed
// KV binding. These are scalar reference kernels (no SIMD); the numbers are a
// correctness-path baseline for future optimization, not a performance claim.

KernelSelector MakeSelector(DataType kv_dtype) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .state_dtype = kv_dtype,
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

struct PreparedParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
    std::byte* data() { return storage.data(); }
};

struct Kv16Cache {
    std::vector<std::byte> key;
    std::vector<std::byte> value;
    size_t token_stride_bytes = 0;
    size_t head_stride_bytes = 0;
};

Kv16Cache MakeCache(size_t kv_heads, size_t capacity, size_t head_dim) {
    const size_t token_stride = head_dim * 2;
    const size_t head_stride = capacity * token_stride;
    Kv16Cache cache;
    cache.token_stride_bytes = token_stride;
    cache.head_stride_bytes = head_stride;
    cache.key.assign(kv_heads * head_stride, std::byte{0});
    cache.value.assign(kv_heads * head_stride, std::byte{0});
    return cache;
}

KVCacheLayerStorageBinding MakeStorage(Kv16Cache& cache, DataType kv_dtype, size_t kv_heads,
                                       size_t capacity, size_t head_dim) {
    return KVCacheLayerStorageBinding{
            .key_data = cache.key.data(),
            .value_data = cache.value.data(),
            .dtype = kv_dtype,
            .num_kv_heads = kv_heads,
            .head_dim = head_dim,
            .token_capacity = capacity,
            .token_stride_bytes = cache.token_stride_bytes,
            .head_stride_bytes = cache.head_stride_bytes,
    };
}

void BenchmarkKVUpdateKV16(benchmark::State& state, DataType kv_dtype) {
    const auto seq_len = static_cast<size_t>(state.range(0));
    constexpr size_t kv_heads = 8;
    constexpr size_t head_dim = 128;
    constexpr size_t capacity = 2048;
    const size_t hidden = kv_heads * head_dim;

    CpuBackend backend;
    auto kernel = backend.PrepareKernel(OpType::kKVCacheUpdate, MakeSelector(kv_dtype),
                                        OpParams{KVCacheUpdateParams{}});
    if (!kernel.ok()) {
        state.SkipWithError(kernel.status().message().c_str());
        return;
    }
    std::vector<float> key_act(seq_len * hidden, 0.5F);
    std::vector<float> value_act(seq_len * hidden, 0.25F);
    const std::array<int64_t, 2> shape = {static_cast<int64_t>(seq_len), static_cast<int64_t>(hidden)};
    const std::array<int64_t, 2> strides = {static_cast<int64_t>(hidden), 1};
    const std::array<TensorView, 2> inputs = {
            TensorView(key_act.data(), DataType::Float32(), shape, strides),
            TensorView(value_act.data(), DataType::Float32(), shape, strides)};
    PreparedParams params;
    if (!kernel->params_builder(KernelParamsBuildContext{.inputs = inputs, .outputs = {}, .attrs = kernel->attrs},
                                params.data())) {
        state.SkipWithError("params build failed");
        return;
    }
    auto cache = MakeCache(kv_heads, capacity, head_dim);
    const KVCacheAppendBinding append{
            .storage = MakeStorage(cache, kv_dtype, kv_heads, capacity, head_dim),
            .begin = 0,
            .end = seq_len};
    const KernelContext ctx{.kernel_params = params.data(), .kv_append = &append, .attrs = kernel->attrs};
    for (auto _: state) {
        const Status status = kernel->fn(ctx);
        if (!status.ok()) {
            state.SkipWithError(status.message().c_str());
            break;
        }
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * seq_len * hidden);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * seq_len * hidden * 2);
}

void BenchmarkAttentionKV16(benchmark::State& state, DataType kv_dtype) {
    const auto seq_len = static_cast<size_t>(state.range(0));
    constexpr size_t kv_heads = 8;
    constexpr size_t q_heads = 8;
    constexpr size_t head_dim = 128;
    constexpr size_t capacity = 2048;
    constexpr size_t context = 1024;
    const size_t hidden = q_heads * head_dim;

    CpuBackend backend;
    auto kernel = backend.PrepareKernel(
            OpType::kAttention, MakeSelector(kv_dtype),
            OpParams{AttentionParams{.num_q_heads = static_cast<int64_t>(q_heads),
                                     .num_kv_heads = static_cast<int64_t>(kv_heads),
                                     .head_dim = static_cast<int64_t>(head_dim)}});
    if (!kernel.ok()) {
        state.SkipWithError(kernel.status().message().c_str());
        return;
    }
    std::vector<float> query(seq_len * hidden, 0.1F);
    std::vector<float> output(seq_len * hidden, 0.0F);
    const std::array<int64_t, 2> shape = {static_cast<int64_t>(seq_len), static_cast<int64_t>(hidden)};
    const std::array<int64_t, 2> strides = {static_cast<int64_t>(hidden), 1};
    const std::array<TensorView, 1> inputs = {TensorView(query.data(), DataType::Float32(), shape, strides)};
    const std::array<MutableTensorView, 1> outputs = {
            MutableTensorView(output.data(), DataType::Float32(), shape, strides)};
    PreparedParams params;
    if (!kernel->params_builder(KernelParamsBuildContext{.inputs = inputs, .outputs = outputs, .attrs = kernel->attrs},
                                params.data())) {
        state.SkipWithError("params build failed");
        return;
    }
    auto cache = MakeCache(kv_heads, capacity, head_dim);
    const KVCacheReadBinding read{
            .storage = MakeStorage(cache, kv_dtype, kv_heads, capacity, head_dim),
            .committed_end = context,
            .visible_end = context,
            .query_begin = context - seq_len,
            .query_end = context};
    const KernelContext ctx{.kernel_params = params.data(), .kv_read = &read, .attrs = kernel->attrs};
    for (auto _: state) {
        const Status status = kernel->fn(ctx);
        if (!status.ok()) {
            state.SkipWithError(status.message().c_str());
            break;
        }
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * seq_len * hidden);
}

void RegisterKv16(benchmark::State& state, DataType kv_dtype, bool attention) {
    if (attention) {
        BenchmarkAttentionKV16(state, kv_dtype);
    } else {
        BenchmarkKVUpdateKV16(state, kv_dtype);
    }
}

} // namespace

BENCHMARK_CAPTURE(RegisterKv16, kvupdate_f16, DataType::Float(16), false)->Arg(1)->Arg(128);
BENCHMARK_CAPTURE(RegisterKv16, kvupdate_bf16, DataType::BFloat(16), false)->Arg(1)->Arg(128);
BENCHMARK_CAPTURE(RegisterKv16, attention_f16, DataType::Float(16), true)->Arg(1)->Arg(128);
BENCHMARK_CAPTURE(RegisterKv16, attention_bf16, DataType::BFloat(16), true)->Arg(1)->Arg(128);