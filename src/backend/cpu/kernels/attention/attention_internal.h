#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ATTENTION_ATTENTION_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ATTENTION_ATTENTION_INTERNAL_H

/// @file attention_internal.h
/// @brief Internal contracts for the scalar CPU FP32 Attention kernel.

#include "aethermind/base/kv_cache_binding.h"
#include "aethermind/base/status.h"

#include <cstdint>

namespace aethermind::cpu::detail {

/// @brief Immutable attention geometry frozen from AttentionParams.
///
/// Phase 1 uses the standard scaled dot-product factor `1 / sqrt(head_dim)`.
/// Model-specific scaling remains outside the current Attention semantic
/// parameters and therefore cannot vary per invocation.
struct AttentionF32KernelMetadata {
    int64_t num_q_heads{};
    int64_t num_kv_heads{};
    int64_t head_dim{};
    float scale{};
};

/// @brief Cold-path-bound query/output views for one Attention step.
///
/// KV storage, visibility frontiers, and query positions are invocation-local
/// state supplied through KernelContext::kv_read, never prepared bindings.
struct AttentionF32KernelArgs {
    const float* query{};
    float* output{};
    int64_t seq_len{};
    int64_t num_q_heads{};
    int64_t num_kv_heads{};
    int64_t head_dim{};
    int64_t query_row_stride{};
    int64_t query_col_stride{};
    int64_t output_row_stride{};
    int64_t output_col_stride{};
    float scale{};
};

/// @brief Runs prevalidated causal scaled dot-product FP32 Attention.
Status RunAttentionF32Reference(const AttentionF32KernelArgs& args,
                                const KVCacheReadBinding& read) noexcept;

/// @brief Runs prevalidated causal scaled dot-product Attention over a KV16 cache.
///
/// Q/output stay F32; K/V are widened from the F16/BF16 cache on read. The
/// reduction is fixed F32 (independent product/add, ascending head_dim and
/// visible token order); this TU is compiled with FMA contraction disabled.
/// `kv_dtype` must be scalar Float(16) or BFloat(16) and match the read
/// binding's storage dtype (validated by the caller).
Status RunAttentionKV16Reference(const AttentionF32KernelArgs& args,
                                 const KVCacheReadBinding& read,
                                 DataType kv_dtype) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_ATTENTION_ATTENTION_INTERNAL_H
