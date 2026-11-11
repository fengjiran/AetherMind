/// Internal declarations for the CPU Embedding kernels.
///
/// Defines the pre-validated compute-ready args structs and reference entry
/// points. Params builders and kernel wrappers live in embedding_entry.cpp;
/// compute lives in embedding_f32_reference.cpp and embedding_w16_reference.cpp.
/// Operators never include this header.

#ifndef AETHERMIND_BACKEND_CPU_KERNELS_EMBEDDING_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_EMBEDDING_INTERNAL_H

#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"

#include <cstddef>

namespace aethermind::cpu::detail {

/// Compute-ready args for the CPU Embedding kernel.
///
/// Produced by the `KernelParamsBuilder` registered with this kernel
/// (BuildEmbeddingF32ReferenceArgs in embedding_entry.cpp) and consumed by
/// the registered kernel entry via KernelContext::kernel_params.
struct EmbeddingF32KernelArgs {
    const void* token_ids_data{};
    DataType token_dtype{};
    const float* weight_data{};
    float* output_data{};
    int64_t token_count{};
    int64_t vocab_size{};
    int64_t hidden_size{};
};

/// @brief Gathers FP32 weight rows selected by token ids.
///
/// `args` must have been produced by the binding-time params builder. A zero
/// `token_count` is a no-op; every token must lie in [0, vocab_size).
Status RunEmbeddingF32Reference(const EmbeddingF32KernelArgs& args) noexcept;

/// Widens one embedding row from its stored 16-bit format into FP32 output.
using EmbeddingRowWiden = void (*)(const std::byte*, std::byte*, size_t) noexcept;

/// Compute-ready args for the CPU FP32 x W16 Embedding kernel.
///
/// Produced by BuildEmbeddingF32W16Args/BuildEmbeddingF32W16PackedIdentityArgs
/// in embedding_entry.cpp and consumed by RunEmbeddingF32W16Reference in
/// embedding_w16_reference.cpp via KernelContext::kernel_params.
/// `weight_row_bytes` and `output_row_bytes` carry the source/output strides
/// because the formats differ in width.
struct EmbeddingF32W16KernelArgs {
    const std::byte* token_ids{};
    DataType token_dtype{};
    const std::byte* weight{};
    std::byte* output{};
    size_t token_count{};
    size_t vocab_size{};
    size_t hidden_size{};
    size_t weight_row_bytes{};
    size_t output_row_bytes{};
    EmbeddingRowWiden widen{};
};

/// @brief Gathers FP16/BF16 rows selected by token ids, widening into FP32.
///
/// `args` must have been produced by the binding-time params builder. A zero
/// `token_count` is a no-op; every token must lie in [0, vocab_size).
Status RunEmbeddingF32W16Reference(const EmbeddingF32W16KernelArgs& args) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_EMBEDDING_INTERNAL_H
