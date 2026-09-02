/// Internal declarations for the CPU Embedding kernel.
///
/// Declares the backend-internal kernel entry point (EmbeddingKernel) and
/// its compute-ready args struct (EmbeddingKernelArgs). Operator code never
/// includes this header; the KernelParamsBuilder indirection keeps
/// operators free of backend internals.

#ifndef AETHERMIND_BACKEND_CPU_KERNELS_EMBEDDING_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_EMBEDDING_INTERNAL_H

#include "aethermind/backend/kernel_context.h"
#include "aethermind/dtypes/data_type.h"

namespace aethermind::cpu::detail {

/// Compute-ready args for the CPU Embedding kernel.
///
/// Produced by the `KernelParamsBuilder` registered with this kernel
/// (BuildEmbeddingArgs in embedding_entry.cpp) and consumed by the
/// subsequent EmbeddingKernel call via KernelContext::kernel_params.
struct EmbeddingKernelArgs {
    const void* token_ids_data{};
    DataType token_dtype{};
    const float* weight_data{};
    float* output_data{};
    int64_t token_count{};
    int64_t vocab_size{};
    int64_t hidden_size{};
};

/// Kernel entry point registered via KernelDescriptor::kernel_func.
AM_NODISCARD Status EmbeddingKernel(const KernelContext& ctx) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_EMBEDDING_INTERNAL_H
