/// Internal declarations for the CPU Embedding kernel.
///
/// Defines the pre-validated FP32 compute-ready args struct
/// (EmbeddingF32KernelArgs). The kernel entry itself is TU-local to
/// embedding_entry.cpp; operators never include this header.

#ifndef AETHERMIND_BACKEND_CPU_KERNELS_EMBEDDING_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_EMBEDDING_INTERNAL_H

#include "aethermind/dtypes/data_type.h"

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

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_EMBEDDING_INTERNAL_H
