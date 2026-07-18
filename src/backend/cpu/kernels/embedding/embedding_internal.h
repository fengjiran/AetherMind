/// Internal declarations for the CPU Embedding kernel.
///
/// Declares the backend-internal kernel entry point (EmbeddingKernel)
/// and its params struct (EmbeddingParams). Operator code never
/// includes this header; the KernelParamsBuilder indirection keeps
/// operators free of backend internals.

#ifndef AETHERMIND_BACKEND_CPU_KERNELS_EMBEDDING_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_EMBEDDING_INTERNAL_H

#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"

namespace aethermind::cpu::detail {

/// Backend-internal params struct for the CPU Embedding kernel.
///
/// Placement-constructed into a stack-allocated buffer by the
/// KernelParamsBuilder registered with this kernel (BuildEmbeddingParams
/// in embedding_entry.cpp) and consumed by the subsequent EmbeddingKernel
/// call via KernelContext::kernel_params.
struct EmbeddingParams {
    TensorView token_ids{};
    TensorView weight{};
    MutableTensorView output{};
};

/// Kernel entry point registered via KernelDescriptor::kernel_func.
AM_NODISCARD Status EmbeddingKernel(const KernelContext& ctx) noexcept;

}// namespace aethermind::cpu::detail

#endif// AETHERMIND_BACKEND_CPU_KERNELS_EMBEDDING_INTERNAL_H
