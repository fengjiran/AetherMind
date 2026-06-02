#ifndef AETHERMIND_BACKEND_CPU_KERNELS_CPU_EMBEDDING_KERNEL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_CPU_EMBEDDING_KERNEL_H

#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"

namespace aethermind {

struct CpuEmbeddingParams {
    TensorView token_ids_{};
    TensorView weight_{};
    MutableTensorView output_{};
};

AM_NODISCARD Status CpuEmbeddingKernel(const KernelContext& ctx) noexcept;

}// namespace aethermind

#endif
