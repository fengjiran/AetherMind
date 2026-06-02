#ifndef AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_REFERENCE_H
#define AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_REFERENCE_H

#include "aethermind/backend/cpu/kernels/cpu_rmsnorm_kernel.h"

namespace aethermind {

inline void ReferenceRmsNorm(const CpuRmsNormKernelArgs& args) noexcept {
    for (int64_t s = 0; s < args.seq_len_; ++s) {
        ProcessStridedRmsNormRowScalar(args, s);
    }
}

}// namespace aethermind

#endif
