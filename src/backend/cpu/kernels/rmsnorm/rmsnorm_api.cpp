//
// Created by richard on 6/5/26.
//
#include "aethermind/backend/cpu/kernels/rmsnorm/cpu_rmsnorm_kernel.h"
#include "rmsnorm_internal.h"

namespace aethermind {

Status LaunchRmsNorm(const RmsNormFp32KernelArgs& args) noexcept {
    return CpuRmsNormKernel(args);
}

}// namespace aethermind