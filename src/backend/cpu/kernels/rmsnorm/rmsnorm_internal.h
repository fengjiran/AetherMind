#ifndef AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H

namespace aethermind {

namespace cpu {

}// namespace cpu

Status CpuRmsNormKernelScalar(const RmsNormFp32KernelArgs& args) noexcept;
Status CpuRmsNormKernel(const RmsNormFp32KernelArgs& args) noexcept;

}// namespace aethermind

#endif// AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H
