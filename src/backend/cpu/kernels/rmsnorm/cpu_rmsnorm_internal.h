#ifndef AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H

namespace aethermind {

namespace cpu {

void micro_kernel_fp32_scalar(float* out, const float* in, const float* w, int d, float eps);
void micro_kernel_fp32_avx2(float* out, const float* in, const float* w, int d, float eps);

}// namespace cpu

}// namespace aethermind

#endif// AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H
