#ifndef AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H

namespace aethermind {

namespace cpu {

AM_ALWAYS_INLINE void micro_kernel_fp32_scalar(float* __restrict__ output,
                                               const float* __restrict__ input,
                                               const float* __restrict__ weight,
                                               int hidden_size,
                                               float eps);

void micro_kernel_fp32_avx2(float* __restrict__ output,
                            const float* __restrict__ input,
                            const float* __restrict__ weight,
                            int hidden_size,
                            float eps);

}// namespace cpu

}// namespace aethermind

#endif// AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H
