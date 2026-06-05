#ifndef AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H

namespace aethermind {

namespace cpu {

}// namespace cpu


struct RmsNormFp32KernelArgs {
    const float* input{};
    const float* weight{};
    float* output{};
    int64_t seq_len{};
    int64_t hidden_size{};
    int64_t input_row_stride{};
    int64_t input_col_stride{1};
    int64_t weight_stride{1};
    int64_t output_row_stride{};
    int64_t output_col_stride{1};
    float epsilon{1.0e-5f};
};

Status RmsNormKernel_CPU_FP32_Scalar(const RmsNormFp32KernelArgs& args) noexcept;
Status RmsNormKernel_CPU_FP32_AVX2(const RmsNormFp32KernelArgs& args) noexcept;

}// namespace aethermind

#endif// AETHERMIND_BACKEND_CPU_KERNELS_RMSNORM_CPU_RMSNORM_INTERNAL_H
