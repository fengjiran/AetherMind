#ifndef AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_H

#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"

#include <cstdint>

namespace aethermind {

namespace cpu {

struct CpuRmsNormParams {
    TensorView input_tensor{};
    TensorView weight_tensor{};
    MutableTensorView output_tensor{};
};

AM_NODISCARD Status CpuRmsNormKernelEntry_FP32_AVX2(const KernelContext& ctx) noexcept;

}// namespace cpu


/**
 * @brief RMSNorm 算子的对外公有参数结构体
 * @note 纯 POD 类型，保证 ABI 稳定。外部调用者(SDK 用户)必须填充此结构体。
 */
struct RmsNormArgs {
    // ==========================================
    // 1. 数据指针 (Type-Erased Pointers)
    // 必须是 void*，为了兼容 FP32, FP16, BF16 等所有可能的类型
    // ==========================================
    const void* input = nullptr;
    const void* weight = nullptr;
    void* output = nullptr;

    // ==========================================
    // 2. 形状维度 (Shapes)
    // ==========================================
    int64_t seq_len = 0;
    int64_t hidden_size = 0;

    // ==========================================
    // 3. 内存步长 (Strides)
    // 保留了所有的 stride 设置，支持切片(Slice)张量的直接传入
    // 默认列步长为 1，代表最内层物理连续
    // ==========================================
    int64_t input_row_stride = 0;
    int64_t input_col_stride = 1;
    int64_t weight_stride = 1;
    int64_t output_row_stride = 0;
    int64_t output_col_stride = 1;

    // ==========================================
    // 4. 超参数与类型标记 (Hyperparameters & Meta)
    // ==========================================
    float eps = 1.0e-5f;
    DataType dtype;// 指示上面的 void* 到底是什么类型

    // ==========================================
    // 5. 临时工作区 (Workspace - 极其重要的面向未来设计)
    // 即使你目前的 FP32 实现不需要临时内存，对外接口也必须预留
    // ==========================================
    void* workspace = nullptr;
    size_t workspace_size = 0;
};

Status LaunchRmsNorm(const RmsNormArgs& args) noexcept;

}// namespace aethermind

#endif
