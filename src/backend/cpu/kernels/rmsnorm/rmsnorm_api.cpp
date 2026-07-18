//
// Created by richard on 6/5/26.
//
#include "aethermind/backend/cpu/kernels/rmsnorm/cpu_rmsnorm_kernel.h"
#include "rmsnorm_internal.h"

namespace aethermind {

Status LaunchRmsNorm(const RmsNormArgs& args) noexcept {
    if (!args.input || !args.weight || !args.output) {
        return Status::InvalidArgument("Pointers cannot be null");
    }

    if (args.dtype.IsFloat32()) {
        cpu::detail::RmsNormFp32KernelArgs kernel_args;
        kernel_args.input = static_cast<const float*>(args.input);
        kernel_args.weight = static_cast<const float*>(args.weight);
        kernel_args.output = static_cast<float*>(args.output);
        kernel_args.seq_len = args.seq_len;
        kernel_args.hidden_size = args.hidden_size;
        kernel_args.input_row_stride = args.input_row_stride;
        kernel_args.input_col_stride = args.input_col_stride;
        kernel_args.weight_stride = args.weight_stride;
        kernel_args.output_row_stride = args.output_row_stride;
        kernel_args.output_col_stride = args.output_col_stride;
        kernel_args.eps = args.eps;
        // Prefer AVX2 when the SDK contract permits it (unit column strides,
        // matching the entry-level selector constraint in
        // RmsNormKernelEntry_FP32_AVX2). Fall back to the scalar path for
        // arbitrary column strides — the scalar micro-kernel handles any
        // stride correctly.
        if (args.input_col_stride == 1 && args.weight_stride == 1 && args.output_col_stride == 1) {
            return cpu::detail::RmsNormKernel_CPU_FP32_AVX2(kernel_args);
        }
        return cpu::detail::RmsNormKernel_CPU_FP32_Scalar(kernel_args);
    }

    return Status::Unimplemented("Not implemented");
}

}// namespace aethermind
