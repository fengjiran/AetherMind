//
// Created by richard on 6/5/26.
//
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "rmsnorm_internal.h"

#include <cmath>
#include <cstring>

namespace aethermind::cpu {

namespace {
const CpuRmsNormParams* GetParams(const void* kernel_params) noexcept {
    return static_cast<const CpuRmsNormParams*>(kernel_params);
}

bool HasUnitColumnStrides(const RmsNormFp32KernelArgs& args) noexcept {
    return args.input_col_stride == 1 && args.weight_stride == 1 && args.output_col_stride == 1;
}

Status ValidateRmsNormEntry(const KernelContext& ctx, RmsNormFp32KernelArgs& args) noexcept {
    float epsilon;
    if (ctx.attrs.size() != sizeof(float)) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires epsilon in KernelContext.attrs");
    }
    std::memcpy(&epsilon, ctx.attrs.data(), sizeof(float));

    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires finite positive epsilon");
    }

    const CpuRmsNormParams* params = GetParams(ctx.kernel_params);
    if (params == nullptr) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires CpuRmsNormParams in KernelContext.kernel_params");
    }

    const TensorView& input = params->input_tensor;
    const TensorView& weight = params->weight_tensor;
    const MutableTensorView& output = params->output_tensor;

    if (!input.is_valid()) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires a valid input TensorView");
    }

    if (!weight.is_valid()) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires a valid weight TensorView");
    }

    if (!output.is_valid()) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires a valid output MutableTensorView");
    }

    if (input.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires float32 input TensorView");
    }

    if (weight.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires float32 weight TensorView");
    }

    if (output.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires float32 output MutableTensorView");
    }

    if (input.rank() != 2) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires rank-2 input TensorView");
    }

    if (weight.rank() != 1) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires rank-1 weight TensorView");
    }

    if (output.rank() != 2) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires rank-2 output MutableTensorView");
    }

    const int64_t seq_len = input.dim(0);
    const int64_t hidden_size = input.dim(1);
    if (seq_len <= 0) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires positive seq_len");
    }

    if (hidden_size <= 0) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires positive hidden_size");
    }

    if (weight.dim(0) != hidden_size) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires weight length to match hidden_size");
    }

    if (output.dim(0) != seq_len || output.dim(1) != hidden_size) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires output shape to match input shape");
    }

    if (input.data() == nullptr) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires non-null input data pointer");
    }

    if (weight.data() == nullptr) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires non-null weight data pointer");
    }

    if (output.data() == nullptr) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires non-null output data pointer");
    }

    if (input.stride(0) <= 0 || input.stride(1) <= 0) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires positive input strides");
    }

    if (weight.stride(0) <= 0) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires positive weight stride");
    }

    if (output.stride(0) <= 0 || output.stride(1) <= 0) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires positive output strides");
    }

    args = RmsNormFp32KernelArgs{
            .input = input.data<float>(),
            .weight = weight.data<float>(),
            .output = output.data<float>(),
            .seq_len = seq_len,
            .hidden_size = hidden_size,
            .input_row_stride = input.stride(0),
            .input_col_stride = input.stride(1),
            .weight_stride = weight.stride(0),
            .output_row_stride = output.stride(0),
            .output_col_stride = output.stride(1),
            .eps = epsilon,
    };
    return Status::Ok();
}
}// namespace

Status CpuRmsNormKernelEntry_FP32_AVX2(const KernelContext& ctx) noexcept {
    RmsNormFp32KernelArgs args;
    if (const Status status = ValidateRmsNormEntry(ctx, args); !status.ok()) {
        return status;
    }
    if (!HasUnitColumnStrides(args)) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry AVX2 requires unit column strides");
    }
    return RmsNormKernel_CPU_FP32_AVX2(args);
}

Status CpuRmsNormKernelEntry_FP32_Scalar(const KernelContext& ctx) noexcept {
    RmsNormFp32KernelArgs args;
    if (const Status status = ValidateRmsNormEntry(ctx, args); !status.ok()) {
        return status;
    }
    return RmsNormKernel_CPU_FP32_Scalar(args);
}

AM_REGISTER_KERNEL(CpuRmsNormFp32Scalar,
                   KernelDescriptor{
                           .op_type = OpType::kRmsNorm,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float32(),
                                   .weight_format = WeightFormat::kPlain,
                                   .isa = IsaLevel::kScalar,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &CpuRmsNormKernelEntry_FP32_Scalar,
                           .name = "cpu::rmsnorm_f32_scalar",
                           .priority = 10,
                   });

AM_REGISTER_KERNEL(CpuRmsNormFp32Avx2,
                   KernelDescriptor{
                           .op_type = OpType::kRmsNorm,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float32(),
                                   .weight_format = WeightFormat::kPlain,
                                   .isa = IsaLevel::kAVX2,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &CpuRmsNormKernelEntry_FP32_AVX2,
                           .name = "cpu::rmsnorm_f32_avx2",
                           .priority = 20,
                   });

}// namespace aethermind::cpu
