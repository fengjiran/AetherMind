#include "aethermind/backend/cpu/kernels/cpu_rmsnorm_kernel.h"

#include "aethermind/backend/kernel_invocation.h"
#include "aethermind/backend/op_kernel_context.h"
#include "aethermind/operators/op_type.h"

#include <cmath>

namespace aethermind {
namespace {

const CpuRmsNormAttrs* GetAttrs(std::span<const std::byte> attrs) noexcept {
    if (attrs.size() != sizeof(CpuRmsNormAttrs)) {
        return nullptr;
    }
    return reinterpret_cast<const CpuRmsNormAttrs*>(attrs.data());
}

const CpuRmsNormParams* GetParams(const void* packed_params) noexcept {
    return static_cast<const CpuRmsNormParams*>(packed_params);
}

}// namespace

Status CpuRmsNormKernel(const KernelInvocation& invocation,
                        const OpKernelContext& op_ctx,
                        const WorkspaceBinding&) noexcept {
    if (invocation.op_type != OpType::kRMSNorm) {
        return Status::InvalidArgument("CpuRmsNormKernel only supports OpType::kRMSNorm");
    }
    if (!op_ctx.device.is_cpu()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires CPU device");
    }

    const CpuRmsNormAttrs* attrs = GetAttrs(op_ctx.attrs);
    if (attrs == nullptr) {
        return Status::InvalidArgument("CpuRmsNormKernel requires CpuRmsNormAttrs in OpKernelContext.attrs");
    }

    const CpuRmsNormParams* params = GetParams(op_ctx.packed_params);
    if (params == nullptr) {
        return Status::InvalidArgument("CpuRmsNormKernel requires CpuRmsNormParams in OpKernelContext.packed_params");
    }

    const TensorView& input = params->Input;
    const TensorView& weight = params->Weight;
    const MutableTensorView& output = params->Output;

    if (!input.is_valid()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires a valid input TensorView");
    }
    if (!weight.is_valid()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires a valid weight TensorView");
    }
    if (!output.is_valid()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires a valid output MutableTensorView");
    }

    if (input.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires input TensorView to have float dtype");
    }
    if (weight.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires weight TensorView to have float dtype");
    }
    if (output.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires output MutableTensorView to have float dtype");
    }

    if (input.rank() != 1 || !input.is_contiguous()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires input TensorView to be contiguous 1D");
    }
    if (weight.rank() != 1 || !weight.is_contiguous()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires weight TensorView to be contiguous 1D");
    }
    if (output.rank() != 1 || !output.is_contiguous()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires output MutableTensorView to be contiguous 1D");
    }

    const int64_t hidden_size_i64 = input.numel();
    if (hidden_size_i64 <= 0) {
        return Status::InvalidArgument("CpuRmsNormKernel requires input TensorView to have a non-zero length");
    }
    if (weight.numel() != hidden_size_i64) {
        return Status::InvalidArgument("CpuRmsNormKernel requires weight TensorView length to match input");
    }
    if (output.numel() != hidden_size_i64) {
        return Status::InvalidArgument("CpuRmsNormKernel requires output MutableTensorView length to match input");
    }

    const size_t hidden_size = static_cast<size_t>(hidden_size_i64);
    const float* const input_data = input.data<float>();
    const float* const weight_data = weight.data<float>();
    float* const output_data = output.data<float>();

    float mean_square = 0.0F;
    for (size_t i = 0; i < hidden_size; ++i) {
        mean_square += input_data[i] * input_data[i];
    }
    mean_square /= static_cast<float>(hidden_size);

    const float inv_rms = 1.0F / std::sqrt(mean_square + attrs->Epsilon);
    for (size_t i = 0; i < hidden_size; ++i) {
        output_data[i] = input_data[i] * inv_rms * weight_data[i];
    }

    return Status::Ok();
}

}// namespace aethermind
