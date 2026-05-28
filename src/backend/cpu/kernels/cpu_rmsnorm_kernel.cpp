#include "aethermind/backend/cpu/kernels/cpu_rmsnorm_kernel.h"

#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_invocation.h"
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
                        const KernelContext& op_ctx,
                        const WorkspaceBinding&) noexcept {
    if (invocation.op_type != OpType::kRmsNorm) {
        return Status::InvalidArgument("CpuRmsNormKernel only supports OpType::kRmsNorm");
    }
    if (!op_ctx.device.is_cpu()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires CPU device");
    }

    const CpuRmsNormAttrs* attrs = GetAttrs(op_ctx.attrs);
    if (attrs == nullptr) {
        return Status::InvalidArgument("CpuRmsNormKernel requires CpuRmsNormAttrs in KernelContext.attrs");
    }

    const CpuRmsNormParams* params = GetParams(op_ctx.packed_params);
    if (params == nullptr) {
        return Status::InvalidArgument("CpuRmsNormKernel requires CpuRmsNormParams in KernelContext.packed_params");
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

    if (input.rank() != 2 || !input.is_contiguous()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires input TensorView to be contiguous 2D [seq_len, hidden]");
    }
    if (weight.rank() != 1 || !weight.is_contiguous()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires weight TensorView to be contiguous 1D");
    }
    if (output.rank() != 2 || !output.is_contiguous()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires output MutableTensorView to be contiguous 2D [seq_len, hidden]");
    }

    const int64_t seq_len = input.dim(0);
    const int64_t hidden_size_i64 = input.dim(1);
    if (seq_len <= 0 || hidden_size_i64 <= 0) {
        return Status::InvalidArgument("CpuRmsNormKernel requires positive seq_len and hidden_size");
    }
    if (output.dim(0) != seq_len || output.dim(1) != hidden_size_i64) {
        return Status::InvalidArgument("CpuRmsNormKernel output shape must match input [seq_len, hidden]");
    }
    if (weight.numel() != hidden_size_i64) {
        return Status::InvalidArgument("CpuRmsNormKernel requires weight length to match hidden_size");
    }

    const auto hidden_size = static_cast<size_t>(hidden_size_i64);
    const float* const input_data = input.data<float>();
    const float* const weight_data = weight.data<float>();
    float* const output_data = output.data<float>();

    for (int64_t s = 0; s < seq_len; ++s) {
        const float* const row_in = input_data + s * hidden_size;
        float* const row_out = output_data + s * hidden_size;

        double mean_square = 0.0F;
        for (size_t i = 0; i < hidden_size; ++i) {
            mean_square += row_in[i] * row_in[i];
        }
        mean_square /= static_cast<float>(hidden_size);

        const float inv_rms = 1.0F / std::sqrt(mean_square + attrs->Epsilon);
        for (size_t i = 0; i < hidden_size; ++i) {
            row_out[i] = row_in[i] * inv_rms * weight_data[i];
        }
    }

    return Status::Ok();
}

}// namespace aethermind
