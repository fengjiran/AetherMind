#include "aethermind/backend/cpu/kernels/cpu_rmsnorm_kernel.h"

#include "aethermind/backend/kernel_invocation.h"
#include "aethermind/backend/op_kernel_context.h"
#include "aethermind/backend/workspace_types.h"
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
    if (params == nullptr || params->Input == nullptr || params->Output == nullptr ||
        params->Weight == nullptr || params->HiddenSize == 0) {
        return Status::InvalidArgument("CpuRmsNormKernel requires valid CpuRmsNormParams");
    }

    float mean_square = 0.0F;
    for (size_t i = 0; i < params->HiddenSize; ++i) {
        mean_square += params->Input[i] * params->Input[i];
    }
    mean_square /= static_cast<float>(params->HiddenSize);

    const float inv_rms = 1.0F / std::sqrt(mean_square + attrs->Epsilon);
    for (size_t i = 0; i < params->HiddenSize; ++i) {
        params->Output[i] = params->Input[i] * inv_rms * params->Weight[i];
    }

    return Status::Ok();
}

}// namespace aethermind
