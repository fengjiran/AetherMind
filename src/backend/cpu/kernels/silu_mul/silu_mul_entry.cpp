#include "aethermind/backend/cpu/kernels/common/broadcast_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "silu_mul_internal.h"

#include <cstddef>
#include <type_traits>

namespace aethermind::cpu::detail {
namespace {

StatusOr<SiluMulF32KernelArgs> ValidateAndBuildF32Args(
        const KernelParamsBuildContext& context) noexcept {
    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("SiluMulKernel requires 2 inputs and 1 output");
    }

    if (inputs[0].dtype() != DataType::Float32() ||
        inputs[1].dtype() != DataType::Float32() ||
        outputs[0].dtype() != DataType::Float32()) {
        return Status::InvalidArgument(
                "SiluMulKernel requires float32 gate, up, and output TensorViews");
    }

    AM_ASSIGN_OR_RETURN(const ElementwiseBroadcastArgs prepared,
                        ValidateAndBuildBroadcastArgs(context, "SiluMulKernel",
                                                      "gate", "up"));
    SiluMulF32KernelArgs args{};
    args.gate_data = static_cast<const float*>(prepared.lhs_data);
    args.up_data = static_cast<const float*>(prepared.rhs_data);
    args.output_data = static_cast<float*>(prepared.output_data);
    args.numel = prepared.numel;
    args.is_flat = prepared.is_flat;
    args.gate_rank = prepared.lhs_rank;
    args.up_rank = prepared.rhs_rank;
    args.output_rank = prepared.output_rank;
    for (int32_t i = 0; i < prepared.lhs_rank; ++i) {
        args.gate_shape[i] = prepared.lhs_shape[i];
        args.gate_strides[i] = prepared.lhs_strides[i];
    }

    for (int32_t i = 0; i < prepared.rhs_rank; ++i) {
        args.up_shape[i] = prepared.rhs_shape[i];
        args.up_strides[i] = prepared.rhs_strides[i];
    }

    for (int32_t i = 0; i < prepared.output_rank; ++i) {
        args.output_shape[i] = prepared.output_shape[i];
        args.output_strides[i] = prepared.output_strides[i];
    }

    return args;
}

Status BuildSiluMulF32ReferenceArgs(const KernelParamsBuildContext& context,
                                    void* params_buffer) noexcept {
    AM_ASSIGN_OR_RETURN(const SiluMulF32KernelArgs args, ValidateAndBuildF32Args(context));
    ::new (params_buffer) SiluMulF32KernelArgs(args);
    return Status::Ok();
}

Status SiluMulF32ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const SiluMulF32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunSiluMulF32Reference(*args);
}

} // namespace

static_assert(std::is_trivially_destructible_v<SiluMulF32KernelArgs>);
static_assert(alignof(SiluMulF32KernelArgs) <= alignof(std::max_align_t));
static_assert(sizeof(SiluMulF32KernelArgs) <= kMaxKernelParamsSize);

AM_REGISTER_KERNEL(CpuSiluMulF32Reference,
                   KernelDescriptor{
                           .op_type = OpType::kSiluMul,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float32(),
                                   .weight_format = WeightFormat::kPlain,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &SiluMulF32ReferenceEntry,
                           .priority = 10,
                           .params_size = sizeof(SiluMulF32KernelArgs),
                           .params_builder = &BuildSiluMulF32ReferenceArgs,
                           .name = "cpu::silu_mul_f32_reference",
                   })

} // namespace aethermind::cpu::detail
