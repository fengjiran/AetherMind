#include "aethermind/backend/cpu/kernels/common/broadcast_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "elementwise_mul_internal.h"

namespace aethermind::cpu::detail {
namespace {

Status ValidateAndBuildF32Args(const KernelParamsBuildContext& context,
                               ElementwiseMulF32KernelArgs& args) noexcept {
    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument(
                "ElementwiseMulKernel requires 2 inputs and 1 output");
    }

    if (inputs[0].dtype() != DataType::Make<float>() ||
        inputs[1].dtype() != DataType::Make<float>() ||
        outputs[0].dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument(
                "ElementwiseMulKernel requires float32 lhs, rhs, and output TensorViews");
    }

    return ValidateAndBuildElementwiseArgs(context, args, "ElementwiseMulKernel");
}

Status BuildElementwiseMulF32ReferenceArgs(const KernelParamsBuildContext& context,
                                           void* params_buffer) noexcept {
    ElementwiseMulF32KernelArgs args;
    AM_RETURN_IF_ERROR(ValidateAndBuildF32Args(context, args));
    ::new (params_buffer) ElementwiseMulF32KernelArgs(args);
    return Status::Ok();
}

Status ElementwiseMulF32ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const ElementwiseMulF32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunElementwiseMulF32Reference(*args);
}

} // namespace

// The prepared args must satisfy the PreparedExecutionBindings params arena contract.
static_assert(std::is_trivially_destructible_v<ElementwiseMulF32KernelArgs>);
static_assert(alignof(ElementwiseMulF32KernelArgs) <= alignof(std::max_align_t));

AM_REGISTER_KERNEL(CpuElementwiseMulF32Reference,
                   KernelDescriptor{
                           .op_type = OpType::kElementwiseMul,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float32(),
                                   .weight_format = WeightFormat::kPlain,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &ElementwiseMulF32ReferenceEntry,
                           .priority = 10,
                           .params_size = sizeof(ElementwiseMulF32KernelArgs),
                           .params_builder = &BuildElementwiseMulF32ReferenceArgs,
                           .name = "cpu::elementwise_mul_f32_reference",
                   })

} // namespace aethermind::cpu::detail