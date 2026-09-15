#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/cpu/kernels/common/broadcast_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "silu_internal.h"

#include <cstddef>
#include <type_traits>

namespace aethermind::cpu::detail {
namespace {

StatusOr<SiluF32KernelArgs> ValidateAndBuildF32Args(
        const KernelParamsBuildContext& context) noexcept {
    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 1 || outputs.size() != 1) {
        return Status::InvalidArgument("SiluKernel requires 1 input and 1 output");
    }

    const TensorView& input = inputs[0];
    const MutableTensorView& output = outputs[0];
    if (!input.is_valid()) {
        return Status::InvalidArgument("SiluKernel requires a valid input TensorView");
    }

    if (!output.is_valid()) {
        return Status::InvalidArgument(
                "SiluKernel requires a valid output MutableTensorView");
    }

    if (input.dtype() != DataType::Float32() || output.dtype() != DataType::Float32()) {
        return Status::InvalidArgument(
                "SiluKernel requires float32 input and output TensorViews");
    }

    if (input.rank() != output.rank()) {
        return Status::InvalidArgument(
                "SiluKernel requires input and output ranks to match");
    }

    if (input.rank() > static_cast<int32_t>(kMaxRank)) {
        return Status::InvalidArgument(
                "SiluKernel rank exceeds maximum supported rank");
    }

    if (input.shape() != output.shape()) {
        return Status::InvalidArgument(
                "SiluKernel requires input and output shapes to match");
    }

    AM_ASSIGN_OR_RETURN(const int64_t numel, CheckedOutputNumel(output.rank(), output.shape()));
    if (numel == 0) {
        return SiluF32KernelArgs{};
    }

    if (input.data() == nullptr || output.data() == nullptr) {
        return Status::InvalidArgument(
                "SiluKernel requires non-null data pointers for non-empty tensors");
    }

    AM_ASSIGN_OR_RETURN(const StridedAddressFootprint input_footprint,
                        BuildStridedAddressFootprint(input, "SiluKernel input"));
    AM_ASSIGN_OR_RETURN(const StridedAddressFootprint output_footprint,
                        BuildStridedAddressFootprint(output, "SiluKernel output"));
    AM_RETURN_IF_ERROR(ValidateLayoutInjectivity(
            "SiluKernel", output_footprint.injectivity(), "output"));

    // The reference loop reads an input element before writing its corresponding
    // output. That permits only an identical logical mapping in-place.
    if (!HaveIdenticalViewMapping(input, output)) {
        AM_RETURN_IF_ERROR(ValidateStridedDisjoint(
                "SiluKernel", output_footprint, "output",
                input_footprint, "input"));
    }

    SiluF32KernelArgs args{};
    args.input_data = static_cast<const float*>(input.data());
    args.output_data = static_cast<float*>(output.data());
    args.numel = numel;
    args.is_flat = input.is_contiguous() && output.is_contiguous();
    args.rank = output.rank();
    for (int32_t axis = 0; axis < args.rank; ++axis) {
        args.shape[axis] = output.shape()[axis];
        args.input_strides[axis] = input.strides()[axis];
        args.output_strides[axis] = output.strides()[axis];
    }
    return args;
}

Status BuildSiluF32ReferenceArgs(const KernelParamsBuildContext& context,
                                 void* params_buffer) noexcept {
    AM_ASSIGN_OR_RETURN(const SiluF32KernelArgs args, ValidateAndBuildF32Args(context));
    ::new (params_buffer) SiluF32KernelArgs(args);
    return Status::Ok();
}

Status SiluF32ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const SiluF32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunSiluF32Reference(*args);
}

} // namespace

static_assert(std::is_trivially_destructible_v<SiluF32KernelArgs>);
static_assert(alignof(SiluF32KernelArgs) <= alignof(std::max_align_t));
static_assert(sizeof(SiluF32KernelArgs) <= kMaxKernelParamsSize);

AM_REGISTER_KERNEL(CpuSiluF32Reference,
                   KernelDescriptor{
                           .op_type = OpType::kSilu,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float32(),
                                   .weight_format = WeightFormat::kPlain,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &SiluF32ReferenceEntry,
                           .priority = 10,
                           .params_size = sizeof(SiluF32KernelArgs),
                           .params_builder = &BuildSiluF32ReferenceArgs,
                           .name = "cpu::silu_f32_reference",
                   })

} // namespace aethermind::cpu::detail
