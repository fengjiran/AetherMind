#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"
#include "argmax_internal.h"
#include "utils/overflow_check.h"

#include <cstddef>
#include <cstring>
#include <new>
#include <span>
#include <type_traits>
#include <vector>

namespace aethermind::cpu::detail {
namespace {

/// Reads the frozen reduction axis produced by `BuildArgmaxMetadata`.
///
/// The axis is stored unnormalized because prepare-kernel time has no input
/// rank; normalization happens in the params builder.
StatusOr<int64_t> ReadFrozenAxis(std::span<const std::byte> attrs) noexcept {
    if (attrs.size() != sizeof(int64_t)) {
        return Status::InvalidArgument(
                "CPU ArgMax requires a frozen int64 axis in KernelContext.attrs");
    }

    int64_t axis = 0;
    std::memcpy(&axis, attrs.data(), sizeof(axis));
    return axis;
}

/// Normalizes a possibly-negative axis against the input rank.
///
/// @return The axis in `[0, input_rank)`, or InvalidArgument when it is out of
///         range or the negative shift overflows.
StatusOr<int64_t> CanonicalizeAxis(int64_t axis, int32_t input_rank) noexcept {
    int64_t canonical_axis = axis;
    if (canonical_axis < 0) {
        int64_t shifted_axis = 0;
        if (CheckOverflowAdd(canonical_axis, static_cast<int64_t>(input_rank), &shifted_axis)) {
            return Status::InvalidArgument("CPU ArgMax axis is out of range for the input rank");
        }
        canonical_axis = shifted_axis;
    }

    if (canonical_axis < 0 || canonical_axis >= static_cast<int64_t>(input_rank)) {
        return Status::InvalidArgument("CPU ArgMax axis is out of range for the input rank");
    }
    return canonical_axis;
}

StatusOr<ArgmaxF32KernelArgs> ValidateAndBuildArgmaxF32Args(
        const KernelParamsBuildContext& context) noexcept {
    AM_ASSIGN_OR_RETURN(const int64_t axis, ReadFrozenAxis(context.attrs));

    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 1 || outputs.size() != 1) {
        return Status::InvalidArgument("CPU ArgMax requires 1 input and 1 output");
    }

    const TensorView& input = inputs[0];
    const MutableTensorView& output = outputs[0];
    if (!input.is_valid()) {
        return Status::InvalidArgument("CPU ArgMax requires a valid input TensorView");
    }

    if (!output.is_valid()) {
        return Status::InvalidArgument("CPU ArgMax requires a valid output MutableTensorView");
    }

    if (input.dtype() != DataType::Float32()) {
        return Status::InvalidArgument(
                "CPU ArgMax reference requires a float32 input TensorView");
    }

    if (output.dtype() != DataType::Int(64)) {
        return Status::InvalidArgument(
                "CPU ArgMax reference requires an int64 output MutableTensorView");
    }

    // TensorView already bounds the rank by ShapeAndStride::kMaxRank, so the
    // per-axis arrays below are always large enough for input and output.
    const int32_t input_rank = input.rank();
    if (input_rank < 1) {
        return Status::InvalidArgument("CPU ArgMax requires input rank >= 1");
    }

    AM_ASSIGN_OR_RETURN(const int64_t reduction_axis, CanonicalizeAxis(axis, input_rank));

    const int32_t output_rank = input_rank - 1;
    if (output.rank() != output_rank) {
        return Status::InvalidArgument(
                "CPU ArgMax requires output rank to be input rank - 1");
    }

    ArgmaxF32KernelArgs built{};
    built.output_rank = output_rank;

    int32_t output_axis = 0;
    for (int32_t dim = 0; dim < input_rank; ++dim) {
        if (static_cast<int64_t>(dim) == reduction_axis) {
            continue;
        }

        if (output.dim(output_axis) != input.dim(dim)) {
            return Status::InvalidArgument(
                    "CPU ArgMax output shape must equal the input shape without the "
                    "reduction axis");
        }

        built.output_shape[static_cast<size_t>(output_axis)] = input.dim(dim);
        built.input_base_strides[static_cast<size_t>(output_axis)] = input.stride(dim);
        built.output_strides[static_cast<size_t>(output_axis)] = output.stride(output_axis);
        ++output_axis;
    }

    const int64_t reduction_size = input.dim(static_cast<int32_t>(reduction_axis));
    if (reduction_size <= 0) {
        return Status::InvalidArgument(
                "CPU ArgMax requires a non-empty reduction axis");
    }
    built.reduction_size = reduction_size;
    built.reduction_stride = input.stride(static_cast<int32_t>(reduction_axis));

    int64_t output_numel = 1;
    for (int32_t dim = 0; dim < output_rank; ++dim) {
        const int64_t extent = built.output_shape[static_cast<size_t>(dim)];
        if (extent == 0) {
            output_numel = 0;
            break;
        }

        int64_t next_numel = 0;
        if (CheckOverflowMul(output_numel, extent, &next_numel)) {
            return Status::InvalidArgument("CPU ArgMax output element count overflow");
        }
        output_numel = next_numel;
    }
    built.output_numel = output_numel;

    if (output_numel == 0) {
        // Nothing to write: the geometry is carried for diagnostics only and the
        // micro-kernel dereferences neither pointer, so null data, zero strides,
        // and overlapping layouts are all acceptable here.
        return built;
    }

    if (input.data() == nullptr || output.data() == nullptr) {
        return Status::InvalidArgument(
                "CPU ArgMax requires non-null data pointers for non-empty tensors");
    }

    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            input, "CPU ArgMax requires positive input strides"));
    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            output, "CPU ArgMax requires positive output strides"));

    AM_ASSIGN_OR_RETURN(const StridedAddressFootprint input_footprint,
                        BuildStridedAddressFootprint(input.data(), input.shape(), input.strides(),
                                                     input.itemsize(), "CPU ArgMax input"));
    AM_ASSIGN_OR_RETURN(const StridedAddressFootprint output_footprint,
                        BuildStridedAddressFootprint(output.data(), output.shape(),
                                                     output.strides(), output.itemsize(),
                                                     "CPU ArgMax output"));

    AM_RETURN_IF_ERROR(ValidateInjectiveLayout(
            "CPU ArgMax", output_footprint.injectivity, "output"));

    // ArgMax changes both dtype and rank, so no output view can be an exact
    // in-place alias of its input and the footprints are compared as byte ranges.
    AM_RETURN_IF_ERROR(ValidateNoFootprintOverlap(
            "CPU ArgMax", output_footprint, "output", input_footprint, "input"));

    built.input = input.data<float>();
    built.output = output.data<int64_t>();
    return built;
}

Status BuildArgmaxF32ReferenceArgs(const KernelParamsBuildContext& context,
                                   void* params_buffer) noexcept {
    AM_ASSIGN_OR_RETURN(const ArgmaxF32KernelArgs args,
                        ValidateAndBuildArgmaxF32Args(context));
    ::new (params_buffer) ArgmaxF32KernelArgs(args);
    return Status::Ok();
}

Status BuildArgmaxMetadata(const OpParams& params,
                           std::vector<std::byte>& attrs) {
    const auto* argmax_params = std::get_if<ArgmaxParams>(&params);
    if (argmax_params == nullptr) {
        return Status::InvalidArgument("ArgMax kernel requires ArgmaxParams");
    }

    const int64_t axis = argmax_params->axis;
    const auto axis_bytes = std::as_bytes(std::span{&axis, size_t{1}});
    attrs.assign(axis_bytes.begin(), axis_bytes.end());
    return Status::Ok();
}

Status ArgmaxF32ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const ArgmaxF32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunArgmaxF32Reference(*args);
}

} // namespace

static_assert(std::is_trivially_destructible_v<ArgmaxF32KernelArgs>);
static_assert(alignof(ArgmaxF32KernelArgs) <= alignof(std::max_align_t));
static_assert(sizeof(ArgmaxF32KernelArgs) <= kMaxKernelParamsSize);

AM_REGISTER_KERNEL(
        CpuArgmaxF32Reference,
        KernelDescriptor{
                .op_type = OpType::kArgmax,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &ArgmaxF32ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(ArgmaxF32KernelArgs),
                .params_builder = &BuildArgmaxF32ReferenceArgs,
                .metadata_builder = &BuildArgmaxMetadata,
                .name = "cpu::argmax_f32_reference"});

} // namespace aethermind::cpu::detail
