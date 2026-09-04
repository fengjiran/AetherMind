#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"
#include "argmax_internal.h"
#include "utils/overflow_check.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <new>
#include <span>
#include <string>
#include <string_view>
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

/// Verifies that the maximum reachable element offset of a view is representable.
///
/// Sums `(extent - 1) * stride` over every axis with overflow checks, catching
/// views whose declared geometry would index past addressable storage.
template<typename TensorLike>
Status ValidateMaxOffsetRepresentable(const TensorLike& tensor,
                                      std::string_view role) noexcept {
    int64_t max_offset = 0;
    for (int32_t dim = 0; dim < tensor.rank(); ++dim) {
        int64_t axis_offset = 0;
        if (CheckOverflowMul(tensor.dim(dim) - 1, tensor.stride(dim), &axis_offset)) {
            return Status::InvalidArgument("CPU ArgMax " + std::string(role) +
                                           " offset overflow");
        }

        int64_t next_offset = 0;
        if (CheckOverflowAdd(max_offset, axis_offset, &next_offset)) {
            return Status::InvalidArgument("CPU ArgMax " + std::string(role) +
                                           " offset overflow");
        }
        max_offset = next_offset;
    }
    return Status::Ok();
}

/// Verifies that the output layout belongs to the supported non-overlapping subset.
///
/// Positive strides alone do not exclude overlap: shape `[2, 2]` with strides
/// `[1, 1]` maps four coordinates onto two slots. Processing the axes with
/// extent > 1 in ascending stride order and requiring each stride to reach past
/// the span covered by the lower axes is a cheap sufficient proof that the
/// coordinate-to-offset map is injective. Transposed, padded, and dense-permuted
/// layouts pass. Failure is reported as Unimplemented rather than InvalidArgument:
/// the test is intentionally conservative and cannot distinguish actual overlap
/// from an injective irregular layout outside the supported subset.
Status ValidateSupportedOutputLayout(const ArgmaxF32KernelArgs& args) noexcept {
    std::array<int32_t, ShapeAndStride::kMaxRank> axes{};
    int32_t axis_count = 0;
    for (int32_t dim = 0; dim < args.output_rank; ++dim) {
        if (args.output_shape[dim] > 1) {
            axes[static_cast<size_t>(axis_count++)] = dim;
        }
    }

    const auto ordered = std::span{axes}.first(static_cast<size_t>(axis_count));
    std::sort(ordered.begin(), ordered.end(), [&](int32_t lhs, int32_t rhs) {
        return args.output_strides[lhs] < args.output_strides[rhs];
    });

    int64_t covered_span = 1;
    for (const int32_t dim: ordered) {
        const int64_t stride = args.output_strides[dim];
        if (stride < covered_span) {
            return Status::Unimplemented(
                    "CPU ArgMax requires a provably non-overlapping output layout");
        }

        int64_t axis_span = 0;
        if (CheckOverflowMul(args.output_shape[dim] - 1, stride, &axis_span)) {
            return Status::InvalidArgument("CPU ArgMax output span overflow");
        }

        int64_t next_span = 0;
        if (CheckOverflowAdd(covered_span, axis_span, &next_span)) {
            return Status::InvalidArgument("CPU ArgMax output span overflow");
        }
        covered_span = next_span;
    }
    return Status::Ok();
}

Status ValidateAndBuildArgmaxF32Args(const KernelParamsBuildContext& context,
                                     ArgmaxF32KernelArgs& args) noexcept {
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
        args = built;
        return Status::Ok();
    }

    if (input.data() == nullptr || output.data() == nullptr) {
        return Status::InvalidArgument(
                "CPU ArgMax requires non-null data pointers for non-empty tensors");
    }

    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            input, "CPU ArgMax requires positive input strides"));
    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            output, "CPU ArgMax requires positive output strides"));

    AM_RETURN_IF_ERROR(ValidateMaxOffsetRepresentable(input, "input"));
    AM_RETURN_IF_ERROR(ValidateMaxOffsetRepresentable(output, "output"));
    AM_RETURN_IF_ERROR(ValidateSupportedOutputLayout(built));

    // Only an identical base pointer is detectable here: TensorView carries no
    // storage bounds, so partial overlap between two distinct base pointers
    // cannot be proven either way. ArgMax changes dtype and rank, so any shared
    // base pointer is rejected outright instead of allowing in-place execution.
    if (input.data() == output.data()) {
        return Status::InvalidArgument("CPU ArgMax output must not alias the input");
    }

    built.input = input.data<float>();
    built.output = output.data<int64_t>();
    args = built;
    return Status::Ok();
}

Status BuildArgmaxF32ReferenceArgs(const KernelParamsBuildContext& context,
                                   void* params_buffer) noexcept {
    ArgmaxF32KernelArgs args{};
    AM_RETURN_IF_ERROR(ValidateAndBuildArgmaxF32Args(context, args));
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
