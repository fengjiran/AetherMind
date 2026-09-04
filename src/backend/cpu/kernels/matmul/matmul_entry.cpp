#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"
#include "matmul_internal.h"
#include "utils/overflow_check.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>

namespace aethermind::cpu::detail {
namespace {

struct AddressRange {
    std::uintptr_t begin{};
    std::uintptr_t end{};
};

template<typename TensorLike>
StatusOr<int64_t> ComputeMaxOffset(const TensorLike& tensor,
                                   const char* message) noexcept {
    int64_t max_offset = 0;
    for (int32_t axis = 0; axis < tensor.rank(); ++axis) {
        const int64_t extent = tensor.dim(axis);
        if (extent == 0) {
            return int64_t{0};
        }

        int64_t axis_offset = 0;
        if (CheckOverflowMul(extent - 1, tensor.stride(axis), &axis_offset) ||
            CheckOverflowAdd(max_offset, axis_offset, &max_offset)) {
            return Status::InvalidArgument(message);
        }
    }
    return max_offset;
}

template<typename TensorLike>
StatusOr<AddressRange> ComputeAddressRange(const TensorLike& tensor,
                                           const char* message) noexcept {
    AM_ASSIGN_OR_RETURN(const int64_t max_offset, ComputeMaxOffset(tensor, message));
    if (max_offset < 0 ||
        static_cast<uint64_t>(max_offset) >
                static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        return Status::InvalidArgument(message);
    }

    const std::uintptr_t max_offset_unsigned = static_cast<std::uintptr_t>(max_offset);
    std::uintptr_t byte_offset = 0;
    if (CheckOverflowMul(max_offset_unsigned,
                         static_cast<std::uintptr_t>(sizeof(float)),
                         &byte_offset)) {
        return Status::InvalidArgument(message);
    }

    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(tensor.data());
    std::uintptr_t last_element = 0;
    std::uintptr_t end = 0;
    if (CheckOverflowAdd(begin, byte_offset, &last_element) ||
        CheckOverflowAdd(last_element, static_cast<std::uintptr_t>(sizeof(float)), &end)) {
        return Status::InvalidArgument(message);
    }
    return AddressRange{.begin = begin, .end = end};
}

bool RangesOverlap(const AddressRange& lhs, const AddressRange& rhs) noexcept {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

template<typename TensorLike>
Status ValidateInjectiveOutputMapping(const TensorLike& output) noexcept {
    struct Axis {
        int64_t extent;
        int64_t stride;
    };

    std::array<Axis, ShapeAndStride::kMaxRank> axes{};
    size_t axis_count = 0;
    for (int32_t axis = 0; axis < output.rank(); ++axis) {
        if (output.dim(axis) > 1) {
            axes[axis_count++] = Axis{
                    .extent = output.dim(axis),
                    .stride = output.stride(axis),
            };
        }
    }

    std::sort(axes.begin(), axes.begin() + static_cast<std::ptrdiff_t>(axis_count),
              [](const Axis& lhs, const Axis& rhs) {
                  return lhs.stride < rhs.stride;
              });

    int64_t covered_span = 1;
    for (size_t axis = 0; axis < axis_count; ++axis) {
        if (axes[axis].stride < covered_span) {
            return Status::InvalidArgument("CPU MatMul output logical elements must not overlap");
        }

        int64_t axis_span = 0;
        if (CheckOverflowMul(axes[axis].extent - 1, axes[axis].stride, &axis_span) ||
            CheckOverflowAdd(covered_span, axis_span, &covered_span)) {
            return Status::InvalidArgument("MatMulKernelEntry output stride span overflow");
        }
    }
    return Status::Ok();
}

Status BuildMatMulF32Metadata(const OpParams& params, std::vector<std::byte>& attrs) {
    const auto* matmul_params = std::get_if<MatMulParams>(&params);
    if (matmul_params == nullptr) {
        return Status::InvalidArgument("MatMul kernel requires MatMulParams");
    }

    attrs.assign(1, static_cast<std::byte>(matmul_params->transpose_rhs ? 1U : 0U));
    return Status::Ok();
}

Status ValidateAndBuildMatMulF32Args(const KernelParamsBuildContext& context,
                                     MatMulF32KernelArgs& args) noexcept {
    if (context.attrs.size() != 1) {
        return Status::InvalidArgument("MatMulKernelEntry requires transpose_rhs metadata");
    }
    const uint8_t transpose_rhs = std::to_integer<uint8_t>(context.attrs[0]);
    if (transpose_rhs > 1) {
        return Status::InvalidArgument("MatMulKernelEntry has invalid transpose_rhs metadata");
    }

    if (context.inputs.size() != 2 || context.outputs.size() != 1) {
        return Status::InvalidArgument("MatMul requires 2 inputs and 1 output");
    }

    const TensorView& lhs = context.inputs[0];
    const TensorView& rhs = context.inputs[1];
    const MutableTensorView& output = context.outputs[0];
    if (!lhs.is_valid() || !rhs.is_valid() || !output.is_valid()) {
        return Status::InvalidArgument("MatMulKernelEntry requires valid TensorViews");
    }

    const DataType fp32 = DataType::Float32();
    if (lhs.dtype() != fp32 || rhs.dtype() != fp32 || output.dtype() != fp32) {
        return Status::InvalidArgument(
                "MatMulKernelEntry requires float32 input and output TensorViews");
    }

    if (lhs.rank() < 2 || rhs.rank() < 2) {
        return Status::InvalidArgument("MatMulKernelEntry requires input rank >= 2");
    }

    const int32_t lhs_rank = lhs.rank();
    const int32_t rhs_rank = rhs.rank();
    const int32_t lhs_batch_rank = lhs_rank - 2;
    const int32_t rhs_batch_rank = rhs_rank - 2;
    const int32_t batch_rank = std::max(lhs_batch_rank, rhs_batch_rank);
    if (output.rank() != batch_rank + 2) {
        return Status::InvalidArgument("MatMulKernelEntry output rank does not match broadcast rank");
    }

    const int64_t m = lhs.dim(lhs_rank - 2);
    const int64_t k = lhs.dim(lhs_rank - 1);
    const int32_t rhs_k_axis = transpose_rhs == 0 ? rhs_rank - 2 : rhs_rank - 1;
    const int32_t rhs_n_axis = transpose_rhs == 0 ? rhs_rank - 1 : rhs_rank - 2;
    if (rhs.dim(rhs_k_axis) != k) {
        return Status::InvalidArgument("MatMul inner dimensions must be equal");
    }
    const int64_t n = rhs.dim(rhs_n_axis);
    if (output.dim(batch_rank) != m || output.dim(batch_rank + 1) != n) {
        return Status::InvalidArgument("MatMulKernelEntry output matrix shape is incorrect");
    }

    MatMulF32KernelArgs built_args{
            .lhs = lhs.data<float>(),
            .rhs = rhs.data<float>(),
            .output = output.data<float>(),
            .batch_count = 1,
            .m = m,
            .n = n,
            .k = k,
            .lhs_m_stride = lhs.stride(lhs_rank - 2),
            .lhs_k_stride = lhs.stride(lhs_rank - 1),
            .rhs_k_stride = rhs.stride(rhs_k_axis),
            .rhs_n_stride = rhs.stride(rhs_n_axis),
            .output_m_stride = output.stride(batch_rank),
            .output_n_stride = output.stride(batch_rank + 1),
            .batch_rank = batch_rank,
    };

    for (int32_t axis = 0; axis < batch_rank; ++axis) {
        const int32_t lhs_axis = axis - (batch_rank - lhs_batch_rank);
        const int32_t rhs_axis = axis - (batch_rank - rhs_batch_rank);
        const int64_t lhs_dim = lhs_axis < 0 ? 1 : lhs.dim(lhs_axis);
        const int64_t rhs_dim = rhs_axis < 0 ? 1 : rhs.dim(rhs_axis);
        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            return Status::InvalidArgument("MatMul batch dimensions must be broadcastable");
        }

        const int64_t batch_dim = lhs_dim == 1 ? rhs_dim : lhs_dim;
        if (output.dim(axis) != batch_dim) {
            return Status::InvalidArgument("MatMulKernelEntry output batch shape is incorrect");
        }

        built_args.batch_dims[axis] = batch_dim;
        built_args.lhs_batch_strides[axis] =
                lhs_axis < 0 || lhs_dim == 1 ? 0 : lhs.stride(lhs_axis);
        built_args.rhs_batch_strides[axis] =
                rhs_axis < 0 || rhs_dim == 1 ? 0 : rhs.stride(rhs_axis);
        built_args.output_batch_strides[axis] = output.stride(axis);

        int64_t next_batch_count = 0;
        if (CheckOverflowMul(built_args.batch_count, batch_dim, &next_batch_count)) {
            return Status::InvalidArgument("MatMulKernelEntry batch count overflow");
        }
        built_args.batch_count = next_batch_count;
    }

    const bool has_output_elements =
            built_args.batch_count != 0 && built_args.m != 0 && built_args.n != 0;
    if (!has_output_elements) {
        args = built_args;
        return Status::Ok();
    }

    AM_RETURN_IF_ERROR(ValidateInjectiveOutputMapping(output));
    AM_ASSIGN_OR_RETURN(const AddressRange output_range,
                        ComputeAddressRange(output, "MatMulKernelEntry output offset overflow"));

    if (built_args.k != 0) {
        AM_ASSIGN_OR_RETURN(const AddressRange lhs_range,
                            ComputeAddressRange(lhs, "MatMulKernelEntry lhs offset overflow"));
        AM_ASSIGN_OR_RETURN(const AddressRange rhs_range,
                            ComputeAddressRange(rhs, "MatMulKernelEntry rhs offset overflow"));
        if (RangesOverlap(output_range, lhs_range) || RangesOverlap(output_range, rhs_range)) {
            return Status::InvalidArgument("CPU MatMul output must not alias input");
        }
    }

    args = built_args;
    return Status::Ok();
}

Status BuildMatMulF32ReferenceArgs(const KernelParamsBuildContext& context,
                                   void* params_buffer) noexcept {
    MatMulF32KernelArgs args;
    AM_RETURN_IF_ERROR(ValidateAndBuildMatMulF32Args(context, args));
    ::new (params_buffer) MatMulF32KernelArgs(args);
    return Status::Ok();
}

Status MatMulF32ReferenceEntry(const KernelContext& context) noexcept {
    const auto* args = static_cast<const MatMulF32KernelArgs*>(context.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunMatMulF32Reference(*args);
}

} // namespace

static_assert(std::is_trivially_destructible_v<MatMulF32KernelArgs>);
static_assert(sizeof(MatMulF32KernelArgs) <= kMaxKernelParamsSize);
static_assert(alignof(MatMulF32KernelArgs) <= alignof(std::max_align_t));

AM_REGISTER_KERNEL(
        CpuMatMulF32Reference,
        KernelDescriptor{
                .op_type = OpType::kMatMul,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &MatMulF32ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(MatMulF32KernelArgs),
                .params_builder = &BuildMatMulF32ReferenceArgs,
                .metadata_builder = &BuildMatMulF32Metadata,
                .name = "cpu::matmul_f32_reference"});

} // namespace aethermind::cpu::detail
