#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"
#include "matmul_internal.h"
#include "utils/overflow_check.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

namespace aethermind::cpu::detail {
namespace {

Status BuildMatMulF32Metadata(const OpParams& params, std::vector<std::byte>& attrs) {
    const auto* matmul_params = std::get_if<MatMulParams>(&params);
    if (matmul_params == nullptr) {
        return Status::InvalidArgument("MatMul kernel requires MatMulParams");
    }

    attrs.assign(1, static_cast<std::byte>(matmul_params->transpose_rhs ? 1U : 0U));
    return Status::Ok();
}

StatusOr<MatMulF32KernelArgs> ValidateAndBuildMatMulF32Args(
        const KernelParamsBuildContext& context) noexcept {
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
        return built_args;
    }

    AM_ASSIGN_OR_RETURN(const StridedAddressFootprint output_footprint,
                        BuildStridedAddressFootprint(output.data(), output.shape(),
                                                     output.strides(), output.itemsize(),
                                                     "MatMulKernelEntry output"));
    AM_RETURN_IF_ERROR(ValidateInjectiveLayout(
            "CPU MatMul", output_footprint.injectivity, "output"));

    if (built_args.k != 0) {
        AM_ASSIGN_OR_RETURN(const StridedAddressFootprint lhs_footprint,
                            BuildStridedAddressFootprint(lhs.data(), lhs.shape(), lhs.strides(),
                                                         lhs.itemsize(), "MatMulKernelEntry lhs"));
        AM_ASSIGN_OR_RETURN(const StridedAddressFootprint rhs_footprint,
                            BuildStridedAddressFootprint(rhs.data(), rhs.shape(), rhs.strides(),
                                                         rhs.itemsize(), "MatMulKernelEntry rhs"));
        AM_RETURN_IF_ERROR(ValidateNoFootprintOverlap(
                "CPU MatMul", output_footprint, "output", lhs_footprint, "lhs"));
        AM_RETURN_IF_ERROR(ValidateNoFootprintOverlap(
                "CPU MatMul", output_footprint, "output", rhs_footprint, "rhs"));
    }

    return built_args;
}

Status BuildMatMulF32ReferenceArgs(const KernelParamsBuildContext& context,
                                   void* params_buffer) noexcept {
    AM_ASSIGN_OR_RETURN(const MatMulF32KernelArgs args,
                        ValidateAndBuildMatMulF32Args(context));
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
