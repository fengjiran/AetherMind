#include "add_internal.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/ops/add_op.h"
#include "utils/overflow_check.h"

#include <algorithm>
#include <span>
#include <string>

namespace aethermind::cpu::detail {
namespace {

bool ValidateBroadcastCompatible(std::span<const int64_t> lhs_shape,
                                 std::span<const int64_t> rhs_shape,
                                 std::span<const int64_t> output_shape) noexcept {
    const auto output_rank = static_cast<int32_t>(output_shape.size());
    const auto lhs_offset = output_rank - static_cast<int32_t>(lhs_shape.size());
    const auto rhs_offset = output_rank - static_cast<int32_t>(rhs_shape.size());

    for (int32_t axis = 0; axis < output_rank; ++axis) {
        const int64_t out_dim = output_shape[axis];
        const int64_t lhs_dim = axis < lhs_offset ? 1 : lhs_shape[axis - lhs_offset];
        const int64_t rhs_dim = axis < rhs_offset ? 1 : rhs_shape[axis - rhs_offset];

        if (lhs_dim < 0 || rhs_dim < 0) {
            return false;
        }

        // Broadcast rule: lhs==1 → rhs; rhs==1 or equal → lhs.
        const int64_t expected = lhs_dim == 1                         ? rhs_dim
                                 : rhs_dim == 1 || lhs_dim == rhs_dim ? lhs_dim
                                                                      : -1;
        if (expected < 0 || out_dim != expected) {
            return false;
        }
    }
    return true;
}

Status ValidateMaxOffset(int32_t rank,
                         std::span<const int64_t> shape,
                         std::span<const int64_t> strides,
                         const char* name) noexcept {
    if (rank == 0) {
        return Status::Ok();
    }

    int64_t max_offset = 0;
    for (int32_t i = 0; i < rank; ++i) {
        if (shape[i] == 0) {
            return Status::Ok();
        }

        int64_t contrib = 0;
        if (CheckOverflowMul(shape[i] - 1, strides[i], &contrib)) {
            return Status::InvalidArgument(
                    std::string("AddKernel ") + name + " offset overflow");
        }

        int64_t new_max = 0;
        if (CheckOverflowAdd(max_offset, contrib, &new_max)) {
            return Status::InvalidArgument(
                    std::string("AddKernel ") + name + " offset overflow");
        }
        max_offset = new_max;
    }
    return Status::Ok();
}

StatusOr<int64_t> CheckedOutputNumel(int32_t rank, std::span<const int64_t> shape) noexcept {
    if (rank == 0) {
        return 1;
    }

    int64_t count = 1;
    for (int32_t i = 0; i < rank; ++i) {
        if (shape[i] == 0) {
            return 0;
        }

        int64_t next = 0;
        if (CheckOverflowMul(count, shape[i], &next)) {
            return Status::InvalidArgument("AddKernel output element count overflow");
        }
        count = next;
    }
    return count;
}

Status ValidateAndBuildCommonAddArgs(
        const KernelParamsBuildContext& context,
        AddKernelArgs& args) noexcept {
    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("Add requires 2 inputs and 1 output");
    }

    const TensorView& lhs = inputs[0];
    const TensorView& rhs = inputs[1];
    const MutableTensorView& output = outputs[0];

    if (!lhs.is_valid()) {
        return Status::InvalidArgument("AddKernel requires a valid lhs TensorView");
    }

    if (!rhs.is_valid()) {
        return Status::InvalidArgument("AddKernel requires a valid rhs TensorView");
    }

    if (!output.is_valid()) {
        return Status::InvalidArgument("AddKernel requires a valid output MutableTensorView");
    }

    const DataType dtype = lhs.dtype();
    if (rhs.dtype() != dtype || output.dtype() != dtype) {
        return Status::InvalidArgument(
                "AddKernel requires matching lhs, rhs, and output dtypes");
    }

    if (!IsAddSupportedDType(dtype)) {
        return Status::InvalidArgument(
                MakeAddUnsupportedDTypeMessage("AddKernel"));
    }

    const int32_t output_rank = output.rank();
    const int32_t expected_rank = std::max(lhs.rank(), rhs.rank());
    if (output_rank != expected_rank) {
        return Status::InvalidArgument("AddKernel output rank must equal max(lhs rank, rhs rank)");
    }

    if (output_rank > static_cast<int32_t>(kMaxRank)) {
        return Status::InvalidArgument("AddKernel output rank exceeds maximum supported rank");
    }

    if (!ValidateBroadcastCompatible(lhs.shape(), rhs.shape(), output.shape())) {
        return Status::InvalidArgument(
                "AddKernel input shapes are not broadcast-compatible with output shape");
    }

    const auto numel_or = CheckedOutputNumel(output_rank, output.shape());
    if (!numel_or.ok()) {
        return numel_or.status();
    }

    const int64_t numel = numel_or.value();
    if (numel == 0) {
        args = AddKernelArgs{};
        return Status::Ok();
    }

    if (lhs.data() == nullptr) {
        return Status::InvalidArgument("AddKernel requires non-null lhs data");
    }

    if (rhs.data() == nullptr) {
        return Status::InvalidArgument("AddKernel requires non-null rhs data");
    }

    if (output.data() == nullptr) {
        return Status::InvalidArgument("AddKernel requires non-null output data");
    }

    {
        auto status = ValidateMaxOffset(lhs.rank(), lhs.shape(), lhs.strides(), "lhs");
        if (!status.ok()) {
            return status;
        }

        status = ValidateMaxOffset(rhs.rank(), rhs.shape(), rhs.strides(), "rhs");
        if (!status.ok()) {
            return status;
        }

        status = ValidateMaxOffset(output.rank(), output.shape(), output.strides(), "output");
        if (!status.ok()) {
            return status;
        }
    }

    args.lhs_data = lhs.data();
    args.rhs_data = rhs.data();
    args.output_data = output.data();
    args.dtype = dtype;
    args.numel = numel;

    // Determine flat-path eligibility.
    args.is_flat = lhs.is_contiguous() && rhs.is_contiguous() && output.is_contiguous() &&
                   lhs.shape() == output.shape() && rhs.shape() == output.shape();

    // Populate broadcast / strided path metadata.
    args.lhs_rank = lhs.rank();
    args.rhs_rank = rhs.rank();
    args.output_rank = output_rank;
    for (int32_t i = 0; i < lhs.rank(); ++i) {
        args.lhs_shape[i] = lhs.shape()[i];
        args.lhs_strides[i] = lhs.strides()[i];
    }

    for (int32_t i = 0; i < rhs.rank(); ++i) {
        args.rhs_shape[i] = rhs.shape()[i];
        args.rhs_strides[i] = rhs.strides()[i];
    }

    for (int32_t i = 0; i < output_rank; ++i) {
        args.output_shape[i] = output.shape()[i];
        args.output_strides[i] = output.strides()[i];
    }

    return Status::Ok();
}

Status BuildAddArgs(const KernelParamsBuildContext& context,
                    void* params_buffer) noexcept {
    AddKernelArgs args;
    AM_RETURN_IF_ERROR(ValidateAndBuildCommonAddArgs(context, args));
    ::new (params_buffer) AddKernelArgs(args);
    return Status::Ok();
}

Status AddKernelEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const AddKernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    if (args->numel == 0) {
        return Status::Ok();
    }
    return RunAddReference(*args);
}

} // namespace

static_assert(std::is_trivially_destructible_v<AddKernelArgs>);
static_assert(alignof(AddKernelArgs) <= alignof(std::max_align_t));

// The five registrations below must cover exactly the dtypes in
// kAddSupportedDTypes; see the static_assert in test_cpu_add_kernel.cpp
// ResolvesThroughCpuBackend for the compile-time check.
AM_REGISTER_KERNEL(
        CpuAddFp32Reference,
        KernelDescriptor{
                .op_type = OpType::kAdd,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &AddKernelEntry,
                .priority = 10,
                .params_size = sizeof(AddKernelArgs),
                .params_builder = &BuildAddArgs,
                .name = "cpu::add_f32_reference",
        });

AM_REGISTER_KERNEL(
        CpuAddFp64Reference,
        KernelDescriptor{
                .op_type = OpType::kAdd,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Double(),
                        .weight_dtype = DataType::Double(),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &AddKernelEntry,
                .priority = 10,
                .params_size = sizeof(AddKernelArgs),
                .params_builder = &BuildAddArgs,
                .name = "cpu::add_f64_reference",
        });

AM_REGISTER_KERNEL(CpuAddBf16Reference,
                   KernelDescriptor{
                           .op_type = OpType::kAdd,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::BFloat(16),
                                   .weight_dtype = DataType::BFloat(16),
                                   .weight_format = WeightFormat::kPlain,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &AddKernelEntry,
                           .priority = 10,
                           .params_size = sizeof(AddKernelArgs),
                           .params_builder = &BuildAddArgs,
                           .name = "cpu::add_bf16_reference",
                   });

AM_REGISTER_KERNEL(CpuAddI32Reference,
                   KernelDescriptor{
                           .op_type = OpType::kAdd,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Int(32),
                                   .weight_dtype = DataType::Int(32),
                                   .weight_format = WeightFormat::kPlain,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &AddKernelEntry,
                           .priority = 10,
                           .params_size = sizeof(AddKernelArgs),
                           .params_builder = &BuildAddArgs,
                           .name = "cpu::add_i32_reference",
                   });

AM_REGISTER_KERNEL(CpuAddI64Reference,
                   KernelDescriptor{
                           .op_type = OpType::kAdd,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Int(64),
                                   .weight_dtype = DataType::Int(64),
                                   .weight_format = WeightFormat::kPlain,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &AddKernelEntry,
                           .priority = 10,
                           .params_size = sizeof(AddKernelArgs),
                           .params_builder = &BuildAddArgs,
                           .name = "cpu::add_i64_reference",
                   });

} // namespace aethermind::cpu::detail
