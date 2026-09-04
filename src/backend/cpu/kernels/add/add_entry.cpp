#include "add_internal.h"
#include "aethermind/backend/cpu/kernels/common/broadcast_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/ops/add_op.h"

#include <string>

namespace aethermind::cpu::detail {
namespace {

Status ValidateAndBuildArgs(const KernelParamsBuildContext& context,
                            AddKernelArgs& args) noexcept {
    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("Add requires 2 inputs and 1 output");
    }

    const DataType dtype = inputs[0].dtype();
    if (inputs[1].dtype() != dtype || outputs[0].dtype() != dtype) {
        return Status::InvalidArgument(
                "AddKernel requires matching lhs, rhs, and output dtypes");
    }

    if (!IsAddSupportedDType(dtype)) {
        return Status::InvalidArgument(
                MakeAddUnsupportedDTypeMessage("AddKernel"));
    }

    AM_RETURN_IF_ERROR(ValidateAndBuildElementwiseArgs(context, args, "AddKernel"));
    args.dtype = dtype;
    return Status::Ok();
}

Status BuildAddArgs(const KernelParamsBuildContext& context,
                    void* params_buffer) noexcept {
    AddKernelArgs args;
    AM_RETURN_IF_ERROR(ValidateAndBuildArgs(context, args));
    ::new (params_buffer) AddKernelArgs(args);
    return Status::Ok();
}

Status AddEntry(const KernelContext& ctx) noexcept {
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
        CpuAddF32Reference,
        KernelDescriptor{
                .op_type = OpType::kAdd,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &AddEntry,
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
                .kernel_func = &AddEntry,
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
                           .kernel_func = &AddEntry,
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
                           .kernel_func = &AddEntry,
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
                           .kernel_func = &AddEntry,
                           .priority = 10,
                           .params_size = sizeof(AddKernelArgs),
                           .params_builder = &BuildAddArgs,
                           .name = "cpu::add_i64_reference",
                   });

} // namespace aethermind::cpu::detail
