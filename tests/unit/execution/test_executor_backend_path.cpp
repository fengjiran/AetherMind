#include "aethermind/backend/cpu/cpu_workspace_arena.h"
#include "aethermind/backend/op_kernel_context.h"
#include "aethermind/execution/executor.h"
#include "aethermind/execution/runtime_binding_context.h"

#include <gtest/gtest.h>

#include <vector>

namespace aethermind {
namespace {

std::vector<int>* g_execution_order = nullptr;
WorkspaceBinding g_last_workspace_binding{};
OpKernelContext g_last_kernel_context{};
KernelInvocation g_last_invocation{};

Status FirstKernel(const KernelInvocation& invocation,
                   const OpKernelContext& op_ctx,
                   const WorkspaceBinding& workspace) noexcept {
    g_last_invocation = invocation;
    g_last_kernel_context = op_ctx;
    g_last_workspace_binding = workspace;
    if (g_execution_order != nullptr) {
        g_execution_order->push_back(1);
    }
    return Status::Ok();
}

Status SecondKernel(const KernelInvocation& invocation,
                    const OpKernelContext& op_ctx,
                    const WorkspaceBinding& workspace) noexcept {
    g_last_invocation = invocation;
    g_last_kernel_context = op_ctx;
    g_last_workspace_binding = workspace;
    if (g_execution_order != nullptr) {
        g_execution_order->push_back(2);
    }
    return Status::Ok();
}

Status FailingKernel(const KernelInvocation&,
                     const OpKernelContext&,
                     const WorkspaceBinding&) noexcept {
    return Status::InvalidArgument("kernel failure");
}

TEST(ExecutorBackendPath, ExecuteRunsFrozenKernelsInPlanOrder) {
    ExecutionPlan plan;
    std::vector<int> execution_order;
    alignas(64) std::byte workspace[256]{};
    CpuWorkspaceArena arena(workspace, sizeof(workspace));
    RuntimeBindingContext bindings(&arena);
    g_execution_order = &execution_order;

    ASSERT_TRUE(plan.AddStep(ExecutionStep{
                                     .op_type = OpType::kRMSNorm,
                                     .invocation = {
                                             .op_type = OpType::kRMSNorm,
                                             .selector = {
                                                     .device_type = DeviceType::kCPU,
                                                     .activation_dtype = DataType::Float32(),
                                                     .weight_dtype = DataType::Float32(),
                                                     .weight_format = WeightFormat::kPlain,
                                                     .isa = IsaLevel::kScalar,
                                                     .phase = ExecPhase::kBoth,
                                             },
                                     },
                                     .fn = &FirstKernel,
                                     .workspace_requirement = {
                                             .bytes = 64,
                                             .alignment = 64,
                                             .offset = 0,
                                     },
                                     .debug_name = "test::first_kernel",
                             })
                        .ok());
    ASSERT_TRUE(plan.AddStep(ExecutionStep{
                                     .op_type = OpType::kRoPE,
                                     .invocation = {
                                             .op_type = OpType::kRoPE,
                                             .selector = {
                                                     .device_type = DeviceType::kCPU,
                                                     .activation_dtype = DataType::Float32(),
                                                     .weight_dtype = DataType::Float32(),
                                                     .weight_format = WeightFormat::kPlain,
                                                     .isa = IsaLevel::kScalar,
                                                     .phase = ExecPhase::kBoth,
                                             },
                                     },
                                     .fn = &SecondKernel,
                                     .workspace_requirement = {
                                             .bytes = 128,
                                             .alignment = 64,
                                             .offset = 64,
                                     },
                                     .debug_name = "test::second_kernel",
                             })
                        .ok());

    const Status status = Executor::Execute(plan, bindings);

    g_execution_order = nullptr;
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(execution_order, (std::vector<int>{1, 2}));
    EXPECT_EQ(g_last_invocation.op_type, OpType::kRoPE);
    EXPECT_TRUE(g_last_kernel_context.device.is_cpu());
    EXPECT_EQ(g_last_kernel_context.workspace, &arena);
    EXPECT_EQ(g_last_workspace_binding.size, 128U);
    EXPECT_EQ(g_last_workspace_binding.data,
              static_cast<void*>(workspace + 64));
}

TEST(ExecutorBackendPath, ExecutePropagatesKernelFailure) {
    ExecutionPlan plan;
    alignas(32) std::byte workspace[64]{};
    CpuWorkspaceArena arena(workspace, sizeof(workspace));
    RuntimeBindingContext bindings(&arena);
    ASSERT_TRUE(plan.AddStep(ExecutionStep{
                                     .op_type = OpType::kRMSNorm,
                                     .invocation = {
                                             .op_type = OpType::kRMSNorm,
                                             .selector = {
                                                     .device_type = DeviceType::kCPU,
                                                     .activation_dtype = DataType::Float32(),
                                                     .weight_dtype = DataType::Float32(),
                                                     .weight_format = WeightFormat::kPlain,
                                                     .isa = IsaLevel::kScalar,
                                                     .phase = ExecPhase::kBoth,
                                             },
                                     },
                                     .fn = &FailingKernel,
                                     .workspace_requirement = {
                                             .bytes = 32,
                                             .alignment = 32,
                                             .offset = 0,
                                     },
                                     .debug_name = "test::failing_kernel",
                             })
                        .ok());

    const Status status = Executor::Execute(plan, bindings);

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ExecutorBackendPath, ExecuteFailsWhenWorkspaceRequirementCannotBeBound) {
    ExecutionPlan plan;
    RuntimeBindingContext bindings;

    ASSERT_TRUE(plan.AddStep(ExecutionStep{
                                     .op_type = OpType::kRMSNorm,
                                     .invocation = {
                                             .op_type = OpType::kRMSNorm,
                                             .selector = {
                                                     .device_type = DeviceType::kCPU,
                                                     .activation_dtype = DataType::Float32(),
                                                     .weight_dtype = DataType::Float32(),
                                                     .weight_format = WeightFormat::kPlain,
                                                     .isa = IsaLevel::kScalar,
                                                     .phase = ExecPhase::kBoth,
                                             },
                                     },
                                     .fn = &FirstKernel,
                                     .workspace_requirement = {
                                             .bytes = 32,
                                             .alignment = 32,
                                             .offset = 0,
                                     },
                                     .debug_name = "test::workspace_required_kernel",
                             })
                        .ok());

    const Status status = Executor::Execute(plan, bindings);

    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

}// namespace
}// namespace aethermind
