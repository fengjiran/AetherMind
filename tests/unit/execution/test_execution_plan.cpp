#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/kernel_invoker.h"

#include <gtest/gtest.h>

#include <span>
#include <vector>

namespace {

using namespace aethermind;

Status FakeKernel(const KernelContext&) noexcept {
    return Status::Ok();
}

int g_kernel_invocations = 0;
int g_built_param = 0;

Status CaptureBuiltParamKernel(const KernelContext& context) noexcept {
    ++g_kernel_invocations;
    if (context.kernel_params == nullptr) {
        return Status::FailedPrecondition("Expected params from the builder");
    }
    g_built_param = *static_cast<const int*>(context.kernel_params);
    return Status::Ok();
}

Status BuildKernelParam(std::span<const TensorView>,
                        std::span<const MutableTensorView>,
                        void* params_buffer) noexcept {
    *static_cast<int*>(params_buffer) = 17;
    return Status::Ok();
}

Status FailToBuildKernelParam(std::span<const TensorView>,
                              std::span<const MutableTensorView>,
                              void*) noexcept {
    return Status::InvalidArgument("test params builder failure");
}

TEST(ExecutionPlan, StoresResolvedKernelByValue) {
    ResolvedKernel kernel{
            .op_type = OpType::kSoftmax,
            .fn = &FakeKernel,
            .attrs = {std::byte{7}},
            .debug_name = "test::fake_kernel",
            .workspace_requirement = {.bytes = 128, .alignment = 64},
    };

    const StatusOr<ExecutionPlan> plan = ExecutionPlan::Create({
            ExecutionStep{
                    .selector = KernelSelector{.device_type = DeviceType::kCPU},
                    .kernel = kernel,
                    .workspace_requirement = {.bytes = 128, .alignment = 64},
            },
    });

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);
    const ExecutionStep& step = plan->steps().front();
    EXPECT_EQ(step.kernel.op_type, OpType::kSoftmax);
    EXPECT_EQ(step.kernel.fn, &FakeKernel);
    EXPECT_EQ(step.kernel.attrs, kernel.attrs);
    EXPECT_NE(step.kernel.attrs.data(), kernel.attrs.data());
    EXPECT_STREQ(step.kernel.debug_name, "test::fake_kernel");
    EXPECT_EQ(step.workspace_requirement.bytes, 128U);
    EXPECT_EQ(step.kernel.workspace_requirement.bytes, step.workspace_requirement.bytes);
    EXPECT_EQ(step.kernel.workspace_requirement.alignment,
              step.workspace_requirement.alignment);
    EXPECT_EQ(step.kernel.workspace_requirement.offset, step.workspace_requirement.offset);
}

TEST(ExecutionPlan, RejectsInvalidResolvedKernel) {
    EXPECT_EQ(ExecutionPlan::Create({ExecutionStep{}}).status().code(),
              StatusCode::kInvalidArgument);

    EXPECT_EQ(ExecutionPlan::Create({ExecutionStep{
                                            .kernel = ResolvedKernel{
                                                    .op_type = OpType::kSoftmax,
                                                    .fn = &FakeKernel,
                                                    .params_size = 1,
                                            },
                                    }})
                      .status()
                      .code(),
              StatusCode::kInvalidArgument);
}

TEST(KernelInvoker, BuildsParamsOnlyForTheKernelCall) {
    g_kernel_invocations = 0;
    g_built_param = 0;
    KernelContext context{};
    const Status status = InvokeKernel(
            ResolvedKernel{
                    .op_type = OpType::kAdd,
                    .fn = &CaptureBuiltParamKernel,
                    .params_builder = &BuildKernelParam,
                    .params_size = sizeof(int),
            },
            context, {}, {});

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(g_kernel_invocations, 1);
    EXPECT_EQ(g_built_param, 17);
}

TEST(KernelInvoker, PropagatesBuilderFailureWithoutCallingKernel) {
    g_kernel_invocations = 0;
    KernelContext context{};
    const Status status = InvokeKernel(
            ResolvedKernel{
                    .op_type = OpType::kAdd,
                    .fn = &CaptureBuiltParamKernel,
                    .params_builder = &FailToBuildKernelParam,
                    .params_size = sizeof(int),
            },
            context, {}, {});

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(g_kernel_invocations, 0);
}

TEST(KernelInvoker, RejectsNullFunctionAndInvalidParamsContract) {
    KernelContext context{};
    EXPECT_EQ(InvokeKernel(ResolvedKernel{.op_type = OpType::kAdd}, context, {}, {})
                      .code(),
              StatusCode::kFailedPrecondition);
    EXPECT_EQ(InvokeKernel(ResolvedKernel{
                                   .op_type = OpType::kAdd,
                                   .fn = &FakeKernel,
                                   .params_size = 1,
                           },
                           context, {}, {})
                      .code(),
              StatusCode::kFailedPrecondition);
}

TEST(ExecutionPlan, AcceptsRuntimeCheckWithinCompactSpecRanges) {
    ResolvedKernel kernel{
            .op_type = OpType::kSoftmax,
            .fn = &FakeKernel,
    };
    const std::vector<int64_t> act_dims{4, 8};
    const std::vector<int64_t> weight_dims{8};
    const ShapeConstraint valid_check{
            .condition = DimEqualConstraint{
                    .lhs = {.tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = 0},
                            .dim_index = 1},
                    .rhs = {.tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = 1},
                            .dim_index = 0},
            },
            .error_context = "hidden size mismatch",
    };

    const StatusOr<ExecutionPlan> plan = ExecutionPlan::Create({
            ExecutionStep{
                    .kernel = kernel,
                    .input_specs = {TensorSpec{.dtype = DataType::Float32(),
                                               .shape = SymbolicShape(IntArrayView{act_dims})},
                                    TensorSpec{.dtype = DataType::Float32(),
                                               .shape = SymbolicShape(IntArrayView{weight_dims})}},
                    .output_specs = {TensorSpec{}},
                    .runtime_checks = {valid_check},
            },
    });

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    EXPECT_EQ(plan->size(), 1U);
}

TEST(ExecutionPlan, RejectsRuntimeCheckReferencingMissingTensorPort) {
    ResolvedKernel kernel{
            .op_type = OpType::kSoftmax,
            .fn = &FakeKernel,
    };
    const ShapeConstraint bad_check{
            .condition = RankEqualConstraint{
                    .port = {.direction = TensorPortType::kInput, .tensor_idx = 1},
                    .target_rank = 2,
            },
            .error_context = "unused",
    };

    const StatusOr<ExecutionPlan> plan = ExecutionPlan::Create({
            ExecutionStep{
                    .kernel = kernel,
                    .input_specs = {TensorSpec{}},
                    .output_specs = {TensorSpec{}},
                    .runtime_checks = {bad_check},
            },
    });

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlan, RejectsRuntimeCheckReferencingDimensionBeyondRank) {
    ResolvedKernel kernel{
            .op_type = OpType::kSoftmax,
            .fn = &FakeKernel,
    };
    const std::vector<int64_t> input_dims{8};
    const ShapeConstraint bad_check{
            .condition = DimPositiveConstraint{
                    .dim = {.tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = 0},
                            .dim_index = 1},
            },
            .error_context = "unused",
    };

    const StatusOr<ExecutionPlan> plan = ExecutionPlan::Create({
            ExecutionStep{
                    .kernel = kernel,
                    .input_specs = {TensorSpec{.dtype = DataType::Float32(),
                                               .shape = SymbolicShape(IntArrayView{input_dims})}},
                    .output_specs = {TensorSpec{}},
                    .runtime_checks = {bad_check},
            },
    });

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlan, SortsStateAliasPlanForStepLookup) {
    ResolvedKernel kernel{
            .op_type = OpType::kSoftmax,
            .fn = &FakeKernel,
    };
    // Deliberately unsorted: Create must converge the ForStep precondition.
    const StatusOr<ExecutionPlan> plan = ExecutionPlan::Create(
            {ExecutionStep{.kernel = kernel}},
            StateAliasPlan{.aliases = {
                                   ResolvedStateAlias{.step_index = 0, .input_port = 0, .output_port = 1},
                                   ResolvedStateAlias{.step_index = 0, .input_port = 0, .output_port = 0},
                           }});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    const auto aliases = plan->state_alias_plan().ForStep(0);
    ASSERT_EQ(aliases.size(), 2U);
    EXPECT_EQ(aliases[0].output_port, 0U);
    EXPECT_EQ(aliases[1].output_port, 1U);
}

TEST(ExecutionPlan, RejectsStateAliasBeyondStepCount) {
    ResolvedKernel kernel{
            .op_type = OpType::kSoftmax,
            .fn = &FakeKernel,
    };

    const StatusOr<ExecutionPlan> plan = ExecutionPlan::Create(
            {ExecutionStep{.kernel = kernel}},
            StateAliasPlan{.aliases = {
                                   ResolvedStateAlias{.step_index = 1},
                           }});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(plan.status().message().find("beyond the plan's step count"),
              std::string::npos);
}

}// namespace
