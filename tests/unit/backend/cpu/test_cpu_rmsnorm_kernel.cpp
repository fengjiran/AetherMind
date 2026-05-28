#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/kernels/cpu_rmsnorm_kernel.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/function_operator.h"
#include "aethermind/operators/rms_norm_op.h"
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>

namespace aethermind {
namespace {

template<size_t N>
TensorView MakeContiguousFloatTensorView(const float (&data)[N],
                                         const int64_t (&shape)[1],
                                         const int64_t (&strides)[1]) {
    return TensorView{data, DataType::Float32(), shape, strides};
}

template<size_t N>
MutableTensorView MakeContiguousMutableFloatTensorView(float (&data)[N],
                                                       const int64_t (&shape)[1],
                                                       const int64_t (&strides)[1]) {
    return MutableTensorView{data, DataType::Float32(), shape, strides};
}

template<size_t N>
    requires requires(CpuRmsNormParams params, TensorView input_view, TensorView weight_view,
                      MutableTensorView output_view) {
        params.Input = input_view;
        params.Weight = weight_view;
        params.Output = output_view;
    }
CpuRmsNormParams MakeCpuRmsNormParams(const float (&input)[N],
                                      const float (&weight)[N],
                                      float (&output)[N],
                                      const int64_t (&shape)[1],
                                      const int64_t (&strides)[1]) {
    const TensorView input_view = MakeContiguousFloatTensorView(input, shape, strides);
    const TensorView weight_view = MakeContiguousFloatTensorView(weight, shape, strides);
    const MutableTensorView output_view = MakeContiguousMutableFloatTensorView(output, shape, strides);

    return CpuRmsNormParams{
            .Input = input_view,
            .Weight = weight_view,
            .Output = output_view,
    };
}

TEST(CpuRmsNormKernel, ComputesExpectedValues) {
    const float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    const float weight[4] = {1.0F, 1.0F, 1.0F, 1.0F};
    float output[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    const int64_t shape[1] = {4};
    const int64_t strides[1] = {1};

    const CpuRmsNormParams params = MakeCpuRmsNormParams(input, weight, output, shape, strides);
    const CpuRmsNormAttrs attrs{.Epsilon = 1.0e-5F};

    const Status status = CpuRmsNormKernel(KernelInvocation{
                                                   .op_type = OpType::kRmsNorm,
                                                   .selector = {.device_type = DeviceType::kCPU},
                                           },
                                           KernelContext{
                                                   .device = Device::CPU(),
                                                   .packed_params = &params,
                                                   .attrs = std::as_bytes(std::span{&attrs, size_t{1}}),
                                           },
                                           WorkspaceBinding{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_NEAR(output[0], 0.365148, 1e-5);
    EXPECT_NEAR(output[1], 0.730297, 1e-5);
    EXPECT_NEAR(output[2], 1.095445, 1e-5);
    EXPECT_NEAR(output[3], 1.460593, 1e-5);
}

TEST(CpuRmsNormKernel, CpuBackendResolvedKernelExecutesThroughExecutor) {
    CpuBackend backend;
    const StatusOr<ResolvedKernel> resolved = backend.ResolveKernelInfo(
            OpType::kRmsNorm,
            KernelSelector{
                    .device_type = DeviceType::kCPU,
                    .activation_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
                    .weight_format = WeightFormat::kPlain,
                    .isa = IsaLevel::kScalar,
                    .phase = ExecPhase::kBoth,
            });
    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();

    const float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    const float weight[4] = {1.0F, 0.5F, 1.5F, 2.0F};
    float output[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    const int64_t shape[1] = {4};
    const int64_t strides[1] = {1};
    const CpuRmsNormParams params = MakeCpuRmsNormParams(input, weight, output, shape, strides);
    const CpuRmsNormAttrs attrs{.Epsilon = 1.0e-5F};
    const auto attrs_bytes = std::as_bytes(std::span{&attrs, size_t{1}});

    ExecutionPlan plan;
    ASSERT_TRUE(plan.AddStep(ExecutionStep{
                                     .op_type = resolved->op_type,
                                     .invocation = {
                                             .op_type = OpType::kRmsNorm,
                                             .selector = {
                                                     .device_type = DeviceType::kCPU,
                                                     .activation_dtype = DataType::Float32(),
                                                     .weight_dtype = DataType::Float32(),
                                                     .weight_format = WeightFormat::kPlain,
                                                     .isa = IsaLevel::kScalar,
                                                      .phase = ExecPhase::kBoth,
                                              },
                                      },
                                      .op = std::make_shared<FunctionOperator>(
                                              resolved->op_type,
                                              resolved->fn,
                                              attrs_bytes,
                                              resolved->debug_name),
                                      .packed_params = &params,
                                      .workspace_requirement = {},
                                      .debug_name = resolved->debug_name,
                              })
                        .ok());

    RuntimeBindingContext bindings;
    const Status status = Executor::Execute(plan, bindings);

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_NEAR(output[0], 0.365148, 1e-5);
    EXPECT_NEAR(output[1], 0.365148, 1e-5);
    EXPECT_NEAR(output[2], 1.643168, 1e-5);
    EXPECT_NEAR(output[3], 2.921186, 1e-5);
}

TEST(CpuRmsNormKernel, ExecutionPlanBuilderRunsThroughRmsNormOperator) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRmsNorm,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
            .op_params = RmsNormOp::Params{.epsilon_ = 1.0e-5F},
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);
    ASSERT_NE(plan->steps().front().op, nullptr);

    const float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    const float weight[4] = {1.0F, 0.5F, 1.5F, 2.0F};
    float output[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    const int64_t shape[1] = {4};
    const int64_t strides[1] = {1};
    const CpuRmsNormParams params = MakeCpuRmsNormParams(input, weight, output, shape, strides);

    ExecutionStep step = plan->steps().front();
    step.packed_params = &params;
    ExecutionPlan executable_plan;
    ASSERT_TRUE(executable_plan.AddStep(step).ok());

    RuntimeBindingContext bindings;
    const Status status = Executor::Execute(executable_plan, bindings);

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_NEAR(output[0], 0.365148, 1e-5);
    EXPECT_NEAR(output[1], 0.365148, 1e-5);
    EXPECT_NEAR(output[2], 1.643168, 1e-5);
    EXPECT_NEAR(output[3], 2.921186, 1e-5);
}

}// namespace
}// namespace aethermind
