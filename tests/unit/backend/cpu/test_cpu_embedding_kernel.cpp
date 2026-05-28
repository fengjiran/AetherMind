#include "aethermind/backend/cpu/kernels/cpu_embedding_kernel.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_invocation.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/embedding_op.h"
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>

namespace aethermind {
namespace {

template<size_t N>
TensorView MakeInt64TensorView(const int64_t (&data)[N],
                               const int64_t (&shape)[1],
                               const int64_t (&strides)[1]) {
    return TensorView{data, DataType::Int(64), shape, strides};
}

template<size_t N>
TensorView MakeFloatTensorView(const float (&data)[N],
                               const int64_t (&shape)[2],
                               const int64_t (&strides)[2]) {
    return TensorView{data, DataType::Float32(), shape, strides};
}

template<size_t N>
MutableTensorView MakeMutableFloatTensorView(float (&data)[N],
                                             const int64_t (&shape)[2],
                                             const int64_t (&strides)[2]) {
    return MutableTensorView{data, DataType::Float32(), shape, strides};
}

CpuEmbeddingParams MakeCpuEmbeddingParams(const int64_t (&token_ids)[3],
                                          const float (&weight)[12],
                                          float (&output)[9],
                                          const int64_t (&token_shape)[1],
                                          const int64_t (&token_strides)[1],
                                          const int64_t (&weight_shape)[2],
                                          const int64_t (&weight_strides)[2],
                                          const int64_t (&output_shape)[2],
                                          const int64_t (&output_strides)[2]) {
    return CpuEmbeddingParams{
            .token_ids_ = MakeInt64TensorView(token_ids, token_shape, token_strides),
            .weight_ = MakeFloatTensorView(weight, weight_shape, weight_strides),
            .output_ = MakeMutableFloatTensorView(output, output_shape, output_strides),
    };
}

TEST(CpuEmbeddingKernel, ComputesExpectedRows) {
    const int64_t token_ids[3] = {2, 0, 3};
    const float weight[12] = {
            1.0F,
            2.0F,
            3.0F,
            4.0F,
            5.0F,
            6.0F,
            7.0F,
            8.0F,
            9.0F,
            10.0F,
            11.0F,
            12.0F,
    };
    float output[9] = {};
    const int64_t token_shape[1] = {3};
    const int64_t token_strides[1] = {1};
    const int64_t weight_shape[2] = {4, 3};
    const int64_t weight_strides[2] = {3, 1};
    const int64_t output_shape[2] = {3, 3};
    const int64_t output_strides[2] = {3, 1};
    const CpuEmbeddingParams params = MakeCpuEmbeddingParams(
            token_ids, weight, output, token_shape, token_strides, weight_shape, weight_strides,
            output_shape, output_strides);

    const Status status = CpuEmbeddingKernel(KernelInvocation{
                                                     .op_type = OpType::kEmbedding,
                                                     .selector = {.device_type = DeviceType::kCPU},
                                             },
                                             KernelContext{
                                                     .device = Device::CPU(),
                                                     .packed_params = &params,
                                             },
                                             WorkspaceBinding{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_FLOAT_EQ(output[0], 7.0F);
    EXPECT_FLOAT_EQ(output[1], 8.0F);
    EXPECT_FLOAT_EQ(output[2], 9.0F);
    EXPECT_FLOAT_EQ(output[3], 1.0F);
    EXPECT_FLOAT_EQ(output[4], 2.0F);
    EXPECT_FLOAT_EQ(output[5], 3.0F);
    EXPECT_FLOAT_EQ(output[6], 10.0F);
    EXPECT_FLOAT_EQ(output[7], 11.0F);
    EXPECT_FLOAT_EQ(output[8], 12.0F);
}

TEST(CpuEmbeddingKernel, ComputesExpectedRowsWithUint32Tokens) {
    const uint32_t token_ids[3] = {2, 0, 3};
    const float weight[12] = {
            1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
            7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F,
    };
    float output[9] = {};
    const int64_t token_shape[1] = {3};
    const int64_t token_strides[1] = {1};
    const int64_t weight_shape[2] = {4, 3};
    const int64_t weight_strides[2] = {3, 1};
    const int64_t output_shape[2] = {3, 3};
    const int64_t output_strides[2] = {3, 1};
    const CpuEmbeddingParams params{
            .token_ids_ = TensorView{token_ids, DataType::UInt(32), token_shape, token_strides},
            .weight_ = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_ = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    };

    const Status status = CpuEmbeddingKernel(KernelInvocation{
                                                     .op_type = OpType::kEmbedding,
                                                     .selector = {.device_type = DeviceType::kCPU},
                                             },
                                             KernelContext{
                                                     .device = Device::CPU(),
                                                     .packed_params = &params,
                                             },
                                             WorkspaceBinding{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_FLOAT_EQ(output[0], 7.0F);
    EXPECT_FLOAT_EQ(output[1], 8.0F);
    EXPECT_FLOAT_EQ(output[2], 9.0F);
    EXPECT_FLOAT_EQ(output[3], 1.0F);
    EXPECT_FLOAT_EQ(output[4], 2.0F);
    EXPECT_FLOAT_EQ(output[5], 3.0F);
    EXPECT_FLOAT_EQ(output[6], 10.0F);
    EXPECT_FLOAT_EQ(output[7], 11.0F);
    EXPECT_FLOAT_EQ(output[8], 12.0F);
}

TEST(CpuEmbeddingKernel, RejectsOutOfRangeTokenId) {
    const int64_t token_ids[3] = {2, 4, 0};
    const float weight[12] = {
            1.0F,
            2.0F,
            3.0F,
            4.0F,
            5.0F,
            6.0F,
            7.0F,
            8.0F,
            9.0F,
            10.0F,
            11.0F,
            12.0F,
    };
    float output[9] = {};
    const int64_t token_shape[1] = {3};
    const int64_t token_strides[1] = {1};
    const int64_t weight_shape[2] = {4, 3};
    const int64_t weight_strides[2] = {3, 1};
    const int64_t output_shape[2] = {3, 3};
    const int64_t output_strides[2] = {3, 1};
    const CpuEmbeddingParams params = MakeCpuEmbeddingParams(
            token_ids, weight, output, token_shape, token_strides, weight_shape, weight_strides,
            output_shape, output_strides);

    const Status status = CpuEmbeddingKernel(KernelInvocation{
                                                     .op_type = OpType::kEmbedding,
                                                     .selector = {.device_type = DeviceType::kCPU},
                                             },
                                             KernelContext{
                                                     .device = Device::CPU(),
                                                     .packed_params = &params,
                                             },
                                             WorkspaceBinding{});

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kOutOfRange);
}

TEST(CpuEmbeddingKernel, ExecutionPlanBuilderRunsThroughEmbeddingOperator) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kEmbedding,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
            .op_params = EmbeddingOp::Params{},
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);
    ASSERT_NE(plan->steps().front().op, nullptr);
    EXPECT_EQ(plan->steps().front().op->Type(), OpType::kEmbedding);

    const int64_t token_ids[3] = {1, 3, 0};
    const float weight[12] = {
            1.0F,
            2.0F,
            3.0F,
            4.0F,
            5.0F,
            6.0F,
            7.0F,
            8.0F,
            9.0F,
            10.0F,
            11.0F,
            12.0F,
    };
    float output[9] = {};
    const int64_t token_shape[1] = {3};
    const int64_t token_strides[1] = {1};
    const int64_t weight_shape[2] = {4, 3};
    const int64_t weight_strides[2] = {3, 1};
    const int64_t output_shape[2] = {3, 3};
    const int64_t output_strides[2] = {3, 1};
    const CpuEmbeddingParams params = MakeCpuEmbeddingParams(
            token_ids, weight, output, token_shape, token_strides, weight_shape, weight_strides,
            output_shape, output_strides);

    ExecutionStep step = plan->steps().front();
    step.packed_params = &params;
    ExecutionPlan executable_plan;
    ASSERT_TRUE(executable_plan.AddStep(step).ok());

    RuntimeBindingContext bindings;
    const Status status = Executor::Execute(executable_plan, bindings);

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_FLOAT_EQ(output[0], 4.0F);
    EXPECT_FLOAT_EQ(output[1], 5.0F);
    EXPECT_FLOAT_EQ(output[2], 6.0F);
    EXPECT_FLOAT_EQ(output[3], 10.0F);
    EXPECT_FLOAT_EQ(output[4], 11.0F);
    EXPECT_FLOAT_EQ(output[5], 12.0F);
    EXPECT_FLOAT_EQ(output[6], 1.0F);
    EXPECT_FLOAT_EQ(output[7], 2.0F);
    EXPECT_FLOAT_EQ(output[8], 3.0F);
}

}// namespace
}// namespace aethermind
