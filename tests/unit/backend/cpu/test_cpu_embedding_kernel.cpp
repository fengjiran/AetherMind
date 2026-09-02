#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/ops/embedding_op.h"
#include "aethermind/runtime/runtime_builder.h"
#include "backend/cpu/kernels/embedding/embedding_internal.h"
#include "execution/test_execution_binding_helpers.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>

namespace {

using namespace aethermind;

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

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

/// Test-side alias of the removed backend params struct: it mirrors the three
/// TensorView slots that the Embedding builder validates at binding time.
struct EmbeddingTestViews {
    TensorView token_ids{};
    TensorView weight{};
    MutableTensorView output{};
};

EmbeddingTestViews MakeEmbeddingParams(const int64_t (&token_ids)[3],
                                       const float (&weight)[12],
                                       float (&output)[9],
                                       const int64_t (&token_shape)[1],
                                       const int64_t (&token_strides)[1],
                                       const int64_t (&weight_shape)[2],
                                       const int64_t (&weight_strides)[2],
                                       const int64_t (&output_shape)[2],
                                       const int64_t (&output_strides)[2]) {
    return EmbeddingTestViews{
            .token_ids = MakeInt64TensorView(token_ids, token_shape, token_strides),
            .weight = MakeFloatTensorView(weight, weight_shape, weight_strides),
            .output = MakeMutableFloatTensorView(output, output_shape, output_strides),
    };
}

StatusOr<ResolvedKernel> PrepareEmbeddingKernel() {
    CpuBackend backend;
    return backend.PrepareKernel(
            OpType::kEmbedding,
            KernelSelector{
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
                    .weight_format = WeightFormat::kPlain,
                    .phase = ExecPhase::kBoth,
            },
            OpParams{EmbeddingParams{}});
}

Status RunEmbedding(const EmbeddingTestViews& views) noexcept {
    const StatusOr<ResolvedKernel> kernel = PrepareEmbeddingKernel();
    if (!kernel.ok()) {
        return kernel.status();
    }

    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
    const std::array<TensorView, 2> inputs{views.token_ids, views.weight};
    const std::array<MutableTensorView, 1> outputs{views.output};
    const Status build_status = kernel->params_builder(
            KernelParamsBuildContext{
                    .inputs = inputs,
                    .outputs = outputs,
                    .attrs = kernel->attrs,
            },
            storage.data());
    if (!build_status.ok()) {
        return build_status;
    }
    return kernel->fn(KernelContext{
            .kernel_params = storage.data(),
            .attrs = kernel->attrs,
    });
}

TEST(EmbeddingKernel, ComputesExpectedRows) {
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
    const EmbeddingTestViews params = MakeEmbeddingParams(
            token_ids, weight, output, token_shape, token_strides, weight_shape, weight_strides,
            output_shape, output_strides);

    const Status status = RunEmbedding(params);

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

TEST(EmbeddingKernel, ComputesExpectedRowsWithUint32Tokens) {
    const uint32_t token_ids[3] = {2, 0, 3};
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
    const EmbeddingTestViews params{
            .token_ids = TensorView{token_ids, DataType::UInt(32), token_shape, token_strides},
            .weight = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    };

    const Status status = RunEmbedding(params);

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

TEST(EmbeddingKernel, AcceptsZeroTokenCount) {
    // Empty token list with null data on zero-element tensors: kernel must
    // succeed without accessing any pointers (TensorView [0] null-data
    // semantics). Weight retains valid data (non-zero numel requires it).
    const float weight[12] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
    const int64_t token_shape[1] = {0};
    const int64_t token_strides[1] = {1};
    const int64_t weight_shape[2] = {4, 3};
    const int64_t weight_strides[2] = {3, 1};
    const int64_t output_shape[2] = {0, 3};
    const int64_t output_strides[2] = {3, 1};
    const EmbeddingTestViews params{
            .token_ids = TensorView{nullptr, DataType::Int(64), token_shape, token_strides},
            .weight = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output = MutableTensorView{nullptr, DataType::Float32(), output_shape, output_strides},
    };

    const Status status = RunEmbedding(params);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(EmbeddingKernel, RejectsOutOfRangeTokenId) {
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
    const EmbeddingTestViews params = MakeEmbeddingParams(
            token_ids, weight, output, token_shape, token_strides, weight_shape, weight_strides,
            output_shape, output_strides);

    const Status status = RunEmbedding(params);

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kOutOfRange);
}

TEST(EmbeddingKernel, ExecutionPlanBuilderRunsResolvedKernel) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const SymbolicShape tokens_spec_shape = StaticShape({3});
    const SymbolicShape weight_spec_shape = StaticShape({4, 3});
    std::vector<TensorSpec> embedding_inputs = {
            TensorSpec{.dtype = DataType::Int(64), .shape = tokens_spec_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_spec_shape},
    };
    const auto analyzed = InferOperator(OpType::kEmbedding,
                                        OpParams{EmbeddingParams{}},
                                        embedding_inputs);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kEmbedding,
            .selector = {
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
                    .weight_format = WeightFormat::kPlain,
                    .phase = ExecPhase::kBoth,
            },
            .input_specs = embedding_inputs,
            .output_specs = analyzed->outputs,
            .runtime_checks = analyzed->runtime_checks,
            .op_params = OpParams{EmbeddingParams{}},
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);
    EXPECT_EQ(plan->steps().front().kernel.op_type, OpType::kEmbedding);

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
    RuntimeBindingContext bindings;
    test::ExecutionBindingCollector binding_collector(*plan, runtime.GetAllocator(Device::CPU()));
    binding_collector.Set(0, StepTensorBinding{
                                     .inputs = {
                                             TensorView{token_ids, DataType::Int(64), token_shape, token_strides},
                                             TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
                                     },
                                     .outputs = {
                                             MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
                                     },
                             });
    ASSERT_TRUE(binding_collector.Install(bindings).ok());
    const Status status = Executor::Execute(*plan, bindings);

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

} // namespace
