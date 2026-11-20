#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/compiler/graph_lowering.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/execution/kernel_invoker.h"
#include "aethermind/graph/graph_op_builder.h"
#include "aethermind/runtime/runtime_builder.h"

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace aethermind;

Status RunDynamicKernel(OpType op, const OpParams& params, TensorView input, MutableTensorView output) {
    const CpuBackend cpu;
    AM_ASSIGN_OR_RETURN(const auto kernel, cpu.PrepareKernel(
                                                   op,
                                                   {.device_type = DeviceType::kCPU,
                                                    .act_dtype = input.dtype(),
                                                    .output_dtype = output.dtype(),
                                                    .weight_dtype = input.dtype(),
                                                    .weight_format = WeightFormat::kPlain,
                                                    .phase = ExecPhase::kDecode},
                                                   params));
    KernelContext context{.attrs = kernel.attrs};
    return BuildAndInvokeKernelForTesting(kernel, context,
                                          std::array{input}, std::array{output});
}

TEST(CpuDynamicOperators, MaterializesDynamicGraphThroughPreparedPlan) {
    const auto f32 = DataType::Float32();
    const auto bf16 = DataType::BFloat(16);
    const auto rows = ShapeSymbol::Create();
    ModelGraph graph;
    const auto input = AddInput(graph, {.dtype = f32, .shape = SymbolicShape({rows, ShapeSymbol::CreateFromValue(2), ShapeSymbol::CreateFromValue(3)})},
                                "input");
    const auto dense = AddReorder(graph, std::nullopt, input, "dense");
    ASSERT_TRUE(dense.ok()) << dense.status().ToString();
    const std::vector<ReshapeDim> target{ReshapeInputDim{0}, ReshapeLiteralDim{6}};
    const auto reshaped = AddReshape(graph, std::nullopt, *dense, target, "flattened");
    ASSERT_TRUE(reshaped.ok()) << reshaped.status().ToString();
    const std::vector<uint32_t> permutation{1, 0};
    const auto permuted = AddPermute(graph, std::nullopt, *reshaped, permutation, "transposed");
    ASSERT_TRUE(permuted.ok()) << permuted.status().ToString();
    const auto reduced = AddCast(graph, std::nullopt, *permuted, bf16, "reduced");
    ASSERT_TRUE(reduced.ok()) << reduced.status().ToString();
    const auto probabilities = graph.AddNode(OpType::kSoftmax, std::nullopt, {*reduced},
                                             {{.payload = ActivationValue{}, .name = "probabilities"}},
                                             SoftmaxParams{.axis = -1});
    ASSERT_TRUE(probabilities.ok()) << probabilities.status().ToString();
    const auto output = AddCast(graph, std::nullopt, probabilities->outputs.front(), f32, "output");
    ASSERT_TRUE(output.ok()) << output.status().ToString();
    graph.MarkOutput(*output);

    const auto lowered = LowerModelGraph(graph);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    Runtime runtime = RuntimeBuilder{}.Build();
    const auto plan = ExecutionPlanBuilder::Build(runtime, *lowered);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->steps().size(), 6U);
    EXPECT_EQ(plan->steps()[4].kernel.name, "cpu::softmax_bf16_reference");
    EXPECT_EQ(plan->total_workspace_bytes(), 0U);

    constexpr std::array<float, 13> data{0, 1, 2, 3, 4, 5, -123, 0, 3, 1, 6, 2, 9};
    constexpr std::array<int64_t, 3> strides{7, 3, 1};
    for (const int64_t n: {2, 1}) {
        const std::array<int64_t, 3> shape{n, 2, 3};
        auto prepared = PrepareExecutionBindings(
                *plan,
                {.read_only = {{.value = plan->model_inputs().front(),
                                .tensor = TensorView(data.data(), f32, shape, strides)}}},
                runtime.GetAllocator(Device::CPU()));
        ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();
        auto context = ExecutionContext::Create(*plan, std::move(*prepared), nullptr);
        ASSERT_TRUE(context.ok()) << context.status().ToString();
        ASSERT_TRUE(Executor::Execute(*plan, *context).ok());
        const auto values = context->prepared_bindings()->values();
        const auto result = values[plan->model_outputs().front().index].ReadView();
        const auto transposed = values[plan->steps()[2].outputs.front().index].ReadView();
        ASSERT_EQ(result.dim(0), 6);
        ASSERT_EQ(result.dim(1), n);
        for (size_t i = 0; i < 6; ++i) {
            for (int64_t row = 0; row < n; ++row) {
                const float value = data[row * 7 + i];
                EXPECT_EQ(transposed.data<float>()[i * n + row], value);
                const double denominator = std::exp(data[i]) + (n == 2 ? std::exp(data[7 + i]) : 0.0);
                EXPECT_NEAR(result.data<float>()[i * n + row], std::exp(value) / denominator, 0.002);
            }
        }
    }
}

TEST(CpuDynamicOperators, ReshapeValidatesLiteralTargetAndAmbiguousInferredAxis) {
    const auto f32 = DataType::Float32();
    const std::array<float, 6> input{};
    std::array<float, 6> output{};
    constexpr std::array<int64_t, 2> source_shape{2, 3}, source_strides{3, 1};
    constexpr std::array<int64_t, 2> wrong_shape{3, 2}, wrong_strides{2, 1};
    const ReshapeParams target{.target_shape = {ReshapeInputDim{0}, ReshapeInferDim{}}};
    EXPECT_EQ(RunDynamicKernel(OpType::kReshape, target, TensorView(input.data(), f32, source_shape, source_strides),
                               MutableTensorView(output.data(), f32, wrong_shape, wrong_strides)),
              StatusCode::kInvalidArgument);

    constexpr std::array<int64_t, 2> empty_shape{0, 3};
    EXPECT_EQ(RunDynamicKernel(OpType::kReshape, target, TensorView(nullptr, f32, empty_shape, source_strides),
                               MutableTensorView(nullptr, f32, empty_shape, source_strides)),
              StatusCode::kInvalidArgument);
}

TEST(CpuDynamicOperators, SymbolicInferredReshapeUsesCallerBoundOutput) {
    const auto f32 = DataType::Float32();
    ModelGraph graph;
    const auto input = AddInput(graph, {.dtype = f32, .shape = SymbolicShape({ShapeSymbol::Create(), ShapeSymbol::CreateFromValue(3)})},
                                "input");
    const std::vector<ReshapeDim> target{ReshapeInferDim{}, ReshapeLiteralDim{2}};
    const auto reshaped = AddReshape(graph, std::nullopt, input, target, "inferred");
    ASSERT_TRUE(reshaped.ok()) << reshaped.status().ToString();
    graph.MarkOutput(*reshaped);
    const auto lowered = LowerModelGraph(graph);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    Runtime runtime = RuntimeBuilder{}.Build();
    const auto plan = ExecutionPlanBuilder::Build(runtime, *lowered);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    constexpr std::array<float, 6> source{1, 2, 3, 4, 5, 6};
    std::array<float, 6> destination{};
    constexpr std::array<int64_t, 2> in_shape{2, 3}, in_strides{3, 1};
    constexpr std::array<int64_t, 2> out_shape{3, 2}, out_strides{2, 1};
    auto prepared = PrepareExecutionBindings(
            *plan,
            {.read_only = {{.value = plan->model_inputs().front(),
                            .tensor = TensorView(source.data(), f32, in_shape, in_strides)}},
             .writable = {{.value = plan->model_outputs().front(),
                           .tensor = MutableTensorView(destination.data(), f32, out_shape, out_strides)}}},
            runtime.GetAllocator(Device::CPU()));
    ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();
    auto context = ExecutionContext::Create(*plan, std::move(*prepared), nullptr);
    ASSERT_TRUE(context.ok()) << context.status().ToString();
    ASSERT_TRUE(Executor::Execute(*plan, *context).ok());
    EXPECT_EQ(destination, source);
}

TEST(CpuDynamicOperators, PermutePreservesOutputPaddingAndReorderRequiresDenseOutput) {
    const auto f16 = DataType::Float(16);
    constexpr std::array<uint16_t, 6> input{0x7d11, 0x8000, 3, 4, 5, 6};
    std::array<uint16_t, 11> output{};
    constexpr std::array<int64_t, 2> shape{2, 3}, strides{3, 1};
    constexpr std::array<int64_t, 2> transposed_shape{3, 2}, strided_output{4, 2};
    const TensorView source(input.data(), f16, shape, strides);
    ASSERT_TRUE(RunDynamicKernel(OpType::kPermute, PermuteParams{{1, 0}}, source,
                                 MutableTensorView(output.data(), f16, transposed_shape, strided_output))
                        .ok());
    EXPECT_EQ(output, (std::array<uint16_t, 11>{0x7d11, 0, 4, 0, 0x8000, 0, 5, 0, 3, 0, 6}));
    EXPECT_EQ(RunDynamicKernel(OpType::kReorder, ReorderParams{}, source,
                               MutableTensorView(output.data(), f16, shape, strided_output)),
              StatusCode::kInvalidArgument);
}

TEST(CpuDynamicOperators, SoftmaxFp16HandlesNonLastAxisAndExactInPlace) {
    const auto f16 = DataType::Float(16);
    std::array<Half, 8> values{Half(1000.0F), Half(-1000.0F), Half(-3.0F), Half(-4.0F),
                               Half(999.0F), Half(-1000.0F), Half(-5.0F), Half(-6.0F)};
    constexpr std::array<int64_t, 2> shape{2, 2}, strides{4, 1};
    ASSERT_TRUE(RunDynamicKernel(OpType::kSoftmax, SoftmaxParams{.axis = -2},
                                 TensorView(values.data(), f16, shape, strides),
                                 MutableTensorView(values.data(), f16, shape, strides))
                        .ok());
    EXPECT_NEAR(static_cast<float>(values[0]), 1.0 / (1.0 + std::exp(-1.0)), 0.00025);
    EXPECT_NEAR(static_cast<float>(values[4]), 1.0 / (1.0 + std::exp(1.0)), 0.00025);
    EXPECT_FLOAT_EQ(static_cast<float>(values[1]), 0.5F);
    EXPECT_FLOAT_EQ(static_cast<float>(values[5]), 0.5F);
    EXPECT_FLOAT_EQ(static_cast<float>(values[2]), -3.0F);
    EXPECT_FLOAT_EQ(static_cast<float>(values[7]), -6.0F);
}

TEST(CpuDynamicOperators, SoftmaxRejectsPartialOverlapAndHandlesNonfiniteSlices) {
    const auto f32 = DataType::Float32();
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::array<float, 9> input{-inf, 10000, inf, 0, -inf, -inf, nan, 1, 0};
    const auto original = input;
    std::array<float, 8> output{};
    constexpr std::array<int64_t, 2> shape{4, 2}, strides{2, 1};
    const TensorView source(input.data(), f32, shape, strides);
    EXPECT_EQ(RunDynamicKernel(OpType::kSoftmax, SoftmaxParams{}, source,
                               MutableTensorView(input.data() + 1, f32, shape, strides)),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(input[1], original[1]);
    ASSERT_TRUE(RunDynamicKernel(OpType::kSoftmax, SoftmaxParams{}, source,
                                 MutableTensorView(output.data(), f32, shape, strides))
                        .ok());
    EXPECT_EQ(output[0], 0.0F);
    EXPECT_EQ(output[1], 1.0F);
    for (size_t i = 2; i < output.size(); ++i) {
        EXPECT_TRUE(std::isnan(output[i]));
    }
}

TEST(CpuDynamicOperators, SoftmaxAcceptsEmptyViewsAndRejectsInvalidConcreteAxis) {
    const auto f32 = DataType::Float32();
    constexpr std::array<int64_t, 2> shape{3, 0}, strides{1, 1};
    const TensorView input(nullptr, f32, shape, strides);
    const MutableTensorView output(nullptr, f32, shape, strides);
    EXPECT_TRUE(RunDynamicKernel(OpType::kSoftmax, SoftmaxParams{}, input, output).ok());
    EXPECT_EQ(RunDynamicKernel(OpType::kSoftmax, SoftmaxParams{.axis = 2}, input, output), StatusCode::kInvalidArgument);
}

} // namespace
