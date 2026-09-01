#include "../test_operator_inference_helpers.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/ops/rmsnorm_op.h"

#include <gtest/gtest.h>

#include <limits>

namespace {
using namespace aethermind;

TEST(AddRmsNormInference, InfersDualOutputsForExactStaticInputs) {
    const TensorSpec inputs[3] = {
            MakeSpec(DataType::Float32(), {2, 8}),
            MakeSpec(DataType::Float32(), {2, 8}),
            MakeSpec(DataType::BFloat(16), {8}),
    };

    const StatusOr<InferenceResult> result = InferOperator(
            OpType::kAddRmsNorm,
            AddRmsNormParams{.eps = 1.0e-5F},
            inputs);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(result->runtime_checks.empty());
    ASSERT_EQ(result->outputs.size(), 2U);
    EXPECT_EQ(result->outputs[0], inputs[0]);
    EXPECT_EQ(result->outputs[1], inputs[0]);
}

TEST(AddRmsNormInference, RejectsWrongParamsArityDtypeAndBroadcasting) {
    const TensorSpec exact[3] = {
            MakeSpec(DataType::Float32(), {2, 8}),
            MakeSpec(DataType::Float32(), {2, 8}),
            MakeSpec(DataType::Float32(), {8}),
    };
    EXPECT_FALSE(InferOperator(OpType::kAddRmsNorm, RmsNormParams{}, exact).ok());
    EXPECT_FALSE(InferOperator(OpType::kAddRmsNorm,
                               AddRmsNormParams{},
                               std::span<const TensorSpec>(exact, 2U))
                         .ok());

    const TensorSpec mixed_dtype[3] = {
            MakeSpec(DataType::Float32(), {2, 8}),
            MakeSpec(DataType::Float(16), {2, 8}),
            MakeSpec(DataType::Float32(), {8}),
    };
    EXPECT_FALSE(InferOperator(OpType::kAddRmsNorm, AddRmsNormParams{}, mixed_dtype).ok());

    const TensorSpec broadcast[3] = {
            MakeSpec(DataType::Float32(), {2, 8}),
            MakeSpec(DataType::Float32(), {8}),
            MakeSpec(DataType::Float32(), {8}),
    };
    const Status status = InferOperator(OpType::kAddRmsNorm, AddRmsNormParams{}, broadcast).status();
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("broadcasting is unsupported"), std::string::npos);

    const TensorSpec static_mismatch[3] = {
            MakeSpec(DataType::Float32(), {2, 8}),
            MakeSpec(DataType::Float32(), {2, 7}),
            MakeSpec(DataType::Float32(), {8}),
    };
    EXPECT_FALSE(InferOperator(OpType::kAddRmsNorm, AddRmsNormParams{}, static_mismatch).ok());
}

TEST(AddRmsNormInference, ValidatesEpsilonRankHiddenAndRmsNormDTypes) {
    const TensorSpec valid[3] = {
            MakeSpec(DataType::Float32(), {2, 8}),
            MakeSpec(DataType::Float32(), {2, 8}),
            MakeSpec(DataType::Float32(), {8}),
    };
    EXPECT_FALSE(InferOperator(OpType::kAddRmsNorm, AddRmsNormParams{.eps = 0.0F}, valid).ok());
    EXPECT_FALSE(InferOperator(OpType::kAddRmsNorm,
                               AddRmsNormParams{.eps = std::numeric_limits<float>::quiet_NaN()},
                               valid)
                         .ok());

    const TensorSpec rank_zero[3] = {
            MakeSpec(DataType::Float32(), {}),
            MakeSpec(DataType::Float32(), {}),
            MakeSpec(DataType::Float32(), {1}),
    };
    EXPECT_FALSE(InferOperator(OpType::kAddRmsNorm, AddRmsNormParams{}, rank_zero).ok());

    const TensorSpec invalid_weight[3] = {
            MakeSpec(DataType::Float32(), {2, 8}),
            MakeSpec(DataType::Float32(), {2, 8}),
            MakeSpec(DataType::Int(32), {8}),
    };
    EXPECT_FALSE(InferOperator(OpType::kAddRmsNorm, AddRmsNormParams{}, invalid_weight).ok());

    const TensorSpec weight_length_mismatch[3] = {
            MakeSpec(DataType::Float32(), {2, 8}),
            MakeSpec(DataType::Float32(), {2, 8}),
            MakeSpec(DataType::Float32(), {7}),
    };
    EXPECT_FALSE(InferOperator(OpType::kAddRmsNorm,
                               AddRmsNormParams{},
                               weight_length_mismatch)
                         .ok());
}

TEST(AddRmsNormInference, DefersExactShapeAndWeightChecksForDistinctSymbols) {
    const ShapeSymbol input_batch = ShapeSymbol::Create();
    const ShapeSymbol residual_batch = ShapeSymbol::Create();
    const ShapeSymbol input_hidden = ShapeSymbol::Create();
    const ShapeSymbol residual_hidden = ShapeSymbol::Create();
    const ShapeSymbol weight_hidden = ShapeSymbol::Create();
    const TensorSpec inputs[3] = {
            {.dtype = DataType::Float32(), .shape = {input_batch, input_hidden}},
            {.dtype = DataType::Float32(), .shape = {residual_batch, residual_hidden}},
            {.dtype = DataType::Float32(), .shape = {weight_hidden}},
    };

    const StatusOr<InferenceResult> result = InferOperator(
            OpType::kAddRmsNorm,
            AddRmsNormParams{},
            inputs);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    // Two exact-shape constraints, positive hidden, and hidden-to-weight equality.
    ASSERT_EQ(result->runtime_checks.size(), 4U);
    const auto* weight_check = std::get_if<DimEqualConstraint>(&result->runtime_checks[3].condition);
    ASSERT_NE(weight_check, nullptr);
    EXPECT_EQ(weight_check->rhs.tensor_port.tensor_idx, 2U);
}

} // namespace
