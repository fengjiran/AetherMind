#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "test_operator_semantics_helpers.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

TEST(RmsNormInference, AcceptsValidStaticInputContract) {
    constexpr RmsNormParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 8}),
            MakeSpec(DataType::Float32(), {8}),
    };

    EXPECT_TRUE(InferOperator(OpType::kRmsNorm, params, inputs).status().ok());
}

TEST(RmsNormInference, InfersStaticOutputContract) {
    constexpr RmsNormParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 8}),
            MakeSpec(DataType::Float32(), {8}),
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kRmsNorm, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 4);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 8);
}

TEST(RmsNormInference, EmitsPositiveAndEqualityConstraintsForDistinctSymbolicHiddenDimensions) {
    constexpr RmsNormParams params;
    const ShapeSymbol seq_len = ShapeSymbol::Create();
    const ShapeSymbol input_hidden = ShapeSymbol::Create();
    const ShapeSymbol weight_hidden = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{seq_len, input_hidden})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{weight_hidden})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kRmsNorm, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    // 1 DimPositiveConstraint (input hidden) + 1 DimEqualConstraint (hidden == weight_len).
    // Weight length positivity is enforced transitively via hidden_size > 0 + hidden == weight_len.
    ASSERT_EQ(inference->runtime_checks.size(), 2U);
    EXPECT_TRUE(std::holds_alternative<DimPositiveConstraint>(inference->runtime_checks[0].condition));
    const ShapeConstraint& constraint = inference->runtime_checks[1];
    ASSERT_TRUE(std::holds_alternative<DimEqualConstraint>(constraint.condition));
    const auto& equal = std::get<DimEqualConstraint>(constraint.condition);
    EXPECT_EQ(equal.lhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(equal.lhs.tensor_port.tensor_idx, 0U);
    EXPECT_EQ(equal.lhs.dim_index, 1U);
    EXPECT_EQ(equal.rhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(equal.rhs.tensor_port.tensor_idx, 1U);
    EXPECT_EQ(equal.rhs.dim_index, 0U);
}

TEST(RmsNormInference, AcceptsSharedSymbolicHiddenDimension) {
    constexpr RmsNormParams params;
    const ShapeSymbol seq_len = ShapeSymbol::Create();
    const ShapeSymbol hidden_size = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{seq_len, hidden_size})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{hidden_size})},
    };

    EXPECT_TRUE(InferOperator(OpType::kRmsNorm, params, inputs).status().ok());
}

TEST(RmsNormInference, RejectsStaticHiddenMismatch) {
    constexpr RmsNormParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 8}),
            MakeSpec(DataType::Float32(), {16}),
    };

    const Status status = InferOperator(OpType::kRmsNorm, params, inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(RmsNormInference, RejectsRankZeroInput) {
    constexpr RmsNormParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {}),
            MakeSpec(DataType::Float32(), {8}),
    };

    const Status status = InferOperator(OpType::kRmsNorm, params, inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(RmsNormInference, RejectsRankZeroWeight) {
    constexpr RmsNormParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 8}),
            MakeSpec(DataType::Float32(), {}),
    };

    const Status status = InferOperator(OpType::kRmsNorm, params, inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(RmsNormInference, AcceptsZeroBatchDimension) {
    // Zero leading batch/sequence dim is valid (empty output); only the last
    // (hidden) dim must be positive (zero-length reduction is undefined).
    constexpr RmsNormParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {0, 8}),
            MakeSpec(DataType::Float32(), {8}),
    };
    EXPECT_TRUE(InferOperator(OpType::kRmsNorm, params, inputs).status().ok());
}

TEST(RmsNormInference, RejectsZeroHiddenDimension) {
    // Zero hidden dim must still be rejected: zero-length reduction is undefined.
    constexpr RmsNormParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 0}),
            MakeSpec(DataType::Float32(), {0}),
    };
    const Status status = InferOperator(OpType::kRmsNorm, params, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(RmsNormInference, EmitsPositiveConstraintForSymbolicHiddenDimension) {
    // Same symbol for hidden and weight_len: AreProvablyEqual → no DimEqualConstraint.
    // Only 1 DimPositiveConstraint for input hidden (input[0] dim[1]).
    // Weight length positivity is enforced transitively.
    constexpr RmsNormParams params;
    const ShapeSymbol hidden = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::CreateFromValue(4), hidden})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{hidden})},
    };
    const auto result = InferOperator(OpType::kRmsNorm, params, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->runtime_checks.size(), 1U);

    const auto& check = result->runtime_checks[0];
    const auto* pos = std::get_if<DimPositiveConstraint>(&check.condition);
    ASSERT_NE(pos, nullptr);
    EXPECT_EQ(pos->dim.tensor_port.tensor_idx, 0U);
    EXPECT_EQ(pos->dim.dim_index, 1U);
}

TEST(RmsNormInference, RejectsNonPositiveEpsilon) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kRmsNorm, RmsNormParams{0.0f}, inputs).ok());
    EXPECT_FALSE(InferOperator(OpType::kRmsNorm, RmsNormParams{-1.0f}, inputs).ok());
}

TEST(RmsNormInference, RejectsNanEpsilon) {
    RmsNormParams p;
    p.eps = std::numeric_limits<float>::quiet_NaN();
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kRmsNorm, p, inputs).ok());
}

TEST(RmsNormInference, PreservesFloat32InputDType) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs.size(), 1U);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(RmsNormInference, PreservesRankOneInputShape) {
    auto input = MakeSpec(DataType::Float32(), {256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape, input.shape);
}

TEST(RmsNormInference, PreservesRankThreeInputShape) {
    auto input = MakeSpec(DataType::Float32(), {2, 4, 256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape, input.shape);
}

TEST(RmsNormInference, PreservesFloat16InputDType) {
    auto input = MakeSpec(DataType::Float(16), {4, 256});
    auto weight = MakeSpec(DataType::Float(16), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(RmsNormInference, PreservesBFloat16InputDType) {
    auto input = MakeSpec(DataType::BFloat(16), {4, 256});
    auto weight = MakeSpec(DataType::BFloat(16), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(RmsNormInference, PreservesFloat8E4M3FnInputDType) {
    auto input = MakeSpec(DataType::Float8E4M3FN(), {4, 256});
    auto weight = MakeSpec(DataType::Float8E4M3FN(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float8E4M3FN());
}

TEST(RmsNormInference, PreservesInputDTypeWithMixedWeightDType) {
    auto input = MakeSpec(DataType::Float(16), {4, 256});
    auto weight = MakeSpec(DataType::BFloat(16), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(RmsNormInference, RejectsUnsupportedInputDType) {
    auto input = MakeSpec(DataType::Int(32), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs).ok());
}

}// namespace
