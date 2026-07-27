#include "test_operator_semantics_helpers.h"

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>
#include <vector>

namespace {
using namespace aethermind;

// Builds a TensorSpec with the given dtype, rank, and fresh symbolic dims.
TensorSpec MakeSymbolicSpec(DataType dtype, size_t rank) {
    std::vector<ShapeSymbol> symbols;
    symbols.reserve(rank);
    for (size_t i = 0; i < rank; ++i) {
        symbols.push_back(ShapeSymbol::Create());
    }
    return {dtype, SymbolicShape(symbols)};
}

// --- Reshape validation ---

TEST(OperatorSemanticsValidate, ReshapeRejectsWrongVariantBeforeInputChecks) {
    // Wrong variant precedence: rejected before any input arity check.
    constexpr std::vector<TensorSpec> empty_inputs;
    const auto result = InferOperator(
            OpType::kReshape, AddParams{}, empty_inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(result.status().message(), "Reshape node requires ReshapeParams");
}

TEST(OperatorSemanticsValidate, ReshapeRejectsNegativeLiteralBeforeInputChecks) {
    // Parameter-only invariant: negative literal must be rejected before
    // input-count validation (no inputs supplied here).
    ReshapeParams params;
    params.target_shape = {ReshapeLiteralDim{-1}};
    constexpr std::vector<TensorSpec> empty_inputs;
    const auto result = InferOperator(OpType::kReshape, params, empty_inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(OperatorSemanticsValidate, ReshapeRejectsDuplicateInferBeforeInputChecks) {
    // Parameter-only invariant: at most one infer marker; rejected before
    // input-count validation.
    ReshapeParams params;
    params.target_shape = {ReshapeInferDim{}, ReshapeInferDim{}};
    const std::vector<TensorSpec> empty_inputs;
    const auto result = InferOperator(OpType::kReshape, params, empty_inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(OperatorSemanticsValidate, ReshapeRejectsWrongInputCount) {
    // Parameter-only invariants are valid; arity check fires next.
    ReshapeParams params;
    params.target_shape = {ReshapeLiteralDim{2}, ReshapeLiteralDim{3}};
    const std::vector<TensorSpec> empty_inputs;
    const auto result = InferOperator(OpType::kReshape, params, empty_inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(OperatorSemanticsValidate, ReshapeRejectsUnrankedInput) {
    ReshapeParams params;
    params.target_shape = {ReshapeLiteralDim{2}, ReshapeLiteralDim{3}};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32())};
    const auto result = InferOperator(OpType::kReshape, params, inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(OperatorSemanticsValidate, ReshapeRejectsOutOfRangeInputAxis) {
    ReshapeParams params;
    // Input has rank 2; axis=2 is out of range.
    params.target_shape = {ReshapeInputDim{2}};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {2, 3})};
    const auto result = InferOperator(OpType::kReshape, params, inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(OperatorSemanticsValidate, ReshapeRejectsStaticVolumeMismatch) {
    // [2,3] volume=6 cannot become [5,2] volume=10.
    ReshapeParams params;
    params.target_shape = {ReshapeLiteralDim{5}, ReshapeLiteralDim{2}};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {2, 3})};
    const auto result = InferOperator(OpType::kReshape, params, inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(OperatorSemanticsValidate, ReshapeRejectsAmbiguousZeroVolumeInfer) {
    // [0,3] -> [0,*]: non-infer product is 0; infer is ambiguous. Reject.
    ReshapeParams params;
    params.target_shape = {ReshapeLiteralDim{0}, ReshapeInferDim{}};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {0, 3})};
    const auto result = InferOperator(OpType::kReshape, params, inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(OperatorSemanticsValidate, ReshapeRejectsStaticOverflow) {
    // Build inputs/dims large enough to overflow uint64_t volume product.
    ReshapeParams params;
    params.target_shape = {ReshapeLiteralDim{2}};
    // Single dim of int64_t max would not overflow; need product overflow.
    // Use two large static dims whose product overflows uint64_t.
    params.target_shape = {ReshapeLiteralDim{0x7FFFFFFFFFFFFFFFLL}, ReshapeLiteralDim{0x7FFFFFFFFFFFFFFFLL}};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {2, 3})};
    const auto result = InferOperator(OpType::kReshape, params, inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kOverflow);
}

// --- Reshape inference ---

TEST(OperatorSemanticsInfer, ReshapeRankZeroSuccess) {
    // [1] -> [] (rank zero, scalar) succeeds with volume 1 == 1.
    ReshapeParams params;
    params.target_shape = {};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {1})};
    const auto result = InferOperator(OpType::kReshape, params, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 1U);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_TRUE(result->outputs[0].shape.IsRankZero());
    // Static volume proven; no deferred runtime check needed.
    EXPECT_EQ(result->runtime_checks.size(), 0U);
}

TEST(OperatorSemanticsInfer, ReshapeRankZeroFailsOnVolumeMismatch) {
    // [2] -> [] (volume 2 != 1) must fail.
    ReshapeParams params;
    params.target_shape = {};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {2})};
    const auto result = InferOperator(OpType::kReshape, params, inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(OperatorSemanticsInfer, ReshapeStaticInferResolvesQuotient) {
    // [2,3,4] -> [@0,*,2]: volume=24, non-infer product=4, quotient=6.
    ReshapeParams params;
    params.target_shape = {ReshapeInputDim{0}, ReshapeInferDim{}, ReshapeLiteralDim{2}};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {2, 3, 4})};
    const auto result = InferOperator(OpType::kReshape, params, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 1U);
    ASSERT_EQ(result->outputs[0].shape.rank().value(), 3U);
    ASSERT_TRUE(result->outputs[0].shape[0].IsStatic());
    ASSERT_TRUE(result->outputs[0].shape[1].IsStatic());
    ASSERT_TRUE(result->outputs[0].shape[2].IsStatic());
    EXPECT_EQ(result->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(result->outputs[0].shape[1].GetStaticValue(), 6);
    EXPECT_EQ(result->outputs[0].shape[2].GetStaticValue(), 2);
    EXPECT_EQ(result->runtime_checks.size(), 0U);
}

TEST(OperatorSemanticsInfer, ReshapeZeroVolumeWithInferQuotient) {
    // [0,3] -> [*,3]: input volume=0, non-infer product=3, quotient=0.
    // Zero-volume success because quotient is statically determined.
    ReshapeParams params;
    params.target_shape = {ReshapeInferDim{}, ReshapeLiteralDim{3}};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {0, 3})};
    const auto result = InferOperator(OpType::kReshape, params, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 1U);
    ASSERT_EQ(result->outputs[0].shape.rank().value(), 2U);
    ASSERT_TRUE(result->outputs[0].shape[0].IsStatic());
    ASSERT_TRUE(result->outputs[0].shape[1].IsStatic());
    EXPECT_EQ(result->outputs[0].shape[0].GetStaticValue(), 0);
    EXPECT_EQ(result->outputs[0].shape[1].GetStaticValue(), 3);
    EXPECT_EQ(result->runtime_checks.size(), 0U);
}

TEST(OperatorSemanticsInfer, ReshapeSymbolicInputReferencePreservesIdentity) {
    // Symbolic input dim referenced via @0 must reuse the same ShapeSymbol
    // (identity equality, not just static equality).
    ReshapeParams params;
    params.target_shape = {ReshapeInputDim{0}, ReshapeLiteralDim{3}};
    const std::vector<TensorSpec> inputs = {MakeSymbolicSpec(DataType::Float32(), 2)};
    const ShapeSymbol input_dim0 = inputs[0].shape[0];
    const auto result = InferOperator(OpType::kReshape, params, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 1U);
    ASSERT_EQ(result->outputs[0].shape.rank().value(), 2U);
    // Symbol identity preserved (same symbolic value).
    EXPECT_EQ(result->outputs[0].shape[0], input_dim0);
    ASSERT_TRUE(result->outputs[0].shape[1].IsStatic());
    EXPECT_EQ(result->outputs[0].shape[1].GetStaticValue(), 3);
}

TEST(OperatorSemanticsInfer, ReshapeSymbolicInferProducesUnknownAndDeferredCheck) {
    // Symbolic input dims + infer marker: infer cannot be statically resolved,
    // emits Unknown() and one deferred VolumeEqualConstraint.
    ReshapeParams params;
    params.target_shape = {ReshapeLiteralDim{2}, ReshapeInferDim{}};
    const std::vector<TensorSpec> inputs = {MakeSymbolicSpec(DataType::Float32(), 2)};
    const auto result = InferOperator(OpType::kReshape, params, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 1U);
    ASSERT_EQ(result->outputs[0].shape.rank().value(), 2U);
    ASSERT_TRUE(result->outputs[0].shape[0].IsStatic());
    EXPECT_EQ(result->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_TRUE(result->outputs[0].shape[1].IsUnknown());
    // Exactly one deferred runtime check persisted.
    ASSERT_EQ(result->runtime_checks.size(), 1U);
    const auto* vol = std::get_if<VolumeEqualConstraint>(&result->runtime_checks[0].condition);
    ASSERT_NE(vol, nullptr);
}

TEST(OperatorSemanticsInfer, ReshapeRepeatedInferenceIsDeterministic) {
    // Repeated inference of the same symbolic-infer graph produces identical
    // outputs and identical deferred runtime checks.
    ReshapeParams params;
    params.target_shape = {ReshapeLiteralDim{2}, ReshapeInferDim{}};
    const std::vector<TensorSpec> inputs = {MakeSymbolicSpec(DataType::Float32(), 2)};

    const auto first = InferOperator(OpType::kReshape, params, inputs);
    ASSERT_TRUE(first.ok()) << first.status().ToString();
    const auto second = InferOperator(OpType::kReshape, params, inputs);
    ASSERT_TRUE(second.ok()) << second.status().ToString();

    ASSERT_EQ(first->outputs.size(), 1U);
    ASSERT_EQ(second->outputs.size(), 1U);
    EXPECT_EQ(first->outputs[0].shape, second->outputs[0].shape);
    ASSERT_EQ(first->runtime_checks.size(), second->runtime_checks.size());
    if (!first->runtime_checks.empty()) {
        EXPECT_EQ(first->runtime_checks[0], second->runtime_checks[0]);
    }
}

TEST(OperatorSemanticsInfer, ReshapePreservesInputDtype) {
    // dtype is preserved without restriction to arithmetic subset.
    ReshapeParams params;
    params.target_shape = {ReshapeLiteralDim{6}};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Int(32), {2, 3})};
    const auto result = InferOperator(OpType::kReshape, params, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->outputs[0].dtype, DataType::Int(32));
}

}// namespace
