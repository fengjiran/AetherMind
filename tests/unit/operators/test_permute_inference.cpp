#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "test_operator_inference_helpers.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

// --- Permute validation precedence ---

TEST(PermuteInference, RejectsWrongVariantBeforeInputChecks) {
    // Wrong variant precedence: rejected before any input arity check.
    constexpr std::vector<TensorSpec> empty_inputs;
    const auto result = InferOperator(
            OpType::kPermute, AddParams{}, empty_inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(result.status().message(), "Permute node requires PermuteParams");
}

TEST(PermuteInference, RejectsDuplicateAxesBeforeInputChecks) {
    // Parameter-only invariant: duplicate axes must be rejected before
    // input-count validation (no inputs supplied here).
    PermuteParams params;
    params.permutation = {0, 0};
    constexpr std::vector<TensorSpec> empty_inputs;
    const auto result = InferOperator(OpType::kPermute, params, empty_inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(PermuteInference, RejectsWrongInputCount) {
    // Parameter-only invariants are valid; arity check fires next.
    PermuteParams params;
    params.permutation = {2, 0, 1};
    constexpr std::vector<TensorSpec> empty_inputs;
    const auto result = InferOperator(OpType::kPermute, params, empty_inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(PermuteInference, RejectsUnrankedInput) {
    PermuteParams params;
    params.permutation = {2, 0, 1};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32())};
    const auto result = InferOperator(OpType::kPermute, params, inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(PermuteInference, RejectsPermutationLengthMismatch) {
    // Input has rank 2; permutation has length 3.
    PermuteParams params;
    params.permutation = {2, 0, 1};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {2, 3})};
    const auto result = InferOperator(OpType::kPermute, params, inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(PermuteInference, RejectsOutOfRangeAxis) {
    // Input has rank 3; permutation references axis 3.
    PermuteParams params;
    params.permutation = {2, 0, 3};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {2, 3, 4})};
    const auto result = InferOperator(OpType::kPermute, params, inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

// --- Permute inference happy paths ---

TEST(PermuteInference, RankZeroIsIdentity) {
    PermuteParams params;
    params.permutation = {};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {})};
    const auto result = InferOperator(OpType::kPermute, params, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 1U);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    ASSERT_TRUE(result->outputs[0].shape.IsRankZero());
    EXPECT_TRUE(result->runtime_checks.empty());
}

TEST(PermuteInference, RankOneIsIdentity) {
    PermuteParams params;
    params.permutation = {0};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {5})};
    const auto result = InferOperator(OpType::kPermute, params, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 1U);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(result->outputs[0].shape.rank(), 1U);
    EXPECT_EQ(result->outputs[0].shape[0].GetStaticValue(), 5);
    EXPECT_TRUE(result->runtime_checks.empty());
}

TEST(PermuteInference, StaticShapePermutation) {
    // input [2,3,4], permutation [2,0,1] => output [4,2,3]
    PermuteParams params;
    params.permutation = {2, 0, 1};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {2, 3, 4})};
    const auto result = InferOperator(OpType::kPermute, params, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 1U);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(result->outputs[0].shape.rank(), 3U);
    EXPECT_EQ(result->outputs[0].shape[0].GetStaticValue(), 4);
    EXPECT_EQ(result->outputs[0].shape[1].GetStaticValue(), 2);
    EXPECT_EQ(result->outputs[0].shape[2].GetStaticValue(), 3);
    EXPECT_TRUE(result->runtime_checks.empty());
}

TEST(PermuteInference, IdentityPermutationPreservesStaticShape) {
    PermuteParams params;
    params.permutation = {0, 1, 2};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {2, 3, 4})};
    const auto result = InferOperator(OpType::kPermute, params, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 1U);
    ASSERT_EQ(result->outputs[0].shape.rank(), 3U);
    EXPECT_EQ(result->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(result->outputs[0].shape[1].GetStaticValue(), 3);
    EXPECT_EQ(result->outputs[0].shape[2].GetStaticValue(), 4);
    EXPECT_TRUE(result->runtime_checks.empty());
}

TEST(PermuteInference, SymbolicShapePermutationPreservesIdentity) {
    // Symbolic [A,B,C] with [2,0,1] => [C,A,B] where output[0] symbol
    // is the same object as input[2] symbol (identity preserved).
    PermuteParams params;
    params.permutation = {2, 0, 1};
    const std::vector<TensorSpec> inputs = {MakeSymbolicSpec(DataType::Float32(), 3)};
    const auto result = InferOperator(OpType::kPermute, params, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 1U);
    ASSERT_EQ(result->outputs[0].shape.rank(), 3U);
    // Symbol identity: output dim j must equal input dim permutation[j].
    EXPECT_EQ(result->outputs[0].shape[0], inputs[0].shape[2]);
    EXPECT_EQ(result->outputs[0].shape[1], inputs[0].shape[0]);
    EXPECT_EQ(result->outputs[0].shape[2], inputs[0].shape[1]);
    EXPECT_TRUE(result->runtime_checks.empty());
}

TEST(PermuteInference, PreservesDtype) {
    // dtype is preserved verbatim (any dtype, not just arithmetic subset).
    PermuteParams params;
    params.permutation = {1, 0};
    const std::vector<TensorSpec> inputs = {MakeSpec(DataType::BFloat(16), {2, 3})};
    const auto result = InferOperator(OpType::kPermute, params, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(PermuteInference, EmitsNoConstraints) {
    // Permute never emits runtime_checks (bijection guarantees volume
    // equality automatically; semantics are statically decidable).
    PermuteParams params;
    params.permutation = {2, 0, 1};
    const std::vector<TensorSpec> inputs = {MakeSymbolicSpec(DataType::Float32(), 3)};
    const auto result = InferOperator(OpType::kPermute, params, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(result->runtime_checks.empty());
}

TEST(PermuteInference, RepeatedInferenceProducesIdenticalOutputAndChecks) {
    // Plan acceptance criterion (line 99): "repeated inference producing
    // identical output/check data". --gtest_repeat=100 is NOT equivalent:
    // it re-runs the same assertions, but does not directly compare two
    // separate InferenceResult objects. This test calls InferOperator twice
    // with the same inputs and params, then directly compares outputs
    // (TensorSpec::operator==) and runtime_checks (both must be empty).
    PermuteParams params;
    params.permutation = {2, 0, 1};

    // Static-shape case: two invocations must produce identical TensorSpec.
    {
        const std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {2, 3, 4})};
        const auto result1 = InferOperator(OpType::kPermute, params, inputs);
        const auto result2 = InferOperator(OpType::kPermute, params, inputs);
        ASSERT_TRUE(result1.ok()) << result1.status().ToString();
        ASSERT_TRUE(result2.ok()) << result2.status().ToString();

        ASSERT_EQ(result1->outputs.size(), result2->outputs.size());
        ASSERT_EQ(result1->outputs.size(), 1U);
        // Direct TensorSpec comparison: dtype + SymbolicShape (which compares
        // the underlying ShapeSymbol vector via operator<=>).
        EXPECT_EQ(result1->outputs[0], result2->outputs[0]);

        ASSERT_EQ(result1->runtime_checks.size(), result2->runtime_checks.size());
        EXPECT_EQ(result1->runtime_checks.size(), 0U);
    }

    // Symbolic-shape case: two invocations with the SAME symbolic input must
    // produce outputs whose ShapeSymbols are the same identity (same object),
    // not just structurally equal. This verifies that InferPermute copies
    // input.shape[permutation[j]] by reference, not by value.
    {
        const std::vector<TensorSpec> inputs = {MakeSymbolicSpec(DataType::Float32(), 3)};
        const auto result1 = InferOperator(OpType::kPermute, params, inputs);
        const auto result2 = InferOperator(OpType::kPermute, params, inputs);
        ASSERT_TRUE(result1.ok()) << result1.status().ToString();
        ASSERT_TRUE(result2.ok()) << result2.status().ToString();

        ASSERT_EQ(result1->outputs.size(), 1U);
        ASSERT_EQ(result2->outputs.size(), 1U);
        // Symbol identity: output[j] from both invocations must be the same
        // ShapeSymbol object (operator== on ShapeSymbol compares identity).
        EXPECT_EQ(result1->outputs[0].shape[0], result2->outputs[0].shape[0]);
        EXPECT_EQ(result1->outputs[0].shape[1], result2->outputs[0].shape[1]);
        EXPECT_EQ(result1->outputs[0].shape[2], result2->outputs[0].shape[2]);

        EXPECT_EQ(result1->runtime_checks.size(), 0U);
        EXPECT_EQ(result2->runtime_checks.size(), 0U);
    }
}

}// namespace
