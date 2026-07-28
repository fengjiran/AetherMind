#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "test_operator_inference_helpers.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

// --- Validation precedence ---

TEST(ReorderInference, WrongVariantRejectedBeforeArity) {
    const std::vector<TensorSpec> empty_inputs;
    const auto result = InferOperator(OpType::kReorder, AddParams{}, empty_inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(result.status().message(), "Reorder node requires ReorderParams");
}

TEST(ReorderInference, ZeroInputsRejected) {
    const std::vector<TensorSpec> empty_inputs;
    const auto result = InferOperator(OpType::kReorder, ReorderParams{}, empty_inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ReorderInference, TwoInputsRejected) {
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {2, 3})};
    const auto result = InferOperator(OpType::kReorder, ReorderParams{}, inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

// --- Valid inference ---

TEST(ReorderInference, StaticShapePreserved) {
    std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {4, 256})};
    const auto result = InferOperator(OpType::kReorder, ReorderParams{}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 1U);
    EXPECT_EQ(result->outputs[0].dtype, inputs[0].dtype);
    EXPECT_EQ(result->outputs[0].shape, inputs[0].shape);
}

TEST(ReorderInference, SymbolicShapeIdentityPreserved) {
    auto s0 = ShapeSymbol::Create();
    auto s1 = ShapeSymbol::Create();
    SymbolicShape shape({s0, s1, ShapeSymbol::CreateFromValue(64)});
    TensorSpec input{DataType::Float(16), shape};
    std::vector<TensorSpec> inputs = {input};

    const auto result = InferOperator(OpType::kReorder, ReorderParams{}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 1U);
    const auto& out = result->outputs[0];
    ASSERT_TRUE(out.shape.IsRanked());
    ASSERT_EQ(*out.shape.rank(), 3U);
    // Exact ShapeSymbol identity: same symbol objects, not just same values.
    EXPECT_EQ(out.shape[0], s0);
    EXPECT_EQ(out.shape[1], s1);
    EXPECT_EQ(out.shape[2], ShapeSymbol::CreateFromValue(64));
    EXPECT_EQ(out.dtype, DataType::Float(16));
}

TEST(ReorderInference, UnrankedInputProducesUnrankedOutput) {
    std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32())};
    const auto result = InferOperator(OpType::kReorder, ReorderParams{}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 1U);
    EXPECT_FALSE(result->outputs[0].shape.IsRanked());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(ReorderInference, RankZeroScalarPreserved) {
    std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {})};
    const auto result = InferOperator(OpType::kReorder, ReorderParams{}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 1U);
    ASSERT_TRUE(result->outputs[0].shape.IsRanked());
    EXPECT_EQ(*result->outputs[0].shape.rank(), 0U);
}

TEST(ReorderInference, ZeroDimensionShapePreserved) {
    std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {0, 3})};
    const auto result = InferOperator(OpType::kReorder, ReorderParams{}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 1U);
    EXPECT_EQ(result->outputs[0].shape, inputs[0].shape);
}

TEST(ReorderInference, NonFloatDtypePreserved) {
    std::vector<TensorSpec> inputs = {MakeSpec(DataType::Int(32), {8})};
    const auto result = InferOperator(OpType::kReorder, ReorderParams{}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 1U);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Int(32));
    EXPECT_EQ(result->outputs[0].shape, inputs[0].shape);
}

TEST(ReorderInference, NoRuntimeChecksEmitted) {
    std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {2, 3, 4})};
    const auto result = InferOperator(OpType::kReorder, ReorderParams{}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(result->runtime_checks.empty());
}

TEST(ReorderInference, ExactlyOneOutput) {
    std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {1})};
    const auto result = InferOperator(OpType::kReorder, ReorderParams{}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->outputs.size(), 1U);
}

TEST(ReorderInference, RepeatedInferenceIsDeterministic) {
    std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {4, 256})};
    const auto r1 = InferOperator(OpType::kReorder, ReorderParams{}, inputs);
    const auto r2 = InferOperator(OpType::kReorder, ReorderParams{}, inputs);
    ASSERT_TRUE(r1.ok());
    ASSERT_TRUE(r2.ok());
    EXPECT_EQ(r1->outputs[0].dtype, r2->outputs[0].dtype);
    EXPECT_EQ(r1->outputs[0].shape, r2->outputs[0].shape);
}

}// namespace
