#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "test_operator_semantics_helpers.h"

#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {
using namespace aethermind;

TEST(OperatorSemanticsValidate, RmsNormValidParams) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_TRUE(InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs).ok());
}

TEST(OperatorSemanticsValidate, RmsNormInvalidEps) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kRmsNorm, RmsNormParams{0.0f}, inputs).ok());
    EXPECT_FALSE(InferOperator(OpType::kRmsNorm, RmsNormParams{-1.0f}, inputs).ok());
}

TEST(OperatorSemanticsValidate, RmsNormNaN) {
    RmsNormParams p;
    p.eps = std::numeric_limits<float>::quiet_NaN();
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kRmsNorm, p, inputs).ok());
}

TEST(OperatorSemanticsInfer, RmsNormFloat32Ok) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(
            OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs.size(), 1);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(OperatorSemanticsInfer, RmsNormInvalidEps) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kRmsNorm, RmsNormParams{0.0f}, inputs).ok());
}

TEST(OperatorSemanticsInfer, RmsNormRejectsRankZero) {
    auto input = MakeSpec(DataType::Float32(), {});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs).ok());
}

TEST(OperatorSemanticsInfer, RmsNormRank1Ok) {
    auto input = MakeSpec(DataType::Float32(), {256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(
            OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape, input.shape);
}

TEST(OperatorSemanticsInfer, RmsNormRank3Ok) {
    auto input = MakeSpec(DataType::Float32(), {2, 4, 256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(
            OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape, input.shape);
}

TEST(OperatorSemanticsInfer, RmsNormFloat16Ok) {
    auto input = MakeSpec(DataType::Float(16), {4, 256});
    auto weight = MakeSpec(DataType::Float(16), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(
            OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(OperatorSemanticsInfer, RmsNormBFloat16Ok) {
    auto input = MakeSpec(DataType::BFloat(16), {4, 256});
    auto weight = MakeSpec(DataType::BFloat(16), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(
            OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(OperatorSemanticsInfer, RmsNormFloat8Ok) {
    auto input = MakeSpec(DataType::Float8E4M3FN(), {4, 256});
    auto weight = MakeSpec(DataType::Float8E4M3FN(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(
            OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float8E4M3FN());
}

TEST(OperatorSemanticsInfer, RmsNormMixedDTypeOk) {
    auto input = MakeSpec(DataType::Float(16), {4, 256});
    auto weight = MakeSpec(DataType::BFloat(16), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(
            OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(OperatorSemanticsInfer, RmsNormWrongDtype) {
    auto input = MakeSpec(DataType::Int(32), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs).ok());
}

}// namespace
