#include "test_operator_semantics_helpers.h"

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>
#include <vector>

namespace {
using namespace aethermind;

TEST(OperatorSemanticsValidate, MatMulValidParams) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_TRUE(InferOperator(OpType::kMatMul, MatMulParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, MatMulRank2Ok) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, MatMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(OperatorSemanticsInfer, MatMulInnerMismatch) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {5, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_FALSE(InferOperator(OpType::kMatMul, MatMulParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, MatMulTransposeRhs) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {4, 3});
    MatMulParams p{true};
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, p, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(OperatorSemanticsInfer, MatMulFloat16Ok) {
    auto lhs = MakeSpec(DataType::Float(16), {2, 3});
    auto rhs = MakeSpec(DataType::Float(16), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, MatMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(OperatorSemanticsInfer, MatMulBFloat16Ok) {
    auto lhs = MakeSpec(DataType::BFloat(16), {2, 3});
    auto rhs = MakeSpec(DataType::BFloat(16), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, MatMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(OperatorSemanticsInfer, MatMulFloat8E4M3Ok) {
    auto lhs = MakeSpec(DataType::Float8E4M3FN(), {2, 3});
    auto rhs = MakeSpec(DataType::Float8E4M3FN(), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, MatMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float8E4M3FN());
}

TEST(OperatorSemanticsInfer, MatMulFloat8E5M2Ok) {
    auto lhs = MakeSpec(DataType::Float8E5M2(), {2, 3});
    auto rhs = MakeSpec(DataType::Float8E5M2(), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, MatMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float8E5M2());
}

TEST(OperatorSemanticsInfer, MatMulInt8Ok) {
    auto lhs = MakeSpec(DataType::Int(8), {2, 3});
    auto rhs = MakeSpec(DataType::Int(8), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, MatMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Int(8));
}

TEST(OperatorSemanticsInfer, MatMulMixedDTypeOk) {
    auto lhs = MakeSpec(DataType::Float(16), {2, 3});
    auto rhs = MakeSpec(DataType::Int(8), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, MatMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(OperatorSemanticsInfer, MatMulWrongLhsDtype) {
    auto lhs = MakeSpec(DataType::Int(32), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_FALSE(InferOperator(OpType::kMatMul, MatMulParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, MatMulWrongRhsDtype) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Double(), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_FALSE(InferOperator(OpType::kMatMul, MatMulParams{}, inputs).ok());
}

}// namespace
