#include "test_operator_semantics_helpers.h"

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>
#include <vector>

namespace {
using namespace aethermind;

TEST(OperatorSemanticsValidate, AddValidParams) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {2, 3});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_TRUE(InferOperator(OpType::kAdd, AddParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, AddFloat32Ok) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {2, 3});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kAdd, AddParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs.size(), 1);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(OperatorSemanticsInfer, AddBroadcast) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {3});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kAdd, AddParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(OperatorSemanticsInfer, AddDtypeMismatch) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Double(), {2, 3});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_FALSE(InferOperator(OpType::kAdd, AddParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, AddWrongVariant) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {2, 3});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_FALSE(InferOperator(OpType::kAdd, RmsNormParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, AddWrongInputCount) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    std::vector<TensorSpec> inputs = {lhs};
    EXPECT_FALSE(InferOperator(OpType::kAdd, AddParams{}, inputs).ok());
}

}// namespace
