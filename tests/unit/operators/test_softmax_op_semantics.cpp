#include "test_operator_semantics_helpers.h"

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>
#include <vector>

namespace {
using namespace aethermind;

TEST(OperatorSemanticsValidate, SoftmaxValidParams) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_TRUE(InferOperator(OpType::kSoftmax, SoftmaxParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, SoftmaxFloat32Ok) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kSoftmax, SoftmaxParams{-1}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(OperatorSemanticsInfer, SoftmaxAxisOutOfRange) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_TRUE(InferOperator(OpType::kSoftmax, SoftmaxParams{5}, inputs).ok());
}

TEST(OperatorSemanticsInfer, SoftmaxUnrankedSkipsAxisCheck) {
    auto input = MakeSpec(DataType::Float32());
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kSoftmax, SoftmaxParams{-1}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_FALSE(result->outputs[0].shape.IsRanked());
}

}// namespace
