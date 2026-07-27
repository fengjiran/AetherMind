#include "test_operator_semantics_helpers.h"

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>
#include <vector>

namespace {
using namespace aethermind;

TEST(OperatorSemanticsValidate, ElementwiseMulValidParams) {
    auto lhs = MakeSpec(DataType::Float32(), {4, 256});
    auto rhs = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_TRUE(InferOperator(OpType::kElementwiseMul, ElementwiseMulParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, ElementwiseMulFloat32Ok) {
    auto lhs = MakeSpec(DataType::Float32(), {4, 256});
    auto rhs = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kElementwiseMul, ElementwiseMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(OperatorSemanticsInfer, ElementwiseMulBFloat16Ok) {
    auto lhs = MakeSpec(DataType::BFloat(16), {4, 256});
    auto rhs = MakeSpec(DataType::BFloat(16), {4, 256});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kElementwiseMul, ElementwiseMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

}// namespace
