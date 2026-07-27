#include "test_operator_semantics_helpers.h"

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>
#include <vector>

namespace {
using namespace aethermind;

TEST(OperatorSemanticsValidate, SiluValidParams) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_TRUE(InferOperator(OpType::kSilu, SiluParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, SiluFloat32Ok) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kSilu, SiluParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(OperatorSemanticsInfer, SiluBFloat16Ok) {
    auto input = MakeSpec(DataType::BFloat(16), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kSilu, SiluParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(OperatorSemanticsInfer, SiluWrongDtype) {
    auto input = MakeSpec(DataType::Int(32), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_FALSE(InferOperator(OpType::kSilu, SiluParams{}, inputs).ok());
}

}// namespace
