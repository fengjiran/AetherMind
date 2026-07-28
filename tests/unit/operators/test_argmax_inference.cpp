#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "test_operator_inference_helpers.h"

#include <gtest/gtest.h>
#include <vector>

namespace {
using namespace aethermind;

TEST(ArgmaxInference, AcceptsValidParams) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_TRUE(InferOperator(OpType::kArgmax, ArgmaxParams{}, inputs).ok());
}

TEST(ArgmaxInference, PreservesFloat32AndReducesRank) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kArgmax, ArgmaxParams{-1}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Int(64));
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 1);
}

TEST(ArgmaxInference, RejectsAxisOutOfRange) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_FALSE(InferOperator(OpType::kArgmax, ArgmaxParams{5}, inputs).ok());
}

}// namespace
