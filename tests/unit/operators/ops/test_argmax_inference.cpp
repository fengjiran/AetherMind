#include "../test_operator_inference_helpers.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>

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
    auto result = InferOperator(
            OpType::kArgmax, ArgmaxParams{-1}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Int(64));
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 1);
}

TEST(ArgmaxInference, RejectsAxisOutOfRange) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_FALSE(InferOperator(OpType::kArgmax, ArgmaxParams{5}, inputs).ok());
}

TEST(ArgmaxInference, AcceptsFloat16) {
    auto input = MakeSpec(DataType::Float(16), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kArgmax, ArgmaxParams{-1}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->outputs[0].dtype, DataType::Int(64));
}

TEST(ArgmaxInference, AcceptsBFloat16) {
    auto input = MakeSpec(DataType::BFloat(16), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kArgmax, ArgmaxParams{-1}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->outputs[0].dtype, DataType::Int(64));
}

TEST(ArgmaxInference, RejectsInt32Input) {
    auto input = MakeSpec(DataType::Int(32), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_FALSE(InferOperator(OpType::kArgmax, ArgmaxParams{-1}, inputs).ok());
}

TEST(ArgmaxInference, RejectsFloat8E4M3FN) {
    // FP8 is intentionally not supported by Argmax (precision concerns in
    // value comparison for index selection).
    auto input = MakeSpec(DataType::Float8E4M3FN(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_FALSE(InferOperator(OpType::kArgmax, ArgmaxParams{-1}, inputs).ok());
}

}// namespace
