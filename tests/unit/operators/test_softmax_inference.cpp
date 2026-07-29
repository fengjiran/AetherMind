#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "test_operator_inference_helpers.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

TEST(SoftmaxInference, AcceptsValidParams) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_TRUE(InferOperator(OpType::kSoftmax, SoftmaxParams{}, inputs).ok());
}

TEST(SoftmaxInference, PreservesFloat32AndRank) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kSoftmax, SoftmaxParams{-1}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(SoftmaxInference, AcceptsAxisOutOfRange) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_TRUE(InferOperator(OpType::kSoftmax, SoftmaxParams{5}, inputs).ok());
}

TEST(SoftmaxInference, UnrankedSkipsAxisCheck) {
    auto input = MakeSpec(DataType::Float32());
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kSoftmax, SoftmaxParams{-1}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_FALSE(result->outputs[0].shape.IsRanked());
}

TEST(SoftmaxInference, AcceptsFloat16) {
    auto input = MakeSpec(DataType::Float(16), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kSoftmax, SoftmaxParams{-1}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(SoftmaxInference, AcceptsBFloat16) {
    auto input = MakeSpec(DataType::BFloat(16), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kSoftmax, SoftmaxParams{-1}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(SoftmaxInference, RejectsInt32Input) {
    auto input = MakeSpec(DataType::Int(32), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_FALSE(InferOperator(OpType::kSoftmax, SoftmaxParams{-1}, inputs).ok());
}

TEST(SoftmaxInference, RejectsFloat8E4M3FN) {
    // FP8 is intentionally not supported by Softmax (precision concerns).
    auto input = MakeSpec(DataType::Float8E4M3FN(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_FALSE(InferOperator(OpType::kSoftmax, SoftmaxParams{-1}, inputs).ok());
}

}// namespace
