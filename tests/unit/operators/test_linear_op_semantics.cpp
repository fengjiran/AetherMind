#include "test_operator_semantics_helpers.h"

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>
#include <vector>

namespace {
using namespace aethermind;

TEST(OperatorSemanticsValidate, LinearValidParams) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_TRUE(InferOperator(OpType::kLinear, LinearParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, LinearRank2Ok) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(OperatorSemanticsInfer, LinearRank1Ok) {
    auto input = MakeSpec(DataType::Float32(), {256});
    auto weight = MakeSpec(DataType::Float32(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 1);
}

TEST(OperatorSemanticsInfer, LinearInFeaturesMismatch) {
    auto input = MakeSpec(DataType::Float32(), {4, 128});
    auto weight = MakeSpec(DataType::Float32(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kLinear, LinearParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, LinearFloat16ActivationOk) {
    auto input = MakeSpec(DataType::Float(16), {4, 256});
    auto weight = MakeSpec(DataType::Float(16), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(OperatorSemanticsInfer, LinearBFloat16ActivationOk) {
    auto input = MakeSpec(DataType::BFloat(16), {4, 256});
    auto weight = MakeSpec(DataType::BFloat(16), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(OperatorSemanticsInfer, LinearFloat8E4M3ActivationOk) {
    auto input = MakeSpec(DataType::Float8E4M3FN(), {4, 256});
    auto weight = MakeSpec(DataType::Float8E4M3FN(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float8E4M3FN());
}

TEST(OperatorSemanticsInfer, LinearFloat8E5M2ActivationOk) {
    auto input = MakeSpec(DataType::Float8E5M2(), {4, 256});
    auto weight = MakeSpec(DataType::Float8E5M2(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float8E5M2());
}

TEST(OperatorSemanticsInfer, LinearInt8WeightOk) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Int(8), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(OperatorSemanticsInfer, LinearInt4WeightOk) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Int(4), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(OperatorSemanticsInfer, LinearMixedDTypeOk) {
    auto input = MakeSpec(DataType::Float(16), {4, 256});
    auto weight = MakeSpec(DataType::Int(8), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(OperatorSemanticsInfer, LinearWrongActivationDtype) {
    auto input = MakeSpec(DataType::Int(32), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kLinear, LinearParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, LinearWrongWeightDtype) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Double(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kLinear, LinearParams{}, inputs).ok());
}

}// namespace
