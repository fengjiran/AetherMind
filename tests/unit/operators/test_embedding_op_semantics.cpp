#include "test_operator_semantics_helpers.h"

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>
#include <vector>

namespace {
using namespace aethermind;

TEST(OperatorSemanticsValidate, EmbeddingValidParams) {
    auto tokens = MakeSpec(DataType::Int(32), {10});
    auto weight = MakeSpec(DataType::Float32(), {32000, 256});
    std::vector<TensorSpec> inputs = {tokens, weight};
    EXPECT_TRUE(InferOperator(OpType::kEmbedding, EmbeddingParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, EmbeddingInt32Ok) {
    auto tokens = MakeSpec(DataType::Int(32), {10});
    auto weight = MakeSpec(DataType::Float32(), {32000, 256});
    std::vector<TensorSpec> inputs = {tokens, weight};
    auto result = InferOperator(OpType::kEmbedding, EmbeddingParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(OperatorSemanticsInfer, EmbeddingWrongWeightDtype) {
    auto tokens = MakeSpec(DataType::Int(32), {10});
    auto weight = MakeSpec(DataType::Int(64), {32000, 256});
    std::vector<TensorSpec> inputs = {tokens, weight};
    EXPECT_FALSE(InferOperator(OpType::kEmbedding, EmbeddingParams{}, inputs).ok());
}

}// namespace
