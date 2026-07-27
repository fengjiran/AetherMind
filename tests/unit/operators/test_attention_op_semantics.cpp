#include "test_operator_semantics_helpers.h"

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>
#include <vector>

namespace {
using namespace aethermind;

TEST(OperatorSemanticsValidate, AttentionValidParams) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_TRUE(InferOperator(OpType::kAttention,
                              AttentionParams{32, 8, 64}, inputs)
                        .ok());
}

TEST(OperatorSemanticsInfer, AttentionFloat32Ok) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    AttentionParams p{32, 8, 64};
    std::vector<TensorSpec> attn_inputs = {q, kCache, vCache};
    auto result = InferOperator(OpType::kAttention, p, attn_inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs.size(), 1);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(OperatorSemanticsInfer, AttentionQLastDimMismatch) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 100});
    auto kCache = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    AttentionParams p{32, 8, 64};
    std::vector<TensorSpec> attn_inputs = {q, kCache, vCache};
    EXPECT_TRUE(InferOperator(OpType::kAttention, p, attn_inputs).ok());
}

TEST(OperatorSemanticsInfer, AttentionWrongCacheDtype) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 32 * 64});
    auto kCache = MakeSpec(DataType::Double(), {1, 8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    AttentionParams p{32, 8, 64};
    std::vector<TensorSpec> attn_inputs = {q, kCache, vCache};
    EXPECT_TRUE(InferOperator(OpType::kAttention, p, attn_inputs).ok());
}

}// namespace
