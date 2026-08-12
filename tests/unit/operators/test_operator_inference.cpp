#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "test_operator_inference_helpers.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

// --- Parameter validation through InferOperator ---

// Wrong variant must be rejected BEFORE input arity checks (parameter-before-input precedence).
TEST(OperatorSemanticsValidate, WrongVariantPrecedesInputValidationForEveryOp) {
    struct TestCase {
        OpType op_type;
        OpParams wrong_params;
        const char* expected_message;
    };
    const TestCase cases[] = {
            {OpType::kEmbedding, AddParams{}, "Embedding node requires EmbeddingParams"},
            {OpType::kRmsNorm, AddParams{}, "RmsNorm node requires RmsNormParams"},
            {OpType::kLinear, AddParams{}, "Linear node requires LinearParams"},
            {OpType::kQkvLinear, AddParams{}, "QkvLinear node requires QkvLinearParams"},
            {OpType::kRoPE, AddParams{}, "RoPE node requires RoPEParams"},
            {OpType::kMatMul, AddParams{}, "MatMul node requires MatMulParams"},
            {OpType::kSoftmax, AddParams{}, "Softmax node requires SoftmaxParams"},
            {OpType::kAdd, RmsNormParams{}, "Add node requires AddParams"},
            {OpType::kSiluMul, AddParams{}, "SiluMul node requires SiluMulParams"},
            {OpType::kKVCacheUpdate, AddParams{}, "KVCacheUpdate node requires KVCacheUpdateParams"},
            {OpType::kAttention, AddParams{}, "Attention node requires AttentionParams"},
            {OpType::kArgmax, AddParams{}, "Argmax node requires ArgmaxParams"},
            {OpType::kSilu, AddParams{}, "Silu node requires SiluParams"},
            {OpType::kElementwiseMul, AddParams{}, "ElementwiseMul node requires ElementwiseMulParams"},
            {OpType::kReshape, AddParams{}, "Reshape node requires ReshapeParams"},
            {OpType::kPermute, AddParams{}, "Permute node requires PermuteParams"},
            {OpType::kReorder, AddParams{}, "Reorder node requires ReorderParams"},
    };

    const std::vector<TensorSpec> empty_inputs;
    for (const auto& test_case: cases) {
        SCOPED_TRACE(ToString(test_case.op_type));
        const auto result = InferOperator(
                test_case.op_type, test_case.wrong_params, empty_inputs);
        ASSERT_FALSE(result.ok());
        EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
        EXPECT_EQ(result.status().message(), test_case.expected_message);
    }
}

TEST(OperatorSemanticsValidate, UnknownOpType) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    const auto result = InferOperator(OpType::kUnknown, AddParams{}, inputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().message(),
              "Unknown op type cannot have validated graph params");
}

// --- Inference tests (existing, unchanged except validation is now embedded) ---

TEST(OperatorSemanticsInfer, UnknownOpType) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> unknown_inputs = {input};
    EXPECT_FALSE(InferOperator(OpType::kUnknown, AddParams{}, unknown_inputs).ok());
}

TEST(OperatorSemanticsMakeCompact, AllContributing) {
    auto schema = GetOperatorSchema(OpType::kAdd);
    ASSERT_TRUE(schema.ok());
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {2, 3})};
    auto compact = MakeCompactInputSpecs(*schema, inputs);
    ASSERT_TRUE(compact.ok());
    EXPECT_EQ(compact->size(), 2);
}

TEST(OperatorSemanticsMakeCompact, FiltersStatePorts) {
    auto schema = GetOperatorSchema(OpType::kKVCacheUpdate);
    ASSERT_TRUE(schema.ok());
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {1, 8, 1, 64}),
            MakeSpec(DataType::Float32(), {1, 8, 1, 64}),
            MakeSpec(DataType::Float32(), {1, 8, 1024, 64}),
            MakeSpec(DataType::Float32(), {1, 8, 1024, 64})};
    auto compact = MakeCompactInputSpecs(*schema, inputs);
    ASSERT_TRUE(compact.ok());
    EXPECT_EQ(compact->size(), 2);
}


TEST(OperatorSemanticsMakeCompact, InputCountMismatch) {
    auto schema = GetOperatorSchema(OpType::kAdd);
    ASSERT_TRUE(schema.ok());
    std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {2, 3})};
    EXPECT_FALSE(MakeCompactInputSpecs(*schema, inputs).ok());
}

}// namespace
