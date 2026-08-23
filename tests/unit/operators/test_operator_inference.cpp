#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "test_operator_inference_helpers.h"

#include <gtest/gtest.h>

#include <string>

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
            {OpType::kGateUpLinear, AddParams{}, "GateUpLinear node requires GateUpLinearParams"},
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

// --- Selector dtype derivation (shared rule with graph lowering) ---

TEST(OperatorSelectorDTypes, DerivesFromActivationAndWeightInputs) {
    const auto schema = GetOperatorSchema(OpType::kRmsNorm);
    ASSERT_TRUE(schema.ok());
    const std::vector<TensorSpec> inputs{
            MakeSpec(DataType::Float32(), {4, 8}),
            MakeSpec(DataType::Float32(), {8}),
    };
    const std::vector<TensorSpec> outputs{
            MakeSpec(DataType::Float32(), {4, 8}),
    };

    const auto derived = DeriveSelectorDTypes(*schema, inputs, outputs);

    ASSERT_TRUE(derived.ok()) << derived.status().ToString();
    EXPECT_EQ(derived->act_dtype, DataType::Float32());
    EXPECT_EQ(derived->weight_dtype, DataType::Float32());
}

TEST(OperatorSelectorDTypes, FallsBackToActivationOutputWhenNoActivationInput) {
    // Embedding has only a kModelInput and a kWeight input; act_dtype must
    // come from the activation output port.
    const auto schema = GetOperatorSchema(OpType::kEmbedding);
    ASSERT_TRUE(schema.ok());
    const std::vector<TensorSpec> inputs{
            MakeSpec(DataType::Int(64), {2}),
            MakeSpec(DataType::Float(16), {32, 8}),
    };
    const std::vector<TensorSpec> outputs{
            MakeSpec(DataType::Float(16), {2, 8}),
    };

    const auto derived = DeriveSelectorDTypes(*schema, inputs, outputs);

    ASSERT_TRUE(derived.ok()) << derived.status().ToString();
    EXPECT_EQ(derived->act_dtype, DataType::Float(16));
    EXPECT_EQ(derived->weight_dtype, DataType::Float(16));
}

TEST(OperatorSelectorDTypes, FallsBackWeightToActivationWhenNoWeightPort) {
    // RoPE has activation inputs plus a kModelInput, but no weight port.
    const auto schema = GetOperatorSchema(OpType::kRoPE);
    ASSERT_TRUE(schema.ok());
    const std::vector<TensorSpec> inputs{
            MakeSpec(DataType::Float32(), {2, 8}),
            MakeSpec(DataType::Float32(), {2, 8}),
            MakeSpec(DataType::Int(64), {2}),
    };
    const std::vector<TensorSpec> outputs{
            MakeSpec(DataType::Float32(), {2, 8}),
            MakeSpec(DataType::Float32(), {2, 8}),
    };

    const auto derived = DeriveSelectorDTypes(*schema, inputs, outputs);

    ASSERT_TRUE(derived.ok()) << derived.status().ToString();
    EXPECT_EQ(derived->act_dtype, DataType::Float32());
    EXPECT_EQ(derived->weight_dtype, DataType::Float32());
}

TEST(OperatorSelectorDTypes, IgnoresStatePorts) {
    const auto schema = GetOperatorSchema(OpType::kKVCacheUpdate);
    ASSERT_TRUE(schema.ok());
    const std::vector<TensorSpec> inputs{
            MakeSpec(DataType::Float32(), {1, 8, 1, 64}),
            MakeSpec(DataType::Float32(), {1, 8, 1, 64}),
            MakeSpec(DataType::Float(16), {1, 8, 1024, 64}),
            MakeSpec(DataType::Float(16), {1, 8, 1024, 64}),
    };
    const std::vector<TensorSpec> outputs{
            MakeSpec(DataType::Float(16), {1, 8, 1024, 64}),
            MakeSpec(DataType::Float(16), {1, 8, 1024, 64}),
    };

    const auto derived = DeriveSelectorDTypes(*schema, inputs, outputs);

    // State ports never contribute; only the activation inputs do.
    ASSERT_TRUE(derived.ok()) << derived.status().ToString();
    EXPECT_EQ(derived->act_dtype, DataType::Float32());
    EXPECT_EQ(derived->weight_dtype, DataType::Float32());
}

TEST(OperatorSelectorDTypes, RejectsUndefinedActivationDType) {
    const auto schema = GetOperatorSchema(OpType::kRmsNorm);
    ASSERT_TRUE(schema.ok());
    const std::vector<TensorSpec> inputs{
            MakeSpec(DataType{}, {4, 8}),
            MakeSpec(DataType::Float32(), {8}),
    };
    const std::vector<TensorSpec> outputs{
            MakeSpec(DataType::Float32(), {4, 8}),
    };

    const auto derived = DeriveSelectorDTypes(*schema, inputs, outputs);

    ASSERT_FALSE(derived.ok());
    EXPECT_EQ(derived.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(derived.status().message().find("undefined"), std::string::npos);
}

TEST(OperatorSelectorDTypes, RejectsInconsistentActivationDTypes) {
    const auto schema = GetOperatorSchema(OpType::kAdd);
    ASSERT_TRUE(schema.ok());
    const std::vector<TensorSpec> inputs{
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float(16), {2, 3}),
    };
    const std::vector<TensorSpec> outputs{
            MakeSpec(DataType::Float32(), {2, 3}),
    };

    const auto derived = DeriveSelectorDTypes(*schema, inputs, outputs);

    ASSERT_FALSE(derived.ok());
    EXPECT_EQ(derived.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(derived.status().message().find("inconsistent"), std::string::npos);
}

TEST(OperatorSelectorDTypes, RejectsSpecCountMismatch) {
    const auto schema = GetOperatorSchema(OpType::kRmsNorm);
    ASSERT_TRUE(schema.ok());
    const std::vector<TensorSpec> inputs{
            MakeSpec(DataType::Float32(), {4, 8}),
    };
    const std::vector<TensorSpec> outputs{
            MakeSpec(DataType::Float32(), {4, 8}),
    };

    const auto derived = DeriveSelectorDTypes(*schema, inputs, outputs);

    ASSERT_FALSE(derived.ok());
    EXPECT_EQ(derived.status().code(), StatusCode::kInvalidArgument);
}

TEST(OperatorSelectorDTypes, RejectsSchemaWithoutActivationSource) {
    const OperatorSchema schema{.op_type = OpType::kUnknown};

    const auto derived = DeriveSelectorDTypes(schema, {}, {});

    ASSERT_FALSE(derived.ok());
    EXPECT_EQ(derived.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(derived.status().message().find("activation dtype source"),
              std::string::npos);
}

}// namespace
