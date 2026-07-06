#include "aethermind/model/graph/operator_schema.h"

#include <gtest/gtest.h>

namespace aethermind {
namespace {

TEST(OperatorSchema, ContainsAllM1Ops) {
    const auto schemas = GetOperatorSchemas();

    ASSERT_EQ(schemas.size(), 13U);
    EXPECT_TRUE(GetOperatorSchema(OpType::kEmbedding).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kRmsNorm).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kLinear).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kRoPE).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kMatMul).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kSoftmax).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kAdd).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kSilu).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kSiluMul).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kElementwiseMul).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kKVCacheUpdate).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kAttention).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kArgmax).ok());
}

TEST(OperatorSchema, RejectsUnknownOpType) {
    const StatusOr<OperatorSchema> schema = GetOperatorSchema(OpType::kUnknown);

    ASSERT_FALSE(schema.ok());
    EXPECT_EQ(schema.status().code(), StatusCode::kInvalidArgument);
}

TEST(OperatorSchema, EmbeddingSchemaUsesModelInputAndWeight) {
    const StatusOr<OperatorSchema> schema = GetOperatorSchema(OpType::kEmbedding);

    ASSERT_TRUE(schema.ok()) << schema.status().ToString();
    ASSERT_EQ(schema->input_ports.size(), 2U);
    EXPECT_EQ(schema->input_ports[0].name, "tokens");
    EXPECT_EQ(schema->input_ports[0].kind, OperatorPortKind::kModelInput);
    EXPECT_EQ(schema->input_ports[1].name, "weight");
    EXPECT_EQ(schema->input_ports[1].kind, OperatorPortKind::kWeight);
    ASSERT_EQ(schema->output_ports.size(), 1U);
    EXPECT_EQ(schema->output_ports[0].name, "output");
    EXPECT_EQ(schema->output_ports[0].kind, OperatorPortKind::kActivation);
}

TEST(OperatorSchema, RopeSchemaUsesNamedDualInputsAndOutputs) {
    const StatusOr<OperatorSchema> schema = GetOperatorSchema(OpType::kRoPE);

    ASSERT_TRUE(schema.ok()) << schema.status().ToString();
    ASSERT_EQ(schema->input_ports.size(), 3U);
    EXPECT_EQ(schema->input_ports[0].name, "q");
    EXPECT_EQ(schema->input_ports[1].name, "k");
    EXPECT_EQ(schema->input_ports[2].name, "position_ids");
    EXPECT_EQ(schema->input_ports[2].kind, OperatorPortKind::kModelInput);
    ASSERT_EQ(schema->output_ports.size(), 2U);
    EXPECT_EQ(schema->output_ports[0].name, "q_rope");
    EXPECT_EQ(schema->output_ports[1].name, "k_rope");
}

TEST(OperatorSchema, WeightedUnaryOpsUseActivationAndWeight) {
    for (const OpType op_type: {OpType::kRmsNorm, OpType::kLinear}) {
        const StatusOr<OperatorSchema> schema = GetOperatorSchema(op_type);
        ASSERT_TRUE(schema.ok()) << ToString(op_type);
        ASSERT_EQ(schema->input_ports.size(), 2U) << ToString(op_type);
        EXPECT_EQ(schema->input_ports[0].kind, OperatorPortKind::kActivation) << ToString(op_type);
        EXPECT_EQ(schema->input_ports[1].kind, OperatorPortKind::kWeight) << ToString(op_type);
        ASSERT_EQ(schema->output_ports.size(), 1U) << ToString(op_type);
        EXPECT_EQ(schema->output_ports[0].kind, OperatorPortKind::kActivation) << ToString(op_type);
    }
}

TEST(OperatorSchema, KVCacheUpdateSchemaUsesStateInputAndOutput) {
    const StatusOr<OperatorSchema> schema = GetOperatorSchema(OpType::kKVCacheUpdate);

    ASSERT_TRUE(schema.ok()) << schema.status().ToString();
    ASSERT_EQ(schema->input_ports.size(), 4U);
    EXPECT_EQ(schema->input_ports[0].name, "k");
    EXPECT_EQ(schema->input_ports[0].kind, OperatorPortKind::kActivation);
    EXPECT_TRUE(schema->input_ports[0].contributes_tensor_spec);
    EXPECT_EQ(schema->input_ports[1].name, "v");
    EXPECT_EQ(schema->input_ports[1].kind, OperatorPortKind::kActivation);
    EXPECT_TRUE(schema->input_ports[1].contributes_tensor_spec);
    EXPECT_EQ(schema->input_ports[2].name, "k_cache_in");
    EXPECT_EQ(schema->input_ports[2].kind, OperatorPortKind::kState);
    EXPECT_FALSE(schema->input_ports[2].contributes_tensor_spec);
    EXPECT_EQ(schema->input_ports[3].name, "v_cache_in");
    EXPECT_EQ(schema->input_ports[3].kind, OperatorPortKind::kState);
    EXPECT_FALSE(schema->input_ports[3].contributes_tensor_spec);
    ASSERT_EQ(schema->output_ports.size(), 2U);
    EXPECT_EQ(schema->output_ports[0].name, "k_cache_out");
    EXPECT_EQ(schema->output_ports[0].kind, OperatorPortKind::kState);
    EXPECT_EQ(schema->output_ports[1].name, "v_cache_out");
    EXPECT_EQ(schema->output_ports[1].kind, OperatorPortKind::kState);
}

TEST(OperatorSchema, AttentionSchemaUsesActivationAndState) {
    const StatusOr<OperatorSchema> schema = GetOperatorSchema(OpType::kAttention);

    ASSERT_TRUE(schema.ok()) << schema.status().ToString();
    ASSERT_EQ(schema->input_ports.size(), 3U);
    EXPECT_EQ(schema->input_ports[0].name, "q");
    EXPECT_EQ(schema->input_ports[0].kind, OperatorPortKind::kActivation);
    EXPECT_TRUE(schema->input_ports[0].contributes_tensor_spec);
    EXPECT_EQ(schema->input_ports[1].name, "k_cache");
    EXPECT_EQ(schema->input_ports[1].kind, OperatorPortKind::kState);
    EXPECT_FALSE(schema->input_ports[1].contributes_tensor_spec);
    EXPECT_EQ(schema->input_ports[2].name, "v_cache");
    EXPECT_EQ(schema->input_ports[2].kind, OperatorPortKind::kState);
    EXPECT_FALSE(schema->input_ports[2].contributes_tensor_spec);
    ASSERT_EQ(schema->output_ports.size(), 1U);
    EXPECT_EQ(schema->output_ports[0].name, "output");
    EXPECT_EQ(schema->output_ports[0].kind, OperatorPortKind::kActivation);
}

TEST(OperatorSchema, ActivationOnlyOpsUseExpectedArities) {
    struct ExpectedArity {
        OpType op_type{};
        size_t inputs = 0;
        size_t outputs = 0;
    };
    constexpr ExpectedArity kExpected[] = {
            ExpectedArity{.op_type = OpType::kRoPE, .inputs = 3, .outputs = 2},
            ExpectedArity{.op_type = OpType::kMatMul, .inputs = 2, .outputs = 1},
            ExpectedArity{.op_type = OpType::kSoftmax, .inputs = 1, .outputs = 1},
            ExpectedArity{.op_type = OpType::kAdd, .inputs = 2, .outputs = 1},
            ExpectedArity{.op_type = OpType::kSilu, .inputs = 1, .outputs = 1},
            ExpectedArity{.op_type = OpType::kSiluMul, .inputs = 2, .outputs = 1},
            ExpectedArity{.op_type = OpType::kElementwiseMul, .inputs = 2, .outputs = 1},
            ExpectedArity{.op_type = OpType::kArgmax, .inputs = 1, .outputs = 1},
    };

    for (const ExpectedArity expected: kExpected) {
        const StatusOr<OperatorSchema> schema = GetOperatorSchema(expected.op_type);
        ASSERT_TRUE(schema.ok()) << ToString(expected.op_type);
        ASSERT_EQ(schema->input_ports.size(), expected.inputs) << ToString(expected.op_type);
        ASSERT_EQ(schema->output_ports.size(), expected.outputs) << ToString(expected.op_type);
        for (size_t index = 0; index < schema->input_ports.size(); ++index) {
            EXPECT_EQ(schema->input_ports[index].index, index) << ToString(expected.op_type);
        }
        EXPECT_EQ(schema->input_ports[0].kind, OperatorPortKind::kActivation) << ToString(expected.op_type);
        if (schema->input_ports.size() > 1) {
            EXPECT_EQ(schema->input_ports[1].kind, OperatorPortKind::kActivation) << ToString(expected.op_type);
        }
        if (expected.op_type == OpType::kRoPE) {
            EXPECT_EQ(schema->input_ports[2].kind, OperatorPortKind::kModelInput) << ToString(expected.op_type);
            EXPECT_EQ(schema->input_ports[2].name, "position_ids") << ToString(expected.op_type);
        }
        for (size_t index = 0; index < schema->output_ports.size(); ++index) {
            EXPECT_EQ(schema->output_ports[index].index, index) << ToString(expected.op_type);
            EXPECT_EQ(schema->output_ports[index].kind, OperatorPortKind::kActivation) << ToString(expected.op_type);
        }
    }
}

}// namespace
}// namespace aethermind
