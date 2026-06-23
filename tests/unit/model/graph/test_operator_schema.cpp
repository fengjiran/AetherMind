#include "aethermind/model/graph/operator_schema.h"

#include <gtest/gtest.h>

namespace aethermind {
namespace {

TEST(OperatorSchema, ContainsAllM1Ops) {
    const auto schemas = GetOperatorSchemas();

    ASSERT_EQ(schemas.size(), 9U);
    EXPECT_TRUE(GetOperatorSchema(OpType::kEmbedding).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kRmsNorm).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kLinear).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kRoPE).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kMatMul).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kSoftmax).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kAdd).ok());
    EXPECT_TRUE(GetOperatorSchema(OpType::kSiluMul).ok());
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
    ASSERT_EQ(schema->input_ports.size(), 2U);
    EXPECT_EQ(schema->input_ports[0].name, "q");
    EXPECT_EQ(schema->input_ports[1].name, "k");
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

TEST(OperatorSchema, ActivationOnlyOpsUseExpectedArities) {
    struct ExpectedArity {
        OpType op_type{};
        size_t inputs = 0;
        size_t outputs = 0;
    };
    constexpr ExpectedArity kExpected[] = {
            ExpectedArity{.op_type = OpType::kRoPE, .inputs = 2, .outputs = 2},
            ExpectedArity{.op_type = OpType::kMatMul, .inputs = 2, .outputs = 1},
            ExpectedArity{.op_type = OpType::kSoftmax, .inputs = 1, .outputs = 1},
            ExpectedArity{.op_type = OpType::kAdd, .inputs = 2, .outputs = 1},
            ExpectedArity{.op_type = OpType::kSiluMul, .inputs = 2, .outputs = 1},
            ExpectedArity{.op_type = OpType::kArgmax, .inputs = 1, .outputs = 1},
    };

    for (const ExpectedArity expected: kExpected) {
        const StatusOr<OperatorSchema> schema = GetOperatorSchema(expected.op_type);
        ASSERT_TRUE(schema.ok()) << ToString(expected.op_type);
        ASSERT_EQ(schema->input_ports.size(), expected.inputs) << ToString(expected.op_type);
        ASSERT_EQ(schema->output_ports.size(), expected.outputs) << ToString(expected.op_type);
        for (size_t index = 0; index < schema->input_ports.size(); ++index) {
            EXPECT_EQ(schema->input_ports[index].index, index) << ToString(expected.op_type);
            EXPECT_EQ(schema->input_ports[index].kind, OperatorPortKind::kActivation) << ToString(expected.op_type);
        }
        for (size_t index = 0; index < schema->output_ports.size(); ++index) {
            EXPECT_EQ(schema->output_ports[index].index, index) << ToString(expected.op_type);
            EXPECT_EQ(schema->output_ports[index].kind, OperatorPortKind::kActivation) << ToString(expected.op_type);
        }
    }
}

}// namespace
}// namespace aethermind
