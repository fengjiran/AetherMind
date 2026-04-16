#include "aethermind/operators/op_type.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(OpTypeToString, AllKnownTypesHaveStringRepresentation) {
    EXPECT_STREQ(ToString(OpType::kUnknown), "Unknown");
    EXPECT_STREQ(ToString(OpType::kEmbedding), "Embedding");
    EXPECT_STREQ(ToString(OpType::kLinear), "Linear");
    EXPECT_STREQ(ToString(OpType::kMatMul), "MatMul");
    EXPECT_STREQ(ToString(OpType::kRMSNorm), "RMSNorm");
    EXPECT_STREQ(ToString(OpType::kRoPE), "RoPE");
    EXPECT_STREQ(ToString(OpType::kAttentionPrefill), "AttentionPrefill");
    EXPECT_STREQ(ToString(OpType::kAttentionDecode), "AttentionDecode");
    EXPECT_STREQ(ToString(OpType::kSiluMul), "SiluMul");
    EXPECT_STREQ(ToString(OpType::kSoftmax), "Softmax");
    EXPECT_STREQ(ToString(OpType::kArgMax), "ArgMax");
}

TEST(OpTypeToString, InvalidValueReturnsUnknown) {
    const OpType invalid = static_cast<OpType>(999);
    EXPECT_STREQ(ToString(invalid), "Unknown");
}

TEST(OpTypeToString, ResultIsNotNull) {
    EXPECT_NE(ToString(OpType::kUnknown), nullptr);
    EXPECT_NE(ToString(OpType::kLinear), nullptr);
    EXPECT_NE(ToString(OpType::kRMSNorm), nullptr);
}

}// namespace