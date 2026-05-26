#include "aethermind/operators/op_type.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(Operators_OpType, AllKnownTypesHaveStringRepresentation) {
    EXPECT_STREQ(ToString(OpType::kUnknown), "Unknown");
    EXPECT_STREQ(ToString(OpType::kEmbedding), "Embedding");
    EXPECT_STREQ(ToString(OpType::kRmsNorm), "RmsNorm");
    EXPECT_STREQ(ToString(OpType::kLinear), "Linear");
    EXPECT_STREQ(ToString(OpType::kMatMul), "MatMul");
    EXPECT_STREQ(ToString(OpType::kRoPE), "RoPE");
    EXPECT_STREQ(ToString(OpType::kAttention), "Attention");
    EXPECT_STREQ(ToString(OpType::kSilu), "Silu");
    EXPECT_STREQ(ToString(OpType::kSiluMul), "SiluMul");
    EXPECT_STREQ(ToString(OpType::kElementwiseMul), "ElementwiseMul");
    EXPECT_STREQ(ToString(OpType::kAdd), "Add");
    EXPECT_STREQ(ToString(OpType::kSoftmax), "Softmax");
    EXPECT_STREQ(ToString(OpType::kArgmax), "Argmax");
}

TEST(Operators_OpType, InvalidValueReturnsUnknown) {
    constexpr auto invalid = static_cast<OpType>(999);
    EXPECT_STREQ(ToString(invalid), "Unknown");
}

TEST(Operators_OpType, ResultIsNotNull) {
    EXPECT_NE(ToString(OpType::kUnknown), nullptr);
    EXPECT_NE(ToString(OpType::kLinear), nullptr);
    EXPECT_NE(ToString(OpType::kRmsNorm), nullptr);
}

}// namespace
