#include "aethermind/model/graph/op_params_serde.h"

#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

namespace aethermind {
namespace {

std::string SerializeToString(const OpParams& params) {
    std::ostringstream os;
    const Status status = SerializeOpParams(params, os);
    EXPECT_TRUE(status.ok()) << status.ToString();
    return os.str();
}

TEST(OpParamsSerde, RoundTripsEveryVariant) {
    const std::vector<OpParams> params{
            std::monostate{},
            EmbeddingParams{},
            RmsNormParams{.eps = 1.0e-6F},
            LinearParams{},
            RoPEParams{.head_dim = 8,
                       .num_attention_heads = 4,
                       .num_key_value_heads = 2,
                       .max_position_embeddings = 128,
                       .theta = 10000.0,
                       .scaling_factor = 2.5,
                       .scaling_type = HfRopeScalingType::kLinear},
            RoPEParams{.head_dim = 16,
                       .num_attention_heads = 8,
                       .num_key_value_heads = 2,
                       .max_position_embeddings = 4096,
                       .theta = 500000.0,
                       .scaling_factor = std::nullopt,
                       .scaling_type = HfRopeScalingType::kNone},
            MatMulParams{.transpose_rhs = true},
            SoftmaxParams{.axis = -1},
            AddParams{},
            SiluParams{},
            SiluMulParams{},
            ElementwiseMulParams{},
            KVCacheUpdateParams{},
            AttentionParams{.num_attention_heads = 4, .num_key_value_heads = 2, .head_dim = 8},
            ArgmaxParams{.axis = -1},
            ReshapeParams{.target_shape = {}},
            ReshapeParams{.target_shape = {ReshapeLiteralDim{2},
                                           ReshapeLiteralDim{3}}},
            ReshapeParams{.target_shape = {ReshapeInputDim{0},
                                           ReshapeInputDim{1},
                                           ReshapeLiteralDim{32},
                                           ReshapeInferDim{}}},
            ReshapeParams{.target_shape = {ReshapeLiteralDim{0},
                                           ReshapeInferDim{}}},
    };

    for (const OpParams& param: params) {
        const std::string serialized = SerializeToString(param);
        const StatusOr<OpParams> parsed = ParseOpParams(serialized);
        ASSERT_TRUE(parsed.ok()) << serialized << " -> " << parsed.status().ToString();
        EXPECT_EQ(SerializeToString(*parsed), serialized);
    }
}

TEST(OpParamsSerde, RejectsUnknownKind) {
    const StatusOr<OpParams> parsed = ParseOpParams("UnknownOp foo=1");

    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.status().code(), StatusCode::kInvalidArgument);
}

TEST(OpParamsSerde, RejectsMissingField) {
    const StatusOr<OpParams> parsed = ParseOpParams("RmsNorm");

    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.status().code(), StatusCode::kInvalidArgument);
}

TEST(OpParamsSerde, RejectsUnexpectedField) {
    const StatusOr<OpParams> parsed = ParseOpParams("Embedding extra=1");

    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.status().code(), StatusCode::kInvalidArgument);
}

TEST(OpParamsSerde, ReshapeKindName) {
    EXPECT_STREQ(OpParamsKindName(OpParams{ReshapeParams{}}), "Reshape");
}

TEST(OpParamsSerde, ReshapeRoundTripsCanonicalForms) {
    // Rank zero: empty brackets.
    {
        const OpParams params{ReshapeParams{.target_shape = {}}};
        const std::string serialized = SerializeToString(params);
        EXPECT_EQ(serialized, "Reshape shape=[]");
        const StatusOr<OpParams> parsed = ParseOpParams(serialized);
        ASSERT_TRUE(parsed.ok()) << parsed.status().ToString();
        EXPECT_EQ(SerializeToString(*parsed), serialized);
    }
    // Static literal dims.
    {
        const OpParams params{ReshapeParams{.target_shape = {ReshapeLiteralDim{2}, ReshapeLiteralDim{3}}}};
        const std::string serialized = SerializeToString(params);
        EXPECT_EQ(serialized, "Reshape shape=[2,3]");
        const StatusOr<OpParams> parsed = ParseOpParams(serialized);
        ASSERT_TRUE(parsed.ok()) << parsed.status().ToString();
        EXPECT_EQ(SerializeToString(*parsed), serialized);
    }
    // Mixed: input-axis references, literal, and infer marker.
    {
        const OpParams params{ReshapeParams{.target_shape = {ReshapeInputDim{0}, ReshapeInputDim{1}, ReshapeLiteralDim{32}, ReshapeInferDim{}}}};
        const std::string serialized = SerializeToString(params);
        EXPECT_EQ(serialized, "Reshape shape=[@0,@1,32,*]");
        const StatusOr<OpParams> parsed = ParseOpParams(serialized);
        ASSERT_TRUE(parsed.ok()) << parsed.status().ToString();
        EXPECT_EQ(SerializeToString(*parsed), serialized);
    }
    // Zero-volume literal + infer.
    {
        const OpParams params{ReshapeParams{.target_shape = {ReshapeLiteralDim{0}, ReshapeInferDim{}}}};
        const std::string serialized = SerializeToString(params);
        EXPECT_EQ(serialized, "Reshape shape=[0,*]");
        const StatusOr<OpParams> parsed = ParseOpParams(serialized);
        ASSERT_TRUE(parsed.ok()) << parsed.status().ToString();
        EXPECT_EQ(SerializeToString(*parsed), serialized);
    }
}

TEST(OpParamsSerde, ReshapeRejectsMalformedShapes) {
    const std::vector<std::string> malformed{
            // Missing field.
            "Reshape",
            // Extra field.
            "Reshape shape=[2,3] extra=1",
            // Missing brackets.
            "Reshape shape=2,3",
            // Leading comma.
            "Reshape shape=[,2,3]",
            // Trailing comma.
            "Reshape shape=[2,3,]",
            // Consecutive commas.
            "Reshape shape=[2,,3]",
            // Empty token between commas.
            "Reshape shape=[2,3,]",
            // Signed/negative literal.
            "Reshape shape=[-1]",
            // @ without unsigned digits.
            "Reshape shape=[@]",
            // @axis overflow uint32.
            "Reshape shape=[@4294967296]",
            // Unknown token syntax (letter).
            "Reshape shape=[abc]",
            // Whitespace inside brackets.
            "Reshape shape=[1, 2]",
            // Duplicate shape field (exactly one shape field required).
            "Reshape shape=[2] shape=[3]",
            // Duplicate shape field with different value.
            "Reshape shape=[2,3] shape=[4,5]",
            // Malformed brackets (unclosed).
            "Reshape shape=[2,3",
            // Malformed brackets (mismatched).
            "Reshape shape=[2,3}",
    };

    for (const std::string& text: malformed) {
        const StatusOr<OpParams> parsed = ParseOpParams(text);
        EXPECT_FALSE(parsed.ok()) << "Expected rejection for: " << text;
        if (!parsed.ok()) {
            EXPECT_EQ(parsed.status().code(), StatusCode::kInvalidArgument) << "Text: " << text;
        }
    }
}

}// namespace
}// namespace aethermind
