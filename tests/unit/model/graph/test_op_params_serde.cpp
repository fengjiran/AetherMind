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

}// namespace
}// namespace aethermind
