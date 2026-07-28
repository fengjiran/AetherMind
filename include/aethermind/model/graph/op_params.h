#ifndef AETHERMIND_MODEL_GRAPH_OP_PARAMS_H
#define AETHERMIND_MODEL_GRAPH_OP_PARAMS_H

#include "aethermind/model/formats/hf/hf_model_config.h"

#include <cstdint>
#include <vector>

namespace aethermind {

struct EmbeddingParams {};

struct RmsNormParams {
    float eps = 1.0e-5f;
};

struct LinearParams {};

struct RoPEParams {
    int64_t head_dim = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t max_position_embeddings = 0;
    double theta = 10000.0;
    std::optional<double> scaling_factor{};
    HfRopeScalingType scaling_type = HfRopeScalingType::kNone;
};

struct MatMulParams {
    bool transpose_rhs = false;
};

struct SoftmaxParams {
    int64_t axis = -1;
};

struct AddParams {};

struct SiluParams {};

struct SiluMulParams {};

struct ElementwiseMulParams {};

struct KVCacheUpdateParams {};

struct AttentionParams {
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
};

struct ArgmaxParams {
    int64_t axis = -1;
};

// Reshape target-shape dimensions.
//
// Reshape preserves dtype and the row-major logical linear order of elements
// while changing only the logical shape; the output volume must equal the
// input volume. Target dimensions are strongly typed instead of overloading
// integer sentinels (e.g. ONNX 0/-1) so that multiple dynamic input
// dimensions remain expressible and importer-specific conventions do not
// leak into the core IR.
//
// Invariants enforced by InferReshape (not by the type system):
//   - literal values are non-negative
//   - input-axis references are canonical non-negative indexes < input rank
//   - at most one ReshapeInferDim is present
//   - an empty target_shape vector means rank zero

// Non-negative literal target dimension (e.g. 32 in shape [2,32]).
struct ReshapeLiteralDim {
    int64_t value = 0;
    friend bool operator==(const ReshapeLiteralDim&, const ReshapeLiteralDim&) = default;
};

// Reference to an input axis (e.g. @0 in shape [@0,32]).
// `axis` is a canonical non-negative index into the input tensor's rank.
struct ReshapeInputDim {
    uint32_t axis = 0;
    friend bool operator==(const ReshapeInputDim&, const ReshapeInputDim&) = default;
};

// Inferred target dimension (e.g. * in shape [2,*,32]).
// Resolved by InferReshape: statically when unique, otherwise Unknown.
struct ReshapeInferDim {
    friend bool operator==(const ReshapeInferDim&, const ReshapeInferDim&) = default;
};

// Strong variant over the three target-dimension alternatives.
using ReshapeDim = std::variant<ReshapeLiteralDim, ReshapeInputDim, ReshapeInferDim>;

// Semantic parameters for OpType::kReshape.
//
// target_shape.size() is the output rank; an empty vector means rank zero.
// At-most-one-infer, non-negative literals, and in-range input-axis
// references are operator-semantic invariants enforced by InferReshape; serde
// deliberately accepts multiple infer markers so the operator-semantic check
// remains the single authority (consistent with existing serde behavior).
struct ReshapeParams {
    std::vector<ReshapeDim> target_shape{};
    friend bool operator==(const ReshapeParams&, const ReshapeParams&) = default;
};

// Semantic parameters for OpType::kPermute.
//
// permutation[j] is the input axis index that maps to output axis j. The
// permutation must be a complete zero-based bijection over [0, input_rank):
// every input axis appears exactly once. An empty permutation means rank
// zero (identity over a scalar). Output dtype follows input dtype.
//
// Invariants enforced by InferPermute (not by the type system):
//   - permutation values are canonical non-negative axis indexes < input_rank
//   - the permutation is a bijection (no repeated axes, no missing axes)
//   - permutation.size() == input rank
struct PermuteParams {
    std::vector<uint32_t> permutation{};
    friend bool operator==(const PermuteParams&, const PermuteParams&) = default;
};

using OpParams = std::variant<std::monostate,
                              EmbeddingParams,
                              RmsNormParams,
                              LinearParams,
                              RoPEParams,
                              MatMulParams,
                              SoftmaxParams,
                              AddParams,
                              SiluParams,
                              SiluMulParams,
                              ElementwiseMulParams,
                              KVCacheUpdateParams,
                              AttentionParams,
                              ArgmaxParams,
                              ReshapeParams,
                              PermuteParams>;

}// namespace aethermind

#endif
