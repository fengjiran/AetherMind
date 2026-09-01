#ifndef AETHERMIND_OPERATORS_OP_PARAMS_H
#define AETHERMIND_OPERATORS_OP_PARAMS_H

/// @file op_params.h
/// @brief Typed parameter variants for operator semantics.
///
/// Defines the OpParams variant and per-operator parameter structs consumed
/// by operator schema inference. Parameter structs are typed value objects;
/// no std::any or stringly-typed fields are used.

#include <cstdint>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

namespace aethermind {

struct EmbeddingParams {};

struct RmsNormParams {
    float eps = 1.0e-5f;
};

struct LinearParams {};

/// @brief Format-agnostic RoPE scaling strategy.
///
/// Only `kNone` (standard RoPE) and `kLinear` are representable on the
/// semantic graph surface. HF-only variants (dynamic NTK, YaRN, llama3,
/// longrope, su, unknown) are rejected by `ModelGraphBuilder::BuildLlamaDense`
/// before any graph mutation, so they can never reach `RoPEParams`.
enum class RoPEScalingType : uint8_t {
    kNone = 0, ///< Standard RoPE; `scaling_factor` must be absent.
    kLinear,   ///< Linear scaling; requires a finite positive `scaling_factor`.
};

/// @brief Returns the canonical string name of a RoPE scaling type.
///
/// @param scaling_type Scaling type to stringify.
/// @return String view of the scaling type name ("none" or "linear").
inline std::string_view ToString(RoPEScalingType scaling_type) noexcept {
    switch (scaling_type) {
        case RoPEScalingType::kNone:
            return "none";
        case RoPEScalingType::kLinear:
            return "linear";
    }
    return "none";
}

/// @brief Semantic parameters for OpType::kRoPE (Rotary Position Embedding).
///
/// Captures the model-level RoPE configuration. InferRoPE validates the
/// graph-time contract against these parameters and input TensorSpecs; it
/// does not inspect position tensor contents or execute the rotation.
///
/// @pre Graph-time invariants enforced by InferRoPE:
///      - scalar params: `head_dim` positive and even; `num_attention_heads`,
///        `num_key_value_heads`, `max_position_embeddings` positive; `theta`
///        finite and positive; `num_attention_heads * head_dim` and
///        `num_key_value_heads * head_dim` do not overflow int64_t
///      - scaling tuple: `scaling_type == kNone` requires `scaling_factor`
///        absent (standard RoPE); `scaling_type == kLinear` requires a
///        present finite `scaling_factor > 0` (factor 1.0 is accepted
///        without normalization); no other scaling types are representable
///        on the RoPEParams surface
///      - input shapes: q and k rank 2 with widths equal to
///        `num_attention_heads * head_dim` and `num_key_value_heads * head_dim`
///        respectively when static (symbolic widths remain legal);
///        position_ids rank 1, Int64, with seq_len reconciled to q/k
///      - input dtypes: q and k share a dtype from {Float32, Float16, BFloat16}
///
/// @post Outputs preserve q and k input TensorSpecs verbatim (dtype + shape).
///       Runtime shape constraints emitted on the InferenceResult are at most
///       one DimPositiveConstraint for symbolic q seq_len (input 0, dim 0)
///       followed by up to two DimEqualConstraint checks reconciling q/k and
///       q/position sequence dims.
///
/// @note Execution boundary: this struct and InferRoPE do NOT inspect position
///       tensor contents. Executable RoPE paths must validate
///       non-negative position IDs, any effective-position bounds defined by
///       the scaling contract, and symbolic q/k widths against params before
///       computation. Loader `allow_rope_scaling` remains a separate policy;
///       semantic acceptance does not imply current end-to-end kernel support.
///       HF-specific RoPE variants are filtered by the model frontend
///       (`ModelGraphBuilder::BuildLlamaDense`), not by InferRoPE.
struct RoPEParams {
    int64_t head_dim = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t max_position_embeddings = 0;
    double theta = 10000.0;
    /// @brief Linear scaling factor, present exactly when `scaling_type == kLinear`.
    std::optional<double> scaling_factor{};
    /// @brief Format-agnostic scaling strategy accepted by semantic inference.
    RoPEScalingType scaling_type = RoPEScalingType::kNone;
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

/// @brief Non-negative literal target dimension (e.g. 32 in shape [2,32]).
///
/// Reshape preserves dtype and the row-major logical linear order of elements
/// while changing only the logical shape; the output volume must equal the
/// input volume. Target dimensions are strongly typed instead of overloading
/// integer sentinels (e.g. ONNX 0/-1) so that multiple dynamic input
/// dimensions remain expressible and importer-specific conventions do not
/// leak into the core IR.
///
/// Invariants enforced by InferReshape (not by the type system):
///   - literal values are non-negative
///   - input-axis references are canonical non-negative indexes < input rank
///   - at most one ReshapeInferDim is present
///   - an empty target_shape vector means rank zero
struct ReshapeLiteralDim {
    int64_t value = 0;
    friend bool operator==(const ReshapeLiteralDim&, const ReshapeLiteralDim&) = default;
};

/// @brief Reference to an input axis (e.g. @0 in shape [@0,32]).
///
/// `axis` is a canonical non-negative index into the input tensor's rank.
struct ReshapeInputDim {
    uint32_t axis = 0;
    friend bool operator==(const ReshapeInputDim&, const ReshapeInputDim&) = default;
};

/// @brief Inferred target dimension (e.g. * in shape [2,*,32]).
///
/// Resolved by InferReshape: statically when unique, otherwise Unknown.
struct ReshapeInferDim {
    friend bool operator==(const ReshapeInferDim&, const ReshapeInferDim&) = default;
};

/// @brief Strong variant over the three target-dimension alternatives.
using ReshapeDim = std::variant<ReshapeLiteralDim, ReshapeInputDim, ReshapeInferDim>;

/// @brief Semantic parameters for OpType::kReshape.
///
/// target_shape.size() is the output rank; an empty vector means rank zero.
/// At-most-one-infer, non-negative literals, and in-range input-axis
/// references are operator-semantic invariants enforced by InferReshape; serde
/// deliberately accepts multiple infer markers so the operator-semantic check
/// remains the single authority (consistent with existing serde behavior).
struct ReshapeParams {
    std::vector<ReshapeDim> target_shape{};
    friend bool operator==(const ReshapeParams&, const ReshapeParams&) = default;
};

/// @brief Semantic parameters for OpType::kPermute.
///
/// permutation[j] is the input axis index that maps to output axis j. The
/// permutation must be a complete zero-based bijection over [0, input_rank):
/// every input axis appears exactly once. An empty permutation means rank
/// zero (identity over a scalar). Output dtype follows input dtype.
///
/// Invariants enforced by InferPermute (not by the type system):
///   - permutation values are canonical non-negative axis indexes < input_rank
///   - the permutation is a bijection (no repeated axes, no missing axes)
///   - permutation.size() == input rank
struct PermuteParams {
    std::vector<uint32_t> permutation{};
    friend bool operator==(const PermuteParams&, const PermuteParams&) = default;
};

/// @brief Semantic parameters for OpType::kReorder.
///
/// Reorder preserves the logical tensor exactly: dtype, shape (with exact
/// ShapeSymbol identity), and every coordinate value Y[i] == X[i] remain
/// unchanged. The operator records a fixed physical destination intent —
/// canonical row-major contiguous storage — without exposing a target-format
/// field or inspecting input strides during semantic inference.
///
/// This is an empty typed parameter object that distinguishes Reorder from
/// logical Permute and dtype conversion. No fields are needed because the
/// destination is always canonical contiguous.
struct ReorderParams {
    friend bool operator==(const ReorderParams&, const ReorderParams&) = default;
};

/// @brief Semantic parameters for a fused Q/K/V linear projection.
///
/// `has_bias=false` means no Q/K/V bias inputs; `true` means exactly Q/K/V
/// bias inputs in that order; partial bias cannot be represented. Weight
/// dtype is not parameter state; future inference requires the three weight
/// TensorSpec dtypes to match.
struct QkvLinearParams {
    int64_t q_out_features = 0;
    int64_t k_out_features = 0;
    int64_t v_out_features = 0;
    bool has_bias = false;
};

/// @brief Semantic parameters for a fused MLP gate/up linear projection.
///
/// Packed weight rows are ordered Gate, then Up. `has_bias=true` would require
/// two bias inputs, which the current two-input schema does not represent.
struct GateUpLinearParams {
    int64_t gate_out_features = 0;
    int64_t up_out_features = 0;
    bool has_bias = false;
};

/// @brief Semantic parameters for fused add + RMS normalization.
///
/// Input order is `[input, residual, weight]` with semantics
/// `output = RmsNorm(input + residual, weight, eps)`; the second output
/// `new_residual` carries `input + residual` for the next block.
struct AddRmsNormParams {
    float eps = 1.0e-5F;
};

/// @brief Typed variant over all per-operator parameter structs.
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
                              PermuteParams,
                              ReorderParams,
                              QkvLinearParams,
                              AddRmsNormParams,
                              GateUpLinearParams>;

} // namespace aethermind

#endif
