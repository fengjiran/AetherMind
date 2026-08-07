#include "aethermind/operators/attention_op.h"
#include "aethermind/operators/operator_inference.h"
#include "utils/overflow_check.h"

namespace aethermind::detail {

namespace {

// Validates AttentionParams fields. Must be called before any input-dependent
// checks since it depends only on params.
Status ValidateAttentionParams(const AttentionParams& p) {
    if (p.num_attention_heads <= 0) {
        return Status::InvalidArgument(
                "Attention num_attention_heads must be positive");
    }

    if (p.num_key_value_heads <= 0) {
        return Status::InvalidArgument(
                "Attention num_key_value_heads must be positive");
    }

    if (p.head_dim <= 0) {
        return Status::InvalidArgument("Attention head_dim must be positive");
    }

    if (p.num_attention_heads % p.num_key_value_heads != 0) {
        return Status::InvalidArgument(
                "Attention num_attention_heads must be divisible by "
                "num_key_value_heads");
    }

    if (int64_t hidden = 0; CheckOverflowMul(p.num_attention_heads, p.head_dim, &hidden)) {
        return Status::InvalidArgument(
                "Attention num_attention_heads * head_dim overflows int64_t");
    }
    return Status::Ok();
}

// Validates that q, k_cache, and v_cache share the same dtype from the
// supported set. Output dtype follows q.
Status ValidateAttentionDTypes(std::span<const TensorSpec> inputs) {
    const TensorSpec& q = inputs[0];
    const TensorSpec& k_cache = inputs[1];
    const TensorSpec& v_cache = inputs[2];

    if (!IsAttentionSupportedDType(q.dtype)) {
        return Status::InvalidArgument(
                MakeAttentionUnsupportedDTypeMessage("Attention q"));
    }

    if (k_cache.dtype != q.dtype) {
        return Status::InvalidArgument(
                "Attention k_cache dtype must match q dtype");
    }

    if (v_cache.dtype != q.dtype) {
        return Status::InvalidArgument(
                "Attention v_cache dtype must match q dtype");
    }
    return Status::Ok();
}

// Validates rank, static shape equations, and symbolic dim rules.
// Populates runtime_checks with DimPositiveConstraint for symbolic seq_len.
Status ValidateAttentionShapes(const AttentionParams& params,
                               std::span<const TensorSpec> inputs,
                               std::vector<ShapeConstraint>& runtime_checks) {
    const SymbolicShape& q_shape = inputs[0].shape;
    const SymbolicShape& k_cache_shape = inputs[1].shape;
    const SymbolicShape& v_cache_shape = inputs[2].shape;

    if (!HasRank(q_shape, 2)) {
        return Status::InvalidArgument(
                "Attention q must be rank 2 [seq_len, hidden]");
    }

    if (!HasRank(k_cache_shape, 3)) {
        return Status::InvalidArgument("Attention k_cache must be rank 3 "
                                       "[num_key_value_heads, cache_len, head_dim]");
    }

    if (!HasRank(v_cache_shape, 3)) {
        return Status::InvalidArgument("Attention v_cache must be rank 3 "
                                       "[num_key_value_heads, cache_len, head_dim]");
    }

    const ShapeSymbol& q_seq_len = q_shape[0];
    const ShapeSymbol& q_hidden = q_shape[1];
    const ShapeSymbol& kv_heads = k_cache_shape[0];
    const ShapeSymbol& cache_len = k_cache_shape[1];
    const ShapeSymbol& head_dim = k_cache_shape[2];

    // Static equation: q.shape[1] == num_attention_heads * head_dim.
    // Overflow already rejected by ValidateAttentionParams.
    if (q_hidden.IsStatic() &&
        q_hidden.GetStaticValue() != params.num_attention_heads * params.head_dim) {
        return Status::InvalidArgument("Attention q hidden dim must equal "
                                       "num_attention_heads * head_dim");
    }

    // Static equation: k_cache.shape[0] == num_key_value_heads.
    if (kv_heads.IsStatic() &&
        kv_heads.GetStaticValue() != params.num_key_value_heads) {
        return Status::InvalidArgument(
                "Attention k_cache num_kv_heads must match params");
    }

    // Static equation: k_cache.shape[2] == head_dim.
    if (head_dim.IsStatic() &&
        head_dim.GetStaticValue() != params.head_dim) {
        return Status::InvalidArgument(
                "Attention k_cache head_dim must match params");
    }

    // k_cache.shape == v_cache.shape (per-dim AreProvablyEqual).
    for (size_t i = 0; i < 3; ++i) {
        if (!AreProvablyEqual(k_cache_shape[i], v_cache_shape[i])) {
            return Status::InvalidArgument(
                    "Attention k_cache and v_cache shapes must be equal");
        }
    }

    // Static zero seq_len rejected.
    if (q_seq_len.IsStatic() && q_seq_len.GetStaticValue() <= 0) {
        return Status::InvalidArgument(
                "Attention q seq_len must be positive when static");
    }

    // Static zero cache_len rejected.
    if (cache_len.IsStatic() && cache_len.GetStaticValue() == 0) {
        return Status::InvalidArgument(
                "Attention cache_len must not be statically zero");
    }

    // Symbolic q seq_len: emit exactly one DimPositiveConstraint
    // (input port 0, dim 0).
    if (!q_seq_len.IsStatic()) {
        runtime_checks.emplace_back(
                DimPositiveConstraint{
                        {.tensor_port = {.direction = TensorPortType::kInput,
                                         .tensor_idx = 0},
                         .dim_index = 0}},
                "Attention q seq_len must be positive");
    }

    return Status::Ok();
}

}// namespace

StatusOr<InferenceResult> InferAttention(const OpParams& params,
                                         std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<AttentionParams>(params)) {
        return Status::InvalidArgument("Attention node requires AttentionParams");
    }
    const auto& attn_params = std::get<AttentionParams>(params);
    AM_RETURN_IF_ERROR(ValidateAttentionParams(attn_params));
    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kAttention, inputs));
    AM_RETURN_IF_ERROR(ValidateAttentionDTypes(inputs));

    std::vector<ShapeConstraint> runtime_checks;
    AM_RETURN_IF_ERROR(ValidateAttentionShapes(attn_params, inputs, runtime_checks));

    const auto& q_spec = inputs[0];
    return InferenceResult{
            .outputs = {q_spec},
            .runtime_checks = std::move(runtime_checks),
    };
}

}// namespace aethermind::detail
