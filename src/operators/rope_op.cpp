#include "aethermind/operators/rope_op.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/utils/overflow_check.h"

namespace aethermind::detail {

namespace {

// Validates RoPEParams scalar invariants. Must be called before any
// input-dependent checks since it depends only on params.
Status ValidateRoPEParams(const RoPEParams& p) {
    if (p.head_dim <= 0) {
        return Status::InvalidArgument("RoPE head_dim must be positive");
    }

    if (p.num_attention_heads <= 0) {
        return Status::InvalidArgument(
                "RoPE num_attention_heads must be positive");
    }

    if (p.num_key_value_heads <= 0) {
        return Status::InvalidArgument(
                "RoPE num_key_value_heads must be positive");
    }

    if (p.max_position_embeddings <= 0) {
        return Status::InvalidArgument(
                "RoPE max_position_embeddings must be positive");
    }

    if (p.head_dim % 2 != 0) {
        return Status::InvalidArgument("RoPE head_dim must be even");
    }

    if (!std::isfinite(p.theta) || p.theta <= 0.0) {
        return Status::InvalidArgument("RoPE theta must be finite and positive");
    }

    if (int64_t hidden = 0; CheckOverflowMul(p.num_attention_heads, p.head_dim, &hidden)) {
        return Status::InvalidArgument(
                "RoPE num_attention_heads * head_dim overflows int64_t");
    }

    if (int64_t hidden = 0; CheckOverflowMul(p.num_key_value_heads, p.head_dim, &hidden)) {
        return Status::InvalidArgument(
                "RoPE num_key_value_heads * head_dim overflows int64_t");
    }
    return Status::Ok();
}

// Validates the RoPE scaling tuple. kNone requires an absent factor; kLinear
// requires a present finite factor > 0. The RoPEScalingType enum is exhaustive
// over the representable surface; HF-only variants are filtered by the model
// frontend before RoPEParams is constructed, so no `default` branch is needed.
Status ValidateRoPEScaling(const RoPEParams& p) {
    switch (p.scaling_type) {
        case RoPEScalingType::kNone:
            if (p.scaling_factor.has_value()) {
                return Status::InvalidArgument(
                        "RoPE scaling_type kNone must not carry a scaling_factor");
            }
            return Status::Ok();
        case RoPEScalingType::kLinear:
            if (!p.scaling_factor.has_value()) {
                return Status::InvalidArgument(
                        "RoPE scaling_type kLinear requires a finite positive "
                        "scaling_factor");
            }
            if (!std::isfinite(*p.scaling_factor)) {
                return Status::InvalidArgument(
                        "RoPE scaling_type kLinear scaling_factor must be finite");
            }
            if (*p.scaling_factor <= 0.0) {
                return Status::InvalidArgument(
                        "RoPE scaling_type kLinear scaling_factor must be positive");
            }
            return Status::Ok();
    }
    AM_UNREACHABLE();
}

// Validates that q and k share the same dtype from the supported set and that
// position_ids is Int64. Output dtype follows q (and k respectively).
Status ValidateRoPEDTypes(std::span<const TensorSpec> inputs) {
    const TensorSpec& q = inputs[0];
    const TensorSpec& k = inputs[1];
    const TensorSpec& position_ids = inputs[2];

    if (!IsRoPESupportedDType(q.dtype)) {
        return Status::InvalidArgument(MakeRoPEUnsupportedDTypeMessage("RoPE q"));
    }

    if (k.dtype != q.dtype) {
        return Status::InvalidArgument("RoPE k dtype must match q dtype");
    }

    if (position_ids.dtype != DataType::Int(64)) {
        return Status::InvalidArgument("RoPE position_ids must be int64");
    }
    return Status::Ok();
}
// Validates rank, static width equations, and sequence reconciliation.
// Populates runtime_checks with deferred shape constraints: at most one
// DimPositiveConstraint for symbolic q seq_len followed by up to two
// DimEqualConstraint checks reconciling q/k and q/position sequence dims.
Status ValidateRoPEShapes(const RoPEParams& params,
                          std::span<const TensorSpec> inputs,
                          std::vector<ShapeConstraint>& runtime_checks) {
    const SymbolicShape& q_shape = inputs[0].shape;
    const SymbolicShape& k_shape = inputs[1].shape;
    const SymbolicShape& pos_shape = inputs[2].shape;

    if (!HasRank(q_shape, 2)) {
        return Status::InvalidArgument(
                "RoPE q must be rank 2 [seq_len, hidden]");
    }

    if (!HasRank(k_shape, 2)) {
        return Status::InvalidArgument(
                "RoPE k must be rank 2 [seq_len, hidden]");
    }

    if (!HasRank(pos_shape, 1)) {
        return Status::InvalidArgument(
                "RoPE position_ids must be rank 1 [seq_len]");
    }

    const ShapeSymbol& q_seq_len = q_shape[0];
    const ShapeSymbol& q_hidden = q_shape[1];
    const ShapeSymbol& k_seq_len = k_shape[0];
    const ShapeSymbol& kv_hidden = k_shape[1];
    const ShapeSymbol& pos_seq_len = pos_shape[0];

    // Static width equations (overflow already rejected by ValidateRoPEParams).
    if (q_hidden.IsStatic() &&
        q_hidden.GetStaticValue() != params.num_attention_heads * params.head_dim) {
        return Status::InvalidArgument(
                "RoPE q hidden dim must equal num_attention_heads * head_dim");
    }

    if (kv_hidden.IsStatic() &&
        kv_hidden.GetStaticValue() != params.num_key_value_heads * params.head_dim) {
        return Status::InvalidArgument(
                "RoPE k hidden dim must equal num_key_value_heads * head_dim");
    }

    // Static q seq_len <= 0 rejected.
    if (q_seq_len.IsStatic() && q_seq_len.GetStaticValue() <= 0) {
        return Status::InvalidArgument(
                "RoPE q seq_len must be positive when static");
    }

    // Symbolic q seq_len: emit exactly one DimPositiveConstraint
    // (input port 0, dim 0) first.
    if (!q_seq_len.IsStatic()) {
        runtime_checks.emplace_back(
                DimPositiveConstraint{
                        {.tensor_port = {.direction = TensorPortType::kInput,
                                         .tensor_idx = 0},
                         .dim_index = 0}},
                "RoPE q seq_len must be positive");
    }

    // Reconcile q[0] with k[0].
    if (!AreProvablyEqual(q_seq_len, k_seq_len)) {
        if (q_seq_len.IsStatic() && k_seq_len.IsStatic()) {
            return Status::InvalidArgument(
                    "RoPE q and k seq_len must be equal");
        }
        runtime_checks.emplace_back(
                DimEqualConstraint{
                        {.tensor_port = {.direction = TensorPortType::kInput,
                                         .tensor_idx = 0},
                         .dim_index = 0},
                        {.tensor_port = {.direction = TensorPortType::kInput,
                                         .tensor_idx = 1},
                         .dim_index = 0}},
                "RoPE q and k seq_len must be equal");
    }

    // Reconcile q[0] with position_ids[0].
    if (!AreProvablyEqual(q_seq_len, pos_seq_len)) {
        if (q_seq_len.IsStatic() && pos_seq_len.IsStatic()) {
            return Status::InvalidArgument(
                    "RoPE q seq_len and position_ids length must be equal");
        }
        runtime_checks.emplace_back(
                DimEqualConstraint{
                        {.tensor_port = {.direction = TensorPortType::kInput,
                                         .tensor_idx = 0},
                         .dim_index = 0},
                        {.tensor_port = {.direction = TensorPortType::kInput,
                                         .tensor_idx = 2},
                         .dim_index = 0}},
                "RoPE q seq_len and position_ids length must be equal");
    }

    return Status::Ok();
}

}// namespace
StatusOr<InferenceResult> InferRoPE(const OpParams& params,
                                    std::span<const TensorSpec> inputs) {
    const auto* rope_params = std::get_if<RoPEParams>(&params);
    if (rope_params == nullptr) {
        return Status::InvalidArgument("RoPE node requires RoPEParams");
    }

    AM_RETURN_IF_ERROR(ValidateRoPEParams(*rope_params));
    AM_RETURN_IF_ERROR(ValidateRoPEScaling(*rope_params));
    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kRoPE, inputs));
    AM_RETURN_IF_ERROR(ValidateRoPEDTypes(inputs));

    std::vector<ShapeConstraint> runtime_checks;
    AM_RETURN_IF_ERROR(ValidateRoPEShapes(*rope_params, inputs, runtime_checks));

    const auto& q_spec = inputs[0];
    const auto& k_spec = inputs[1];
    return InferenceResult{
            .outputs = {q_spec, k_spec},
            .runtime_checks = std::move(runtime_checks),
    };
}

}// namespace aethermind::detail