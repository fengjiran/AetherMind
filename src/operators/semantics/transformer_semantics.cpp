#include "aethermind/dtypes/data_type.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/shape_symbol.h"
#include "aethermind/shape_inference/tensor_spec.h"
#include "operator_semantics_internal.h"

#include <span>
#include <string>
#include <vector>

namespace aethermind::detail {

namespace {
AM_NODISCARD bool IsRmsNormSupportedDType(const DataType& dtype) noexcept {
    return dtype.IsFloat32() || dtype.IsFloat16() || dtype.IsBFloat16() || dtype.IsFloat8();
}
}// namespace

StatusOr<InferenceResult> AnalyzeRmsNorm(const OpParams& /*params*/, std::span<const TensorSpec> inputs) {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "RmsNorm expects exactly 2 inputs, got " + std::to_string(inputs.size()));
    }

    const auto& input_spec = inputs[0];
    const auto& weight_spec = inputs[1];
    if (!IsRmsNormSupportedDType(input_spec.dtype)) {
        return Status::InvalidArgument(
                "RmsNorm input dtype not supported: " + ToString(input_spec.dtype));
    }
    if (!IsRmsNormSupportedDType(weight_spec.dtype)) {
        return Status::InvalidArgument(
                "RmsNorm weight dtype not supported: " + ToString(weight_spec.dtype));
    }

    if (!HasRank(input_spec.shape, 2)) {
        return Status::InvalidArgument("RmsNorm input must be rank-2 [seq_len, hidden]");
    }

    if (!HasRank(weight_spec.shape, 1)) {
        return Status::InvalidArgument("RmsNorm weight must be rank-1");
    }

    const ShapeSymbol& seq_len = input_spec.shape[0];
    if (!IsPositiveIfStatic(seq_len)) {
        return Status::InvalidArgument("RmsNorm seq_len must be positive");
    }

    const ShapeSymbol& hidden_size = input_spec.shape[1];
    if (!IsPositiveIfStatic(hidden_size)) {
        return Status::InvalidArgument("RmsNorm hidden size must be positive");
    }

    const ShapeSymbol& weight_len = weight_spec.shape[0];
    if (!IsPositiveIfStatic(weight_len)) {
        return Status::InvalidArgument("RmsNorm weight length must be positive");
    }

    if (!UnifyShapeSymbol(hidden_size, weight_len).ok()) {
        return Status::InvalidArgument(
                "RmsNorm weight length must equal input last dimension");
    }

    std::vector<ShapeConstraint> runtime_checks;
    if (hidden_size != weight_len) {
        runtime_checks.push_back({
                .condition = DimEqualConstraint{
                        .lhs = {
                                .tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 0},
                                .dim_index = 1,
                        },
                        .rhs = {
                                .tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = 1},
                                .dim_index = 0,
                        }},
                .error_context = "RmsNorm hidden dimension must match weight length",
        });
    }

    return InferenceResult{
            .outputs = {input_spec},
            .runtime_checks = std::move(runtime_checks),
    };
}

StatusOr<InferenceResult> AnalyzeRoPE(const OpParams& /*params*/, std::span<const TensorSpec> inputs) {
    if (inputs.size() != 3) {
        return Status::InvalidArgument(
                "RoPE expects exactly 3 inputs (q, k, position_ids), got " +
                std::to_string(inputs.size()));
    }

    const auto& q_spec = inputs[0];
    const auto& k_spec = inputs[1];
    const auto& pos_spec = inputs[2];

    if (q_spec.dtype != DataType::Float32() || k_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("RoPE only supports float32 q/k inputs in Phase 1");
    }

    if (pos_spec.dtype != DataType::Int(64)) {
        return Status::InvalidArgument("RoPE position_ids must be int64");
    }

    // GQA-compatible: q and k may have different last dimensions
    // (q_last = num_attention_heads * head_dim, k_last = num_key_value_heads * head_dim).
    // Only require rank equality and batch-dimension consistency.
    if (q_spec.shape.IsRanked() && k_spec.shape.IsRanked()) {
        if (q_spec.shape.rank() != k_spec.shape.rank()) {
            return Status::InvalidArgument("RoPE q and k must have the same rank");
        }

        if (const auto rank = q_spec.shape.rank().value(); rank >= 1) {
            for (size_t i = 0; i < rank - 1; ++i) {
                if (q_spec.shape[i] != k_spec.shape[i]) {
                    return Status::InvalidArgument(
                            "RoPE q and k batch dimensions must be identical");
                }
            }
        }
    }

    return InferenceResult{
            .outputs = {q_spec, k_spec},
    };
}

StatusOr<InferenceResult> AnalyzeAttention(const OpParams& /*params*/, std::span<const TensorSpec> inputs) {
    if (inputs.size() != 3) {
        return Status::InvalidArgument(
                "Attention expects exactly 3 inputs (q, kCache, vCache), got " + std::to_string(inputs.size()));
    }

    const auto& q_spec = inputs[0];

    if (q_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("Attention only supports float32 q input in Phase 1");
    }

    return InferenceResult{
            .outputs = {q_spec},
    };
}

StatusOr<InferenceResult> AnalyzeKVCacheUpdate(const OpParams& /*params*/, std::span<const TensorSpec> inputs) {
    if (inputs.size() != 4) {
        return Status::InvalidArgument(
                "KVCacheUpdate expects exactly 4 inputs (k, v, kCacheIn, vCacheIn), got " + std::to_string(inputs.size()));
    }

    const auto& k_spec = inputs[0];
    const auto& v_spec = inputs[1];
    const auto& k_cache_in_spec = inputs[2];
    const auto& v_cache_in_spec = inputs[3];

    if (k_spec.dtype != DataType::Float32() || v_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("KVCacheUpdate only supports float32 k/v inputs in Phase 1");
    }

    return InferenceResult{
            .outputs = {k_cache_in_spec, v_cache_in_spec},
    };
}

}// namespace aethermind::detail
