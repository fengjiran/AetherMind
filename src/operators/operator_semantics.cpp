#include "aethermind/operators/operator_semantics.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/model/graph/operator_schema.h"
#include "semantics/operator_semantics_internal.h"

#include <cmath>

namespace aethermind {
namespace {

template<typename Params>
Status RequireParams(const OpParams& params, const char* message) {
    if (!std::holds_alternative<Params>(params)) {
        return Status::InvalidArgument(message);
    }
    return Status::Ok();
}

Status ValidateRmsNormParams(const OpParams& params) {
    const auto* typed = std::get_if<RmsNormParams>(&params);
    if (typed == nullptr) {
        return Status::InvalidArgument("RmsNorm node requires RmsNormParams");
    }

    if (!std::isfinite(typed->eps) || typed->eps <= 0.0F) {
        return Status::InvalidArgument("RmsNormParams eps must be finite and positive");
    }
    return Status::Ok();
}

Status ValidateRoPEParams(const OpParams& params) {
    const auto* typed = std::get_if<RoPEParams>(&params);
    if (typed == nullptr) {
        return Status::InvalidArgument("RoPE node requires RoPEParams");
    }

    if (typed->head_dim <= 0 || typed->num_attention_heads <= 0 ||
        typed->num_key_value_heads <= 0 || typed->max_position_embeddings <= 0) {
        return Status::InvalidArgument("RoPEParams dimensions must be positive");
    }

    if (!std::isfinite(typed->theta) || typed->theta <= 0.0) {
        return Status::InvalidArgument("RoPEParams theta must be finite and positive");
    }
    return Status::Ok();
}

}// namespace

Status ValidateOperatorParams(OpType op_type, const OpParams& params) {
    switch (op_type) {
        case OpType::kEmbedding:
            return RequireParams<EmbeddingParams>(
                    params, "Embedding node requires EmbeddingParams");
        case OpType::kRmsNorm:
            return ValidateRmsNormParams(params);
        case OpType::kLinear:
            return RequireParams<LinearParams>(params, "Linear node requires LinearParams");
        case OpType::kRoPE:
            return ValidateRoPEParams(params);
        case OpType::kMatMul:
            return RequireParams<MatMulParams>(params, "MatMul node requires MatMulParams");
        case OpType::kSoftmax:
            return RequireParams<SoftmaxParams>(params, "Softmax node requires SoftmaxParams");
        case OpType::kAdd:
            return RequireParams<AddParams>(params, "Add node requires AddParams");
        case OpType::kSiluMul:
            return RequireParams<SiluMulParams>(params, "SiluMul node requires SiluMulParams");
        case OpType::kKVCacheUpdate:
            return RequireParams<KVCacheUpdateParams>(
                    params, "KVCacheUpdate node requires KVCacheUpdateParams");
        case OpType::kAttention:
            return RequireParams<AttentionParams>(
                    params, "Attention node requires AttentionParams");
        case OpType::kArgmax:
            return RequireParams<ArgmaxParams>(params, "Argmax node requires ArgmaxParams");
        case OpType::kSilu:
            return RequireParams<SiluParams>(params, "Silu node requires SiluParams");
        case OpType::kElementwiseMul:
            return RequireParams<ElementwiseMulParams>(
                    params, "ElementwiseMul node requires ElementwiseMulParams");
        case OpType::kUnknown:
            return Status::InvalidArgument("Unknown op type cannot have validated graph params");
    }
    return Status::InvalidArgument("Unknown op type");
}

StatusOr<InferenceResult> AnalyzeOperator(OpType op_type,
                                          const OpParams& params,
                                          std::span<const TensorSpec> inputs) {
    AM_RETURN_IF_ERROR(ValidateOperatorParams(op_type, params));
    switch (op_type) {
        case OpType::kEmbedding:
            return detail::AnalyzeEmbedding(params, inputs);
        case OpType::kRmsNorm:
            return detail::AnalyzeRmsNorm(params, inputs);
        case OpType::kLinear:
            return detail::AnalyzeLinear(params, inputs);
        case OpType::kRoPE:
            return detail::AnalyzeRoPE(params, inputs);
        case OpType::kMatMul:
            return detail::AnalyzeMatMul(params, inputs);
        case OpType::kSoftmax:
            return detail::AnalyzeSoftmax(params, inputs);
        case OpType::kAdd:
            return detail::AnalyzeAdd(params, inputs);
        case OpType::kSiluMul:
            return detail::AnalyzeSiluMul(params, inputs);
        case OpType::kKVCacheUpdate:
            return detail::AnalyzeKVCacheUpdate(params, inputs);
        case OpType::kAttention:
            return detail::AnalyzeAttention(params, inputs);
        case OpType::kArgmax:
            return detail::AnalyzeArgmax(params, inputs);
        case OpType::kSilu:
            return detail::AnalyzeSilu(params, inputs);
        case OpType::kElementwiseMul:
            return detail::AnalyzeElementwiseMul(params, inputs);
        case OpType::kUnknown:
            return Status::InvalidArgument("Unknown op type cannot be analyzed");
    }
    return Status::InvalidArgument("Unknown op type");
}

StatusOr<std::vector<TensorSpec>> MakeCompactInputSpecs(const OperatorSchema& schema,
                                                        std::span<const TensorSpec> all_inputs) {
    if (schema.input_ports.size() != all_inputs.size()) {
        return Status::InvalidArgument(
                "MakeCompactInputSpecs: input count mismatch, schema expects " + std::to_string(schema.input_ports.size()) + " inputs but got " + std::to_string(all_inputs.size()));
    }
    std::vector<TensorSpec> compact;
    compact.reserve(all_inputs.size());
    for (const auto& port: schema.input_ports) {
        if (port.index >= all_inputs.size()) {
            return Status::InvalidArgument(
                    "MakeCompactInputSpecs: port index " + std::to_string(port.index) + " out of bounds for " + std::to_string(all_inputs.size()) + " inputs");
        }
        if (port.contributes_tensor_spec) {
            compact.push_back(all_inputs[port.index]);
        }
    }
    return compact;
}

}// namespace aethermind
