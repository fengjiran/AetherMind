#include "aethermind/operators/operator_inference.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/shape_inference/broadcast.h"

namespace aethermind {

StatusOr<InferenceResult> InferOperator(OpType op_type,
                                        const OpParams& params,
                                        std::span<const TensorSpec> inputs) {
    // Variant and parameter validation is performed at the beginning of each
    // detail::Infer* function dispatched below — there is no separate
    // pre-validation step. kUnknown and any out-of-range op_type are rejected
    // directly here.
    switch (op_type) {
        case OpType::kEmbedding:
            return detail::InferEmbedding(params, inputs);
        case OpType::kRmsNorm:
            return detail::InferRmsNorm(params, inputs);
        case OpType::kLinear:
            return detail::InferLinear(params, inputs);
        case OpType::kRoPE:
            return detail::InferRoPE(params, inputs);
        case OpType::kMatMul:
            return detail::InferMatMul(params, inputs);
        case OpType::kSoftmax:
            return detail::InferSoftmax(params, inputs);
        case OpType::kAdd:
            return detail::InferAdd(params, inputs);
        case OpType::kSiluMul:
            return detail::InferSiluMul(params, inputs);
        case OpType::kKVCacheUpdate:
            return detail::InferKVCacheUpdate(params, inputs);
        case OpType::kAttention:
            return detail::InferAttention(params, inputs);
        case OpType::kArgmax:
            return detail::InferArgmax(params, inputs);
        case OpType::kSilu:
            return detail::InferSilu(params, inputs);
        case OpType::kElementwiseMul:
            return detail::InferElementwiseMul(params, inputs);
        case OpType::kUnknown:
            return Status::InvalidArgument(
                    "Unknown op type cannot have validated graph params");
    }
    return Status::InvalidArgument("Unknown op type");
}

StatusOr<std::vector<TensorSpec>> MakeCompactInputSpecs(const OperatorSchema& schema,
                                                        std::span<const TensorSpec> all_inputs) {
    if (schema.input_ports.size() != all_inputs.size()) {
        return Status::InvalidArgument(
                "MakeCompactInputSpecs: input count mismatch, schema expects " +
                std::to_string(schema.input_ports.size()) + " inputs but got " +
                std::to_string(all_inputs.size()));
    }

    std::vector<TensorSpec> compact;
    compact.reserve(all_inputs.size());
    for (const auto& port: schema.input_ports) {
        if (port.index >= all_inputs.size()) {
            return Status::InvalidArgument(
                    "MakeCompactInputSpecs: port index " +
                    std::to_string(port.index) + " out of bounds for " +
                    std::to_string(all_inputs.size()) + " inputs");
        }
        if (port.contributes_tensor_spec) {
            compact.push_back(all_inputs[port.index]);
        }
    }
    return compact;
}

namespace detail {

StatusOr<InferenceResult> InferBroadcastBinary(const OpParams& /*params*/,
                                               std::span<const TensorSpec> inputs,
                                               std::string_view op_name) {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                std::string(op_name) + " requires exactly 2 inputs");
    }
    const TensorSpec& lhs_spec = inputs[0];
    const TensorSpec& rhs_spec = inputs[1];
    if (lhs_spec.dtype != DataType::Float32() && lhs_spec.dtype != DataType::BFloat(16)) {
        return Status::InvalidArgument(
                std::string(op_name) + " lhs must be float32 or bfloat16");
    }
    if (rhs_spec.dtype != DataType::Float32() && rhs_spec.dtype != DataType::BFloat(16)) {
        return Status::InvalidArgument(
                std::string(op_name) + " rhs must be float32 or bfloat16");
    }
    auto broadcast_result = InferBroadcastShape(
            lhs_spec.shape, rhs_spec.shape);
    if (!broadcast_result.ok()) {
        return broadcast_result.status();
    }

    InferenceResult result;
    result.outputs.push_back({lhs_spec.dtype, broadcast_result->output_shape});
    for (const auto& deferred: broadcast_result->deferred_axes) {
        result.runtime_checks.push_back(
                {DimBroadcastableConstraint{
                         {{TensorPortType::kInput, 0},
                          deferred.lhs_axis},
                         {{TensorPortType::kInput, 1},
                          deferred.rhs_axis}},
                 std::string(op_name) + " dimensions must be broadcastable"});
    }
    return result;
}

}// namespace detail

}// namespace aethermind
