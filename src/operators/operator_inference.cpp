#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/op_params.h"

#include <optional>
#include <string>

namespace aethermind {
namespace {

// Records a role candidate dtype, rejecting undefined dtypes and conflicting
// dtypes across ports of the same role (activation/weight).
Status SetSelectorDTypeCandidate(const TensorSpec& spec,
                                 std::optional<DataType>& candidate,
                                 std::string_view role) {
    if (spec.dtype.IsUndefined()) {
        return Status::InvalidArgument(
                "undefined " + std::string(role) + " dtype");
    }

    if (candidate.has_value() && *candidate != spec.dtype) {
        return Status::InvalidArgument(
                "inconsistent " + std::string(role) + " dtypes in one operator");
    }
    candidate = spec.dtype;
    return Status::Ok();
}

} // namespace

StatusOr<InferenceResult> InferOperator(OpType op_type,
                                        const OpParams& params,
                                        std::span<const TensorSpec> inputs) {
    // Variant/parameter validation and input count validation are performed at
    // the beginning of each detail::Infer* function dispatched below. Variant
    // validation precedes input count validation so that a wrong OpParams
    // variant is reported before any structural input check.
    switch (op_type) {
        case OpType::kEmbedding:
            return detail::InferEmbedding(params, inputs);
        case OpType::kRmsNorm:
            return detail::InferRmsNorm(params, inputs);
        case OpType::kAddRmsNorm:
            return detail::InferAddRmsNorm(params, inputs);
        case OpType::kLinear:
            return detail::InferLinear(params, inputs);
        case OpType::kQkvLinear:
            return detail::InferQkvLinear(params, inputs);
        case OpType::kGateUpLinear:
            return detail::InferGateUpLinear(params, inputs);
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
        case OpType::kReshape:
            return detail::InferReshape(params, inputs);
        case OpType::kPermute:
            return detail::InferPermute(params, inputs);
        case OpType::kReorder:
            return detail::InferReorder(params, inputs);
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
    for (size_t i = 0; i < schema.input_ports.size(); ++i) {
        if (schema.input_ports[i].contributes_tensor_spec) {
            compact.push_back(all_inputs[i]);
        }
    }
    return compact;
}

StatusOr<SelectorDTypes> DeriveSelectorDTypes(const OperatorSchema& schema,
                                              std::span<const TensorSpec> input_specs,
                                              std::span<const TensorSpec> output_specs) {
    if (input_specs.size() != schema.input_ports.size() ||
        output_specs.size() != schema.output_ports.size()) {
        return Status::InvalidArgument(
                "DeriveSelectorDTypes: spec count does not match schema ports");
    }

    std::optional<DataType> act_dtype;
    std::optional<DataType> weight_dtype;
    for (size_t i = 0; i < schema.input_ports.size(); ++i) {
        const auto& port = schema.input_ports[i];
        if (!port.contributes_tensor_spec) {
            continue;
        }

        if (port.kind == OperatorPortKind::kActivation) {
            AM_RETURN_IF_ERROR(SetSelectorDTypeCandidate(
                    input_specs[i], act_dtype, "activation"));
        } else if (port.kind == OperatorPortKind::kWeight) {
            AM_RETURN_IF_ERROR(SetSelectorDTypeCandidate(
                    input_specs[i], weight_dtype, "weight"));
        }
    }

    if (!act_dtype.has_value()) {
        for (size_t i = 0; i < schema.output_ports.size(); ++i) {
            if (schema.output_ports[i].kind == OperatorPortKind::kActivation) {
                AM_RETURN_IF_ERROR(SetSelectorDTypeCandidate(
                        output_specs[i], act_dtype, "activation"));
            }
        }
    }

    if (!act_dtype.has_value()) {
        return Status::InvalidArgument(
                "operator schema has no contributing activation dtype source");
    }

    return SelectorDTypes{
            .act_dtype = *act_dtype,
            .weight_dtype = weight_dtype.value_or(*act_dtype),
    };
}

Status ValidateInferenceInputCount(OpType op_type,
                                   std::span<const TensorSpec> inputs) {
    auto schema = GetOperatorSchema(op_type);
    AM_RETURN_IF_ERROR(schema.status());

    if (inputs.size() != schema->input_ports.size()) {
        return Status::InvalidArgument(
                std::string(ToString(op_type)) + " expects exactly " +
                std::to_string(schema->input_ports.size()) + " inputs, got " +
                std::to_string(inputs.size()));
    }
    return Status::Ok();
}
} // namespace aethermind
