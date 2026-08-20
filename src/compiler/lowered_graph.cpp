#include "aethermind/compiler/lowered_graph.h"
#include "aethermind/operators/operator_schema.h"

#include <algorithm>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace aethermind {
namespace {

Status ValidateValueId(const LoweredGraph& lowered,
                       GraphValueId id,
                       std::string_view context) {
    if (id.index >= lowered.values().size()) {
        return Status::Internal("ValidateLoweredGraph: invalid GraphValueId for " +
                                std::string(context));
    }
    return Status::Ok();
}

Status ValidateStateAlias(const LoweredGraph& lowered,
                          const LoweredStateAlias& alias,
                          size_t alias_index,
                          std::set<std::tuple<size_t, uint32_t, uint32_t>>& pairs,
                          std::set<std::pair<size_t, uint32_t>>& inputs,
                          std::set<std::pair<size_t, uint32_t>>& outputs) {
    const std::string prefix = "ValidateLoweredGraph: state alias " +
                               std::to_string(alias_index) + ": ";
    if (alias.step_index >= lowered.steps().size()) {
        return Status::Internal(prefix + "step index out of range");
    }

    const LoweredStep& step = lowered.steps()[alias.step_index];
    const auto schema = GetOperatorSchema(step.spec.op_type);
    if (!schema.ok()) {
        return Status::Internal(prefix + "step has no registered operator schema");
    }

    if (alias.input_port >= schema->input_ports.size() ||
        alias.output_port >= schema->output_ports.size()) {
        return Status::Internal(prefix + "schema port index out of range");
    }

    if (schema->input_ports[alias.input_port].kind != OperatorPortKind::kState ||
        schema->output_ports[alias.output_port].kind != OperatorPortKind::kState) {
        return Status::Internal(prefix + "alias ports must both be state ports");
    }

    bool declared = std::ranges::any_of(
            schema->state_alias_ports,
            [&schema, &alias](const StateAliasPortPair& pair) {
                const auto input = FindInputPortIndex(*schema, pair.input_port);
                const auto output = FindOutputPortIndex(*schema, pair.output_port);
                return input.ok() && output.ok() && input.value() == alias.input_port &&
                       output.value() == alias.output_port;
            });

    if (!declared) {
        return Status::Internal(
                prefix + "alias is not declared by the operator schema");
    }

    if (alias.input_port >= step.binding.input_values.size() ||
        step.binding.input_values[alias.input_port] != alias.input ||
        alias.output_port >= step.binding.output_values.size() ||
        step.binding.output_values[alias.output_port] != alias.output) {
        return Status::Internal(
                prefix + "binding does not match its recorded GraphValueIds");
    }
    AM_RETURN_IF_ERROR(ValidateValueId(lowered, alias.input, "state alias input"));
    AM_RETURN_IF_ERROR(ValidateValueId(lowered, alias.output, "state alias output"));

    const auto* input_state = std::get_if<StateValue>(&lowered.values()[alias.input.index].payload);
    const auto* output_state = std::get_if<StateValue>(&lowered.values()[alias.output.index].payload);
    if (input_state == nullptr || output_state == nullptr) {
        return Status::Internal(
                prefix + "aliased values must both carry StateValue payloads");
    }

    if (input_state->binding != output_state->binding) {
        return Status::Internal(
                prefix + "state bindings must describe the same state slot");
    }

    if (const auto pair =
                std::make_tuple(alias.step_index, alias.input_port, alias.output_port);
        !pairs.insert(pair).second) {
        return Status::Internal(prefix + "duplicate alias declaration");
    }

    if (!inputs.insert({alias.step_index, alias.input_port}).second ||
        !outputs.insert({alias.step_index, alias.output_port}).second) {
        return Status::Internal(prefix + "conflicts with another alias port");
    }
    return Status::Ok();
}

}// namespace

Status LoweredGraph::Builder::Validate() const {
    LoweredGraph lowered;
    lowered.steps_ = steps;
    lowered.values_ = values;
    lowered.model_inputs_ = model_inputs;
    lowered.model_outputs_ = model_outputs;
    lowered.state_aliases_ = state_aliases;
    return ValidateLoweredGraph(lowered);
}

StatusOr<LoweredGraph> LoweredGraph::Builder::Build() && {
    LoweredGraph lowered;
    lowered.steps_ = std::move(steps);
    lowered.values_ = std::move(values);
    lowered.model_inputs_ = std::move(model_inputs);
    lowered.model_outputs_ = std::move(model_outputs);
    lowered.state_aliases_ = std::move(state_aliases);
    AM_RETURN_IF_ERROR(ValidateLoweredGraph(lowered));
    return lowered;
}

Status ValidateLoweredGraph(const LoweredGraph& lowered) {
    std::unordered_set<uint32_t> nodes;
    for (size_t i = 0; i < lowered.steps().size(); ++i) {
        const auto& step = lowered.steps()[i];
        const auto schema = GetOperatorSchema(step.spec.op_type);
        if (!schema.ok()) {
            return Status::Internal(
                    "ValidateLoweredGraph: step " + std::to_string(i) +
                    " has no registered operator schema");
        }

        if (!nodes.insert(step.binding.node.index).second) {
            return Status::Internal(
                    "ValidateLoweredGraph: duplicate GraphNodeId in steps");
        }

        if (step.binding.input_values.size() != schema->input_ports.size() ||
            step.spec.input_specs.size() != schema->input_ports.size()) {
            return Status::Internal(
                    "ValidateLoweredGraph: step " + std::to_string(i) +
                    " input binding/spec arity differs from schema");
        }

        if (step.binding.output_values.size() != schema->output_ports.size() ||
            step.spec.output_specs.size() != schema->output_ports.size()) {
            return Status::Internal(
                    "ValidateLoweredGraph: step " + std::to_string(i) +
                    " output binding/spec arity differs from schema");
        }

        if (step.spec.selector.act_dtype.IsUndefined() ||
            step.spec.selector.weight_dtype.IsUndefined()) {
            return Status::Internal(
                    "ValidateLoweredGraph: step " + std::to_string(i) +
                    " has undefined selector dtype");
        }

        for (size_t j = 0; j < step.binding.input_values.size(); ++j) {
            const auto id = step.binding.input_values[j];
            AM_RETURN_IF_ERROR(ValidateValueId(lowered, id, "step input"));
            if (step.spec.input_specs[j] != lowered.values()[id.index].spec) {
                return Status::Internal(
                        "ValidateLoweredGraph: input spec does not match value metadata");
            }
        }

        for (size_t j = 0; j < step.binding.output_values.size(); ++j) {
            const GraphValueId id = step.binding.output_values[j];
            AM_RETURN_IF_ERROR(ValidateValueId(lowered, id, "step output"));
            if (step.spec.output_specs[j] != lowered.values()[id.index].spec) {
                return Status::Internal(
                        "ValidateLoweredGraph: output spec does not match value metadata");
            }
        }
    }

    for (const auto id: lowered.model_inputs()) {
        AM_RETURN_IF_ERROR(ValidateValueId(lowered, id, "model input"));
        if (!std::holds_alternative<ModelInputValue>(lowered.values()[id.index].payload)) {
            return Status::Internal(
                    "ValidateLoweredGraph: model input does not carry ModelInputValue");
        }
    }

    for (const auto id: lowered.model_outputs()) {
        AM_RETURN_IF_ERROR(ValidateValueId(lowered, id, "model output"));
    }

    std::set<std::tuple<size_t, uint32_t, uint32_t>> pairs;
    std::set<std::pair<size_t, uint32_t>> inputs;
    std::set<std::pair<size_t, uint32_t>> outputs;
    for (size_t i = 0; i < lowered.state_aliases().size(); ++i) {
        AM_RETURN_IF_ERROR(ValidateStateAlias(
                lowered, lowered.state_aliases()[i], i, pairs, inputs, outputs));
    }
    return Status::Ok();
}

}// namespace aethermind
