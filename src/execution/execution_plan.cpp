#include "aethermind/execution/execution_plan.h"
#include "aethermind/operators/operator_schema.h"
#include "utils/variant_utils.h"

#include <algorithm>
#include <atomic>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace aethermind {
namespace {

std::atomic<uint64_t> g_next_binding_key{1};

ExecutionPlanBindingKey MakeBindingKey() noexcept {
    uint64_t value = g_next_binding_key.fetch_add(1, std::memory_order_relaxed);
    if (value == 0) {
        value = g_next_binding_key.fetch_add(1, std::memory_order_relaxed);
    }
    return {.value = value};
}

Status ValidateRuntimeCheckReferences(const ShapeConstraint& check,
                                      const std::vector<TensorSpec>& input_specs,
                                      const std::vector<TensorSpec>& output_specs) {
    auto validate_port = [&](const TensorPort& port) -> Status {
        switch (port.direction) {
            case TensorPortType::kInput:
                if (port.tensor_idx >= input_specs.size()) {
                    return Status::InvalidArgument(
                            "Runtime check references input tensor " +
                            std::to_string(port.tensor_idx) +
                            " beyond the step's compact input spec range");
                }
                return Status::Ok();
            case TensorPortType::kOutput:
                if (port.tensor_idx >= output_specs.size()) {
                    return Status::InvalidArgument(
                            "Runtime check references output tensor " +
                            std::to_string(port.tensor_idx) +
                            " beyond the step's compact output spec range");
                }
                return Status::Ok();
        }
        return Status::InvalidArgument(
                "Runtime check references a tensor port with an invalid direction");
    };

    auto validate_dim = [&](const DimLocator& locator) -> Status {
        AM_RETURN_IF_ERROR(validate_port(locator.tensor_port));
        const SymbolicShape& shape = locator.tensor_port.direction == TensorPortType::kInput
                                             ? input_specs[locator.tensor_port.tensor_idx].shape
                                             : output_specs[locator.tensor_port.tensor_idx].shape;
        if (const auto rank = shape.rank();
            rank.has_value() && locator.dim_index >= *rank) {
            return Status::InvalidArgument(
                    "Runtime check references dimension " +
                    std::to_string(locator.dim_index) +
                    " beyond the referenced spec's rank");
        }
        return Status::Ok();
    };

    const auto visitor = overloaded{
            [&](const DimEqualConstraint& c) {
                AM_RETURN_IF_ERROR(validate_dim(c.lhs));
                return validate_dim(c.rhs);
            },
            [&](const DimBroadcastableConstraint& c) {
                AM_RETURN_IF_ERROR(validate_dim(c.lhs));
                return validate_dim(c.rhs);
            },
            [&](const VolumeEqualConstraint& c) {
                for (const DimLocator& dim: c.lhs_dims) {
                    AM_RETURN_IF_ERROR(validate_dim(dim));
                }

                for (const DimLocator& dim: c.rhs_dims) {
                    AM_RETURN_IF_ERROR(validate_dim(dim));
                }
                return Status::Ok();
            },
            [&](const RankEqualConstraint& c) {
                return validate_port(c.port);
            },
            [&](const RankAtLeastConstraint& c) {
                return validate_port(c.port);
            },
            [&](const DimPositiveConstraint& c) {
                return validate_dim(c.dim);
            }};
    return std::visit(visitor, check.condition);
}

bool MatchesInputKind(ExecutionValueKind value_kind, OperatorPortKind port_kind) noexcept {
    switch (port_kind) {
        case OperatorPortKind::kModelInput:
            return value_kind == ExecutionValueKind::kModelInput;
        case OperatorPortKind::kActivation:
            return value_kind == ExecutionValueKind::kActivation ||
                   value_kind == ExecutionValueKind::kModelInput ||
                   value_kind == ExecutionValueKind::kConstant;
        case OperatorPortKind::kWeight:
            return value_kind == ExecutionValueKind::kWeight;
        case OperatorPortKind::kConstant:
            return value_kind == ExecutionValueKind::kConstant;
        case OperatorPortKind::kState:
            return value_kind == ExecutionValueKind::kState;
    }
    return false;
}

bool MatchesOutputKind(ExecutionValueKind value_kind, OperatorPortKind port_kind) noexcept {
    switch (port_kind) {
        case OperatorPortKind::kActivation:
            return value_kind == ExecutionValueKind::kActivation;
        case OperatorPortKind::kState:
            return value_kind == ExecutionValueKind::kState;
        case OperatorPortKind::kModelInput:
        case OperatorPortKind::kWeight:
        case OperatorPortKind::kConstant:
            return false;
    }
    return false;
}

Status ValidateValueId(const std::vector<ExecutionValueDesc>& values,
                       ExecutionValueId id,
                       std::string_view context) {
    if (id.index >= values.size()) {
        return Status::InvalidArgument(
                "ExecutionPlan " + std::string(context) +
                " references value " + std::to_string(id.index) +
                " beyond the value table");
    }
    return Status::Ok();
}

template<typename Port>
std::vector<uint32_t> ExpectedKernelPorts(std::span<const Port> ports) {
    std::vector<uint32_t> indices;
    indices.reserve(ports.size());
    for (size_t i = 0; i < ports.size(); ++i) {
        if constexpr (std::is_same_v<Port, OperatorInputPort>) {
            if (ports[i].kind != OperatorPortKind::kState) {
                indices.push_back(static_cast<uint32_t>(i));
            }
        } else if (ports[i].kind != OperatorPortKind::kState) {
            indices.push_back(static_cast<uint32_t>(i));
        }
    }
    return indices;
}

}// namespace

StatusOr<ExecutionPlan> ExecutionPlan::Create(std::vector<ExecutionValueDesc> values,
                                              std::vector<ExecutionValueId> model_inputs,
                                              std::vector<ExecutionValueId> model_outputs,
                                              std::vector<ExecutionStep> steps,
                                              StateAliasPlan state_alias_plan,
                                              WorkspacePlanLayout workspace_layout) {
    ExecutionPlan plan;
    plan.values_ = std::move(values);
    plan.model_inputs_ = std::move(model_inputs);
    plan.model_outputs_ = std::move(model_outputs);
    plan.binding_key_ = MakeBindingKey();
    plan.workspace_layout_ = workspace_layout;
    plan.steps_.reserve(steps.size());

    std::vector<bool> seen_model_inputs(plan.values_.size());
    for (const auto id: plan.model_inputs_) {
        AM_RETURN_IF_ERROR(ValidateValueId(plan.values_, id, "model input"));
        if (seen_model_inputs[id.index]) {
            return Status::InvalidArgument(
                    "ExecutionPlan model_inputs contains a duplicate value");
        }

        seen_model_inputs[id.index] = true;
        if (plan.values_[id.index].kind != ExecutionValueKind::kModelInput) {
            return Status::InvalidArgument(
                    "ExecutionPlan model input does not have kModelInput kind");
        }
    }

    std::vector<bool> seen_model_outputs(plan.values_.size());
    for (const ExecutionValueId id: plan.model_outputs_) {
        AM_RETURN_IF_ERROR(ValidateValueId(plan.values_, id, "model output"));
        if (seen_model_outputs[id.index]) {
            return Status::InvalidArgument(
                    "ExecutionPlan model_outputs contains a duplicate value");
        }
        seen_model_outputs[id.index] = true;
    }

    for (auto& step: steps) {
        AM_RETURN_IF_ERROR(plan.AddStep(std::move(step)));
    }

    std::vector<std::optional<size_t>> activation_producers(plan.values_.size());
    for (size_t i = 0; i < plan.steps_.size(); ++i) {
        const auto& step = plan.steps_[i];
        for (const auto [index]: step.outputs) {
            if (plan.values_[index].kind != ExecutionValueKind::kActivation) {
                continue;
            }

            if (activation_producers[index].has_value()) {
                return Status::InvalidArgument(
                        "ExecutionPlan activation value has more than one producer");
            }
            activation_producers[index] = i;
        }

        for (const auto [index]: step.inputs) {
            if (plan.values_[index].kind != ExecutionValueKind::kActivation) {
                continue;
            }

            if (const auto producer = activation_producers[index];
                !producer.has_value() || *producer >= i) {
                return Status::InvalidArgument(
                        "ExecutionPlan activation input must be produced by an earlier step");
            }
        }
    }

    for (size_t i = 0; i < plan.values_.size(); ++i) {
        const auto kind = plan.values_[i].kind;
        if (kind == ExecutionValueKind::kModelInput && !seen_model_inputs[i]) {
            return Status::InvalidArgument(
                    "ExecutionPlan kModelInput value is missing from model_inputs");
        }

        if (kind == ExecutionValueKind::kActivation && !activation_producers[i].has_value()) {
            return Status::InvalidArgument(
                    "ExecutionPlan activation value has no producer");
        }
    }

    for (const auto& [i, input_port, output_port]:
         state_alias_plan.aliases) {
        if (i >= plan.steps_.size()) {
            return Status::InvalidArgument(
                    "State alias references step " + std::to_string(i) +
                    " beyond the plan's step count");
        }

        const auto& step = plan.steps_[i];
        if (input_port >= step.inputs.size() || output_port >= step.outputs.size()) {
            return Status::InvalidArgument(
                    "State alias references a semantic port beyond its execution step");
        }

        if (plan.values_[step.inputs[input_port].index].kind != ExecutionValueKind::kState ||
            plan.values_[step.outputs[output_port].index].kind != ExecutionValueKind::kState) {
            return Status::InvalidArgument(
                    "State alias must reference kState execution values");
        }
    }

    std::ranges::stable_sort(
            state_alias_plan.aliases,
            [](const ResolvedStateAlias& lhs, const ResolvedStateAlias& rhs) noexcept {
                return lhs.step_index < rhs.step_index;
            });
    plan.state_alias_plan_ = std::move(state_alias_plan);
    return plan;
}

Status ExecutionPlan::AddStep(ExecutionStep step) {
    if (step.kernel.op_type == OpType::kUnknown) {
        return Status::InvalidArgument(
                "Execution step kernel op_type cannot be kUnknown");
    }

    if (step.kernel.fn == nullptr) {
        return Status::InvalidArgument(
                "Execution step kernel function cannot be null");
    }

    if (step.kernel.params_builder == nullptr && step.kernel.params_size != 0) {
        return Status::InvalidArgument(
                "Execution step kernel params_size must be zero without a params builder");
    }

    if (step.kernel.params_builder != nullptr &&
        (step.kernel.params_size == 0 || step.kernel.params_size > kMaxKernelParamsSize)) {
        return Status::InvalidArgument(
                "Execution step kernel params_size is outside the supported range");
    }

    if (!IsValidWorkspaceAlignment(step.workspace_requirement.alignment)) {
        return Status::InvalidArgument(
                "Execution step workspace alignment must be a non-zero power of two");
    }

    if (step.kernel.workspace_requirement.bytes != step.workspace_requirement.bytes ||
        step.kernel.workspace_requirement.alignment != step.workspace_requirement.alignment ||
        step.kernel.workspace_requirement.lifetime != step.workspace_requirement.lifetime ||
        step.kernel.workspace_requirement.reusable != step.workspace_requirement.reusable ||
        step.kernel.workspace_requirement.offset != step.workspace_requirement.offset) {
        return Status::InvalidArgument(
                "Execution step workspace requirement must match its prepared kernel");
    }

    const auto schema = GetOperatorSchema(step.kernel.op_type);
    if (!schema.ok()) {
        return Status::InvalidArgument(
                "Execution step has no registered operator schema");
    }

    if (step.inputs.size() != schema->input_ports.size() ||
        step.outputs.size() != schema->output_ports.size()) {
        return Status::InvalidArgument(
                "Execution step semantic operand arity differs from its operator schema");
    }

    const auto expected_input_ports =
            ExpectedKernelPorts<OperatorInputPort>(schema->input_ports);
    const auto expected_output_ports =
            ExpectedKernelPorts<OperatorOutputPort>(schema->output_ports);
    if (step.kernel_input_ports != expected_input_ports ||
        step.kernel_output_ports != expected_output_ports) {
        return Status::InvalidArgument(
                "Execution step kernel ports do not match its semantic schema");
    }

    for (size_t i = 0; i < step.inputs.size(); ++i) {
        AM_RETURN_IF_ERROR(ValidateValueId(values_, step.inputs[i], "step input"));
        if (!MatchesInputKind(values_[step.inputs[i].index].kind,
                              schema->input_ports[i].kind)) {
            return Status::InvalidArgument(
                    "Execution step input value kind does not match its semantic port");
        }
    }

    for (size_t i = 0; i < step.outputs.size(); ++i) {
        AM_RETURN_IF_ERROR(ValidateValueId(values_, step.outputs[i], "step output"));
        if (!MatchesOutputKind(values_[step.outputs[i].index].kind,
                               schema->output_ports[i].kind)) {
            return Status::InvalidArgument(
                    "Execution step output value kind does not match its semantic port");
        }
    }

    std::vector<TensorSpec> compact_input_specs;
    compact_input_specs.reserve(step.kernel_input_ports.size());
    for (uint32_t port: step.kernel_input_ports) {
        compact_input_specs.push_back(values_[step.inputs[port].index].spec);
    }
    std::vector<TensorSpec> compact_output_specs;
    compact_output_specs.reserve(step.kernel_output_ports.size());
    for (uint32_t port: step.kernel_output_ports) {
        compact_output_specs.push_back(values_[step.outputs[port].index].spec);
    }

    for (const auto& check: step.runtime_checks) {
        AM_RETURN_IF_ERROR(ValidateRuntimeCheckReferences(
                check, compact_input_specs, compact_output_specs));
    }

    steps_.push_back(std::move(step));
    return Status::Ok();
}
}// namespace aethermind
