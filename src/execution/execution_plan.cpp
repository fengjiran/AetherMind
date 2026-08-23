#include "aethermind/execution/execution_plan.h"

#include "utils/variant_utils.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>

namespace aethermind {
namespace {

/// @brief Validates that a runtime shape check's port references fall within
///        the step's compact input/output spec ranges.
///
/// Runtime checks are evaluated against the runtime tensor bindings in the
/// hot path, where malformed references are rejected without aborting. This
/// plan-build check rejects them up front so Execute never surfaces them.
Status ValidateRuntimeCheckReferences(const ShapeConstraint& check,
                                      const std::vector<TensorSpec>& input_specs,
                                      const std::vector<TensorSpec>& output_specs) {
    const auto validate_port = [&](const TensorPort& port) -> Status {
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

    const auto validate_dim = [&](const DimLocator& locator) -> Status {
        AM_RETURN_IF_ERROR(validate_port(locator.tensor_port));
        const SymbolicShape& shape =
                locator.tensor_port.direction == TensorPortType::kInput
                        ? input_specs[locator.tensor_port.tensor_idx].shape
                        : output_specs[locator.tensor_port.tensor_idx].shape;
        const auto rank = shape.rank();
        if (rank.has_value() && locator.dim_index >= *rank) {
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
            },
    };
    return std::visit(visitor, check.condition);
}

}// namespace

StatusOr<ExecutionPlan> ExecutionPlan::Create(std::vector<ExecutionStep> steps,
                                              StateAliasPlan state_alias_plan,
                                              WorkspacePlanLayout workspace_layout) {
    ExecutionPlan plan;
    plan.workspace_layout_ = workspace_layout;
    plan.steps_.reserve(steps.size());

    for (ExecutionStep& step: steps) {
        AM_RETURN_IF_ERROR(plan.AddStep(std::move(step)));
    }

    // StateAliasPlan is a public data struct; converge its invariants here:
    // aliases must reference existing steps and be sorted by step_index so
    // ForStep's binary search is well-defined.
    for (const ResolvedStateAlias& alias: state_alias_plan.aliases) {
        if (alias.step_index >= plan.steps_.size()) {
            return Status::InvalidArgument(
                    "State alias references step " +
                    std::to_string(alias.step_index) +
                    " beyond the plan's step count");
        }
    }
    std::ranges::sort(state_alias_plan.aliases,
                      [](const ResolvedStateAlias& lhs,
                         const ResolvedStateAlias& rhs) noexcept {
                          return std::tie(lhs.step_index, lhs.input_port, lhs.output_port) <
                                 std::tie(rhs.step_index, rhs.input_port, rhs.output_port);
                      });
    plan.state_alias_plan_ = std::move(state_alias_plan);

    return plan;
}

Status ExecutionPlan::AddStep(ExecutionStep step) {
    if (step.kernel.op_type == OpType::kUnknown) {
        return Status::InvalidArgument("Execution step kernel op_type cannot be kUnknown");
    }
    if (step.kernel.fn == nullptr) {
        return Status::InvalidArgument("Execution step kernel function cannot be null");
    }
    if (step.kernel.params_builder == nullptr && step.kernel.params_size != 0) {
        return Status::InvalidArgument(
                "Execution step kernel params_size must be zero without a params builder");
    }
    if (step.kernel.params_builder != nullptr &&
        (step.kernel.params_size == 0 || step.kernel.params_size > kMaxKernelParamsSize)) {
        return Status::InvalidArgument("Execution step kernel params_size is outside the supported range");
    }
    if (!IsValidWorkspaceAlignment(step.workspace_requirement.alignment)) {
        return Status::InvalidArgument("Execution step workspace alignment must be a non-zero power of two");
    }
    if (step.kernel.workspace_requirement.bytes != step.workspace_requirement.bytes ||
        step.kernel.workspace_requirement.alignment != step.workspace_requirement.alignment ||
        step.kernel.workspace_requirement.lifetime != step.workspace_requirement.lifetime ||
        step.kernel.workspace_requirement.reusable != step.workspace_requirement.reusable ||
        step.kernel.workspace_requirement.offset != step.workspace_requirement.offset) {
        return Status::InvalidArgument(
                "Execution step workspace requirement must match its prepared kernel");
    }
    for (const ShapeConstraint& check: step.runtime_checks) {
        AM_RETURN_IF_ERROR(ValidateRuntimeCheckReferences(check,
                                                          step.input_specs,
                                                          step.output_specs));
    }

    steps_.push_back(std::move(step));
    return Status::Ok();
}

const std::vector<ExecutionStep>& ExecutionPlan::steps() const noexcept {
    return steps_;
}

size_t ExecutionPlan::size() const noexcept {
    return steps_.size();
}

const StateAliasPlan& ExecutionPlan::state_alias_plan() const noexcept {
    return state_alias_plan_;
}

size_t ExecutionPlan::total_workspace_bytes() const noexcept {
    return workspace_layout_.total_bytes;
}

size_t ExecutionPlan::workspace_alignment() const noexcept {
    return workspace_layout_.required_alignment;
}

}// namespace aethermind
