#include "aethermind/execution/execution_plan.h"

#include <utility>

namespace aethermind {

StatusOr<ExecutionPlan> ExecutionPlan::Create(std::vector<ExecutionStep> steps,
                                              StateAliasPlan state_alias_plan) {
    ExecutionPlan plan;
    plan.state_alias_plan_ = std::move(state_alias_plan);
    plan.steps_.reserve(steps.size());

    for (ExecutionStep& step: steps) {
        AM_RETURN_IF_ERROR(plan.AddStep(std::move(step)));
    }

    return plan;
}

Status ExecutionPlan::AddStep(ExecutionStep step) {
    if (step.op == nullptr) {
        return Status::InvalidArgument("Execution step operator cannot be null");
    }
    if (step.op->GetResolvedKernel().fn == nullptr) {
        return Status::InvalidArgument("Execution step operator resolved kernel function cannot be null");
    }
    if (!IsValidWorkspaceAlignment(step.workspace_requirement.alignment)) {
        return Status::InvalidArgument("Execution step workspace alignment must be a non-zero power of two");
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

}// namespace aethermind
