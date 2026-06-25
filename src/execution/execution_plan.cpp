#include "aethermind/execution/execution_plan.h"

namespace aethermind {

Status ExecutionPlan::AddStep(const ExecutionStep& step) {
    if (step.op == nullptr) {
        return Status::InvalidArgument("Execution step operator cannot be null");
    }
    if (step.op->GetResolvedKernel().fn == nullptr) {
        return Status::InvalidArgument("Execution step operator resolved kernel function cannot be null");
    }
    if (!IsValidWorkspaceAlignment(step.workspace_requirement.alignment)) {
        return Status::InvalidArgument("Execution step workspace alignment must be a non-zero power of two");
    }

    steps_.push_back(step);
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
