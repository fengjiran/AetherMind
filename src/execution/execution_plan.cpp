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
