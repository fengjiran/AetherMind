#include "aethermind/execution/execution_plan.h"

#include <cstring>

namespace aethermind {

Status ExecutionPlan::AddStep(const ExecutionStep& step) {
    if (step.fn == nullptr) {
        return Status::InvalidArgument("Execution step kernel function cannot be null");
    }
    if (!IsValidWorkspaceAlignment(step.workspace_requirement.alignment)) {
        return Status::InvalidArgument("Execution step workspace alignment must be a non-zero power of two");
    }

    ExecutionStep stored_step = step;
    if (step.attrs.empty()) {
        stored_step.attrs = {};
    } else {
        auto& owned_attrs = owned_attrs_.emplace_back(step.attrs.begin(), step.attrs.end());
        stored_step.attrs = std::span<const std::byte>(owned_attrs.data(), owned_attrs.size());
    }

    steps_.push_back(stored_step);
    return Status::Ok();
}

Status ExecutionPlan::AddStep(const ResolvedKernel& kernel,
                              const WorkspaceRequirement& workspace_requirement,
                              const void* packed_params) {
    return AddStep(ExecutionStep{
            .op_type = kernel.op_type,
            .fn = kernel.fn,
            .packed_params = packed_params,
            .workspace_requirement = workspace_requirement,
            .attrs = kernel.attrs,
            .debug_name = kernel.debug_name,
    });
}

const std::vector<ExecutionStep>& ExecutionPlan::steps() const noexcept {
    return steps_;
}

size_t ExecutionPlan::size() const noexcept {
    return steps_.size();
}

}// namespace aethermind
