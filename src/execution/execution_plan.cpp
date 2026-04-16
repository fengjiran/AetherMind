#include "aethermind/execution/execution_plan.h"

#include <cstring>

namespace aethermind {

Status ExecutionPlan::AddStep(const OperatorName& op_name,
                              const ResolvedKernel& kernel) {
    if (kernel.fn == nullptr) {
        return Status::InvalidArgument("Resolved kernel function cannot be null");
    }

    ResolvedKernel stored_kernel = kernel;
    if (kernel.attrs.empty()) {
        stored_kernel.attrs = {};
    } else {
        auto& owned_attrs = owned_attrs_.emplace_back(kernel.attrs.begin(), kernel.attrs.end());
        stored_kernel.attrs = std::span<const std::byte>(owned_attrs.data(), owned_attrs.size());
    }

    steps_.push_back(ExecutionStep{
            .op_name = op_name,
            .kernel = stored_kernel,
    });
    return Status::Ok();
}

const std::vector<ExecutionStep>& ExecutionPlan::steps() const noexcept {
    return steps_;
}

size_t ExecutionPlan::size() const noexcept {
    return steps_.size();
}

}// namespace aethermind
